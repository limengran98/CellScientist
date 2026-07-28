from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .artifacts import atomic_write_json, write_jsonl
from .controllers import build_controller
from .data import TaskData, load_task_data
from .evaluator import evaluate_predictions, objective_value
from .hrt import build_hrt_record
from .llm_client import OpenAICompatiblePolicy
from .models import ModelCache
from .protocol import (
    canonical_json,
    load_config,
    sha256_text,
    source_fingerprint,
    task_specs,
    verify_lock,
)
from .schemas import (
    CandidateConfig,
    ContractError,
    Proposal,
    TaskSpec,
    TrialObservation,
)
from .search_space import CandidateSpace


def _sha256_optional(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _train_target_mean(data: TaskData) -> np.ndarray:
    train = np.asarray(data.partitions.train, dtype=np.int64)
    target = np.asarray(data.morphology_post[train], dtype=np.float32).copy()
    target[~np.isfinite(target)] = np.nan
    mean = np.nanmean(target, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    return mean.astype(np.float32)


def _indices(data: TaskData, partition: str) -> np.ndarray:
    values = getattr(data.partitions, partition)
    return np.asarray(values, dtype=np.int64)


def _best_feedback_observation(
    history: Sequence[TrialObservation],
    primary_metric: str,
) -> TrialObservation:
    return max(
        history,
        key=lambda observation: objective_value(
            observation.feedback_metrics,
            primary_metric,
        ),
    )


def _retention_objective(
    observation: TrialObservation,
    selection_scores: Mapping[str, Mapping[str, float]],
    primary_metric: str,
    selection_rule: str,
) -> float:
    selection_value = objective_value(
        dict(selection_scores[observation.candidate.candidate_id]),
        primary_metric,
    )
    if selection_rule == "selection_only":
        return selection_value
    if selection_rule == "feedback_selection_mean":
        feedback_value = objective_value(
            observation.feedback_metrics,
            primary_metric,
        )
        return 0.5 * (feedback_value + selection_value)
    raise ContractError(f"Unknown search.selection_rule: {selection_rule}")


def _evaluate_partition(
    data: TaskData,
    model: Any,
    partition: str,
    train_mean: np.ndarray,
) -> Dict[str, float]:
    index = _indices(data, partition)
    predictions = model.predict(data, index)
    return evaluate_predictions(
        y_true=data.morphology_post[index],
        y_pred=predictions,
        morphology_pre=data.morphology_pre[index],
        train_target_mean=train_mean,
    )


def _runtime_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    try:
        import sklearn

        snapshot["scikit_learn"] = sklearn.__version__
    except Exception:
        snapshot["scikit_learn"] = "unavailable"
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snapshot["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        snapshot["torch"] = "unavailable"
        snapshot["cuda_available"] = False
    return snapshot


class ControlledRun:
    def __init__(
        self,
        root: Path,
        config: Mapping[str, Any],
        protocol_hash: str,
        data: TaskData,
        controller_name: str,
        initialization: str,
        seed: int,
        output_dir: Path,
        source_bundle_sha256: str,
    ) -> None:
        self.root = root
        self.config = config
        self.protocol_hash = protocol_hash
        self.data = data
        self.controller_name = controller_name
        self.initialization = initialization
        self.seed = int(seed)
        self.output_dir = output_dir
        self.source_bundle_sha256 = source_bundle_sha256
        self.space = CandidateSpace()
        cache_root = root / str(config["outputs"]["cache"])
        self.cache = ModelCache(cache_root, protocol_hash)
        self.primary_metric = str(config["search"]["primary_metric"])
        self.selection_rule = str(
            config["search"].get("selection_rule", "selection_only")
        )
        if self.selection_rule not in {
            "selection_only",
            "feedback_selection_mean",
        }:
            raise ContractError(
                f"Unknown search.selection_rule: {self.selection_rule}"
            )
        self.model_seed = int(config["model"]["model_seed_offset"]) + self.seed
        self.train_mean = _train_target_mean(data)
        self.llm_policy = OpenAICompatiblePolicy(config["llm"])
        if (
            controller_name == "cellscientist"
            and self.llm_policy.enabled
            and bool(config["llm"].get("require_credential", False))
            and not os.environ.get(str(config["llm"]["api_key_env"]))
        ):
            raise ContractError(
                "Formal LLM execution requires credential environment variable "
                f"{config['llm']['api_key_env']}"
            )
        self.raw_llm_rows: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []

    def _record_proposal_trace(
        self,
        candidate: CandidateConfig,
        step: int,
        proposal: Optional[Proposal],
    ) -> Optional[Dict[str, Any]]:
        if proposal is None or (
            proposal.prompt_text is None
            and proposal.response_text is None
            and not proposal.llm_http_attempts
        ):
            return None
        usage = proposal.llm_usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        row = {
            "step": step,
            "controller": self.controller_name,
            "candidate_id": candidate.candidate_id,
            "address": proposal.address,
            "source": proposal.source,
            "operation": proposal.operation,
            "parent_candidate_id": proposal.parent_candidate_id,
            "fallback_used": proposal.fallback_used,
            "valid_decision": not proposal.fallback_used,
            "executed": False,
            "prompt": proposal.prompt_text,
            "response": proposal.response_text,
            "prompt_sha256": _sha256_optional(proposal.prompt_text),
            "response_sha256": _sha256_optional(proposal.response_text),
            "usage": usage,
            "usage_reported": proposal.llm_usage is not None,
            "http_attempts": [
                dict(attempt) for attempt in proposal.llm_http_attempts
            ],
            "error": proposal.llm_error,
        }
        self.raw_llm_rows.append(row)
        return row

    def _set_transition_provenance(
        self,
        observation: TrialObservation,
        parent: CandidateConfig,
        proposal: Proposal,
    ) -> None:
        try:
            changed_address = self.space.changed_address(
                parent,
                observation.candidate,
            )
        except ContractError:
            observation.changed_address = None
            observation.transition_admissible = False
            return
        observation.changed_address = changed_address
        if proposal.source.startswith("runner_atomic_recovery:"):
            observation.transition_admissible = proposal.address == changed_address
        elif self.controller_name == "cellscientist":
            observation.transition_admissible = proposal.address == changed_address
        else:
            observation.transition_admissible = False

    def evaluate_feedback(
        self,
        candidate: CandidateConfig,
        step: int,
        proposal: Optional[Proposal] = None,
        parent_candidate_id: Optional[str] = None,
        record_proposal_trace: bool = True,
    ) -> TrialObservation:
        trace_row = (
            self._record_proposal_trace(candidate, step, proposal)
            if record_proposal_trace
            else None
        )
        started = time.monotonic()
        model, cache_hit, fit_seconds = self.cache.fit_or_load(
            self.data,
            candidate,
            self.config["model"],
            self.model_seed,
        )
        metrics = _evaluate_partition(
            self.data,
            model,
            "feedback",
            self.train_mean,
        )
        actual_elapsed = time.monotonic() - started
        evaluation_seconds = max(0.0, actual_elapsed - (0.0 if cache_hit else fit_seconds))
        accounted_elapsed = fit_seconds + evaluation_seconds
        if trace_row is not None:
            trace_row["executed"] = True
        return TrialObservation(
            step=step,
            candidate=candidate,
            feedback_metrics=metrics,
            route_address=None if proposal is None else proposal.address,
            proposal_source="initial_h0" if proposal is None else proposal.source,
            elapsed_seconds=accounted_elapsed,
            cache_hit=cache_hit,
            actual_evaluation_seconds=actual_elapsed,
            fallback_used=False if proposal is None else proposal.fallback_used,
            prompt_sha256=None if proposal is None else _sha256_optional(proposal.prompt_text),
            response_sha256=None
            if proposal is None
            else _sha256_optional(proposal.response_text),
            parent_candidate_id=parent_candidate_id,
        )

    def _search(self) -> List[TrialObservation]:
        search_started = time.monotonic()
        budget = int(self.config["search"]["budget"])
        h0 = self.space.initial(self.initialization)
        controller = build_controller(
            self.controller_name,
            self.space,
            self.seed,
            self.primary_metric,
            self.llm_policy,
        )
        initial_observation = self.evaluate_feedback(h0, 1)
        initial_observation.cumulative_seconds = initial_observation.elapsed_seconds
        history = [initial_observation]
        seen = {h0.candidate_id}
        for step in range(2, budget + 1):
            step_started = time.monotonic()
            try:
                default_parent_candidate_id = _best_feedback_observation(
                    history,
                    self.primary_metric,
                ).candidate.candidate_id
                proposal = controller.propose(history, seen, step)
                transition_parent_candidate_id = (
                    proposal.parent_candidate_id
                    if proposal.parent_candidate_id is not None
                    else default_parent_candidate_id
                )
                parent_candidate = self.space.by_id(transition_parent_candidate_id)
                trace_row = self._record_proposal_trace(
                    proposal.candidate,
                    step,
                    proposal,
                )
                if proposal.candidate.candidate_id in seen:
                    raise ContractError(
                        f"Controller repeated candidate: {proposal.candidate.candidate_id}"
                    )
                observation = self.evaluate_feedback(
                    proposal.candidate,
                    step,
                    proposal,
                    parent_candidate_id=transition_parent_candidate_id,
                    record_proposal_trace=False,
                )
                if trace_row is not None:
                    trace_row["executed"] = True
                self._set_transition_provenance(
                    observation,
                    parent_candidate,
                    proposal,
                )
                observation.proposal_seconds = max(
                    0.0,
                    time.monotonic()
                    - step_started
                    - observation.actual_evaluation_seconds,
                )
                observation.cumulative_seconds = (
                    history[-1].cumulative_seconds
                    + observation.proposal_seconds
                    + observation.elapsed_seconds
                )
                history.append(observation)
                seen.add(proposal.candidate.candidate_id)
            except Exception as exc:
                self.failures.append(
                    {
                        "step": step,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                parent_observation = _best_feedback_observation(
                    history,
                    self.primary_metric,
                )
                remaining = [
                    candidate
                    for candidate in self.space.all_neighbors(
                        parent_observation.candidate
                    )
                    if candidate.candidate_id not in seen
                ]
                if not remaining:
                    break
                fallback = remaining[(self.seed + step) % len(remaining)]
                fallback_address = self.space.changed_address(
                    parent_observation.candidate,
                    fallback,
                )
                proposal = Proposal(
                    candidate=fallback,
                    address=fallback_address,
                    source=f"runner_atomic_recovery:{type(exc).__name__}",
                    fallback_used=True,
                    llm_error=str(exc),
                )
                observation = self.evaluate_feedback(
                    fallback,
                    step,
                    proposal,
                    parent_candidate_id=parent_observation.candidate.candidate_id,
                )
                self._set_transition_provenance(
                    observation,
                    parent_observation.candidate,
                    proposal,
                )
                observation.proposal_seconds = max(
                    0.0,
                    time.monotonic()
                    - step_started
                    - observation.actual_evaluation_seconds,
                )
                observation.cumulative_seconds = (
                    history[-1].cumulative_seconds
                    + observation.proposal_seconds
                    + observation.elapsed_seconds
                )
                history.append(observation)
                seen.add(fallback.candidate_id)
        return history

    def _score_candidates(
        self,
        history: Sequence[TrialObservation],
        partition: str,
    ) -> Dict[str, Dict[str, float]]:
        scores: Dict[str, Dict[str, float]] = {}
        for observation in history:
            model, _, _ = self.cache.fit_or_load(
                self.data,
                observation.candidate,
                self.config["model"],
                self.model_seed,
            )
            scores[observation.candidate.candidate_id] = _evaluate_partition(
                self.data,
                model,
                partition,
                self.train_mean,
            )
        return scores

    def execute(self) -> Dict[str, Any]:
        started = time.monotonic()
        history = self._search()
        selection_scores = self._score_candidates(history, "selection")
        selected = max(
            history,
            key=lambda observation: _retention_objective(
                observation,
                selection_scores,
                self.primary_metric,
                self.selection_rule,
            ),
        )
        model, _, _ = self.cache.fit_or_load(
            self.data,
            selected.candidate,
            self.config["model"],
            self.model_seed,
        )
        test_metrics = _evaluate_partition(self.data, model, "test", self.train_mean)
        feedback_best = max(
            history,
            key=lambda observation: objective_value(
                observation.feedback_metrics,
                self.primary_metric,
            ),
        )

        checkpoints: Dict[str, Any] = {}
        for raw_budget in self.config["search"]["budget_checkpoints"]:
            budget = min(int(raw_budget), len(history))
            candidates = history[:budget]
            winner = max(
                candidates,
                key=lambda observation: _retention_objective(
                    observation,
                    selection_scores,
                    self.primary_metric,
                    self.selection_rule,
                ),
            )
            checkpoint_model, _, _ = self.cache.fit_or_load(
                self.data,
                winner.candidate,
                self.config["model"],
                self.model_seed,
            )
            checkpoints[str(raw_budget)] = {
                "actual_budget": budget,
                "candidate_id": winner.candidate.candidate_id,
                "selection_metrics": selection_scores[winner.candidate.candidate_id],
                "test_metrics": _evaluate_partition(
                    self.data,
                    checkpoint_model,
                    "test",
                    self.train_mean,
                ),
            }
        wallclock_checkpoints: Dict[str, Any] = {}
        for raw_seconds in self.config["search"].get(
            "wallclock_checkpoints_seconds", []
        ):
            seconds = float(raw_seconds)
            eligible = [
                observation
                for observation in history
                if observation.cumulative_seconds <= seconds
            ]
            if not eligible:
                eligible = [history[0]]
            winner = max(
                eligible,
                key=lambda observation: _retention_objective(
                    observation,
                    selection_scores,
                    self.primary_metric,
                    self.selection_rule,
                ),
            )
            checkpoint_model, _, _ = self.cache.fit_or_load(
                self.data,
                winner.candidate,
                self.config["model"],
                self.model_seed,
            )
            wallclock_checkpoints[str(raw_seconds)] = {
                "wallclock_limit_seconds": seconds,
                "actual_budget": len(eligible),
                "candidate_id": winner.candidate.candidate_id,
                "selection_metrics": selection_scores[
                    winner.candidate.candidate_id
                ],
                "test_metrics": _evaluate_partition(
                    self.data,
                    checkpoint_model,
                    "test",
                    self.train_mean,
                ),
            }

        http_attempts = [
            attempt
            for row in self.raw_llm_rows
            for attempt in row.get("http_attempts", [])
        ]
        llm_request_steps = len(self.raw_llm_rows)
        llm_http_attempts = len(http_attempts)
        llm_http_successes = sum(
            bool(attempt.get("http_success")) for attempt in http_attempts
        )
        llm_valid_decisions = sum(
            bool(row.get("valid_decision")) for row in self.raw_llm_rows
        )
        llm_usage = {
            key: int(
                sum(
                    int((attempt.get("provider_usage") or {}).get(key, 0))
                    for attempt in http_attempts
                )
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        llm_missing_usage_successes = sum(
            bool(attempt.get("http_success"))
            and not bool(attempt.get("usage_reported"))
            for attempt in http_attempts
        )
        transition_rows = history[1:]
        result = {
            "schema_version": 2,
            "protocol_id": self.config["protocol_id"],
            "protocol_sha256": self.protocol_hash,
            "source_bundle_sha256": self.source_bundle_sha256,
            "mode": self.config["mode"],
            "task": self.data.partition_summary,
            "controller": self.controller_name,
            "initialization": self.initialization,
            "seed": self.seed,
            "model_seed": self.model_seed,
            "primary_metric": self.primary_metric,
            "selection_rule": self.selection_rule,
            "configured_budget": int(self.config["search"]["budget"]),
            "evaluated_candidates": len(history),
            "llm_request_steps": llm_request_steps,
            "llm_http_attempts": llm_http_attempts,
            "llm_http_successes": llm_http_successes,
            "llm_valid_decisions": llm_valid_decisions,
            "llm_attempts": llm_http_attempts,
            "llm_calls": llm_http_successes,
            "llm_fallbacks": int(
                sum(bool(row["fallback_used"]) for row in self.raw_llm_rows)
            ),
            "llm_usage": llm_usage,
            "llm_missing_usage_successes": llm_missing_usage_successes,
            "llm_http_latency_seconds": float(
                sum(
                    float(attempt.get("latency_seconds") or 0.0)
                    for attempt in http_attempts
                )
            ),
            "raw_llm_valid_rate": (
                float(llm_valid_decisions / llm_request_steps)
                if llm_request_steps
                else None
            ),
            "executed_candidate_compliance_rate": (
                float(
                    sum(
                        bool(observation.transition_admissible)
                        for observation in transition_rows
                    )
                    / len(transition_rows)
                )
                if transition_rows
                else 1.0
            ),
            "selected_candidate": selected.candidate.to_dict(),
            "selected_candidate_id": selected.candidate.candidate_id,
            "selected_candidate_step": selected.step,
            "selected_candidate_accounted_seconds": selected.cumulative_seconds,
            "feedback_best_step": feedback_best.step,
            "feedback_best_accounted_seconds": feedback_best.cumulative_seconds,
            "selection_metrics": selection_scores[selected.candidate.candidate_id],
            "test_metrics": test_metrics,
            "selection_to_test_gap": float(
                selection_scores[selected.candidate.candidate_id][
                    self.primary_metric
                ]
                - test_metrics[self.primary_metric]
            ),
            "budget_checkpoints": checkpoints,
            "wallclock_checkpoints": wallclock_checkpoints,
            "hrt": build_hrt_record(
                self.data,
                history,
                self.protocol_hash,
                self.config["data"],
            ),
            "trajectory": [observation.to_dict() for observation in history],
            "failures": self.failures,
            "elapsed_seconds": time.monotonic() - started,
            "runtime": _runtime_snapshot(),
        }
        return result


def _protocol_hash(
    config: Mapping[str, Any],
    lock: Optional[Mapping[str, Any]],
) -> str:
    if lock is not None:
        return str(lock["lock_sha256"])
    return sha256_text(canonical_json(config))


def _find_task(config: Mapping[str, Any], task_id: str) -> TaskSpec:
    for spec in task_specs(config):
        if spec.task_id == task_id:
            return spec
    raise ContractError(f"Unknown task ID: {task_id}")


def run_one(
    root: Path,
    config_path: Path,
    task_id: str,
    controller: str,
    initialization: str,
    seed: int,
    lock_path: Optional[Path] = None,
    overwrite_development: bool = False,
    verified_lock: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if controller != "cellscientist":
        raise ContractError("This public release contains only the CellScientist controller")
    config = load_config(config_path)
    lock: Optional[Mapping[str, Any]] = None
    if config["mode"] == "formal":
        if verified_lock is not None:
            lock = verified_lock
        elif lock_path is None:
            raise ContractError("Formal execution requires --lock")
        else:
            lock = verify_lock(config_path, lock_path, root)
    protocol_hash = _protocol_hash(config, lock)
    spec = _find_task(config, task_id)
    output_dir = (
        root
        / str(config["outputs"]["root"])
        / task_id
        / initialization
        / controller
        / f"seed-{seed}"
    )
    result_path = output_dir / "run_result.json"
    if result_path.exists():
        if config["mode"] == "formal":
            with result_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            expected = {
                "protocol_sha256": protocol_hash,
                "controller": controller,
                "initialization": initialization,
                "seed": int(seed),
            }
            observed = {key: existing.get(key) for key in expected}
            if observed != expected:
                raise ContractError(
                    "Existing formal result does not match the frozen run key: "
                    f"{result_path}"
                )
            return existing
        if not overwrite_development:
            raise ContractError(f"Refusing to overwrite an existing run: {result_path}")
    data = load_task_data(spec, config["data"], config["search"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = source_fingerprint(root)
    source_bundle_sha256 = sha256_text(canonical_json(sources))
    manifest = {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": protocol_hash,
        "source_bundle_sha256": source_bundle_sha256,
        "sources": sources,
        "task_id": task_id,
        "controller": controller,
        "initialization": initialization,
        "seed": int(seed),
        "started_at_unix": time.time(),
        "runtime": _runtime_snapshot(),
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    run = ControlledRun(
        root=root,
        config=config,
        protocol_hash=protocol_hash,
        data=data,
        controller_name=controller,
        initialization=initialization,
        seed=seed,
        output_dir=output_dir,
        source_bundle_sha256=source_bundle_sha256,
    )
    try:
        result = run.execute()
    except Exception as exc:
        failure = {
            **manifest,
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(output_dir / "run_failure.json", failure)
        raise
    if run.raw_llm_rows:
        write_jsonl(output_dir / "llm_trace.jsonl", run.raw_llm_rows)
    atomic_write_json(result_path, result)
    return result


def run_matrix(
    root: Path,
    config_path: Path,
    lock_path: Optional[Path] = None,
    matrix_name: str = "primary",
    task_filter: Optional[Sequence[str]] = None,
    controller_filter: Optional[Sequence[str]] = None,
    initialization_filter: Optional[Sequence[str]] = None,
    seed_filter: Optional[Sequence[int]] = None,
    overwrite_development: bool = False,
    jobs: int = 1,
) -> List[Dict[str, Any]]:
    config = load_config(config_path)
    if jobs < 1:
        raise ContractError("jobs must be a positive integer")
    verified_lock: Optional[Mapping[str, Any]] = None
    if config["mode"] == "formal":
        if lock_path is None:
            raise ContractError("Formal matrix execution requires --lock")
        verified_lock = verify_lock(config_path, lock_path, root)
    if matrix_name == "primary":
        matrix = config["search"]
        configured_tasks = [spec.task_id for spec in task_specs(config)]
    elif matrix_name == "stress":
        if matrix_name not in config:
            raise ContractError(
                f"The selected config has no {matrix_name} matrix"
            )
        matrix = config[matrix_name]
        configured_tasks = [str(value) for value in matrix["tasks"]]
    else:
        raise ContractError(f"Unknown matrix: {matrix_name}")

    known_tasks = {spec.task_id for spec in task_specs(config)}
    unknown_tasks = set(configured_tasks).difference(known_tasks)
    if unknown_tasks:
        raise ContractError(
            f"Matrix {matrix_name} references unknown tasks: {sorted(unknown_tasks)}"
        )
    tasks = [
        task_id
        for task_id in configured_tasks
        if task_filter is None or task_id in set(task_filter)
    ]
    controllers = [
        value
        for value in matrix["controllers"]
        if controller_filter is None or value in set(controller_filter)
    ]
    initializations = [
        value
        for value in matrix["initializations"]
        if initialization_filter is None or value in set(initialization_filter)
    ]
    seeds = [
        int(value)
        for value in matrix["seeds"]
        if seed_filter is None or int(value) in set(seed_filter)
    ]
    if not tasks or not controllers or not initializations or not seeds:
        raise ContractError(
            f"Matrix {matrix_name} is empty after applying command-line filters"
        )
    shards = [
        (task_id, initialization, seed)
        for task_id in tasks
        for initialization in initializations
        for seed in seeds
    ]
    kwargs = [
        {
            "root": root,
            "config_path": config_path,
            "task_id": task_id,
            "initialization": initialization,
            "seed": seed,
            "controllers": tuple(controllers),
            "lock_path": lock_path,
            "overwrite_development": overwrite_development,
            "verified_lock": verified_lock,
        }
        for task_id, initialization, seed in shards
    ]
    if jobs == 1:
        completed = []
        for item in kwargs:
            completed.extend(_run_shard(**item))
        return completed
    completed: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for rows in executor.map(_run_shard_from_mapping, kwargs):
            completed.extend(rows)
    return completed


def _run_shard_from_mapping(values: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return _run_shard(**values)


def _run_shard(
    root: Path,
    config_path: Path,
    task_id: str,
    initialization: str,
    seed: int,
    controllers: Sequence[str],
    lock_path: Optional[Path],
    overwrite_development: bool,
    verified_lock: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    completed: List[Dict[str, Any]] = []
    for controller in controllers:
        completed.append(
            run_one(
                root=root,
                config_path=config_path,
                task_id=task_id,
                controller=controller,
                initialization=initialization,
                seed=seed,
                lock_path=lock_path,
                overwrite_development=overwrite_development,
                verified_lock=verified_lock,
            )
        )
    return completed
