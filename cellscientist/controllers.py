"""The CellScientist discrepancy-conditioned revision controller.

This release intentionally contains only the proposed controller.  It does
not include baselines, ablations, or comparative-search policies.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Set

from .evaluator import objective_value
from .llm_client import OpenAICompatiblePolicy
from .schemas import CandidateConfig, ContractError, Proposal, TrialObservation
from .search_space import CandidateSpace


def _best_observation(
    history: Sequence[TrialObservation], primary_metric: str
) -> TrialObservation:
    return max(
        history,
        key=lambda item: objective_value(item.feedback_metrics, primary_metric),
    )


def _candidate_rows(candidates: Sequence[CandidateConfig]) -> List[Dict[str, object]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            **candidate.to_dict(),
            "estimated_cost": candidate.estimated_cost,
        }
        for candidate in candidates
    ]


def _canonical_atomic_edit(
    candidates: Sequence[CandidateConfig], address: str
) -> CandidateConfig:
    """Instantiate one legal atomic edit at an already selected address."""
    if not candidates:
        raise ContractError("No legal unseen candidate at the selected address")
    preferences = {
        "conditioning": {"pre_smiles_dose": 0, "pre_smiles": 1, "pre_dose": 2, "pre": 3},
        "target": {"delta": 0, "direct": 1},
        "representation": {64: 0, 128: 1, 32: 2, 16: 3},
        "decoder": {64: 0, 128: 1, 32: 2, 16: 3},
        "regularization": {1.0: 0, 0.1: 1, 0.01: 2, 10.0: 3, 100.0: 4, 1000.0: 5},
    }
    field = {
        "conditioning": "input_mode",
        "target": "target_mode",
        "representation": "x_components",
        "decoder": "y_components",
        "regularization": "alpha",
    }[address]
    return min(
        candidates,
        key=lambda candidate: (
            preferences[address][getattr(candidate, field)],
            candidate.candidate_id,
        ),
    )


class CellScientistController:
    """Routes diagnostics to an allow-listed typed component address.

    A deterministic stage ranks legal addresses from the protected task state,
    feedback metrics, and retained history.  The LLM then selects one of at
    most two address--candidate pairs.  Parsing, address consistency, and the
    candidate transition are checked deterministically by the runner.
    """

    name = "cellscientist"

    def __init__(
        self,
        space: CandidateSpace,
        seed: int,
        primary_metric: str,
        llm_policy: Optional[OpenAICompatiblePolicy],
    ) -> None:
        self.space = space
        self.seed = int(seed)
        self.primary_metric = primary_metric
        self.llm_policy = llm_policy

    def rank_addresses(
        self, history: Sequence[TrialObservation], seen: Set[str]
    ) -> List[str]:
        incumbent = _best_observation(history, self.primary_metric)
        candidate = incumbent.candidate
        metrics = incumbent.feedback_metrics
        failed_addresses = {
            observation.route_address
            for observation in history
            if observation.step > incumbent.step
            and observation.route_address is not None
            and objective_value(observation.feedback_metrics, self.primary_metric)
            <= objective_value(incumbent.feedback_metrics, self.primary_metric)
        }

        def capacity_gain_demonstrated(address: str) -> bool:
            field = {"representation": "x_components", "decoder": "y_components"}[address]
            current_capacity = int(getattr(candidate, field))
            fields = ("input_mode", "target_mode", "x_components", "y_components", "alpha")
            for observation in history:
                previous = observation.candidate
                if int(getattr(previous, field)) >= current_capacity:
                    continue
                if any(
                    getattr(previous, other) != getattr(candidate, other)
                    for other in fields
                    if other != field
                ):
                    continue
                if objective_value(observation.feedback_metrics, self.primary_metric) < objective_value(
                    incumbent.feedback_metrics, self.primary_metric
                ):
                    return True
            return False

        priority: List[str] = []
        if candidate.input_mode != "pre_smiles_dose":
            priority.append("conditioning")
        if candidate.target_mode != "delta" or metrics.get("pcc", 0.0) + 0.05 < metrics.get("response_pcc", 0.0):
            priority.append("target")
        if metrics.get("top50_response_pcc", 0.0) < metrics.get("response_pcc", 0.0):
            priority.append("decoder")
        if candidate.x_components < 64 or (
            candidate.x_components == 64 and capacity_gain_demonstrated("representation")
        ):
            priority.append("representation")
        if candidate.y_components < 64 or (
            candidate.y_components == 64 and capacity_gain_demonstrated("decoder")
        ):
            priority.append("decoder")
        priority.extend(["regularization", "target", "conditioning"])

        ranked: List[str] = []
        for address in priority:
            if address in ranked or address in failed_addresses:
                continue
            if any(
                neighbor.candidate_id not in seen
                for neighbor in self.space.neighbors(candidate, address)
            ):
                ranked.append(address)
        if not ranked:
            raise ContractError("No legal address remains for a CellScientist revision")
        return ranked

    def propose(
        self, history: Sequence[TrialObservation], seen: Set[str], step: int
    ) -> Proposal:
        del step
        incumbent = _best_observation(history, self.primary_metric)
        allowed_addresses = self.rank_addresses(history, seen)[:2]
        candidate_to_address: Dict[str, str] = {}
        routed_candidates: List[CandidateConfig] = []
        for address in allowed_addresses:
            legal = [
                item
                for item in self.space.neighbors(incumbent.candidate, address)
                if item.candidate_id not in seen
            ]
            instantiated = _canonical_atomic_edit(legal, address)
            routed_candidates.append(instantiated)
            candidate_to_address[instantiated.candidate_id] = address
        fallback = routed_candidates[0]
        fallback_address = allowed_addresses[0]
        prompt = (
            "A deterministic diagnostic stage has shortlisted at most two typed "
            "edit addresses. A constrained revision stage has instantiated one "
            "legal atomic edit per address. Attribute the discrepancy to one "
            "shortlisted address and return its paired candidate. Prefer the "
            "earlier ranked address unless the feedback clearly supports the second. "
            "Any other address or candidate is forbidden.\n"
            f"current={json.dumps(incumbent.candidate.to_dict(), sort_keys=True)}\n"
            f"feedback={json.dumps(incumbent.feedback_metrics, sort_keys=True)}\n"
            f"ranked_allowed_addresses={json.dumps(allowed_addresses)}\n"
            f"candidate_to_address={json.dumps(candidate_to_address, sort_keys=True)}\n"
            f"candidates={json.dumps(_candidate_rows(routed_candidates), sort_keys=True)}\n"
            "valid_outputs="
            f"{json.dumps([{'candidate_id': item.candidate_id, 'address': candidate_to_address[item.candidate_id]} for item in routed_candidates], sort_keys=True)}\n"
            "Return exactly one object copied verbatim from valid_outputs."
        )
        if self.llm_policy is None or not self.llm_policy.enabled:
            return Proposal(
                candidate=fallback,
                address=fallback_address,
                source="cellscientist:deterministic",
            )
        try:
            decision = self.llm_policy.choose(
                prompt,
                [item.candidate_id for item in routed_candidates],
                allowed_addresses,
            )
            candidate = self.space.by_id(decision.candidate_id)
            if decision.address != candidate_to_address[candidate.candidate_id]:
                raise ContractError("candidate/address pairing mismatch")
            return Proposal(
                candidate=candidate,
                address=decision.address,
                source=f"cellscientist:{self.llm_policy.model}",
                prompt_text=decision.prompt_text,
                response_text=decision.response_text,
                llm_usage=decision.usage,
                llm_http_attempts=decision.http_attempts,
            )
        except ContractError as exc:
            return Proposal(
                candidate=fallback,
                address=fallback_address,
                source=f"cellscientist:fallback:{type(exc).__name__}",
                fallback_used=True,
                prompt_text=prompt,
                response_text=None if self.llm_policy is None else self.llm_policy.last_response_text,
                llm_usage=None if self.llm_policy is None else self.llm_policy.last_usage,
                llm_http_attempts=() if self.llm_policy is None else tuple(dict(row) for row in self.llm_policy.last_call_trace),
                llm_error=str(exc),
            )


def build_controller(
    name: str,
    space: CandidateSpace,
    seed: int,
    primary_metric: str,
    llm_policy: Optional[OpenAICompatiblePolicy],
) -> CellScientistController:
    if name != "cellscientist":
        raise ContractError("This public release contains only the CellScientist controller")
    return CellScientistController(space, seed, primary_metric, llm_policy)
