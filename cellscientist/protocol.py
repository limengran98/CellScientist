from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .schemas import ContractError, TaskSpec


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    data_root_env = str(value.get("data", {}).get("root_env", "CELLSCIENTIST_DATA_ROOT"))
    data_root = os.environ.get(data_root_env)
    for task in value.get("data", {}).get("tasks", []):
        raw_path = Path(str(task["path"]))
        if raw_path.is_absolute():
            continue
        if not data_root:
            raise ContractError(
                f"Set {data_root_env} to the directory containing the BBBC HDF5 files"
            )
        task["path"] = str(Path(data_root) / raw_path)
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("mode") not in {"development", "formal"}:
        raise ContractError("mode must be development or formal")
    tasks = config.get("data", {}).get("tasks", [])
    if not tasks:
        raise ContractError("At least one task is required")
    for raw in tasks:
        path = Path(raw["path"])
        if not path.is_file():
            raise ContractError(f"Dataset does not exist: {path}")
        if raw["split"] not in {"plate", "smiles"}:
            raise ContractError(f"Unsupported split: {raw['split']}")
    budget = int(config.get("search", {}).get("budget", 0))
    if budget < 1:
        raise ContractError("search.budget must be positive")
    checkpoints = config["search"].get("budget_checkpoints", [])
    if not checkpoints or max(int(x) for x in checkpoints) > budget:
        raise ContractError("budget checkpoints must be nonempty and within budget")
    train_folds = {int(value) for value in config["data"].get("train_folds", [])}
    feedback_fold = int(config["data"]["feedback_fold"])
    test_fold = int(config["data"]["test_fold"])
    if not train_folds:
        raise ContractError("At least one training fold is required")
    if feedback_fold in train_folds or test_fold in train_folds:
        raise ContractError("Training, feedback, and test folds must be distinct")
    if feedback_fold == test_fold:
        raise ContractError("Feedback and test folds must be distinct")
    backend = str(config.get("model", {}).get("backend", "sklearn"))
    if backend not in {"sklearn", "torch_cpu", "torch_cuda"}:
        raise ContractError(f"Unsupported model.backend: {backend}")
    allowed_controllers = {"cellscientist"}
    controllers = set(config["search"].get("controllers", []))
    unknown = controllers.difference(allowed_controllers)
    if unknown:
        raise ContractError(f"Unsupported controllers: {sorted(unknown)}")
    allowed_initializations = {"standard_h0"}
    initializations = set(config["search"].get("initializations", []))
    if not initializations or initializations.difference(allowed_initializations):
        raise ContractError("search.initializations contains an unsupported value")


def task_specs(config: Mapping[str, Any]) -> Iterable[TaskSpec]:
    for raw in config["data"]["tasks"]:
        yield TaskSpec(
            dataset=str(raw["dataset"]),
            split=str(raw["split"]),
            path=Path(raw["path"]),
            group_key=str(raw["group_key"]),
        )


def source_fingerprint(root: Path) -> Dict[str, str]:
    digest_map: Dict[str, str] = {}
    for path in sorted((root / "cellscientist").glob("*.py")):
        digest_map[str(path.relative_to(root))] = sha256_file(path)
    return digest_map


def build_lock(config_path: Path, root: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    if config["mode"] != "formal":
        raise ContractError("Only a formal config may be locked")
    if config.get("llm", {}).get("selection_status") != "frozen":
        raise ContractError(
            "Formal LLM selection is still provisional. Complete the "
            "development-only model comparison and set "
            "llm.selection_status=frozen before creating the lock."
        )
    datasets: Dict[str, Dict[str, Any]] = {}
    for spec in task_specs(config):
        stat = spec.path.stat()
        datasets[spec.task_id] = {
            "path": str(spec.path),
            "size": stat.st_size,
            "sha256": sha256_file(spec.path),
        }
    sources = source_fingerprint(root)
    config_hash = sha256_text(canonical_json(config))
    payload: Dict[str, Any] = {
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "datasets": datasets,
        "sources": sources,
        "source_bundle_sha256": sha256_text(canonical_json(sources)),
    }
    payload["lock_sha256"] = sha256_text(canonical_json(payload))
    return payload


def write_lock(config_path: Path, lock_path: Path, root: Path) -> Dict[str, Any]:
    lock = build_lock(config_path, root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = lock_path.with_suffix(lock_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, lock_path)
    return lock


def verify_lock(config_path: Path, lock_path: Path, root: Path) -> Dict[str, Any]:
    if not lock_path.is_file():
        raise ContractError(f"Formal protocol lock is missing: {lock_path}")
    with lock_path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)
    observed = build_lock(config_path, root)
    if canonical_json(expected) != canonical_json(observed):
        raise ContractError(
            "Formal protocol lock mismatch. Code, config, runtime, or data changed."
        )
    return expected
