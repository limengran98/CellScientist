from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import h5py
import numpy as np

from .schemas import ContractError, PartitionIndices, TaskSpec


@dataclass
class TaskData:
    spec: TaskSpec
    morphology_pre: np.ndarray
    morphology_post: np.ndarray
    dose: np.ndarray
    smiles: np.ndarray
    plate_id: np.ndarray
    split_id: np.ndarray
    partitions: PartitionIndices
    partition_summary: Dict[str, Any]

    @property
    def n_samples(self) -> int:
        return int(self.morphology_pre.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.morphology_pre.shape[1])


def _decode_strings(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [
                item.decode("utf-8", errors="replace")
                if isinstance(item, (bytes, np.bytes_))
                else str(item)
                for item in values
            ],
            dtype=object,
        )
    return values.astype(str)


def _stable_group_split(
    indices: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = sorted({str(groups[index]) for index in indices})
    if len(unique_groups) < 2:
        raise ContractError(
            "Feedback fold has fewer than two groups; independent selection is impossible"
        )
    ordered = sorted(
        unique_groups,
        key=lambda value: hashlib.sha256(
            f"{seed}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    feedback_groups = set(ordered[::2])
    selection_groups = set(ordered[1::2])
    if not selection_groups:
        moved = sorted(feedback_groups)[-1]
        feedback_groups.remove(moved)
        selection_groups.add(moved)
    feedback = indices[
        np.asarray([str(groups[index]) in feedback_groups for index in indices])
    ]
    selection = indices[
        np.asarray([str(groups[index]) in selection_groups for index in indices])
    ]
    if feedback.size == 0 or selection.size == 0:
        raise ContractError("Deterministic group split produced an empty partition")
    return feedback, selection


def _subsample(
    indices: np.ndarray,
    maximum: Optional[int],
    seed: int,
) -> np.ndarray:
    if maximum is None or len(indices) <= maximum:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=int(maximum), replace=False)
    return np.sort(chosen.astype(np.int64))


def load_task_data(
    spec: TaskSpec,
    data_config: Mapping[str, Any],
    search_config: Mapping[str, Any],
) -> TaskData:
    with h5py.File(spec.path, "r") as handle:
        if "combined" not in handle:
            raise ContractError(f"{spec.path} has no combined group")
        group = handle["combined"]
        required = {
            "morphology_pre",
            "morphology_post",
            "dose",
            "smiles",
            "plate_id",
            "split_id",
        }
        missing = required.difference(group.keys())
        if missing:
            raise ContractError(f"{spec.path} is missing datasets: {sorted(missing)}")
        morphology_pre = np.asarray(group["morphology_pre"], dtype=np.float32)
        morphology_post = np.asarray(group["morphology_post"], dtype=np.float32)
        dose = np.asarray(group["dose"], dtype=np.float32)
        smiles = _decode_strings(np.asarray(group["smiles"]))
        plate_id = _decode_strings(np.asarray(group["plate_id"]))
        split_id = np.asarray(group["split_id"], dtype=np.int64)

    if morphology_pre.shape != morphology_post.shape:
        raise ContractError("Pre/post morphology shapes differ")
    n_samples = morphology_pre.shape[0]
    for name, values in {
        "dose": dose,
        "smiles": smiles,
        "plate_id": plate_id,
        "split_id": split_id,
    }.items():
        if len(values) != n_samples:
            raise ContractError(f"{name} length does not match morphology rows")

    train_folds = np.asarray(data_config["train_folds"], dtype=np.int64)
    feedback_fold = int(data_config["feedback_fold"])
    test_fold = int(data_config["test_fold"])
    train = np.flatnonzero(np.isin(split_id, train_folds))
    fold4 = np.flatnonzero(split_id == feedback_fold)
    test = np.flatnonzero(split_id == test_fold)
    split_groups = plate_id if spec.group_key == "plate_id" else smiles
    feedback, selection = _stable_group_split(
        fold4,
        split_groups,
        int(data_config["partition_seed"]),
    )

    max_train = search_config.get("max_train_samples")
    max_eval = search_config.get("max_eval_samples")
    seed = int(data_config["partition_seed"])
    train = _subsample(train, max_train, seed + 1)
    feedback = _subsample(feedback, max_eval, seed + 2)
    selection = _subsample(selection, max_eval, seed + 3)
    test = _subsample(test, max_eval, seed + 4)

    partition_sets = [set(x.tolist()) for x in (train, feedback, selection, test)]
    for left in range(len(partition_sets)):
        for right in range(left + 1, len(partition_sets)):
            if partition_sets[left].intersection(partition_sets[right]):
                raise ContractError("Partition overlap detected")
    partition_groups = [
        set(split_groups[index].tolist())
        for index in (train, feedback, selection, test)
    ]
    for left in range(len(partition_groups)):
        for right in range(left + 1, len(partition_groups)):
            overlap = partition_groups[left].intersection(partition_groups[right])
            if overlap:
                raise ContractError(
                    "Group leakage detected between protected partitions: "
                    f"{len(overlap)} shared {spec.group_key} values"
                )

    summary = {
        "task_id": spec.task_id,
        "source": str(spec.path),
        "n_samples": int(n_samples),
        "n_features": int(morphology_pre.shape[1]),
        "partition_sizes": {
            "train": int(len(train)),
            "feedback": int(len(feedback)),
            "selection": int(len(selection)),
            "test": int(len(test)),
        },
        "group_key": spec.group_key,
        "group_counts": {
            "train": int(len(set(split_groups[train].tolist()))),
            "feedback": int(len(set(split_groups[feedback].tolist()))),
            "selection": int(len(set(split_groups[selection].tolist()))),
            "test": int(len(set(split_groups[test].tolist()))),
        },
    }
    return TaskData(
        spec=spec,
        morphology_pre=morphology_pre,
        morphology_post=morphology_post,
        dose=dose,
        smiles=smiles,
        plate_id=plate_id,
        split_id=split_id,
        partitions=PartitionIndices(
            train=tuple(int(x) for x in train),
            feedback=tuple(int(x) for x in feedback),
            selection=tuple(int(x) for x in selection),
            test=tuple(int(x) for x in test),
        ),
        partition_summary=summary,
    )
