from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .data import TaskData
from .schemas import EDIT_ADDRESSES, TrialObservation


# Fixed dependency records used to interpret an edit trajectory. They are
# preregistered and do not dynamically rewire during a run.
COMPONENT_DEPENDENCIES = {
    "conditioning": ["representation", "target"],
    "target": ["decoder"],
    "representation": ["decoder", "regularization"],
    "decoder": ["regularization"],
    "regularization": [],
}


def build_hrt_record(
    data: TaskData,
    observations: Sequence[TrialObservation],
    protocol_hash: str,
    data_config: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "representation": "persistent_typed_task_state_and_dependency_record",
        "dynamic_topology": False,
        "protected_task_state": {
            "task_id": data.spec.task_id,
            "dataset": data.spec.dataset,
            "split": data.spec.split,
            "group_key": data.spec.group_key,
            "train_folds": list(data_config["train_folds"]),
            "feedback_fold": int(data_config["feedback_fold"]),
            "test_fold": int(data_config["test_fold"]),
            "protocol_sha256": protocol_hash,
        },
        "editable_component_addresses": list(EDIT_ADDRESSES),
        "dependencies": COMPONENT_DEPENDENCIES,
        "states": [
            {
                "step": observation.step,
                "candidate_id": observation.candidate.candidate_id,
                "candidate": observation.candidate.to_dict(),
                "parent_candidate_id": observation.parent_candidate_id,
                "routed_address": observation.route_address,
                "proposal_source": observation.proposal_source,
                "changed_address": observation.changed_address,
                "admissible": observation.transition_admissible,
            }
            for observation in observations
        ],
    }
