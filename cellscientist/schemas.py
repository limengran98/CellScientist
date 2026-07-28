from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


class ContractError(ValueError):
    """Raised when a protocol, proposal, or artifact violates the study contract."""


ALLOWED_INPUT_MODES = ("pre", "pre_dose", "pre_smiles", "pre_smiles_dose")
ALLOWED_TARGET_MODES = ("direct", "delta")
ALLOWED_X_COMPONENTS = (16, 32, 64, 128)
ALLOWED_Y_COMPONENTS = (16, 32, 64, 128)
ALLOWED_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
EDIT_ADDRESSES = (
    "conditioning",
    "target",
    "representation",
    "decoder",
    "regularization",
)


@dataclass(frozen=True)
class CandidateConfig:
    input_mode: str
    target_mode: str
    x_components: int
    y_components: int
    alpha: float

    def __post_init__(self) -> None:
        if self.input_mode not in ALLOWED_INPUT_MODES:
            raise ContractError(f"Invalid input_mode: {self.input_mode}")
        if self.target_mode not in ALLOWED_TARGET_MODES:
            raise ContractError(f"Invalid target_mode: {self.target_mode}")
        if self.x_components not in ALLOWED_X_COMPONENTS:
            raise ContractError(f"Invalid x_components: {self.x_components}")
        if self.y_components not in ALLOWED_Y_COMPONENTS:
            raise ContractError(f"Invalid y_components: {self.y_components}")
        if float(self.alpha) not in ALLOWED_ALPHAS:
            raise ContractError(f"Invalid alpha: {self.alpha}")

    @property
    def candidate_id(self) -> str:
        alpha = str(float(self.alpha)).replace(".", "p")
        return (
            f"in-{self.input_mode}__target-{self.target_mode}"
            f"__x-{self.x_components}__y-{self.y_components}__a-{alpha}"
        )

    @property
    def estimated_cost(self) -> float:
        input_factor = {
            "pre": 1.0,
            "pre_dose": 1.02,
            "pre_smiles": 1.10,
            "pre_smiles_dose": 1.12,
        }[self.input_mode]
        return input_factor * self.x_components * self.y_components

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateConfig":
        return cls(
            input_mode=str(value["input_mode"]),
            target_mode=str(value["target_mode"]),
            x_components=int(value["x_components"]),
            y_components=int(value["y_components"]),
            alpha=float(value["alpha"]),
        )


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    split: str
    path: Path
    group_key: str

    @property
    def task_id(self) -> str:
        return f"{self.dataset}_{self.split}"


@dataclass
class TrialObservation:
    step: int
    candidate: CandidateConfig
    feedback_metrics: Dict[str, float]
    route_address: Optional[str]
    proposal_source: str
    elapsed_seconds: float
    cache_hit: bool
    actual_evaluation_seconds: float = 0.0
    proposal_seconds: float = 0.0
    cumulative_seconds: float = 0.0
    fallback_used: bool = False
    prompt_sha256: Optional[str] = None
    response_sha256: Optional[str] = None
    parent_candidate_id: Optional[str] = None
    changed_address: Optional[str] = None
    transition_admissible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "candidate": self.candidate.to_dict(),
            "candidate_id": self.candidate.candidate_id,
            "feedback_metrics": dict(self.feedback_metrics),
            "route_address": self.route_address,
            "proposal_source": self.proposal_source,
            "elapsed_seconds": self.elapsed_seconds,
            "actual_evaluation_seconds": self.actual_evaluation_seconds,
            "proposal_seconds": self.proposal_seconds,
            "cumulative_seconds": self.cumulative_seconds,
            "cache_hit": self.cache_hit,
            "fallback_used": self.fallback_used,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "parent_candidate_id": self.parent_candidate_id,
            "changed_address": self.changed_address,
            "transition_admissible": self.transition_admissible,
        }


@dataclass(frozen=True)
class Proposal:
    candidate: CandidateConfig
    address: Optional[str]
    source: str
    fallback_used: bool = False
    prompt_text: Optional[str] = None
    response_text: Optional[str] = None
    llm_usage: Optional[Dict[str, int]] = None
    llm_http_attempts: Tuple[Dict[str, Any], ...] = ()
    llm_error: Optional[str] = None
    parent_candidate_id: Optional[str] = None
    operation: Optional[str] = None


@dataclass(frozen=True)
class PartitionIndices:
    train: Tuple[int, ...]
    feedback: Tuple[int, ...]
    selection: Tuple[int, ...]
    test: Tuple[int, ...]
