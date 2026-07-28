from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Sequence

from .schemas import (
    ALLOWED_ALPHAS,
    ALLOWED_INPUT_MODES,
    ALLOWED_TARGET_MODES,
    ALLOWED_X_COMPONENTS,
    ALLOWED_Y_COMPONENTS,
    EDIT_ADDRESSES,
    CandidateConfig,
    ContractError,
)


class CandidateSpace:
    def __init__(self) -> None:
        self._candidates = tuple(
            CandidateConfig(
                input_mode=input_mode,
                target_mode=target_mode,
                x_components=x_components,
                y_components=y_components,
                alpha=alpha,
            )
            for input_mode, target_mode, x_components, y_components, alpha in itertools.product(
                ALLOWED_INPUT_MODES,
                ALLOWED_TARGET_MODES,
                ALLOWED_X_COMPONENTS,
                ALLOWED_Y_COMPONENTS,
                ALLOWED_ALPHAS,
            )
        )
        self._by_id = {candidate.candidate_id: candidate for candidate in self._candidates}

    @property
    def candidates(self) -> Sequence[CandidateConfig]:
        return self._candidates

    def by_id(self, candidate_id: str) -> CandidateConfig:
        try:
            return self._by_id[candidate_id]
        except KeyError as exc:
            raise ContractError(f"Unknown candidate ID: {candidate_id}") from exc

    def initial(self, name: str) -> CandidateConfig:
        if name != "standard_h0":
            raise ContractError("This public release provides the registered standard initialization only")
        return CandidateConfig(
            input_mode="pre_dose",
            target_mode="delta",
            x_components=32,
            y_components=32,
            alpha=10.0,
        )

    def neighbors(
        self,
        candidate: CandidateConfig,
        address: str,
    ) -> List[CandidateConfig]:
        if address not in EDIT_ADDRESSES:
            raise ContractError(f"Unknown edit address: {address}")
        values: Iterable[object]
        field: str
        if address == "conditioning":
            field, values = "input_mode", ALLOWED_INPUT_MODES
        elif address == "target":
            field, values = "target_mode", ALLOWED_TARGET_MODES
        elif address == "representation":
            field, values = "x_components", ALLOWED_X_COMPONENTS
        elif address == "decoder":
            field, values = "y_components", ALLOWED_Y_COMPONENTS
        else:
            field, values = "alpha", ALLOWED_ALPHAS
        neighbors = [
            replace(candidate, **{field: value})
            for value in values
            if value != getattr(candidate, field)
        ]
        return sorted(neighbors, key=lambda item: item.candidate_id)

    def all_neighbors(self, candidate: CandidateConfig) -> List[CandidateConfig]:
        unique: Dict[str, CandidateConfig] = {}
        for address in EDIT_ADDRESSES:
            for neighbor in self.neighbors(candidate, address):
                unique[neighbor.candidate_id] = neighbor
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def changed_address(
        before: CandidateConfig,
        after: CandidateConfig,
    ) -> str:
        mapping = {
            "input_mode": "conditioning",
            "target_mode": "target",
            "x_components": "representation",
            "y_components": "decoder",
            "alpha": "regularization",
        }
        changed = [
            address
            for field, address in mapping.items()
            if getattr(before, field) != getattr(after, field)
        ]
        if len(changed) != 1:
            raise ContractError(
                f"Expected one atomic edit, observed addresses: {changed}"
            )
        return changed[0]
