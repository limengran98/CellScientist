from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schemas import EDIT_ADDRESSES, CandidateConfig, ContractError
from .search_space import CandidateSpace


CANDIDATE_FIELDS = {
    "input_mode",
    "target_mode",
    "x_components",
    "y_components",
    "alpha",
}
REQUIRED_RUNTIME_METRICS = {"pcc", "response_pcc", "mse"}
REPAIRABLE_CODES = {
    "protected_state_mutation",
    "candidate_interface",
    "route_mismatch",
}


@dataclass(frozen=True)
class LCACase:
    case_id: str
    family: str
    expected_violation: bool
    expected_codes: Tuple[str, ...]
    stage: str
    repairable: bool
    request: Dict[str, Any]
    estimated_evaluation_cost_units: float


def _protected_state() -> Dict[str, Any]:
    return {
        "task_id": "BBBC036_plate",
        "dataset": "BBBC036",
        "split": "plate",
        "group_key": "plate_id",
        "train_folds": [1, 2],
        "feedback_fold": 3,
        "test_fold": 4,
        "primary_metric": "response_pcc",
    }


def _runtime_payload() -> Dict[str, Any]:
    return {
        "metrics": {
            "pcc": 0.51,
            "response_pcc": 0.32,
            "mse": 1.7,
        },
        "output_shape": [128, 256],
        "expected_output_shape": [128, 256],
    }


def _request(
    current: CandidateConfig,
    proposed: CandidateConfig,
    route_address: str,
) -> Dict[str, Any]:
    protected = _protected_state()
    return {
        "protected_before": protected,
        "protected_after": copy.deepcopy(protected),
        "current_candidate": current.to_dict(),
        "proposed_candidate": proposed.to_dict(),
        "candidate_id": proposed.candidate_id,
        "route_address": route_address,
        "controller_mode": "routed",
        "seen_candidate_ids": [current.candidate_id],
        "repair_count": 0,
        "repair_budget": 3,
        "runtime": _runtime_payload(),
    }


def validate_lca_request(request: Mapping[str, Any]) -> List[str]:
    violations: List[str] = []
    if request.get("protected_before") != request.get("protected_after"):
        violations.append("protected_state_mutation")

    raw_current = request.get("current_candidate")
    raw_proposed = request.get("proposed_candidate")
    if not isinstance(raw_current, Mapping) or not isinstance(
        raw_proposed,
        Mapping,
    ):
        violations.append("candidate_interface")
        current = None
        proposed = None
    else:
        if set(raw_current) != CANDIDATE_FIELDS or set(
            raw_proposed
        ) != CANDIDATE_FIELDS:
            violations.append("candidate_interface")
        try:
            current = CandidateConfig.from_mapping(raw_current)
            proposed = CandidateConfig.from_mapping(raw_proposed)
        except (ContractError, KeyError, TypeError, ValueError):
            violations.append("candidate_domain")
            current = None
            proposed = None

    changed_address: Optional[str] = None
    if current is not None and proposed is not None:
        if request.get("candidate_id") != proposed.candidate_id:
            violations.append("candidate_id_mismatch")
        try:
            changed_address = CandidateSpace.changed_address(current, proposed)
        except ContractError:
            violations.append("non_atomic_edit")
        if proposed.candidate_id in set(request.get("seen_candidate_ids", [])):
            violations.append("duplicate_candidate")

    mode = request.get("controller_mode")
    route = request.get("route_address")
    if mode not in {"routed", "flat"}:
        violations.append("controller_interface")
    elif mode == "routed":
        if route not in EDIT_ADDRESSES or (
            changed_address is not None and route != changed_address
        ):
            violations.append("route_mismatch")
    elif route is not None:
        violations.append("route_mismatch")

    try:
        repair_count = int(request.get("repair_count"))
        repair_budget = int(request.get("repair_budget"))
        if repair_budget < 1 or repair_count >= repair_budget:
            violations.append("repair_budget_exhausted")
    except (TypeError, ValueError):
        violations.append("repair_budget_exhausted")

    runtime = request.get("runtime")
    if not isinstance(runtime, Mapping):
        violations.extend(["output_shape_mismatch", "runtime_nonfinite"])
    else:
        if runtime.get("output_shape") != runtime.get("expected_output_shape"):
            violations.append("output_shape_mismatch")
        metrics = runtime.get("metrics")
        if not isinstance(metrics, Mapping) or not REQUIRED_RUNTIME_METRICS.issubset(
            metrics
        ):
            violations.append("runtime_nonfinite")
        else:
            try:
                if not all(
                    math.isfinite(float(metrics[key]))
                    for key in REQUIRED_RUNTIME_METRICS
                ):
                    violations.append("runtime_nonfinite")
            except (TypeError, ValueError):
                violations.append("runtime_nonfinite")

    return list(dict.fromkeys(violations))


def repair_lca_request(
    request: Mapping[str, Any],
    violation_codes: Sequence[str],
) -> Optional[Dict[str, Any]]:
    codes = set(violation_codes)
    if not codes or not codes.issubset(REPAIRABLE_CODES):
        return None
    repaired = copy.deepcopy(dict(request))
    if "protected_state_mutation" in codes:
        repaired["protected_after"] = copy.deepcopy(
            repaired["protected_before"]
        )
    if "candidate_interface" in codes:
        for key in ("current_candidate", "proposed_candidate"):
            value = repaired.get(key)
            if isinstance(value, Mapping):
                repaired[key] = {
                    field: value[field]
                    for field in CANDIDATE_FIELDS
                    if field in value
                }
    if "route_mismatch" in codes:
        try:
            current = CandidateConfig.from_mapping(
                repaired["current_candidate"]
            )
            proposed = CandidateConfig.from_mapping(
                repaired["proposed_candidate"]
            )
            repaired["route_address"] = CandidateSpace.changed_address(
                current,
                proposed,
            )
        except (ContractError, KeyError, TypeError, ValueError):
            return None
    return repaired if not validate_lca_request(repaired) else None


def lca_cases() -> List[LCACase]:
    space = CandidateSpace()
    current = CandidateConfig(
        "pre_dose",
        "delta",
        32,
        32,
        10.0,
    )
    address_candidates = {
        address: space.neighbors(current, address)[0]
        for address in EDIT_ADDRESSES
    }
    cases: List[LCACase] = []

    # Twenty clean negative cases: four legal variants for each typed address.
    for variant in range(4):
        for address_index, address in enumerate(EDIT_ADDRESSES):
            options = space.neighbors(current, address)
            proposed = options[variant % len(options)]
            request = _request(current, proposed, address)
            cases.append(
                LCACase(
                    case_id=f"N{address_index + 1:02d}-{variant + 1}",
                    family="clean",
                    expected_violation=False,
                    expected_codes=(),
                    stage="pre_execution",
                    repairable=False,
                    request=request,
                    estimated_evaluation_cost_units=float(
                        proposed.estimated_cost / 1024.0
                    ),
                )
            )

    # Forty positive cases: five deterministic variants for each violation
    # family. Fault labels are registered in the harness before validation.
    families = (
        ("protected", "protected_state_mutation", "pre_execution", True),
        ("interface", "candidate_interface", "pre_execution", True),
        ("domain", "candidate_domain", "pre_execution", False),
        ("atomicity", "non_atomic_edit", "pre_execution", False),
        ("route", "route_mismatch", "pre_execution", True),
        ("output", "output_shape_mismatch", "post_execution", False),
        ("runtime", "runtime_nonfinite", "post_execution", False),
        ("budget", "repair_budget_exhausted", "pre_execution", False),
    )
    for family_index, (family, code, stage, repairable) in enumerate(families):
        for variant in range(5):
            address = EDIT_ADDRESSES[variant]
            proposed = address_candidates[address]
            request = _request(current, proposed, address)
            if family == "protected":
                request["protected_after"]["test_fold"] = 5 + variant
            elif family == "interface":
                request["proposed_candidate"]["undeclared_field"] = variant
            elif family == "domain":
                request["proposed_candidate"]["alpha"] = -1.0 - variant
            elif family == "atomicity":
                request["proposed_candidate"]["input_mode"] = "pre_smiles_dose"
                request["proposed_candidate"]["target_mode"] = "direct"
                request["candidate_id"] = CandidateConfig.from_mapping(
                    request["proposed_candidate"]
                ).candidate_id
            elif family == "route":
                request["route_address"] = EDIT_ADDRESSES[
                    (variant + 1) % len(EDIT_ADDRESSES)
                ]
            elif family == "output":
                request["runtime"]["output_shape"] = [129 + variant, 256]
            elif family == "runtime":
                request["runtime"]["metrics"]["response_pcc"] = float("nan")
            elif family == "budget":
                request["repair_count"] = request["repair_budget"]
            cases.append(
                LCACase(
                    case_id=f"P{family_index + 1:02d}-{variant + 1}",
                    family=family,
                    expected_violation=True,
                    expected_codes=(code,),
                    stage=stage,
                    repairable=repairable,
                    request=request,
                    estimated_evaluation_cost_units=float(
                        proposed.estimated_cost / 1024.0
                    ),
                )
            )
    return sorted(cases, key=lambda case: case.case_id)


def run_lca_audit() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in lca_cases():
        observed_codes = validate_lca_request(case.request)
        detected = bool(observed_codes)
        repaired_request = repair_lca_request(case.request, observed_codes)
        repair_success = repaired_request is not None
        safely_blocked = case.expected_violation and detected and not repair_success
        avoided = (
            case.expected_violation
            and detected
            and case.stage == "pre_execution"
        )
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "expected_violation": case.expected_violation,
                "expected_codes": list(case.expected_codes),
                "observed_codes": observed_codes,
                "stage": case.stage,
                "detected": detected,
                "accepted": not detected,
                "repairable": case.repairable,
                "repair_success": repair_success,
                "safely_blocked": safely_blocked,
                "avoided_invalid_evaluation": avoided,
                "estimated_evaluation_cost_units": (
                    case.estimated_evaluation_cost_units
                ),
            }
        )

    positives = [row for row in rows if row["expected_violation"]]
    negatives = [row for row in rows if not row["expected_violation"]]
    repairable = [row for row in positives if row["repairable"]]
    exact_code_matches = [
        set(row["expected_codes"]) == set(row["observed_codes"])
        for row in positives
    ]
    return {
        "schema_version": 1,
        "case_definition": (
            "Deterministic contract cases registered before execution; "
            "positive cases contain one injected violation and negative cases "
            "are legal controls."
        ),
        "n_positive_cases": len(positives),
        "n_negative_cases": len(negatives),
        "violation_recall": float(
            np.mean([row["detected"] for row in positives])
        ),
        "clean_specificity": float(
            np.mean([row["accepted"] for row in negatives])
        ),
        "exact_violation_code_accuracy": float(np.mean(exact_code_matches)),
        "repair_success_rate": float(
            np.mean([row["repair_success"] for row in repairable])
        ),
        "containment_rate": float(
            np.mean(
                [
                    row["repair_success"] or row["safely_blocked"]
                    for row in positives
                ]
            )
        ),
        "avoided_invalid_evaluations": int(
            sum(row["avoided_invalid_evaluation"] for row in rows)
        ),
        "avoided_estimated_cost_units": float(
            sum(
                row["estimated_evaluation_cost_units"]
                for row in rows
                if row["avoided_invalid_evaluation"]
            )
        ),
        "post_execution_violations_caught": int(
            sum(
                row["detected"] and row["stage"] == "post_execution"
                for row in positives
            )
        ),
        "family_summary": {
            family: {
                "n": len(family_rows),
                "recall": float(
                    np.mean([row["detected"] for row in family_rows])
                ),
                "repair_success_rate": (
                    float(
                        np.mean(
                            [
                                row["repair_success"]
                                for row in family_rows
                                if row["repairable"]
                            ]
                        )
                    )
                    if any(row["repairable"] for row in family_rows)
                    else None
                ),
            }
            for family in sorted({row["family"] for row in positives})
            for family_rows in [
                [row for row in positives if row["family"] == family]
            ]
        },
        "rows": rows,
    }
