"""Registered-fault audit for CellScientist's component-address routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .controllers import CellScientistController
from .llm_client import OpenAICompatiblePolicy
from .schemas import CandidateConfig, TrialObservation
from .search_space import CandidateSpace


@dataclass(frozen=True)
class RoutingCase:
    case_id: str
    expected_address: str
    candidate: CandidateConfig
    metrics: Dict[str, float]


def _metrics(response_pcc: float, top50_response_pcc: float) -> Dict[str, float]:
    return {
        "pcc": 0.58,
        "sample_pcc": 0.36,
        "mse": 2.7,
        "r2": 0.16,
        "centered_pcc": 0.50,
        "response_pcc": response_pcc,
        "response_sample_pcc": max(0.0, response_pcc - 0.05),
        "response_mse": 2.7,
        "top20_response_pcc": max(0.0, top50_response_pcc - 0.04),
        "top50_response_pcc": top50_response_pcc,
    }


def held_out_cases() -> List[RoutingCase]:
    """Fifteen typed violations: three instances for each editable address."""
    prototypes = (
        ("conditioning", CandidateConfig("pre", "delta", 64, 64, 1.0), _metrics(0.18, 0.20)),
        ("target", CandidateConfig("pre_smiles_dose", "direct", 64, 64, 1.0), _metrics(0.04, 0.06)),
        ("representation", CandidateConfig("pre_smiles_dose", "delta", 32, 64, 1.0), _metrics(0.24, 0.27)),
        ("decoder", CandidateConfig("pre_smiles_dose", "delta", 64, 32, 1.0), _metrics(0.31, 0.13)),
        ("regularization", CandidateConfig("pre_smiles_dose", "delta", 64, 64, 100.0), _metrics(0.34, 0.37)),
    )
    rows: List[RoutingCase] = []
    for replica in range(3):
        for address, candidate, metrics in prototypes:
            adjusted = dict(metrics)
            adjusted["pcc"] += replica * 0.001
            rows.append(
                RoutingCase(
                    case_id=f"{address[:3]}-{replica + 1}",
                    expected_address=address,
                    candidate=candidate,
                    metrics=adjusted,
                )
            )
    return rows


def _observation(case: RoutingCase) -> TrialObservation:
    return TrialObservation(
        step=1,
        candidate=case.candidate,
        feedback_metrics=dict(case.metrics),
        route_address=None,
        proposal_source="registered_fault",
        elapsed_seconds=0.0,
        cache_hit=False,
    )


def _repaired(case: RoutingCase, candidate: CandidateConfig) -> bool:
    values = {
        "conditioning": candidate.input_mode == "pre_smiles_dose",
        "target": candidate.target_mode == "delta",
        "representation": candidate.x_components == 64,
        "decoder": candidate.y_components == 64,
        "regularization": candidate.alpha == 1.0,
    }
    return values[case.expected_address]


def run_routing_audit(
    seed: int = 11, llm_policy: Optional[OpenAICompatiblePolicy] = None
) -> Dict[str, Any]:
    """Run a self-contained, no-baseline routing and repair audit.

    If ``llm_policy`` is omitted, the registered top-ranked address is used;
    supplying an enabled policy exercises the complete constrained LLM route.
    """
    space = CandidateSpace()
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(held_out_cases()):
        controller = CellScientistController(
            space=space,
            seed=seed + index,
            primary_metric="response_pcc",
            llm_policy=llm_policy,
        )
        history = [_observation(case)]
        seen: Set[str] = {case.candidate.candidate_id}
        ranked = controller.rank_addresses(history, seen)
        proposal = controller.propose(history, seen, step=2)
        rows.append(
            {
                "case_id": case.case_id,
                "expected_address": case.expected_address,
                "ranked_addresses": ranked[:2],
                "selected_address": proposal.address,
                "selected_candidate_id": proposal.candidate.candidate_id,
                "top1_correct": ranked[0] == case.expected_address,
                "top2_correct": case.expected_address in ranked[:2],
                "repair_success": proposal.address == case.expected_address
                and _repaired(case, proposal.candidate),
                "fallback_used": proposal.fallback_used,
            }
        )
    count = len(rows)
    return {
        "audit": "registered_component_routing",
        "n_held_out_cases": count,
        "routing_mode": "llm_constrained" if llm_policy and llm_policy.enabled else "registered_top_rank",
        "top1_accuracy": sum(row["top1_correct"] for row in rows) / count,
        "top2_accuracy": sum(row["top2_correct"] for row in rows) / count,
        "repair_success": sum(row["repair_success"] for row in rows) / count,
        "fallback_rate": sum(row["fallback_used"] for row in rows) / count,
        "cases": rows,
    }
