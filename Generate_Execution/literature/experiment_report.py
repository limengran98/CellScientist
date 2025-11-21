# -*- coding: utf-8 -*-
"""
Experiment report generator.

Given a metrics.json dict that may contain multiple models / variants,
this module writes an ICML/ICLR/NeurIPS-style experiment_report.md
under the trial directory.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import os

__all__ = ["write_experiment_report"]


def _extract_methods_from_metrics(metrics_obj: Dict[str, Any],
                                  metric_keys=None) -> Dict[str, Dict[str, Any]]:
    """
    Normalize metrics.json into a mapping: method_name -> {metric_name -> value}.

    Heuristics:
      - Prefer top-level containers like "models"/"methods"/"variants"
      - Otherwise treat any top-level key whose value is a dict containing known metric keys as a method
      - As a last resort, treat the whole metrics_obj as a single method named "main"
    """
    if metric_keys is None:
        metric_keys = ["MSE", "PCC", "R2", "DEG_PCC", "MSE_DM", "PCC_DM", "R2_DM"]

    methods: Dict[str, Dict[str, Any]] = {}

    # 1) explicit containers
    for container_key in ("models", "methods", "variants"):
        block = metrics_obj.get(container_key)
        if isinstance(block, dict):
            for name, val in block.items():
                if isinstance(val, dict) and any(k in val for k in metric_keys):
                    methods[name] = val
            if methods:
                return methods

    # 2) top-level dicts that look like metric groups
    for name, val in metrics_obj.items():
        if isinstance(val, dict) and any(k in val for k in metric_keys):
            methods[name] = val

    if methods:
        return methods

    # 3) single-method fallback
    return {"main": dict(metrics_obj or {})}


def _choose_baseline_name(methods: Dict[str, Dict[str, Any]]) -> str:
    """
    Heuristic baseline selection:
      - Prefer any name containing 'baseline' (case-insensitive)
      - Otherwise prefer names containing 'base'
      - Fallback: the first key in insertion order
    """
    names = list(methods.keys())
    lower_to_name = {n.lower(): n for n in names}
    for ln, n in lower_to_name.items():
        if "baseline" in ln:
            return n
    for ln, n in lower_to_name.items():
        if "base" in ln:
            return n
    return names[0] if names else "main"


def _format_metric_value(val: Any) -> str:
    try:
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        v = float(val)
        return f"{v:.4f}"
    except Exception:
        return "" if val is None else str(val)


def _compute_relative_improvement(current: Any,
                                  baseline: Any,
                                  metric_name: str) -> Optional[float]:
    """
    Compute relative improvement (in %) of 'current' vs 'baseline' for a given metric.

    For MSE-style metrics (names starting with 'MSE'), lower is better:
        impr% = (baseline - current) / |baseline| * 100

    For correlation / R2-style metrics, higher is better:
        impr% = (current - baseline) / |baseline| * 100
    """
    try:
        c = float(current)
        b = float(baseline)
    except Exception:
        return None
    if b == 0:
        return None

    lower_is_better = metric_name.upper().startswith("MSE")
    if lower_is_better:
        impr = (b - c) / abs(b) * 100.0
    else:
        impr = (c - b) / abs(b) * 100.0
    return impr


def write_experiment_report(trial_dir: str,
                            metrics_obj: Dict[str, Any],
                            *,
                            primary_metric: Optional[str] = None) -> str:
    """
    Build an ICML/ICLR/NeurIPS-style experiment report in Markdown and save it as
    `experiment_report.md` under `trial_dir`.

    The report includes:
      - clear description of compared models/variants
      - a Markdown table with MSE/PCC/R2/DEG_PCC/MSE_DM/PCC_DM/R2_DM for all methods
      - relative improvement (percentage) on the primary metric vs. baseline
      - a logically structured contribution analysis section
    """
    metric_keys = ["MSE", "PCC", "R2", "DEG_PCC", "MSE_DM", "PCC_DM", "R2_DM"]
    methods = _extract_methods_from_metrics(metrics_obj, metric_keys)

    if not methods:
        raise ValueError("No methods or metrics found in metrics.json; cannot build experiment_report.md.")

    baseline_name = _choose_baseline_name(methods)

    # Determine primary metric: fall back to MSE if not provided or missing
    pm = (primary_metric or "MSE").strip()
    if pm not in metric_keys:
        metric_keys = list(metric_keys) + [pm]

    baseline_metrics = methods.get(baseline_name, {})
    baseline_pm_val = baseline_metrics.get(pm)

    lines: list[str] = []

    # --- Header ---
    lines.append("# Experiment Report")
    lines.append("")
    lines.append(f"- Trial directory: `{trial_dir}`")
    lines.append(f"- Baseline configuration: **{baseline_name}**")
    lines.append(f"- Primary metric: **{pm}**")
    lines.append("")

    # --- Compared models / variants ---
    lines.append("## Compared models and variants")
    lines.append("")
    lines.append("We evaluate the following models and ablation variants under a shared dataset,")
    lines.append("training pipeline, and evaluation protocol:")
    lines.append("")
    for name, vals in methods.items():
        nice_name = name.replace("_", " ").replace("-", " ").title()
        if name == baseline_name:
            role = (
                "This configuration serves as the **baseline**, providing the reference "
                "level of performance for all subsequent variants."
            )
        else:
            role = (
                "This configuration is a **variant** that modifies one or more architectural, "
                "regularization, or optimization components relative to the baseline."
            )
        lines.append(f"- **{nice_name}** — {role}")
    lines.append("")

    # --- Quantitative table ---
    header = ["Model", "MSE", "PCC", "R2", "DEG_PCC", "MSE_DM", "PCC_DM", "R2_DM",
              f"Δ% ({pm}) vs baseline"]
    lines.append("## Quantitative results")
    lines.append("")
    lines.append(
        "The following table summarizes the aggregate performance of all methods on the global "
        "metrics (MSE, PCC, R2, DEG_PCC) and the differential metrics (MSE_DM, PCC_DM, R2_DM). "
        "The rightmost column reports relative improvement on the primary metric with respect "
        "to the baseline configuration."
    )
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for name, vals in methods.items():
        row: list[str] = [name]
        for mk in ["MSE", "PCC", "R2", "DEG_PCC", "MSE_DM", "PCC_DM", "R2_DM"]:
            row.append(_format_metric_value(vals.get(mk)))
        pm_val = vals.get(pm)
        rel = None
        if baseline_pm_val is not None and pm_val is not None:
            rel = _compute_relative_improvement(pm_val, baseline_pm_val, pm)
        row.append(f"{rel:.1f}%" if isinstance(rel, float) else "")
        lines.append("| " + " | ".join(row) + " |")

    # --- Relative improvements & best performers ---
    lines.append("")
    lines.append("## Relative improvements and metric-wise best performers")
    lines.append("")
    lines.append(
        "Interpreting the table, we focus on changes in the primary metric and on which "
        "configurations dominate the others for each metric:"
    )

    best_per_metric: Dict[str, tuple[str, Any]] = {}

    # Lower-is-better metrics
    for mk in ("MSE", "MSE_DM"):
        best_name = None
        best_val = None
        for name, vals in methods.items():
            v = vals.get(mk)
            try:
                v = float(v)
            except Exception:
                continue
            if best_val is None or v < best_val:
                best_val, best_name = v, name
        if best_name is not None:
            best_per_metric[mk] = (best_name, best_val)

    # Higher-is-better metrics
    for mk in ("PCC", "R2", "DEG_PCC", "PCC_DM", "R2_DM"):
        best_name = None
        best_val = None
        for name, vals in methods.items():
            v = vals.get(mk)
            try:
                v = float(v)
            except Exception:
                continue
            if best_val is None or v > best_val:
                best_val, best_name = v, name
        if best_name is not None:
            best_per_metric[mk] = (best_name, best_val)

    if best_per_metric:
        lines.append("")
        lines.append("Across metrics, the strongest configurations are:")
        for mk, (bn, bv) in best_per_metric.items():
            lines.append(f"- **{mk}** — best achieved by **{bn}** with {_format_metric_value(bv)}.")
    else:
        lines.append("")
        lines.append(
            "Due to missing or non-numeric values, best-performing models per metric could not "
            "be reliably determined."
        )

    # --- Contribution analysis ---
    lines.append("")
    lines.append("## Contribution analysis: architecture, training strategy, and ablations")
    lines.append("")
    lines.append(
        "From an architectural viewpoint, the non-baseline variants can be interpreted as "
        "incremental modifications to the baseline design. Variants that consistently improve "
        "over the baseline on both global and differential metrics suggest that the added "
        "components contribute positively to representational capacity, optimization stability, "
        "or domain-specific inductive bias."
    )
    lines.append("")
    lines.append(
        "When a variant decreases error-based metrics (MSE, MSE_DM) while simultaneously "
        "increasing correlation-based metrics (PCC, R2, DEG_PCC, PCC_DM, R2_DM), it indicates "
        "that the model not only reduces absolute discrepancy but also preserves the rank "
        "ordering and effect sizes across perturbations. This pattern is characteristic of "
        "better-calibrated architectures, more robust training strategies (e.g., improved "
        "learning-rate schedules or regularization), and appropriate handling of the underlying "
        "biological signal."
    )
    lines.append("")
    lines.append(
        "Conversely, if a variant improves the primary metric but degrades one or more "
        "differential metrics, it may be overfitting global trends at the expense of "
        "condition-specific responses. Such behavior typically motivates targeted ablation "
        "studies—removing specific modules, altering normalization schemes, or simplifying the "
        "loss—to disentangle which components are genuinely responsible for the observed gains."
    )
    lines.append("")
    lines.append(
        "Overall, the comparative results provide a quantitative basis for selecting a "
        "preferred model configuration for downstream use. In a NeurIPS/ICML/ICLR-style "
        "experimental section, one would typically extend this analysis with controlled "
        "ablations (e.g., removing the novel architecture block, turning off auxiliary losses, "
        "or modifying the training curriculum) and report how each change affects both the "
        "primary and differential metrics. The patterns observed here should guide those "
        "follow-up experiments: variants that dominate the baseline on most metrics are "
        "strong candidates for further refinement and deployment."
    )

    out_path = os.path.join(trial_dir, "experiment_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path
