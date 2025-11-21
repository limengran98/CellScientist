# -*- coding: utf-8 -*-
"""
Experiment report generator (Deep Metric Hunter & NaN-Safe Version).
Fixes:
1. "Empty Table" issue: Automatically calculates averages from per-fold data if aggregates are missing.
2. JSON Safety: Converts NaN/Infinity to valid JSON types to prevent API 400 errors.
3. Robust Fallback: Generates a clean Markdown table via Python even if LLM fails.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List
import os, json
import math
import numpy as np
from scipy import stats

try:
    from .prompt_builder import chat_text
except ImportError:
    from prompt_builder import chat_text

__all__ = ["write_experiment_report"]

# =============================================================================
# 1. Data Cleaning & Sanitization
# =============================================================================

def _sanitize_json_values(obj: Any) -> Any:
    """
    Recursively clean data to be strict JSON compliant.
    - Converts np.bool_ -> bool, np.integer -> int, np.floating -> float
    - Converts NaN/Inf -> None (JSON null) or string
    """
    # 1. Numpy Scalars
    if hasattr(obj, "item"):
        obj = obj.item()
        
    # 2. Float Special Values (NaN, Inf)
    if isinstance(obj, float):
        if math.isnan(obj):
            return None # JSON null
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
        
    # 3. Recursive Containers
    if isinstance(obj, dict):
        return {k: _sanitize_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_json_values(x) for x in obj]
        
    return obj

def _simplify_metrics_for_llm(obj: Any, depth: int = 0) -> Any:
    """
    Remove massive arrays to fit in context, and sanitize types.
    """
    obj = _sanitize_json_values(obj)
    
    if depth > 4: return "..."
    
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Filter out heavy logging data
            if k.lower() in ["per_fold", "folds", "history", "predictions", "per_fold_details", "config", "epochs_trained"]:
                continue
            new_obj[k] = _simplify_metrics_for_llm(v, depth + 1)
        return new_obj
    elif isinstance(obj, list):
        if len(obj) > 5: return f"List(len={len(obj)})"
        return [_simplify_metrics_for_llm(x, depth + 1) for x in obj]
    return obj

def _normalize_metrics(metrics_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the metrics object to find models."""
    target = metrics_obj
    if "models" in metrics_obj: target = metrics_obj["models"]
    elif "methods" in metrics_obj: target = metrics_obj["methods"]
    
    cleaned = {}
    for k, v in target.items():
        # Heuristic: A valid model dict usually has metric keys or 'aggregate'/'per_fold'
        if isinstance(v, dict) and k not in ["winner", "trial_dir", "config"]:
             cleaned[k] = v
    return cleaned if cleaned else target

def _guess_baseline_name(models_dict: Dict[str, Any]) -> str:
    keys = list(models_dict.keys())
    # 1. Search for specific keywords
    for k in keys:
        if "baseline" in k.lower() or "linear" in k.lower() or "ridge" in k.lower():
            return k
    # 2. Search for 'base'
    for k in keys: 
        if "base" in k.lower():
            return k
    # 3. Fallback to first key
    return keys[0] if keys else "unknown"

# =============================================================================
# 2. Smart Metric Extraction & Statistics
# =============================================================================

def _smart_get_metric(model_data: Dict[str, Any], metric_key: str) -> Optional[float]:
    """
    The "Metric Hunter": Finds a metric value no matter where it is hiding.
    Priority:
    1. model['aggregate'][key]
    2. model[key] (flat)
    3. mean(model['per_fold'][...][key]) (Auto-aggregation)
    """
    # Strategy 1: Check explicit aggregate container
    if "aggregate" in model_data and isinstance(model_data["aggregate"], dict):
        val = model_data["aggregate"].get(metric_key)
        if val is not None: return float(val)

    # Strategy 2: Check flat key
    if metric_key in model_data:
        val = model_data[metric_key]
        if isinstance(val, (int, float)): return float(val)

    # Strategy 3: Auto-aggregate from per_fold (Savior for crashed notebooks)
    fold_container = None
    if "per_fold" in model_data: fold_container = model_data["per_fold"]
    elif "folds" in model_data: fold_container = model_data["folds"]
    
    if fold_container and isinstance(fold_container, dict):
        values = []
        for fold_data in fold_container.values():
            if isinstance(fold_data, dict):
                # Deep search in fold
                val = fold_data.get(metric_key)
                if val is None:
                    # Try nested 'metrics' or 'val'
                    nested = fold_data.get('metrics', fold_data.get('val', {}))
                    val = nested.get(metric_key)
                
                if val is not None:
                    values.append(float(val))
        
        if values:
            return float(np.mean(values))

    return None

def _extract_fold_values(model_data: Dict[str, Any], metric_key: str) -> Optional[List[float]]:
    """Extract list of values for T-Test."""
    container = None
    if "per_fold" in model_data: container = model_data["per_fold"]
    elif "folds" in model_data: container = model_data["folds"]
    
    if container and isinstance(container, dict):
        try:
            # Sort keys to ensure alignment between models
            sorted_keys = sorted(container.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
            values = []
            for k in sorted_keys:
                fold_res = container[k]
                val = fold_res.get(metric_key)
                if val is None:
                    nested = fold_res.get('metrics', fold_res.get('val', {}))
                    val = nested.get(metric_key)
                
                if val is not None:
                    values.append(float(val))
            if len(values) >= 2: return values
        except Exception: pass
    return None

def _compute_statistics(models_data: Dict[str, Any], baseline_name: str, primary_metric: str) -> Dict[str, Any]:
    stats_summary = {}
    if baseline_name not in models_data: return {}
    
    base_samples = _extract_fold_values(models_data[baseline_name], primary_metric)
    if not base_samples: return {}

    for name, data in models_data.items():
        if name == baseline_name: continue
        curr_samples = _extract_fold_values(data, primary_metric)
        
        if curr_samples and len(curr_samples) == len(base_samples):
            try:
                # Use nan_policy='omit' to handle potential NaNs safely
                res = stats.ttest_rel(curr_samples, base_samples, nan_policy='omit')
                p_val = res.pvalue
                
                if not math.isnan(p_val):
                    is_sig = bool(p_val < 0.05)
                    stats_summary[name] = {
                        "metric": primary_metric,
                        "p_value": float(p_val),
                        "significant": is_sig,
                        "verdict": "Significant" if is_sig else "Not Significant"
                    }
            except Exception: pass
    return stats_summary

# =============================================================================
# 3. Report Generators (Python Fallback & LLM)
# =============================================================================

def _fallback_static_report(trial_dir: str, baseline_name: str, pm: str, metrics_data: Dict, stats_data: Dict) -> str:
    """
    Generates a guaranteed Markdown table using Python logic.
    This runs if the LLM fails or times out.
    """
    out_path = os.path.join(trial_dir, "experiment_report.md")
    lines = [f"# Experiment Report (Auto-Generated)", "", f"**Primary Metric**: {pm} | **Baseline**: {baseline_name}", ""]
    
    lines.append("## Quantitative Results")
    headers = ["Model", "MSE", "PCC", "R2", "MSE_DM", "PCC_DM", f"Delta ({pm})", "P-Value"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Get baseline score for delta calculation
    base_val = _smart_get_metric(metrics_data.get(baseline_name, {}), pm)

    for name, data in metrics_data.items():
        if name.startswith("_") or name == "winner": continue
        row = [f"**{name}**"]
        
        # 1. Fill Metrics using the "Metric Hunter"
        for m in ["MSE", "PCC", "R2", "MSE_DM", "PCC_DM"]:
            val = _smart_get_metric(data, m)
            row.append(f"{val:.4f}" if val is not None else "-")
        
        # 2. Calculate Delta
        curr_val = _smart_get_metric(data, pm)
        d_str = "-"
        if base_val is not None and curr_val is not None and base_val != 0:
            try:
                # Higher is better (PCC) -> (Curr - Base) / |Base|
                # Lower is better (MSE) -> (Base - Curr) / |Base|
                is_error_metric = any(x in pm.upper() for x in ["MSE", "RMSE", "LOSS", "MAE"])
                
                if is_error_metric:
                    imp = (base_val - curr_val) / abs(base_val) * 100
                else:
                    imp = (curr_val - base_val) / abs(base_val) * 100
                d_str = f"{imp:+.2f}%"
            except: pass
        row.append(d_str)

        # 3. Fill Stats
        p_str = "-"
        if name in stats_data and "p_value" in stats_data[name]:
            p = stats_data[name]["p_value"]
            sig = "*" if stats_data[name].get("significant") else ""
            p_str = f"{p:.4f}{sig}"
        row.append(p_str)
        
        lines.append("| " + " | ".join(row) + " |")

    # Add explanations
    lines.append("")
    lines.append("### Analysis Notes")
    lines.append(f"- **Primary Metric ({pm})**: Used to calculate 'Delta' and determine the winner.")
    lines.append("- **P-Value**: Calculated using a Paired T-Test across folds vs. Baseline. (* = p < 0.05)")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path

def write_experiment_report(trial_dir: str,
                            metrics_obj: Dict[str, Any],
                            cfg: Dict[str, Any],
                            primary_metric: Optional[str] = None) -> str:
    
    # 1. Robust Loading & Normalization
    models_data = _normalize_metrics(metrics_obj)
    baseline_name = _guess_baseline_name(models_data)
    pm = primary_metric or "PCC"

    # 2. Calculate Statistics
    stats_results = _compute_statistics(models_data, baseline_name, pm)
    
    # 3. Sanitize for LLM
    slim_data = _simplify_metrics_for_llm(models_data)
    if stats_results:
        slim_data["_STATISTICAL_TESTS_"] = {
            "target_metric": pm,
            "baseline": baseline_name,
            "results": _sanitize_json_values(stats_results)
        }
    
    # 4. Inject Computed Aggregates (The "Metric Hunter" Injection)
    # This ensures LLM sees the numbers even if they were buried in per_fold or flat keys
    for m_name, m_data in models_data.items():
        if m_name not in slim_data: continue
        if "aggregate" not in slim_data[m_name]:
            slim_data[m_name]["aggregate"] = {}
        
        for k in ["MSE", "PCC", "R2", "MSE_DM", "PCC_DM"]:
            val = _smart_get_metric(m_data, k)
            if val is not None:
                slim_data[m_name]["aggregate"][k] = val

    metrics_str = json.dumps(slim_data, indent=2)
    print(f"[REPORT] Metrics payload prepared ({len(metrics_str)} chars). Generating analysis...")

    # 5. LLM Prompt
    system_prompt = f"""
You are a Senior Computational Biologist. Write an `experiment_report.md`.

**Task**:
1.  **Winner**: Determine solely by Primary Metric **{pm}**.
2.  **Table**: `| Model | MSE | PCC | R2 | MSE_DM | PCC_DM | Δ% ({pm}) | P-Value |`.
    - P-Value: From `_STATISTICAL_TESTS_`. Mark significant (p<0.05) with *.
3.  **Analysis**:
    - Interpret the biological significance of PCC_DM (Differential Metric).
    - Discuss if the Innovation is statistically significant vs Baseline.

**Tone**: Scientific.
"""
    user_content = f"```json\n{metrics_str}\n```\n\nWrite report."

    try:
        # Call LLM
        report_content = chat_text(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            cfg=cfg,
            timeout=600, # 10 minutes
            temperature=0.5,
            max_tokens=4000
        )
        
        # Clean response
        cleaned = report_content.strip()
        if cleaned.startswith("```markdown"): cleaned = cleaned[11:]
        elif cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        
        out_path = os.path.join(trial_dir, "experiment_report.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned.strip())
        return out_path

    except Exception as e:
        print(f"[REPORT][WARN] LLM Generation Failed ({e}). Switching to Native Python Fallback.")
        return _fallback_static_report(trial_dir, baseline_name, pm, models_data, stats_results)