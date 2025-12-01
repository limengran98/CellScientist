# -*- coding: utf-8 -*-
"""
Experiment report generator (Deep Metric Hunter & NaN-Safe Version).
Refactored to load prompts from YAML with robust path discovery and extended metrics.
"""

# design_execution/experiment_report.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
import os, json, re, math
import numpy as np
from scipy import stats

try:
    import yaml
except ImportError:
    yaml = None

# [MODIFIED] Use new centralized LLM Utils
try:
    from .llm_utils import chat_text
except ImportError:
    pass

__all__ = ["write_experiment_report"]

# =============================================================================
# 0. Helper: Template Expansion
# =============================================================================

def _expand_vars(text: str, context: Dict[str, str]) -> str:
    """Recursively expand ${VAR} in string."""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(context.get(m.group(1), m.group(0))), text)

def _load_prompt_template(pm: str, metrics_str: str) -> tuple[str, str]:
    """
    Attempts to load prompts/experiment_report.yaml.
    Returns (system_prompt, user_content).
    """
    # 1. Define Candidate Paths to search for the yaml
    candidates = [
        os.path.join(os.getcwd(), "prompts", "experiment_report.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "experiment_report.yaml"),
        os.path.join(os.path.dirname(__file__), "prompts", "experiment_report.yaml")
    ]
    
    prompt_path = None
    for p in candidates:
        if os.path.exists(p):
            prompt_path = p
            break

    # 2. Default Hardcoded Prompts (Fallback)
    default_system = f"""
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
    default_user = f"```json\n{metrics_str}\n```\n\nWrite report."

    # 3. Try to load from YAML if found
    if prompt_path:
        if yaml is None:
            print(f"[REPORT] ⚠️ Found {prompt_path}, but 'PyYAML' is not installed. Using fallback.")
        else:
            try:
                print(f"[REPORT] 📄 Loading prompt template from: {prompt_path}")
                with open(prompt_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                
                context = {
                    "primary_metric": pm,
                    "metrics_json": metrics_str
                }
                
                sys_tmpl = data.get("system", default_system)
                usr_tmpl = data.get("user_template", default_user)
                
                return _expand_vars(sys_tmpl, context), _expand_vars(usr_tmpl, context)
                
            except Exception as e:
                print(f"[REPORT][WARN] Failed to parse YAML: {e}. Using fallback.")
    else:
        print("[REPORT] ⚠️ 'experiment_report.yaml' not found in search paths. Using hardcoded fallback.")

    return default_system, default_user

# =============================================================================
# 1. Data Cleaning & Sanitization
# =============================================================================

def _sanitize_json_values(obj: Any) -> Any:
    """Recursively clean data to be strict JSON compliant."""
    if hasattr(obj, "item"): obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj): return None 
        if math.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict): return {k: _sanitize_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_sanitize_json_values(x) for x in obj]
    return obj

def _simplify_metrics_for_llm(obj: Any, depth: int = 0) -> Any:
    """Strips detailed per-fold data to save tokens."""
    obj = _sanitize_json_values(obj)
    if depth > 4: return "..."
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k.lower() in ["per_fold", "folds", "history", "predictions", "per_fold_details", "config", "epochs_trained"]:
                continue
            new_obj[k] = _simplify_metrics_for_llm(v, depth + 1)
        return new_obj
    elif isinstance(obj, list):
        if len(obj) > 5: return f"List(len={len(obj)})"
        return [_simplify_metrics_for_llm(x, depth + 1) for x in obj]
    return obj

def _normalize_metrics(metrics_obj: Dict[str, Any]) -> Dict[str, Any]:
    target = metrics_obj
    if "models" in metrics_obj: target = metrics_obj["models"]
    elif "methods" in metrics_obj: target = metrics_obj["methods"]
    
    cleaned = {}
    for k, v in target.items():
        if isinstance(v, dict) and k not in ["winner", "trial_dir", "config"]:
             cleaned[k] = v
    return cleaned if cleaned else target

def _guess_baseline_name(models_dict: Dict[str, Any]) -> str:
    keys = list(models_dict.keys())
    for k in keys:
        if "baseline" in k.lower() or "linear" in k.lower() or "ridge" in k.lower(): return k
    for k in keys: 
        if "base" in k.lower(): return k
    return keys[0] if keys else "unknown"

# =============================================================================
# 2. Smart Metric Extraction & Statistics
# =============================================================================

def _smart_get_metric(model_data: Dict[str, Any], metric_key: str) -> Optional[float]:
    if "aggregate" in model_data and isinstance(model_data["aggregate"], dict):
        val = model_data["aggregate"].get(metric_key)
        if val is not None: return float(val)

    if metric_key in model_data:
        val = model_data[metric_key]
        if isinstance(val, (int, float)): return float(val)

    fold_container = None
    if "per_fold" in model_data: fold_container = model_data["per_fold"]
    elif "folds" in model_data: fold_container = model_data["folds"]
    
    if fold_container and isinstance(fold_container, dict):
        values = []
        for fold_data in fold_container.values():
            if isinstance(fold_data, dict):
                val = fold_data.get(metric_key)
                if val is None:
                    nested = fold_data.get('metrics', fold_data.get('val', {}))
                    val = nested.get(metric_key)
                if val is not None:
                    values.append(float(val))
        if values: return float(np.mean(values))
    return None

def _extract_fold_values(model_data: Dict[str, Any], metric_key: str) -> Optional[List[float]]:
    container = None
    if "per_fold" in model_data: container = model_data["per_fold"]
    elif "folds" in model_data: container = model_data["folds"]
    
    if container and isinstance(container, dict):
        try:
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
# 3. Report Generators
# =============================================================================

def _fallback_static_report(trial_dir: str, baseline_name: str, pm: str, metrics_data: Dict, stats_data: Dict) -> str:
    """Python-based markdown generator fallback."""
    out_path = os.path.join(trial_dir, "experiment_report.md")
    lines = [f"# Experiment Report (Auto-Generated)", "", f"**Primary Metric**: {pm} | **Baseline**: {baseline_name}", ""]
    
    lines.append("## Quantitative Results")
    headers = ["Model", "MSE", "PCC", "R2", "MSE_DM", "PCC_DM", f"Delta ({pm})", "P-Value"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    base_val = _smart_get_metric(metrics_data.get(baseline_name, {}), pm)

    for name, data in metrics_data.items():
        if name.startswith("_") or name == "winner": continue
        row = [f"**{name}**"]
        
        for m in ["MSE", "PCC", "R2", "MSE_DM", "PCC_DM"]:
            val = _smart_get_metric(data, m)
            row.append(f"{val:.4f}" if val is not None else "-")
        
        curr_val = _smart_get_metric(data, pm)
        d_str = "-"
        if base_val is not None and curr_val is not None and base_val != 0:
            try:
                is_error_metric = any(x in pm.upper() for x in ["MSE", "RMSE", "LOSS", "MAE"])
                if is_error_metric: imp = (base_val - curr_val) / abs(base_val) * 100
                else: imp = (curr_val - base_val) / abs(base_val) * 100
                d_str = f"{imp:+.2f}%"
            except: pass
        row.append(d_str)

        p_str = "-"
        if name in stats_data and "p_value" in stats_data[name]:
            p = stats_data[name]["p_value"]
            sig = "*" if stats_data[name].get("significant") else ""
            p_str = f"{p:.4f}{sig}"
        row.append(p_str)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("### Analysis Notes")
    lines.append(f"- **Primary Metric ({pm})**: Used to calculate 'Delta' and determine the winner.")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path

def write_experiment_report(trial_dir: str,
                            metrics_obj: Dict[str, Any],
                            cfg: Dict[str, Any],
                            primary_metric: Optional[str] = None) -> str:
    
    # 1. Robust Loading & Normalization
    models_data = _normalize_metrics(metrics_obj)
    
    # [CRITICAL FIX] Strict check: If no models/metrics, do not hallucinate a report.
    if not models_data:
        print("[REPORT] ⚠️ Metrics object is empty (No models found). Generating empty report.", flush=True)
        out_path = os.path.join(trial_dir, "experiment_report.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Experiment Report\n\n**Status**: No metrics data available.\n")
        return out_path

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
    
    # 4. Inject Computed Aggregates
    # This step ensures that if metrics are only in per_fold, they are promoted to aggregate
    # so the LLM can actually see them in the simplified JSON.
    metrics_allowlist = [
        "MSE", "PCC", "R2", "MSE_DM", "PCC_DM", "R2_DM",
        "DEG_RMSE_20", "DEG_RMSE_50", "DEG_PCC_20", "DEG_PCC_50"
    ]
    
    for m_name, m_data in models_data.items():
        if m_name not in slim_data: continue
        if "aggregate" not in slim_data[m_name]:
            slim_data[m_name]["aggregate"] = {}
        
        for k in metrics_allowlist:
            val = _smart_get_metric(m_data, k)
            if val is not None:
                slim_data[m_name]["aggregate"][k] = val

    metrics_str = json.dumps(slim_data, indent=2)
    print(f"[REPORT] Metrics payload prepared ({len(metrics_str)} chars). Generating analysis...")

    # 5. Load Prompts (Dynamic from YAML with Robust Search)
    system_prompt, user_content = _load_prompt_template(pm, metrics_str)

    try:
        # Call LLM
        report_content = chat_text(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            llm_config=cfg.get("llm", {}),  # [FIX] Use llm_config key to match utils
            timeout=600,
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