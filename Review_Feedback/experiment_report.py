# -*- coding: utf-8 -*-
"""
Experiment report generator (Dynamic & Universal).
Refactored to support Mean +/- SD formatting and robust stats.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List, Union
import os, json, re, argparse, sys
import math
import numpy as np
from copy import deepcopy
from scipy import stats

# Add current dir to path for imports
sys.path.append(os.getcwd())

# [FIX] Enforce PyYAML requirement
try:
    import yaml
except ImportError:
    # Warning only, as we now have a high-quality default template
    print("[WARN] PyYAML not installed. Using built-in default template.")
    yaml = None

try:
    from llm_utils import chat_text
except ImportError:
    try:
        from .llm_utils import chat_text
    except ImportError:
        chat_text = None

try:
    from config_loader import load_full_config
except ImportError:
    load_full_config = None

__all__ = ["write_experiment_report"]

# =============================================================================
# 1. Dynamic Data Cleaning
# =============================================================================

def _smart_prune(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and (obj != obj): return None # NaN check
        return obj

    if isinstance(obj, list):
        return f"<List len={len(obj)} omitted>"

    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Prune large internal structures to save context
            if k.lower() in ["per_fold", "folds", "history", "predictions", "gradients", "per_fold_details", "config"]:
                continue
            cleaned_val = _smart_prune(v)
            new_dict[k] = cleaned_val
        return new_dict
    
    return str(obj)

# =============================================================================
# 2. Statistical Helpers & Formatters
# =============================================================================

def _recursive_find_metric(data: Any, target_key: str) -> Optional[float]:
    """Recursively search for a metric value (float) in a nested dict."""
    if isinstance(data, dict):
        # Direct match?
        for k, v in data.items():
            if k.lower() == target_key.lower():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
        # Recurse
        for v in data.values():
            if isinstance(v, (dict, list)):
                val = _recursive_find_metric(v, target_key)
                if val is not None: return val
    return None

def _extract_fold_values(model_data: Dict[str, Any], metric_key: str) -> Optional[List[float]]:
    """Extract list of values per fold for a given metric."""
    container = model_data.get("per_fold") or model_data.get("folds")
    if not container or not isinstance(container, dict):
        return None
    
    values = []
    # Sort keys numerically if possible (0, 1, 2...)
    sorted_keys = sorted(container.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    
    for k in sorted_keys:
        val = _recursive_find_metric(container[k], metric_key)
        if val is not None: values.append(val)
        
    return values if len(values) >= 2 else None

def _format_mean_sd(model_data: Dict[str, Any], metric_key: str) -> Union[str, float, None]:
    """
    Returns string 'Mean ± SD' if fold data exists, else returns float value or None.
    """
    # 1. Try to calculate from folds
    fold_vals = _extract_fold_values(model_data, metric_key)
    if fold_vals:
        mean_val = np.mean(fold_vals)
        std_val = np.std(fold_vals, ddof=1) # Sample SD
        return f"{mean_val:.4f} ± {std_val:.4f}"
    
    # 2. Fallback to pre-calculated aggregate
    val = _recursive_find_metric(model_data.get("aggregate", {}), metric_key)
    if val is not None:
        return float(val)

    # 3. Fallback to recursive search in model root
    return _recursive_find_metric(model_data, metric_key)

def _inject_statistics_and_formatting(metrics_data: Dict[str, Any], primary_metric: str) -> Dict[str, Any]:
    """
    1. Calculates Delta and P-Values vs Baseline.
    2. Formats all key metrics as 'Mean ± SD'.
    """
    data_copy = deepcopy(metrics_data)
    root = data_copy.get("models", data_copy.get("methods", data_copy))
    if not isinstance(root, dict): return data_copy

    # Identify Baseline
    baseline_key = next((k for k in root.keys() if "baseline" in k.lower()), list(root.keys())[0])
    baseline_data = root.get(baseline_key, {})
    
    # Get Baseline Primary Metric for Comparison
    base_val = _recursive_find_metric(baseline_data, primary_metric)
    base_folds = _extract_fold_values(baseline_data, primary_metric)

    # Metrics to format
    metrics_to_format = [
        "MSE", "PCC", "R2", "MSE_DM", "PCC_DM", "R2_DM",
        "DEG_RMSE_20", "DEG_RMSE_50", "DEG_PCC_20", "DEG_PCC_50"
    ]
    # Ensure primary metric is included
    if primary_metric not in metrics_to_format:
        metrics_to_format.append(primary_metric)

    for name, model_data in root.items():
        # --- A. Statistical Comparison ---
        stats_info = {
            "is_baseline": (name == baseline_key),
            "primary_metric_name": primary_metric,
            "delta_vs_baseline": "-",
            "p_value_vs_baseline": "-"
        }

        curr_val = _recursive_find_metric(model_data, primary_metric)
        
        # Calculate Delta %
        if curr_val is not None and base_val is not None and base_val != 0:
            delta = (curr_val - base_val) / abs(base_val) * 100
            stats_info["delta_vs_baseline"] = f"{delta:+.2f}%"

        # Calculate P-Value
        # Note: We need to access original un-modified data to get fold lists
        orig_model = metrics_data.get("models", {}).get(name, {})
        curr_folds = _extract_fold_values(orig_model, primary_metric)
        
        if name != baseline_key and base_folds and curr_folds and len(base_folds) == len(curr_folds):
            try:
                _, p = stats.ttest_rel(curr_folds, base_folds)
                mark = "*" if p < 0.05 else ""
                stats_info["p_value_vs_baseline"] = f"{p:.4e}{mark}"
            except: pass

        model_data["_STATISTICS_HELPER_"] = stats_info

        # --- B. Format Metrics (Mean ± SD) ---
        # We inject these formatted strings into the 'aggregate' block or root
        if "aggregate" not in model_data:
            model_data["aggregate"] = {}
            
        for m_key in metrics_to_format:
            # Get formatted string using the original data (to ensure we have folds)
            fmt_val = _format_mean_sd(orig_model, m_key)
            if fmt_val is not None:
                model_data["aggregate"][m_key] = fmt_val

    return data_copy

# =============================================================================
# 3. Main Execution
# =============================================================================

def write_experiment_report(trial_dir: str,
                            metrics_obj: Dict[str, Any],
                            cfg: Dict[str, Any],
                            primary_metric: Optional[str] = None) -> str:
    
    if chat_text is None:
        print("[REPORT] LLM utils missing. Cannot generate.")
        return ""

    pm = primary_metric or "PCC"
    
    # 1. Inject Stats & Format Metrics
    data_processed = _inject_statistics_and_formatting(metrics_obj, pm)

    # 2. Smart Prune (Removes raw folds, keeps formatted aggregates)
    clean_payload = _smart_prune(data_processed)

    # 3. Prompt Loading
    metrics_json_str = json.dumps(clean_payload, indent=2)

    sys_tmpl, user_tmpl = _load_prompt_template(pm)
    
    # [FIX] Ensure metrics_json is replaced in the user template
    user_content = _expand_vars(user_tmpl, {"metrics_json": metrics_json_str})

    # 4. Generate
    try:
        debug_dir = os.path.join(trial_dir, "llm_report_debug")
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "payload_full_dynamic.json"), "w") as f:
            f.write(metrics_json_str)

        # Copy config and set timeout
        run_cfg = deepcopy(cfg)
        if "llm" not in run_cfg: run_cfg["llm"] = {}
        run_cfg["llm"]["timeout"] = 600

        print(f"[REPORT] Generating report using metric: {pm}")
        response = chat_text(
            [{"role": "system", "content": sys_tmpl}, 
             {"role": "user", "content": user_content}],
            cfg=run_cfg,
            temperature=0.3,
            debug_dir=debug_dir 
        )
        
        # Clean Output
        cleaned = re.sub(r"^```[a-z]*\n", "", response.strip())
        cleaned = re.sub(r"\n```$", "", cleaned)
        
        out_path = os.path.join(trial_dir, "experiment_report.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
            
        print(f"[REPORT] Report generated successfully: {out_path}")
        return out_path

    except Exception as e:
        print(f"[REPORT] Error: {e}")
        import traceback
        traceback.print_exc()
        return ""

# =============================================================================
# 4. Helpers & Main
# =============================================================================

def _expand_vars(text: str, context: Dict[str, str]) -> str:
    if not text: return ""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(context.get(m.group(1), m.group(0))), text)

def _load_prompt_template(pm: str) -> tuple[str, str]:
    """
    Loads prompt from yaml. 
    Falls back to a high-quality BUILT-IN template if file load fails.
    """
    candidates = [
        os.path.join(os.getcwd(), "prompts", "experiment_report.yaml"),
        os.path.join(os.path.dirname(__file__), "prompts", "experiment_report.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "prompts", "experiment_report.yaml")
    ]
    
    # [UPDATED] High-Quality Built-in Default
    # Directly implements the user's requested template structure.
    sys_t = (
        f"You are a Senior Computational Biologist. Write an `experiment_report.md`.\n\n"
        f"**Task**:\n"
        f"1.  **Winner**: Determine solely by Primary Metric **{pm}**.\n\n"
        f"2.  **Table**: Generate a comprehensive performance table:\n"
        f"    `| Model | MSE | PCC | R2 | DEG_RMSE_20 | DEG_RMSE_50 | DEG_PCC_20 | DEG_PCC_50 | MSE_DM | PCC_DM | R2_DM | Δ% ({pm}) | P-Value |`\n"
        f"    - **Format**: Values must be displayed as **\"Mean ± SD\"** (e.g., `0.8500 ± 0.0020`) as provided in the JSON. Do not strip the standard deviation.\n"
        f"    - **DEG Metrics**: Calculated on Top-K (20/50) most changed features.\n"
        f"    - **P-Value**: From `_STATISTICAL_TESTS_`. Mark significant (p<0.05) with *.\n\n"
        f"3.  **Analysis (Concise & Holistic)**:\n"
        f"    - **Global & Differential**: Briefly assess overall fit (Global MSE/PCC) and the model's ability to predict variation from the mean (Differential metrics).\n"
        f"    - **Variance**: Comment on the stability of the models based on the Standard Deviation (SD). High SD indicates instability across folds.\n"
        f"    - **SOTA Mechanism (DEG)**: Focus on the Top-20/50 metrics. Does the Innovation better capture the *trend* (DEG_PCC) and *magnitude* (DEG_RMSE) of critical biological changes compared to the Baseline?\n"
        f"    - **Verdict**: Conclude if the Innovation offers a statistically significant and biologically meaningful improvement.\n\n"
        f"**Tone**: Scientific, objective, and concise."
    )
    
    user_t = "```json\n${metrics_json}\n```"
    
    # 1. Check PyYAML availability
    if not yaml:
        print("[WARN] PyYAML not loaded. Using built-in default prompt.")
        return sys_t, user_t

    # 2. Search for file
    found_path = None
    for p in candidates:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
        # Not finding the file is acceptable now, as we have a good default.
        print(f"[INFO] 'experiment_report.yaml' not found. Using built-in default prompt.")
        return sys_t, user_t

    # 3. Load
    try:
        print(f"[REPORT] Loading prompt from: {found_path}")
        with open(found_path, "r", encoding="utf-8") as f: 
            data = yaml.safe_load(f)
        
        # Apply system template substitutions immediately
        loaded_sys = data.get("system")
        if loaded_sys:
            sys_t = _expand_vars(loaded_sys, {"primary_metric": pm})
        
        loaded_user = data.get("user_template")
        if loaded_user:
            user_t = loaded_user
            
        return sys_t, user_t
        
    except Exception as e:
        print(f"[ERROR] Failed to parse YAML: {found_path}. Error: {e}")
        print("[WARN] Reverting to built-in default prompt.")
        return sys_t, user_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target", default="PCC")
    args = parser.parse_args()

    if not os.path.exists(args.metrics): sys.exit(f"Missing: {args.metrics}")
    
    with open(args.metrics, 'r') as f: m_data = json.load(f)
    
    if load_full_config: cfg = load_full_config(args.config)
    else:
        with open(args.config, 'r') as f: cfg = json.load(f)

    out_dir = args.output or os.path.dirname(os.path.abspath(args.metrics))
    write_experiment_report(out_dir, m_data, cfg, primary_metric=args.target)

if __name__ == "__main__":
    main()