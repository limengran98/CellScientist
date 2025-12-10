# -*- coding: utf-8 -*-
"""
Experiment report generator (Robust Hybrid Version).
Features:
1. Dynamic Metrics Processing (Mean ± SD).
2. LLM-based Analysis (Primary).
3. Python-based Static Fallback (Guaranteed Table Format).
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

try:
    import yaml
except ImportError:
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
# 1. Data Helpers & Statistics
# =============================================================================

def _recursive_find_metric(data: Any, target_key: str) -> Optional[float]:
    """Recursively search for a metric value (float) in a nested dict."""
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() == target_key.lower():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
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
    sorted_keys = sorted(container.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    
    for k in sorted_keys:
        val = _recursive_find_metric(container[k], metric_key)
        if val is not None: values.append(val)
        
    return values if len(values) >= 2 else None

def _format_mean_sd(model_data: Dict[str, Any], metric_key: str) -> str:
    """
    Returns string 'Mean ± SD' if fold data exists, else returns float formatted string or '-'.
    """
    # 1. Try fold data
    fold_vals = _extract_fold_values(model_data, metric_key)
    if fold_vals:
        mean_val = np.mean(fold_vals)
        std_val = np.std(fold_vals, ddof=1)
        return f"{mean_val:.4f} ± {std_val:.4f}"
    
    # 2. Try aggregate or direct
    val = _recursive_find_metric(model_data.get("aggregate", {}), metric_key)
    if val is None:
        val = _recursive_find_metric(model_data, metric_key)
        
    if val is not None:
        return f"{val:.4f}"
    
    return "-"

def _smart_prune(obj: Any) -> Any:
    """Prepare JSON for LLM by removing bulky raw data."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and (obj != obj): return None 
        return obj
    if isinstance(obj, list):
        return f"<List len={len(obj)} omitted>"
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k.lower() in ["per_fold", "folds", "history", "predictions", "gradients", "config"]:
                continue
            new_dict[k] = _smart_prune(v)
        return new_dict
    return str(obj)

def _prepare_metrics_payload(metrics_obj: Dict[str, Any], primary_metric: str) -> Dict[str, Any]:
    """Injects formatted stats into the metrics object for the LLM/Report."""
    data = deepcopy(metrics_obj)
    root = data.get("models", data.get("methods", data))
    
    # Define the exact columns user wants
    target_metrics = [
        "MSE", "PCC", "R2", 
        "DEG_RMSE_20", "DEG_RMSE_50", "DEG_PCC_20", "DEG_PCC_50", 
        "MSE_DM", "PCC_DM", "R2_DM"
    ]
    if primary_metric not in target_metrics: target_metrics.append(primary_metric)

    baseline_key = next((k for k in root.keys() if "baseline" in k.lower()), list(root.keys())[0])
    base_folds = _extract_fold_values(root.get(baseline_key, {}), primary_metric)

    for name, model_data in root.items():
        if "aggregate" not in model_data: model_data["aggregate"] = {}
        
        # 1. Format Mean ± SD for all target metrics
        for m in target_metrics:
            model_data["aggregate"][m] = _format_mean_sd(root[name], m) # Use root[name] to access folds

        # 2. P-Value Calculation
        curr_folds = _extract_fold_values(root[name], primary_metric)
        p_str = "-"
        if name != baseline_key and base_folds and curr_folds and len(base_folds) == len(curr_folds):
            try:
                _, p = stats.ttest_rel(curr_folds, base_folds)
                mark = "*" if p < 0.05 else ""
                p_str = f"{p:.4e}{mark}"
            except: pass
        
        model_data["aggregate"]["p_value_vs_baseline"] = p_str

    return data

# =============================================================================
# 2. Static Fallback Report Generator (The Fix)
# =============================================================================

def _fallback_static_report(trial_dir: str, metrics_payload: Dict[str, Any], pm: str) -> str:
    """
    Generates the markdown report via Python logic if LLM fails.
    Strictly adheres to the requested table format.
    """
    root = metrics_payload.get("models", metrics_payload.get("methods", metrics_payload))
    baseline_key = next((k for k in root.keys() if "baseline" in k.lower()), list(root.keys())[0])
    base_val = _recursive_find_metric(root.get(baseline_key, {}), pm)

    lines = [
        f"# Experiment Report (Static Fallback)", 
        "", 
        f"**Primary Metric**: {pm} | **Baseline**: {baseline_key}", 
        ""
    ]
    
    # --- TABLE GENERATION ---
    lines.append("## Quantitative Results (Mean ± SD)")
    
    # EXACT Columns requested by user
    headers = [
        "Model", 
        "MSE", "PCC", "R2", 
        "DEG_RMSE_20", "DEG_RMSE_50", "DEG_PCC_20", "DEG_PCC_50", 
        "MSE_DM", "PCC_DM", "R2_DM", 
        f"Delta ({pm})", "P-Value"
    ]
    
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    metric_keys = [
        "MSE", "PCC", "R2", 
        "DEG_RMSE_20", "DEG_RMSE_50", "DEG_PCC_20", "DEG_PCC_50", 
        "MSE_DM", "PCC_DM", "R2_DM"
    ]

    for name, data in root.items():
        if name == "winner" or name.startswith("_"): continue
        
        agg = data.get("aggregate", {})
        row = [f"**{name}**"]
        
        # 1. Standard Metrics
        for k in metric_keys:
            row.append(str(agg.get(k, "-")))
        
        # 2. Delta %
        curr_val = _recursive_find_metric(data, pm)
        d_str = "-"
        if base_val is not None and curr_val is not None and base_val != 0:
            imp = (curr_val - base_val) / abs(base_val) * 100
            d_str = f"{imp:+.2f}%"
        row.append(d_str)

        # 3. P-Value
        row.append(str(agg.get("p_value_vs_baseline", "-")))

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("### Note")
    lines.append("- This report was generated by the static fallback engine (LLM unavailable).")
    lines.append("- **DEG Metrics**: Calculated on Top-20/50 most changed features.")
    
    out_path = os.path.join(trial_dir, "experiment_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return out_path

# =============================================================================
# 3. Prompt & LLM Handling
# =============================================================================

def _load_prompt_template(pm: str) -> tuple[str, str]:
    # Search paths for yaml
    candidates = [
        os.path.join(os.getcwd(), "prompts", "experiment_report.yaml"),
        os.path.join(os.path.dirname(__file__), "prompts", "experiment_report.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "experiment_report.yaml"),
    ]
    
    # DEFAULT PROMPT (Updated to match your requirements exactly)
    sys_t = f"""
You are a Senior Computational Biologist. Write an `experiment_report.md`.
**Task**:
1. **Table**: Generate a table with these EXACT columns:
   `| Model | MSE | PCC | R2 | DEG_RMSE_20 | DEG_RMSE_50 | DEG_PCC_20 | DEG_PCC_50 | MSE_DM | PCC_DM | R2_DM | Δ% ({pm}) | P-Value |`
   - Use "Mean ± SD" format.
2. **Analysis**: Discuss biological significance of DEG (Top-K genes) and Differential Metrics.
"""
    user_t = "```json\n${metrics_json}\n```"

    for p in candidates:
        if os.path.exists(p):
            if yaml:
                try:
                    with open(p, "r", encoding="utf-8") as f: 
                        data = yaml.safe_load(f)
                    # Expand vars inside the prompt text immediately
                    sys_t = data.get("system", sys_t).replace("${primary_metric}", pm)
                    user_t = data.get("user_template", user_t)
                    print(f"[REPORT] Loaded prompt from: {p}")
                    break
                except Exception as e:
                    print(f"[REPORT] Error parsing {p}: {e}")
            else:
                print(f"[REPORT] Found {p} but PyYAML not installed.")

    return sys_t, user_t

# =============================================================================
# 4. Main Entry Point
# =============================================================================

def write_experiment_report(trial_dir: str,
                            metrics_obj: Dict[str, Any],
                            cfg: Dict[str, Any],
                            primary_metric: Optional[str] = None) -> str:
    
    pm = primary_metric or "PCC"
    
    # 1. Prepare Data (Calc Stats + Format Strings)
    # This ensures both LLM and Fallback see "0.123 ± 0.001"
    full_payload = _prepare_metrics_payload(metrics_obj, pm)
    clean_payload = _smart_prune(full_payload)
    metrics_json_str = json.dumps(clean_payload, indent=2)

    # 2. Try LLM Generation
    if chat_text:
        try:
            sys_tmpl, user_tmpl = _load_prompt_template(pm)
            
            # Simple variable expansion
            user_content = user_tmpl.replace("${metrics_json}", metrics_json_str)

            run_cfg = deepcopy(cfg)
            if "llm" not in run_cfg: run_cfg["llm"] = {}
            run_cfg["llm"]["timeout"] = 600 # Ensure enough time
            
            print("[REPORT] Sending to LLM...")
            response = chat_text(
                [{"role": "system", "content": sys_tmpl}, 
                 {"role": "user", "content": user_content}],
                cfg=run_cfg,
                temperature=0.3
            )
            
            if response:
                # Cleanup markdown fences
                cleaned = re.sub(r"^```[a-z]*\n", "", response.strip())
                cleaned = re.sub(r"\n```$", "", cleaned)
                
                out_path = os.path.join(trial_dir, "experiment_report.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                print(f"[REPORT] LLM Report generated: {out_path}")
                return out_path
            else:
                print("[REPORT] LLM returned empty response. Switching to Fallback.")

        except Exception as e:
            print(f"[REPORT] LLM Generation Failed: {e}")
    else:
        print("[REPORT] LLM utils not available. Switching to Fallback.")

    # 3. Static Fallback (Guaranteed Output)
    print("[REPORT] Generating Static Fallback Report...")
    return _fallback_static_report(trial_dir, clean_payload, pm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target", default="PCC")
    args = parser.parse_args()

    if not os.path.exists(args.metrics): sys.exit(f"Missing: {args.metrics}")
    
    with open(args.metrics, 'r') as f: m_data = json.load(f)
    
    cfg = {}
    if load_full_config: 
        cfg = load_full_config(args.config)
    else:
        with open(args.config, 'r') as f: cfg = json.load(f)

    out_dir = args.output or os.path.dirname(os.path.abspath(args.metrics))
    write_experiment_report(out_dir, m_data, cfg, primary_metric=args.target)

if __name__ == "__main__":
    main()