# -*- coding: utf-8 -*-
"""
Experiment report generator (Dynamic & Universal).
Fixed: Removed invalid 'timeout' argument in chat_text call.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List
import os, json, re, argparse, sys
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
# 1. Dynamic Data Cleaning
# =============================================================================

def _smart_prune(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and (obj != obj): return None
        return obj

    if isinstance(obj, list):
        return f"<List len={len(obj)} omitted>"

    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k.lower() in ["per_fold", "folds", "history", "predictions", "gradients", "per_fold_details"]:
                continue
            cleaned_val = _smart_prune(v)
            new_dict[k] = cleaned_val
        return new_dict
    
    return str(obj)

# =============================================================================
# 2. Statistical Helpers
# =============================================================================

def _recursive_find_metric(data: Any, target_key: str) -> Optional[float]:
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
    container = model_data.get("per_fold") or model_data.get("folds")
    if not container or not isinstance(container, dict):
        return None
    
    values = []
    sorted_keys = sorted(container.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    for k in sorted_keys:
        val = _recursive_find_metric(container[k], metric_key)
        if val is not None: values.append(val)
    return values if len(values) >= 2 else None

def _inject_statistics(metrics_data: Dict[str, Any], primary_metric: str) -> Dict[str, Any]:
    data_copy = deepcopy(metrics_data)
    root = data_copy.get("models", data_copy.get("methods", data_copy))
    if not isinstance(root, dict): return data_copy

    baseline_key = next((k for k in root.keys() if "baseline" in k.lower()), list(root.keys())[0])
    baseline_data = root.get(baseline_key, {})
    base_val = _recursive_find_metric(baseline_data, primary_metric)
    base_folds = _extract_fold_values(baseline_data, primary_metric)


    for name, model_data in root.items():
        stats_info = {
            "is_baseline": (name == baseline_key),
            "primary_metric_name": primary_metric,
            "delta_vs_baseline": "-",
            "p_value_vs_baseline": "-"
        }

        curr_val = _recursive_find_metric(model_data, primary_metric)
        if curr_val is not None and base_val is not None and base_val != 0:
            delta = (curr_val - base_val) / abs(base_val) * 100
            stats_info["delta_vs_baseline"] = f"{delta:+.2f}%"

        orig_model = metrics_data.get("models", {}).get(name, {})
        curr_folds = _extract_fold_values(orig_model, primary_metric)
        
        if name != baseline_key and base_folds and curr_folds and len(base_folds) == len(curr_folds):
            try:
                _, p = stats.ttest_rel(curr_folds, base_folds)
                mark = "*" if p < 0.05 else ""
                stats_info["p_value_vs_baseline"] = f"{p:.4e}{mark}"
            except: pass

        model_data["_STATISTICS_HELPER_"] = stats_info

    return data_copy

# =============================================================================
# 3. Main Execution (FIXED)
# =============================================================================

def write_experiment_report(trial_dir: str,
                            metrics_obj: Dict[str, Any],
                            cfg: Dict[str, Any],
                            primary_metric: Optional[str] = None) -> str:
    
    if chat_text is None:
        print("[REPORT] LLM utils missing. Cannot generate.")
        return ""

    pm = primary_metric or "PCC"
    
    # 1. Inject Stats
    data_with_stats = _inject_statistics(metrics_obj, pm)

    # 2. Smart Prune
    clean_payload = _smart_prune(data_with_stats)

    # 3. Prompt Loading
    metrics_json_str = json.dumps(clean_payload, indent=2)

    sys_tmpl, user_tmpl = _load_prompt_template(pm)
    user_content = _expand_vars(user_tmpl, {"metrics_json": metrics_json_str})

    # 4. Generate (FIXED: Timeout handling)
    try:
        debug_dir = os.path.join(trial_dir, "llm_report_debug")
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "payload_full_dynamic.json"), "w") as f:
            f.write(metrics_json_str)

        # [FIX] Copy config and set timeout there, do not pass as kwarg
        run_cfg = deepcopy(cfg)
        if "llm" not in run_cfg: run_cfg["llm"] = {}
        run_cfg["llm"]["timeout"] = 600  # Set timeout in config

        response = chat_text(
            [{"role": "system", "content": sys_tmpl}, 
             {"role": "user", "content": user_content}],
            cfg=run_cfg,         # Pass modified config
            temperature=0.3,     # This is allowed
            debug_dir=debug_dir 
        )
        
        # Clean Output
        cleaned = re.sub(r"^```[a-z]*\n", "", response.strip())
        cleaned = re.sub(r"\n```$", "", cleaned)
        
        out_path = os.path.join(trial_dir, "experiment_report.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
            
        print(f"[REPORT] ✅ Report generated: {out_path}")
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
    candidates = [
        os.path.join(os.getcwd(), "prompts", "experiment_report.yaml"),
        os.path.join(os.path.dirname(__file__), "prompts", "experiment_report.yaml"),
    ]
    sys_t, user_t = f"Analyze results. Metric: {pm}", "```json\n${metrics_json}\n```"
    
    for p in candidates:
        if os.path.exists(p) and yaml:
            try:
                with open(p, "r", encoding="utf-8") as f: data = yaml.safe_load(f)
                sys_t = _expand_vars(data.get("system", sys_t), {"primary_metric": pm})
                user_t = data.get("user_template", user_t)
                break
            except: pass
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