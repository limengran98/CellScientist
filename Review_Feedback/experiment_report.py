# -*- coding: utf-8 -*-
"""
Experiment report generator (Deep Metric Hunter & NaN-Safe Version).
Refactored to load prompts from YAML with robust path discovery and extended metrics.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List
import os, json, re, math
import numpy as np
from scipy import stats

try:
    import yaml
except ImportError:
    yaml = None

# [FIX] Point to the correct utils file (llm_utils.py) present in your zip
try:
    from llm_utils import chat_text
except ImportError:
    # Try relative import if running as a package
    try:
        from .llm_utils import chat_text
    except ImportError:
        print("[REPORT] WARN: llm_utils not found. LLM features disabled.")
        chat_text = None

__all__ = ["write_experiment_report"]

# =============================================================================
# 0. Helper: Template Expansion
# =============================================================================

def _expand_vars(text: str, context: Dict[str, str]) -> str:
    """Recursively expand ${VAR} in string."""
    if not text: return ""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(context.get(m.group(1), m.group(0))), text)

def _load_prompt_template(pm: str, metrics_str: str) -> tuple[str, str]:
    """
    Attempts to load prompts/experiment_report.yaml with robust path search.
    """
    # 1. Define Candidate Paths
    # We check multiple locations to be robust against where the script is run from.
    candidates = [
        # Priority 1: Current Working Directory (standard execution)
        os.path.join(os.getcwd(), "prompts", "experiment_report.yaml"),
        
        # Priority 2: Script Directory (Fallback)
        os.path.join(os.path.dirname(__file__), "prompts", "experiment_report.yaml"),

        # Priority 3: Parent/prompts (Common in project structures)
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "experiment_report.yaml")
    ]
    
    prompt_path = None
    for p in candidates:
        if os.path.exists(p):
            prompt_path = p
            break

    # 2. Default Fallback (Hardcoded)
    default_system = f"Analyze results. Primary Metric: {pm}. Output Markdown."
    default_user = f"```json\n{metrics_str}\n```"

    # 3. Load from YAML
    if prompt_path and yaml:
        try:
            print(f"[REPORT] 📄 Loading prompt template from: {prompt_path}")
            with open(prompt_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            context = {"primary_metric": pm, "metrics_json": metrics_str}
            sys_tmpl = data.get("system", default_system)
            usr_tmpl = data.get("user_template", default_user)
            return _expand_vars(sys_tmpl, context), _expand_vars(usr_tmpl, context)
        except Exception as e:
            print(f"[REPORT][WARN] YAML Load Error: {e}")

    if not prompt_path:
        print("[REPORT] ⚠️ 'experiment_report.yaml' not found in search paths. Using Hardcoded Fallback.")
    
    return default_system, default_user

# =============================================================================
# 1. Data Cleaning & Stats Utils
# =============================================================================

def _sanitize_json_values(obj: Any) -> Any:
    if hasattr(obj, "item"): obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj): return None
        if math.isinf(obj): return "Infinity"
    if isinstance(obj, dict): return {k: _sanitize_json_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_sanitize_json_values(x) for x in obj]
    return obj

def _simplify_metrics_for_llm(obj: Any, depth: int = 0) -> Any:
    obj = _sanitize_json_values(obj)
    if depth > 4: return "..."
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Filter out bulky keys to save tokens
            if k in ["per_fold", "history", "predictions", "per_fold_details", "config"]: continue
            new_obj[k] = _simplify_metrics_for_llm(v, depth + 1)
        return new_obj
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
    return keys[0] if keys else "unknown"

# [NEW] Helper to safely extract metric from complex nested dicts
def _smart_get_metric(model_data: Dict[str, Any], metric_key: str) -> Optional[float]:
    # 1. Check Aggregate
    if "aggregate" in model_data and isinstance(model_data["aggregate"], dict):
        val = model_data["aggregate"].get(metric_key)
        if val is not None: return float(val)
    # 2. Check Root
    if metric_key in model_data:
        val = model_data[metric_key]
        if isinstance(val, (int, float)): return float(val)
    # 3. Calculate from Per-Fold (Rescue missing data)
    fold_container = model_data.get("per_fold") or model_data.get("folds")
    if fold_container and isinstance(fold_container, dict):
        vals = []
        for f in fold_container.values():
            if isinstance(f, dict):
                v = f.get(metric_key, f.get("metrics", {}).get(metric_key))
                if v is not None: vals.append(float(v))
        if vals: return float(np.mean(vals))
    return None

# [NEW] Helper to extract fold lists for T-Tests
def _extract_fold_values(model_data: Dict[str, Any], metric_key: str) -> Optional[List[float]]:
    container = model_data.get("per_fold") or model_data.get("folds")
    if container and isinstance(container, dict):
        # Sort by fold index to ensure alignment
        sorted_keys = sorted(container.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        values = []
        for k in sorted_keys:
            fold_res = container[k]
            val = fold_res.get(metric_key, fold_res.get("metrics", {}).get(metric_key))
            if val is not None: values.append(float(val))
        if len(values) >= 2: return values
    return None

# [NEW] Compute P-Values
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
                    stats_summary[name] = {
                        "metric": primary_metric,
                        "p_value": float(p_val),
                        "significant": bool(p_val < 0.05)
                    }
            except Exception: pass
    return stats_summary

# =============================================================================
# 2. Main Execution
# =============================================================================

def write_experiment_report(trial_dir: str,
                            metrics_obj: Dict[str, Any],
                            cfg: Dict[str, Any],
                            primary_metric: Optional[str] = None) -> str:
    
    if chat_text is None: return "[ERROR] LLM utils missing"

    # 1. Prepare Data
    pm = primary_metric or "PCC"
    models_data = _normalize_metrics(metrics_obj)
    baseline_name = _guess_baseline_name(models_data)
    
    # 2. [NEW] Compute Statistics (P-Values)
    stats_results = _compute_statistics(models_data, baseline_name, pm)

    # 3. Simplify for LLM
    slim_data = _simplify_metrics_for_llm(models_data)
    
    # 4. [NEW] Inject Stats & Enforce DEG Metrics
    if stats_results:
        slim_data["_STATISTICAL_TESTS_"] = {
            "baseline": baseline_name,
            "results": _sanitize_json_values(stats_results)
        }

    # CRITICAL: Whitelist to ensure DEG metrics appear in 'aggregate'
    # Even if simplify() removed detailed fold info, we pull these back up to the top.
    metrics_allowlist = [
        "MSE", "PCC", "R2", "MSE_DM", "PCC_DM", "R2_DM",
        "DEG_RMSE_20", "DEG_RMSE_50", "DEG_PCC_20", "DEG_PCC_50"
    ]
    
    for m_name, m_data in models_data.items():
        if m_name not in slim_data: continue
        if "aggregate" not in slim_data[m_name]:
            slim_data[m_name]["aggregate"] = {}
        
        # Force promote these metrics if they are missing in aggregate
        for k in metrics_allowlist:
            if k not in slim_data[m_name]["aggregate"]:
                val = _smart_get_metric(m_data, k)
                if val is not None:
                    slim_data[m_name]["aggregate"][k] = val

    metrics_str = json.dumps(slim_data, indent=2)
    print(f"[REPORT] Payload size: {len(metrics_str)} chars. Baseline: {baseline_name}")

    # 5. Load Prompts & Generate
    system_prompt, user_content = _load_prompt_template(pm, metrics_str)
    
    try:
        # Use debug_dir to save raw LLM interaction for inspection
        debug_dir = os.path.join(trial_dir, "llm_report_debug")
        os.makedirs(debug_dir, exist_ok=True)

        report_content = chat_text(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            cfg=cfg, 
            timeout=600,
            temperature=0.5,
            debug_dir=debug_dir 
        )
        
        # Clean Output
        cleaned = report_content.strip()
        if cleaned.startswith("```markdown"): cleaned = cleaned[11:]
        elif cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        
        out_path = os.path.join(trial_dir, "experiment_report.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned.strip())
            
        print(f"[REPORT] ✅ Report saved: {out_path}")
        return out_path

    except Exception as e:
        print(f"[REPORT][ERROR] Generation Failed: {e}")
        return os.path.join(trial_dir, "experiment_report_failed.txt")