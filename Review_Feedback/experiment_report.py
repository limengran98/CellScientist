# -*- coding: utf-8 -*-
"""
Experiment report generator.
[FIXED] Uses prompt_builder and correctly loads YAML prompts.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import os, json, re, math
import numpy as np
from scipy import stats

# [FIX] Use User's Prompt Builder
try:
    from prompt_builder import chat_text
except ImportError:
    print("[REPORT] WARN: prompt_builder not found. LLM features disabled.")
    chat_text = None

try:
    import yaml
except ImportError:
    yaml = None

__all__ = ["write_experiment_report"]

# =============================================================================
# 0. Helper: Template Expansion & Loading
# =============================================================================

def _expand_vars(text: str, context: Dict[str, str]) -> str:
    """Recursively expand ${VAR} in string."""
    if not text: return ""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(context.get(m.group(1), m.group(0))), text)

def _load_prompt_template(pm: str, metrics_str: str) -> tuple[str, str]:
    """
    [FIXED] Robust YAML loading using absolute paths.
    """
    # 1. Define Candidate Paths (Priority: CWD/prompts -> ScriptDir/prompts)
    cwd_path = os.path.join(os.getcwd(), "prompts", "experiment_report.yaml")
    script_path = os.path.join(os.path.dirname(__file__), "prompts", "experiment_report.yaml")
    
    prompt_path = None
    if os.path.exists(cwd_path): prompt_path = cwd_path
    elif os.path.exists(script_path): prompt_path = script_path

    # 2. Define Context for Variable Replacement
    context = {
        "primary_metric": pm,
        "metrics_json": metrics_str
    }

    # 3. Load from YAML if available
    if prompt_path and yaml:
        try:
            print(f"[REPORT] Loading prompt template from: {prompt_path}")
            with open(prompt_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            sys_tmpl = data.get("system", "")
            usr_tmpl = data.get("user_template", "")
            
            # Expand variables
            return _expand_vars(sys_tmpl, context), _expand_vars(usr_tmpl, context)
            
        except Exception as e:
            print(f"[REPORT][WARN] YAML Load Error: {e}. Using fallback.")

    # 4. Fallback (Hardcoded) - Only if YAML fails
    print("[REPORT] Using FALLBACK hardcoded prompts (Check your yaml path!)")
    default_system = (
        f"You are a Senior Scientist. Analyze the results. Primary Metric: {pm}.\n"
        "Output a markdown report with a table and analysis."
    )
    default_user = f"Metrics:\n```json\n{metrics_str}\n```"
    return default_system, default_user

# =============================================================================
# 1. Data Cleaning
# =============================================================================

def _sanitize_json_values(obj: Any) -> Any:
    if hasattr(obj, "item"): obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj): return None
        if math.isinf(obj): return "Infinity"
    if isinstance(obj, dict): return {k: _sanitize_json_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_sanitize_json_values(x) for x in obj]
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

def _simplify_metrics_for_llm(obj: Any, depth: int = 0) -> Any:
    obj = _sanitize_json_values(obj)
    if depth > 3: return "..."
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k in ["per_fold", "history", "predictions", "per_fold_details"]: continue
            new_obj[k] = _simplify_metrics_for_llm(v, depth + 1)
        return new_obj
    return obj

def _guess_baseline_name(models_dict: Dict[str, Any]) -> str:
    keys = list(models_dict.keys())
    for k in keys:
        if "baseline" in k.lower() or "linear" in k.lower() or "ridge" in k.lower(): return k
    return keys[0] if keys else "unknown"

# =============================================================================
# 2. Main Entry
# =============================================================================

def write_experiment_report(trial_dir: str,
                            metrics_obj: Dict[str, Any],
                            cfg: Dict[str, Any],
                            primary_metric: Optional[str] = None) -> str:
    
    if chat_text is None:
        return "[ERROR] prompt_builder missing"

    # 1. Prepare Data
    pm = primary_metric or "PCC"
    models_data = _normalize_metrics(metrics_obj)
    baseline_name = _guess_baseline_name(models_data)
    
    # Sanitize and Simplify
    slim_data = _simplify_metrics_for_llm(models_data)
    
    # Inject Stats Meta info
    slim_data["_META_"] = {
        "baseline": baseline_name,
        "target_metric": pm
    }

    metrics_str = json.dumps(slim_data, indent=2)

    # 2. Load Prompts (Strictly from YAML)
    system_prompt, user_content = _load_prompt_template(pm, metrics_str)

    # 3. Call LLM using prompt_builder
    print(f"[REPORT] Generating analysis for {os.path.basename(trial_dir)}...")
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_content}
    ]

    try:
        # [FIX] Use chat_text from prompt_builder
        # Pass trial_dir as debug_dir so we can see raw LLM I/O in the folder
        report_content = chat_text(
            messages,
            cfg=cfg,
            timeout=600,
            debug_dir=os.path.join(trial_dir, "llm_report_debug"), 
            temperature=0.5,
            max_tokens=4000
        )
        
        # 4. Clean & Save
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