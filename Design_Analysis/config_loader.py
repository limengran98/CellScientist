#!/usr/bin/env python3
# config_loader.py
# [NEW] Centralized config loader for JSON config + YAML prompts directory.
# Ensures all config loading (main, orchestrator, runner) uses one source.

import json
import os
import re
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except ImportError:
    print("PyYAML is required. Please install it with: pip install pyyaml", flush=True)
    raise

def _resolve_placeholders(cfg: dict) -> dict:
    """Recursively replace ${dataset_name} in all string fields using top-level keys."""
    ds = cfg.get("dataset_name", "default_dataset")
    def _subst(v):
        if isinstance(v, str):
            return re.sub(r"\$\{dataset_name\}", ds, v)
        if isinstance(v, dict):
            return {k: _subst(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_subst(x) for x in v]
        return v
    return _subst(cfg)

def load_prompts_from_dir(prompts_dir: Path) -> Dict[str, Any]:
    """Scans a directory for .yml files and loads them into a dict."""
    prompts = {}
    if not prompts_dir.is_dir():
        print(f"Warning: Prompts directory not found: {prompts_dir}", flush=True)
        return prompts

    for p_file in prompts_dir.glob("*.yml"):
        try:
            prompt_name = p_file.stem  # e.g., "notebook_generation"
            with open(p_file, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load prompt file {p_file.name}: {e}", flush=True)
            
    return prompts

def load_app_config(config_path: str, prompts_dir_path: str) -> Dict[str, Any]:
    """
    Loads the main JSON config and merges all prompts from the prompts directory.
    """
    p_config = Path(config_path).resolve()
    p_prompts = Path(prompts_dir_path).resolve()
    
    # 1. Load main JSON config
    if not p_config.exists():
        raise FileNotFoundError(f"Configuration file not found: {p_config}")
    with open(p_config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 2. Load all prompts from the prompts directory
    prompts_data = load_prompts_from_dir(p_prompts)
    if not prompts_data:
        print(f"Warning: No prompts loaded from {p_prompts}. Check directory.", flush=True)

    # 3. Inject prompts into the config structure at the top level
    cfg['prompts'] = prompts_data
    
    # 4. Inject specific prompts into their expected legacy locations
    #    to minimize code changes in other modules.
    nb_cfg = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    
    # - Notebook generation user prompt
    if 'notebook_generation' in prompts_data:
        # [MODIFIED] Check if user_prompt exists before getting it
        if 'user_prompt' in prompts_data['notebook_generation']:
            nb_cfg.setdefault('prompt', prompts_data['notebook_generation'].get('user_prompt'))

    # - Review prompts
    if 'review' in prompts_data:
        review_cfg = (nb_cfg.get('multi') or {}).get('review') or {}
        review_llm_cfg = review_cfg.get('llm') or {}
        if 'system_prompt' in prompts_data['review']:
            review_llm_cfg.setdefault('system_prompt', prompts_data['review'].get('system_prompt'))
        if 'critique_template' in prompts_data['review']:
            review_llm_cfg.setdefault('critique_prompt_template', prompts_data['review'].get('critique_template'))
        review_cfg['llm'] = review_llm_cfg

    # 5. Resolve placeholders (must be done after all injections)
    cfg = _resolve_placeholders(cfg)
    
    return cfg