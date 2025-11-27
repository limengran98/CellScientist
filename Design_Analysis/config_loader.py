#!/usr/bin/env python3
# config_loader.py
import json, os, re
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except ImportError:
    yaml = None

def _resolve_placeholders(cfg: dict) -> dict:
    ds = cfg.get("dataset_name", "default_dataset")
    def _subst(v):
        if isinstance(v, str): return re.sub(r"\$\{dataset_name\}", ds, v)
        if isinstance(v, dict): return {k: _subst(x) for k, x in v.items()}
        if isinstance(v, list): return [_subst(x) for x in v]
        return v
    return _subst(cfg)

def load_prompts_from_dir(prompts_dir: Path) -> Dict[str, Any]:
    prompts = {}
    if not prompts_dir.is_dir():
        return prompts
    
    if yaml:
        for p_file in prompts_dir.glob("*.yml"):
            try:
                with open(p_file, 'r', encoding='utf-8') as f:
                    prompts[p_file.stem] = yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Error loading {p_file.name}: {e}", flush=True)
    return prompts

def load_app_config(config_path: str, prompts_dir_path: str) -> Dict[str, Any]:
    p_config = Path(config_path).resolve()
    p_prompts = Path(prompts_dir_path).resolve() if prompts_dir_path else p_config.parent / "prompts"

    if not p_config.exists():
        raise FileNotFoundError(f"Config file missing: {p_config}")

    with open(p_config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    prompts_data = load_prompts_from_dir(p_prompts)
    cfg['prompts'] = prompts_data

    nb_cfg = cfg.get("phases", {}).get("task_analysis", {}).get("llm_notebook", {})
    if 'notebook_generation' in prompts_data:
        p_gen = prompts_data['notebook_generation']
        if isinstance(p_gen, dict) and 'user_prompt' in p_gen:
             nb_cfg.setdefault('prompt', p_gen['user_prompt'])

    return _resolve_placeholders(cfg)