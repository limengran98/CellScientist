#!/usr/bin/env python3
# config_loader.py
import json, os, re
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

def _get_repo_root() -> Path:
    """Determine the repository root directory (CellScientist/)."""
    # Assuming config_loader.py is in CellScientist/Design_Analysis/config_loader.py
    # Parent = Design_Analysis, Parent.Parent = CellScientist
    return Path(__file__).resolve().parent.parent

def _resolve_placeholders(cfg: dict) -> dict:
    ds = cfg.get("dataset_name", "default_dataset")
    def _subst(v):
        if isinstance(v, str): return re.sub(r"\$\{dataset_name\}", ds, v)
        if isinstance(v, dict): return {k: _subst(x) for k, x in v.items()}
        if isinstance(v, list): return [_subst(x) for x in v]
        return v
    return _subst(cfg)

def _looks_like_path(s: str) -> bool:
    if not isinstance(s, str): return False
    t = s.strip()
    if not t: return False
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", t): return False # URL
    if t.startswith("mailto:"): return False
    if re.match(r"^[a-zA-Z]:\\", t) or t.startswith("/"): return True # Abs
    if "/" in t or "\\" in t or t.startswith("./") or t.startswith("../"): return True
    return False

def _resolve_relative_paths(cfg: dict, config_dir: Path) -> dict:
    """
    Robustly resolve paths.
    - If path starts with '../', resolve relative to REPO_ROOT (CellScientist/).
    - Otherwise, resolve relative to config_dir (Design_Analysis/).
    """
    repo_root = _get_repo_root()
    
    PATH_KEYS = {
        "out_dir", "export_dir",
        "data", "paper", "preprocess", "out", "out_exec", "h5_out",
        "literature_dir", "literature_knowledge_json",
    }

    def _resolve_str(v: str) -> str:
        t = (v or "").strip()
        if not t: return v
        if not _looks_like_path(t): return v
        
        # Already absolute?
        p = Path(t)
        if p.is_absolute():
            return str(p)
            
        # Handle "../" specifically to anchor to repo root
        if t.startswith("../") or t.startswith("..\\"):
            # Strip the leading "../" and join with repo root
            # This avoids ambiguity of where ".." assumes we are
            clean_path = t[3:] # remove ../
            return str((repo_root / clean_path).resolve())
        
        # Default: relative to config file location
        return str((config_dir / p).resolve())

    def _walk(obj, parent_key: Optional[str] = None):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, str) and (k in PATH_KEYS or parent_key == "paths"):
                    out[k] = _resolve_str(v)
                else:
                    out[k] = _walk(v, k)
            return out
        if isinstance(obj, list):
            return [_walk(x, parent_key) for x in obj]
        return obj

    return _walk(cfg)

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

    cfg = _resolve_placeholders(cfg)
    cfg = _resolve_relative_paths(cfg, config_dir=p_config.parent)
    return cfg