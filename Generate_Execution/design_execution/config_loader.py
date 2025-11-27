# design_execution/config_loader.py
import os, json, re
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except ImportError:
    yaml = None

def _expand_vars(obj: Any, env: Dict[str, str]) -> Any:
    """Recursively expand ${VAR} in dictionary values."""
    if isinstance(obj, dict):
        return {k: _expand_vars(v, env) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_vars(v, env) for v in obj]
    elif isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), obj)
    return obj

def load_yaml_prompts(prompts_dir: str) -> Dict[str, Any]:
    """Loads all .yaml files from prompts directory."""
    prompts = {}
    p_dir = Path(prompts_dir)
    if not p_dir.exists() or not yaml:
        return prompts
    
    for p_file in p_dir.glob("*.yaml"):
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                prompts[p_file.stem] = yaml.safe_load(f)
        except Exception as e:
            print(f"[CONFIG] Warning: Failed to load prompt {p_file.name}: {e}", flush=True)
    
    # Also support .yml extension
    for p_file in p_dir.glob("*.yml"):
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                prompts[p_file.stem] = yaml.safe_load(f)
        except Exception:
            pass
            
    return prompts

def load_full_config(config_path: str) -> Dict[str, Any]:
    """
    Loads main JSON config, expands env vars, and loads prompts.
    """
    # 1. Load JSON
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # 2. Expand Variables
    env = dict(os.environ)
    env.update(cfg) # Config values can reference each other if needed, or just env
    cfg = _expand_vars(cfg, env)
    
    # 3. Load Prompts
    # Assuming prompts are in ../prompts relative to this file or current working dir
    # We prefer the current working directory's prompts folder usually.
    prompts_path = os.path.join(os.getcwd(), "prompts")
    cfg["prompts"] = load_yaml_prompts(prompts_path)
    
    return cfg