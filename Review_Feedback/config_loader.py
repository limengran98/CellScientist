# config_loader.py
import os, json, re
from typing import Dict, Any

try:
    import yaml
except ImportError:
    yaml = None

def _expand_vars(obj: Any, env: Dict[str, str]) -> Any:
    """Recursively expand ${VAR}."""
    if isinstance(obj, dict):
        return {k: _expand_vars(v, env) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_vars(v, env) for v in obj]
    elif isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), obj)
    return obj

def load_full_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # Environment + Config Self-Reference
    env = dict(os.environ)
    env.update(cfg) # Allow referring to simple top-level keys
    
    return _expand_vars(cfg, env)