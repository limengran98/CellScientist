#!/usr/bin/env python3
# cellscientist_phase_1.py — multi-only entry
# Optimized for real-time logging and robustness.

import os
import sys
import io
import json
import contextlib
import importlib
from pathlib import Path

# [CRITICAL] Force line buffering for stdout/stderr to ensure logs appear immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# --- PATH SETUP ---
# Ensure the directory containing this script is in sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Ensure the repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# [NEW] Use the centralized config loader
try:
    from config_loader import load_app_config
except ImportError:
    # Fallback in case of path issues
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_loader", os.path.join(THIS_DIR, "config_loader.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    load_app_config = mod.load_app_config

def _load_runner_clean():
    """
    Import runner robustly.
    [FIXED] Removed hardcoded 'CellScientist' package path to allow running in renamed folders.
    """
    try:
        # 1. Try direct import (works if THIS_DIR is in sys.path)
        import llm_notebook_runner as mod
        return getattr(mod, 'run_llm_notebook_from_file')
    except ImportError:
        # 2. Fallback: Manually load from file path
        try:
            spec = importlib.util.spec_from_file_location(
                "llm_notebook_runner", 
                os.path.join(THIS_DIR, "llm_notebook_runner.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, 'run_llm_notebook_from_file')
        except Exception as e:
            print(f"[ERROR] Could not import llm_notebook_runner: {e}", flush=True)
            raise

# [STABLE] keep this for orchestrator
def run_pipeline_basic(cfg_path: str, phase_name: str = 'task_analysis') -> str:
    prompts_dir = os.path.join(THIS_DIR, 'prompts')
    
    run_llm_notebook_from_file = _load_runner_clean()
    
    # Run the notebook generation & execution
    # Note: run_llm_notebook_from_file handles the heavy lifting
    executed_path = run_llm_notebook_from_file(
        config_path=cfg_path, 
        prompts_dir_path=str(prompts_dir),
        phase_name=phase_name
    )
    return executed_path

def main():
    cfg_path_arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS_DIR, 'design_analysis_config.json')
    cfg_path = os.path.abspath(cfg_path_arg)
    prompts_path = os.path.join(THIS_DIR, 'prompts')
    
    if len(sys.argv) > 1:
        print(f"ℹ️  Using custom config path: {cfg_path}", flush=True)
        print(f"ℹ️  Assuming prompts dir: {prompts_path}", flush=True)

    try:
        cfg = load_app_config(cfg_path, prompts_path)
    except Exception as e:
        print(f"❌ Config Load Error: {e}")
        # Try loading simple json if centralized loader fails
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)

    nb_cfg = (((cfg.get('phases') or {}).get('task_analysis') or {}).get('llm_notebook') or {})
    multi = nb_cfg.get('multi') or {}
    
    # [NEW] Force Inject API Key into Environment for safety
    # This ensures all subprocesses and threads can find the key
    llm_cfg = nb_cfg.get('llm') or {}
    config_key = llm_cfg.get('api_key')
    if config_key:
        os.environ["OPENAI_API_KEY"] = config_key
        # Mask key for security in logs
        print(f"🔑 API key injected from config: True (Masked: ...{config_key[-4:]})", flush=True)
    else:
        print("⚠️ API key NOT found in config! Relying on existing env vars.", flush=True)

    num_runs_config = int(multi.get('num_runs', 1))
    pv_len = len(multi.get('prompt_variants') or [])
    
    out_dir = (multi.get('out_dir') or os.path.join(THIS_DIR, 'hypergraph_runs'))
    
    print('🗂  Config:', cfg_path, flush=True)
    print('📂 Prompts:', prompts_path, flush=True)
    print('📦 Out dir (multi):', out_dir, flush=True)
    print(f"🔢 Runs (config): {num_runs_config} | Prompt Variants: {pv_len} (Adaptive logic in orchestrator)", flush=True)
    
    # [MODIFIED] Robust import for orchestrator
    print('🧪 Running orchestration...', flush=True)
    try:
        from hypergraph_orchestrator import orchestrate
        orchestrate(cfg_path, prompts_path)
    except ImportError:
        # Fallback if function name differs in `hypergraph_orchestrator.py`
        try:
            from hypergraph_orchestrator import run_hypergraph_orchestration
            run_hypergraph_orchestration(
                cfg_path=cfg_path,
                prompts_dir=prompts_path,
                output_dir=out_dir
            )
        except Exception as e:
            print(f"❌ Orchestrator Import Failed: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()