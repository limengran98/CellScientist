#!/usr/bin/env python3
# cellscientist_phase_1.py — multi-only entry
# Optimized for real-time logging and robustness.

import os, sys, io, json, contextlib, importlib
from pathlib import Path

# [CRITICAL] Force line buffering for stdout/stderr to ensure logs appear immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# [NEW] Use the centralized config loader
from config_loader import load_app_config

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def _load_runner_clean():
    """
    Import runner. 
    [MODIFIED] Removed output suppression (redirect_stdout) so logs show up.
    """
    # Removed contextlib.redirect_stdout logic to allow runner logs to appear
    mod = importlib.import_module('CellScientist.Design_Analysis.llm_notebook_runner')
    return getattr(mod, 'run_llm_notebook_from_file')

# [STABLE] keep this for orchestrator
def run_pipeline_basic(cfg_path: str, phase_name: str = 'task_analysis') -> str:
    prompts_dir = os.path.join(THIS_DIR, 'prompts')
    
    run_llm_notebook_from_file = _load_runner_clean()
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

    cfg = load_app_config(cfg_path, prompts_path)
    
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
    
    # [MODIFIED] Removed closed_loop logic, always run standard orchestration
    from hypergraph_orchestrator import orchestrate
    
    print('🧪 Running orchestration...', flush=True)
    orchestrate(cfg_path, prompts_path)

if __name__ == '__main__':
    main()