#!/usr/bin/env python3
# cellscientist_phase_1.py — multi-only entry (num_runs=1 == single logically)
# Keeps run_pipeline_basic() so hypergraph_orchestrator can import it.

import os, sys, io, json, contextlib, importlib
from pathlib import Path

# [NEW] Use the centralized config loader
from config_loader import load_app_config

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def _load_runner_clean():
    """Import runner quietly (suppress llm_notebook_runner prints)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        mod = importlib.import_module('CellScientist.Design_Analysis.llm_notebook_runner')
    return getattr(mod, 'run_llm_notebook_from_file')

# [STABLE] keep this for orchestrator
def run_pipeline_basic(cfg_path: str, phase_name: str = 'task_analysis') -> str:
    # [MODIFIED] Pass both config path and prompts path
    # Assume prompts dir is sibling of config file's parent (i.e., in root)
    prompts_dir = os.path.join(THIS_DIR, 'prompts')
    
    run_llm_notebook_from_file = _load_runner_clean()
    # Pass both paths. The runner will load them.
    executed_path = run_llm_notebook_from_file(
        config_path=cfg_path, 
        prompts_dir_path=str(prompts_dir),
        phase_name=phase_name
    )
    return executed_path

def main():
    # [MODIFIED] Default config path and new prompts path are both in root
    cfg_path_arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS_DIR, 'design_analysis_config.json')
    cfg_path = os.path.abspath(cfg_path_arg)
    prompts_path = os.path.join(THIS_DIR, 'prompts')
    
    if len(sys.argv) > 1:
        print(f"ℹ️  Using custom config path: {cfg_path}")
        print(f"ℹ️  Assuming prompts dir: {prompts_path}")

    # [MODIFIED] Use centralized loader with both paths
    cfg = load_app_config(cfg_path, prompts_path)
    
    nb_cfg = (((cfg.get('phases') or {}).get('task_analysis') or {}).get('llm_notebook') or {})
    multi = nb_cfg.get('multi') or {}
    review = (multi.get('review') or {})

    out_dir = (multi.get('out_dir') or os.path.join(THIS_DIR, 'hypergraph_runs'))
    
    # [MODIFIED] Adaptive num_runs logic is now in hypergraph_orchestrator
    # This print will just reflect what's in the config, the orchestrator will print the adaptive logic
    num_runs_config = int(multi.get('num_runs', 1)) # Get config value or default
    pv_len = len(multi.get('prompt_variants') or [])
    
    model = (nb_cfg.get('llm') or {}).get('model')
    api_key_env = (nb_cfg.get('llm') or {}).get('api_key_env', 'OPENAI_API_KEY')
    base_url_env = (nb_cfg.get('llm') or {}).get('base_url_env', 'OPENAI_BASE_URL')

    print('🗂  Config:', cfg_path, flush=True)
    print('📂 Prompts:', prompts_path, flush=True)
    print('📦 Out dir (multi):', out_dir, flush=True)
    print(f"🔢 Runs (config): {num_runs_config} | Prompt Variants: {pv_len} (Adaptive logic in orchestrator)")
    print('🧠 LLM model:', model, flush=True)
    print('🔑 API key loaded?:', bool(os.environ.get(api_key_env)), flush=True)
    print('🌐 Base URL:', os.environ.get(base_url_env, 'default'), flush=True)
    print('ℹ️  Mode: MULTI orchestration only (set num_runs=1 to behave like single).', flush=True)

    from hypergraph_orchestrator import orchestrate, closed_loop_orchestrate
    
    # [MODIFIED] Pass prompts_dir to orchestrator functions
    if bool(review.get('closed_loop') or review.get('closed_loop_enabled', False)):
        max_cycles = int(review.get('max_cycles', 1))
        print(f'🧠 Closed-loop enabled → cycles={max_cycles}', flush=True)
        closed_loop_orchestrate(cfg_path, prompts_path, max_cycles=max_cycles)
    else:
        print('🧪 Single-pass orchestration (review/export handled inside orchestrator if enabled)', flush=True)
        orchestrate(cfg_path, prompts_path)

if __name__ == '__main__':
    # Unbuffered output to ensure logs appear
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()