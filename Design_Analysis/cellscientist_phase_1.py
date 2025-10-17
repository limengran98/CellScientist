#!/usr/bin/env python3
# cellscientist_phase_1.py — multi-only entry (num_runs=1 == single logically)
# Keeps run_pipeline_basic() so hypergraph_orchestrator can import it.

import os, sys, io, json, contextlib, importlib
from pathlib import Path


def resolve_placeholders(cfg: dict) -> dict:
    """Recursively replace ${dataset_name} in all string fields using top-level keys."""
    import re as _re
    ds = cfg.get("dataset_name", "default_dataset")
    def _subst(v):
        if isinstance(v, str):
            return _re.sub(r"\$\{dataset_name\}", ds, v)
        if isinstance(v, dict):
            return {k: _subst(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_subst(x) for x in v]
        return v
    return _subst(cfg)

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
    run_llm_notebook_from_file = _load_runner_clean()
    executed_path = run_llm_notebook_from_file(cfg_path, phase_name=phase_name)
    return executed_path

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS_DIR, 'design_analysis_config.json')
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg = resolve_placeholders(cfg)
    nb_cfg = (((cfg.get('phases') or {}).get('task_analysis') or {}).get('llm_notebook') or {})
    multi = nb_cfg.get('multi') or {}
    review = (multi.get('review') or {})

    out_dir = (multi.get('out_dir') or os.path.join(THIS_DIR, 'hypergraph_runs'))
    num_runs = int(multi.get('num_runs', max(1, len(multi.get('prompt_variants') or []) or 1)))
    pv_len = len(multi.get('prompt_variants') or [])
    model = (nb_cfg.get('llm') or {}).get('model')
    api_key_env = (nb_cfg.get('llm') or {}).get('api_key_env', 'OPENAI_API_KEY')
    base_url_env = (nb_cfg.get('llm') or {}).get('base_url_env', 'OPENAI_BASE_URL')

    print('🗂  Config:', cfg_path, flush=True)
    print('📦 Out dir (multi):', out_dir, flush=True)
    print('🔢 num_runs:', num_runs, '| prompt_variants:', pv_len, flush=True)
    print('🧠 LLM model:', model, flush=True)
    print('🔑 API key loaded?:', bool(os.environ.get(api_key_env)), flush=True)
    print('🌐 Base URL:', os.environ.get(base_url_env, 'default'), flush=True)
    print('ℹ️  Mode: MULTI orchestration only (set num_runs=1 to behave like single).', flush=True)

    from hypergraph_orchestrator import orchestrate, closed_loop_orchestrate

    if bool(review.get('closed_loop') or review.get('closed_loop_enabled', False)):
        max_cycles = int(review.get('max_cycles', 1))
        print(f'🧠 Closed-loop enabled → cycles={max_cycles}', flush=True)
        closed_loop_orchestrate(cfg_path, max_cycles=max_cycles)
    else:
        print('🧪 Single-pass orchestration (review/export handled inside orchestrator if enabled)', flush=True)
        orchestrate(cfg_path)

if __name__ == '__main__':
    # Unbuffered output to ensure logs appear
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()
