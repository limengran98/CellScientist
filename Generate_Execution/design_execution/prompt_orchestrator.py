# design_execution/prompt_orchestrator.py
import os, json, shutil, datetime
from typing import Dict, Any, Optional
import nbformat

from .prompt_generator import generate_notebook_content
from .prompt_executor import run_notebook_with_autofix
from .experiment_report import write_experiment_report

# Import Visualization Tool
try:
    from .prompt_viz import write_hypergraph_viz
except ImportError:
    write_hypergraph_viz = None

# =============================================================================
# Helper: Path Management
# =============================================================================

def _get_save_root(cfg: Dict[str, Any]) -> str:
    return cfg.get("prompt_branch", {}).get("save_root", cfg["paths"]["design_execution_root"])

def _get_latest_trial(cfg: Dict[str, Any]) -> Optional[str]:
    root = os.path.join(_get_save_root(cfg), "prompt")
    if not os.path.exists(root): return None
    # Filter for directories starting with prompt_ or workspace_
    subs = sorted([p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))])
    if subs:
        return os.path.join(root, subs[-1])
    return None

def _audit_intermediate_files(trial_dir: str):
    """
    [NEW] Force audit and print files in intermediate directory to ensure visibility of the process.
    """
    inter_dir = os.path.join(trial_dir, "intermediate")
    print(f"\n[ORCH] 🔎 Auditing Intermediate Results in: {inter_dir}", flush=True)
    
    if not os.path.exists(inter_dir):
        print("   ⚠️ Directory NOT created by Notebook. (Did the model skip saving?)", flush=True)
        return

    files = []
    for root, _, filenames in os.walk(inter_dir):
        for f in filenames:
            path = os.path.join(root, f)
            try:
                size_kb = os.path.getsize(path) / 1024
            except OSError:
                size_kb = 0
            rel_path = os.path.relpath(path, trial_dir)
            files.append((rel_path, size_kb))
    
    if not files:
        print("   ⚠️ Directory exists but is EMPTY.", flush=True)
    else:
        # Sort by name
        files.sort()
        for fname, fsize in files:
            print(f"   📄 {fname:<40} | {fsize:>6.1f} KB", flush=True)
    print("", flush=True) # Spacer

# =============================================================================
# Phases
# =============================================================================

def phase_generate(
    cfg: Dict[str, Any], 
    spec_path: str, 
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """Phase 1: Generate Notebook."""
    # [FIX] Make debug_prompt directory unique to avoid collisions between concurrent processes
    ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    debug_folder_name = f"debug_prompt_{ts_now}_{pid}"
    
    debug_dir = os.path.join(cfg["paths"]["design_execution_root"], debug_folder_name)
    out_root = _get_save_root(cfg)
    
    nb, _user_prompt, strategy_md = generate_notebook_content(cfg, spec_path, debug_dir)
    
    # [MODIFIED] Logic for directory naming
    if run_name:
        # Loop mode: Use fixed workspace
        trial_dir = os.path.join(out_root, "prompt", run_name)
        # Clean up previous run data if exists to avoid mixing
        if os.path.exists(trial_dir):
            try:
                shutil.rmtree(trial_dir)
            except OSError as e:
                print(f"[ORCH][WARN] Failed to clean workspace {trial_dir}: {e}", flush=True)
    else:
        # Standard mode: Timestamp + PID for safety
        trial_dir = os.path.join(out_root, "prompt", f"prompt_run_{ts_now}_{pid}")
        
    os.makedirs(trial_dir, exist_ok=True)
    
    nb_path = os.path.join(trial_dir, "notebook_prompt.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    if strategy_md:
        with open(os.path.join(trial_dir, "research_strategy.md"), "w", encoding="utf-8") as f:
            f.write(strategy_md)

    print(f"[ORCH] Generation Complete. Trial: {trial_dir}", flush=True)
    return {"trial_dir": trial_dir, "notebook_path": nb_path}

def phase_execute(cfg: Dict[str, Any], trial_dir: Optional[str] = None) -> Dict[str, Any]:
    """Phase 2: Execute Notebook with Auto-Fix."""
    tdir = trial_dir or _get_latest_trial(cfg)
    if not tdir:
        raise RuntimeError("No trial directory found.")
        
    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    if not os.path.exists(nb_path):
        raise RuntimeError(f"Notebook not found: {nb_path}")
        
    # Run
    final_exec = run_notebook_with_autofix(nb_path, tdir, cfg)
    
    # Generate Hypergraph Visualization
    if write_hypergraph_viz:
        print(f"[ORCH] Generating Hypergraph Visualization...", flush=True)
        viz_out = write_hypergraph_viz(tdir, nb_path, fmt="mermaid")
        if viz_out.get("mermaid"):
            print(f"[ORCH] Viz saved: {viz_out['mermaid']}", flush=True)
            
    # [NEW] Audit Intermediate Files
    _audit_intermediate_files(tdir)
    
    # Load Metrics
    metrics = {}
    m_path = os.path.join(tdir, "metrics.json")
    if os.path.exists(m_path):
        try:
            with open(m_path, "r") as f: metrics = json.load(f)
        except: pass
        
    return {"trial_dir": tdir, "exec_notebook": final_exec, "metrics": metrics}

def phase_analyze(cfg: Dict[str, Any], trial_dir: Optional[str] = None) -> Dict[str, Any]:
    """Phase 3: Generate Report."""
    tdir = trial_dir or _get_latest_trial(cfg)
    if not tdir:
        raise RuntimeError("No trial directory found.")
    
    m_path = os.path.join(tdir, "metrics.json")
    if not os.path.exists(m_path):
        print(f"[ORCH] No metrics.json in {tdir}. Skipping report.", flush=True)
        return {}
        
    try:
        with open(m_path, "r") as f: metrics = json.load(f)
        
        pm = ((cfg.get("experiment") or {}).get("primary_metric") or "PCC")
        
        report_path = write_experiment_report(tdir, metrics, cfg, primary_metric=pm)
        print(f"[ORCH] Report written: {report_path}", flush=True)
        return {"report_path": report_path}
    except Exception as e:
        print(f"[ORCH] Analysis failed: {e}", flush=True)
        return {}

def run_full_pipeline(
    cfg: Dict[str, Any], 
    spec_path: str,
    run_name: Optional[str] = None # [NEW] Pass down run_name
) -> Dict[str, Any]:
    
    print("[ORCH] === STEP 1: GENERATE ===", flush=True)
    # Pass run_name to control folder creation/overwriting
    gen_res = phase_generate(cfg, spec_path, run_name=run_name)
    
    print("[ORCH] === STEP 2: EXECUTE ===", flush=True)
    exec_res = phase_execute(cfg, gen_res["trial_dir"])
    
    print("[ORCH] === STEP 3: ANALYZE ===", flush=True)
    phase_analyze(cfg, gen_res["trial_dir"])
    
    return exec_res