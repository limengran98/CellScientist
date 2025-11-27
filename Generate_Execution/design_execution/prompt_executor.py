# design_execution/prompt_executor.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os, json, re, hashlib
import nbformat
from nbclient import NotebookClient
from copy import deepcopy

# Import centralized LLM tools
from .llm_utils import chat_json

# =============================================================================
# Execution Utils
# =============================================================================

def collect_cell_errors(nb: nbformat.NotebookNode) -> List[Dict[str, Any]]:
    errs = []
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code": continue
        for out in (c.get("outputs") or []):
            if out.get("output_type") == "error":
                tb = out.get("traceback", [])
                tb_str = "\n".join(tb) if isinstance(tb, list) else str(tb)
                tb_clean = re.sub(r'\x1b\[[0-9;]*m', '', tb_str)
                errs.append({
                    "cell_index": i,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback": tb_clean,
                })
    return errs

def execute_once(
    nb: nbformat.NotebookNode,
    workdir: str,
    timeout: int = 1800,
) -> Tuple[nbformat.NotebookNode, List[Dict[str, Any]]]:
    
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    os.environ["OUTPUT_DIR"] = workdir

    nb_to_run = deepcopy(nb)
    
    # Inject Guard for sys.exit
    guard_code = (
        "# [AUTO-FIX] Guard Cell\n"
        "import sys, os\n"
        "def _guard_exit(*args, **kwargs):\n"
        "    raise RuntimeError(\"SysExitBlocked: Use raise ValueError() instead.\")\n"
        "sys.exit = _guard_exit\n"
    )
    nb_to_run.cells.insert(0, nbformat.v4.new_code_cell(guard_code))

    client = NotebookClient(
        nb_to_run, timeout=timeout, kernel_name="python3",
        allow_errors=True, resources={"metadata": {"path": workdir}},
    )
    
    try:
        exec_nb = client.execute()
    except Exception as e:
        print(f"[EXEC] Warning: Notebook execution crashed: {e}", flush=True)
        exec_nb = client.nb

    # Remove guard cell from result
    if len(exec_nb.cells) > 0:
        exec_nb.cells.pop(0)

    errors = collect_cell_errors(exec_nb)
    return exec_nb, errors

# =============================================================================
# Auto-Fix Logic
# =============================================================================

def _apply_edits(nb: nbformat.NotebookNode, edits: List[Dict[str, Any]]) -> None:
    for ed in edits:
        try:
            idx = int(ed.get("cell_index"))
            src = ed.get("source")
            if idx >= 0 and idx < len(nb.cells) and src:
                nb.cells[idx]["source"] = src
        except: pass

def _llm_auto_fix_once(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    llm_cfg: Dict[str, Any],
    autofix_system_prompt: str
) -> Tuple[nbformat.NotebookNode, bool]:
    
    # 1. Construct Prompt
    error_list = []
    for e in errors:
        error_list.append({
            "cell_index": e['cell_index'],
            "error": f"{e.get('ename')}: {e.get('evalue')}",
            "traceback": (e.get("traceback") or "")[-2000:]
        })
    
    # Prepare Failing Code Context
    indices = {e['cell_index'] for e in errors}
    failing_cells = []
    for i in indices:
        if 0 <= i < len(nb.cells):
            src = nb.cells[i].source
            if len(src) > 8000: src = src[:8000] + "\n# ... [TRUNCATED]"
            failing_cells.append({"cell_index": i, "source": src})

    user_payload = {
        "task": "Fix specific notebook cells.",
        "errors": error_list,
        "failing_code": failing_cells,
        "instruction": "Return FULL corrected source for the failing cells."
    }

    messages = [
        {"role": "system", "content": autofix_system_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2)}
    ]

    # 2. Call LLM (Temperature 0.0 for precision)
    try:
        spec = chat_json(
            messages, 
            llm_config=llm_cfg,
            temperature=0.0, 
            timeout=600
        )
    except Exception as e:
        print(f"[FIX] LLM Call Failed: {e}", flush=True)
        return nb, False

    edits = spec.get("edits") or []
    if not edits:
        return nb, False

    # 3. Apply
    _apply_edits(nb, edits)
    return nb, True

def run_notebook_with_autofix(
    nb_path: str,
    workdir: str,
    cfg: Dict[str, Any]
) -> str:
    """
    Main entry for execution with auto-fix loop.
    """
    exec_cfg = cfg.get("exec", {})
    timeout = int(exec_cfg.get("timeout_seconds", 3600))
    max_rounds = int(exec_cfg.get("max_fix_rounds", 3))
    
    nb_orig = nbformat.read(nb_path, as_version=4)
    print(f"[EXEC] Initial Execution: {nb_path}", flush=True)
    
    exec_nb, errors = execute_once(nb_orig, workdir, timeout=timeout)
    
    out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    
    # Auto-Fix Loop
    round_idx = 0
    
    # Get prompt from config
    prompts_map = cfg.get("prompts", {})
    autofix_prompt = prompts_map.get("autofix", {}).get("system_prompt", "You are a Python Expert. Fix the code. Return JSON.")

    while errors and round_idx < max_rounds:
        round_idx += 1
        print(f"\n[FIX] === ROUND {round_idx}/{max_rounds} ===", flush=True)
        print(f"[FIX] {len(errors)} errors detected. Attempting LLM fix...", flush=True)
        
        # Use Executed NB for fixing to keep state? 
        # Usually better to fix original Source and re-run clean to avoid state pollution.
        # But we need error context.
        # Strategy: Patch `exec_nb`, then re-run.
        
        fixed_nb, patched = _llm_auto_fix_once(
            deepcopy(exec_nb), 
            errors, 
            cfg.get("llm", {}),
            autofix_system_prompt=autofix_prompt
        )
        
        if not patched:
            print("[FIX] LLM returned no effective patches. Stopping.", flush=True)
            break
            
        print("[FIX] Patches applied. Re-executing...", flush=True)
        exec_nb, errors = execute_once(fixed_nb, workdir, timeout=timeout)
        
        if not errors:
            print("[FIX] Success! No errors remaining.", flush=True)
            break

    # Save Final
    nbformat.write(exec_nb, out_path)
    print(f"[EXEC] Saved final result: {out_path}", flush=True)
    if errors:
        print(f"[EXEC] Finished with {len(errors)} errors remaining.", flush=True)
        
    return out_path