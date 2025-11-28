# design_execution/prompt_executor.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, json, re, hashlib
import nbformat
from nbclient import NotebookClient
from copy import deepcopy

# Import centralized LLM tools
from .llm_utils import chat_json

# =============================================================================
# 1. Heuristic Logic (Ported & Simplified from Old Version)
# =============================================================================

def _get_heuristic_patch(evalue: str) -> Optional[str]:
    """Return a hardcoded patch snippet for common dumb errors."""
    msg = (evalue or "").lower()
    
    # Case 1: NaN / Infinity in input
    if "input contains nan" in msg or "input contains infinity" in msg:
        return (
            "# [AUTO-FIX] Heuristic: Impute NaNs and Clip Infinity\n"
            "import numpy as np, pandas as pd\n"
            "from sklearn.impute import SimpleImputer\n"
            "print('[AUTO-FIX] Applying heuristic data cleaning...')\n"
            "def _clean_data(X):\n"
            "    if hasattr(X, 'fillna'): X = X.fillna(0)\n"
            "    X = np.nan_to_num(X, nan=0.0, posinf=None, neginf=None)\n"
            "    return X\n"
            "for _nm in ['X', 'X_train', 'X_test', 'y', 'y_train', 'y_test']:\n"
            "    if _nm in globals():\n"
            "        globals()[_nm] = _clean_data(globals()[_nm])\n"
        )
    
    # Case 2: String to Float conversion failure
    if "could not convert string to float" in msg:
        return (
            "# [AUTO-FIX] Heuristic: Force Numeric Conversion\n"
            "import pandas as pd, numpy as np\n"
            "print('[AUTO-FIX] Filtering non-numeric columns...')\n"
            "def _force_numeric(df):\n"
            "    if isinstance(df, pd.DataFrame):\n"
            "        return df.select_dtypes(include=[np.number])\n"
            "    return df\n"
            "for _nm in ['X', 'X_train', 'X_test']:\n"
            "    if _nm in globals(): globals()[_nm] = _force_numeric(globals()[_nm])\n"
        )

    return None

def _apply_heuristics(nb: nbformat.NotebookNode, errors: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
    """Try to apply heuristics before calling LLM."""
    changed_count = 0
    patched_indices = []
    
    for e in errors:
        idx = int(e["cell_index"])
        patch = _get_heuristic_patch(e.get("evalue", ""))
        
        if patch and 0 <= idx < len(nb.cells):
            cell = nb.cells[idx]
            if cell.cell_type == "code" and patch not in cell.source:
                cell.source += "\n\n" + patch
                changed_count += 1
                patched_indices.append(idx)
                
    return changed_count, patched_indices

# =============================================================================
# 2. Execution Utils
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

    if len(exec_nb.cells) > 0:
        exec_nb.cells.pop(0)

    errors = collect_cell_errors(exec_nb)
    return exec_nb, errors

# =============================================================================
# 3. Auto-Fix Logic (Enhanced with Hash Check)
# =============================================================================

def _compute_cell_hash(source: str) -> str:
    """Compute MD5 hash of cell content ignoring whitespace."""
    return hashlib.md5(source.strip().encode('utf-8')).hexdigest()

def _apply_llm_edits(nb: nbformat.NotebookNode, edits: List[Dict[str, Any]]) -> int:
    """Apply edits and return number of effective changes (Hash Verified)."""
    effective_changes = 0
    for ed in edits:
        try:
            idx = int(ed.get("cell_index"))
            new_src = ed.get("source")
            
            if idx >= 0 and idx < len(nb.cells) and new_src:
                old_src = nb.cells[idx]["source"]
                
                # HASH CHECK: Only apply if content actually changed
                if _compute_cell_hash(old_src) != _compute_cell_hash(new_src):
                    nb.cells[idx]["source"] = new_src
                    effective_changes += 1
                else:
                    print(f"[FIX] ⚠️ Skipping edit for Cell {idx}: Content identical.", flush=True)
        except: pass
    return effective_changes

def _llm_auto_fix_once(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    llm_cfg: Dict[str, Any],
    autofix_system_prompt: str
) -> Tuple[nbformat.NotebookNode, bool]:
    
    # 1. Context Construction
    error_list = []
    for e in errors:
        error_list.append({
            "cell_index": e['cell_index'],
            "error": f"{e.get('ename')}: {e.get('evalue')}",
            "traceback": (e.get("traceback") or "")[-2000:]
        })
    
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

    # 2. Call LLM
    try:
        spec = chat_json(messages, llm_config=llm_cfg, temperature=0.0, timeout=600)
    except Exception as e:
        print(f"[FIX] LLM Call Failed: {e}", flush=True)
        return nb, False

    edits = spec.get("edits") or []
    if not edits:
        return nb, False

    # 3. Apply with Verification
    changes = _apply_llm_edits(nb, edits)
    return nb, (changes > 0)

def run_notebook_with_autofix(
    nb_path: str,
    workdir: str,
    cfg: Dict[str, Any]
) -> str:
    """
    Main entry for execution with Robust Auto-Fix (Heuristic -> LLM).
    """
    exec_cfg = cfg.get("exec", {})
    timeout = int(exec_cfg.get("timeout_seconds", 3600))
    max_rounds = int(exec_cfg.get("max_fix_rounds", 3))
    
    nb_orig = nbformat.read(nb_path, as_version=4)
    print(f"[EXEC] Initial Execution: {nb_path}", flush=True)
    
    exec_nb, errors = execute_once(nb_orig, workdir, timeout=timeout)
    out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    
    round_idx = 0
    prompts_map = cfg.get("prompts", {})
    autofix_prompt = prompts_map.get("autofix", {}).get("system_prompt", "You are a Python Expert. Fix the code. Return JSON.")

    while errors and round_idx < max_rounds:
        round_idx += 1
        print(f"\n[FIX] === ROUND {round_idx}/{max_rounds} ===", flush=True)
        print(f"[FIX] {len(errors)} errors. Analyzing...", flush=True)

        # STEP A: Try Heuristics First (Fast & Robust)
        # We work on a copy to avoid corrupting state if heuristics fail partially
        nb_heuristic = deepcopy(exec_nb)
        h_changes, _ = _apply_heuristics(nb_heuristic, errors)
        
        if h_changes > 0:
            print(f"[FIX] ⚡ Applied {h_changes} heuristic patches (NaN/Type fixes). Re-executing...", flush=True)
            exec_nb, errors = execute_once(nb_heuristic, workdir, timeout=timeout)
            if not errors:
                print("[FIX] Success via Heuristics!", flush=True)
                break
        
        # STEP B: If errors persist, use LLM
        if errors:
            print(f"[FIX] 🧠 Heuristics insufficient. Calling LLM...", flush=True)
            # Use deepcopy to ensure we don't mutate if LLM fails
            fixed_nb, patched = _llm_auto_fix_once(
                deepcopy(exec_nb), 
                errors, 
                cfg.get("llm", {}),
                autofix_system_prompt=autofix_prompt
            )
            
            if not patched:
                print("[FIX] 🛑 LLM could not provide effective edits (Hash Check Failed). Stopping.", flush=True)
                break
                
            print("[FIX] LLM patches applied. Re-executing...", flush=True)
            exec_nb, errors = execute_once(fixed_nb, workdir, timeout=timeout)
        
        if not errors:
            print("[FIX] Success! No errors remaining.", flush=True)
            break

    nbformat.write(exec_nb, out_path)
    print(f"[EXEC] Saved final result: {out_path}", flush=True)
    if errors:
        print(f"[EXEC] Finished with {len(errors)} errors remaining.", flush=True)
        
    return out_path