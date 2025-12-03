# nb_autofix.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json, re, hashlib
import nbformat
from nbclient import NotebookClient
from copy import deepcopy

# [OPTIMIZATION] Use the new robust utilities
from llm_utils import chat_json

# =============================================================================
# 1. Debug Utilities (RESTORED)
# =============================================================================

def _debug_cell_mapping(nb: nbformat.NotebookNode, errors: List[Dict[str, Any]]):
    """Visually confirm error locations."""
    print("\n[AUTO-FIX] === CELL INDEX MAP (DEBUG) ===")
    err_indices = {int(e['cell_index']) for e in errors}
    for i, c in enumerate(nb.cells):
        marker = " [ERROR] >>" if i in err_indices else "           "
        ctype = c.get('cell_type', 'unk')[:4].upper()
        # Get first line of code for context
        src = c.get('source', '').strip().split('\n')[0][:60]
        print(f"{marker} Idx {i:02d} | {ctype} | {src}")
    print("==========================================\n")

# =============================================================================
# 2. Execution Logic
# =============================================================================

def collect_cell_errors(nb: nbformat.NotebookNode) -> List[Dict[str, Any]]:
    errs = []
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code": continue
        for out in (c.get("outputs") or []):
            if out.get("output_type") == "error":
                tb_str = "\n".join(out.get("traceback", []))
                tb_clean = re.sub(r'\x1b\[[0-9;]*m', '', tb_str)
                errs.append({
                    "cell_index": i,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback": tb_clean,
                })
    return errs

# [MODIFIED] Added cuda_device_id parameter
def execute_once(nb: nbformat.NotebookNode, workdir: str, timeout: int = 1800, cuda_device_id: Optional[int] = None) -> Tuple[nbformat.NotebookNode, List[Dict]]:
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    os.environ["OUTPUT_DIR"] = workdir
    
    # [NEW] Set CUDA Device if provided
    if cuda_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    
    nb_run = deepcopy(nb)
    # Guard against sys.exit
    guard = "# [AUTO-FIX] Guard\nimport sys\nsys.exit = lambda *x: None\n"
    nb_run.cells.insert(0, nbformat.v4.new_code_cell(guard))

    client = NotebookClient(nb_run, timeout=timeout, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": workdir}})
    
    try: client.execute()
    except Exception as e: print(f"[EXEC] Partial exec: {e}")
    
    if len(client.nb.cells) > 0: client.nb.cells.pop(0) # Remove guard
    return client.nb, collect_cell_errors(client.nb)

# =============================================================================
# 3. Heuristics (PRESERVED)
# =============================================================================

def _known_fix_snippet(evalue: str) -> Optional[str]:
    msg = (evalue or "").lower()
    if "could not convert string to float" in msg:
        return "\n# [AUTO-FIX]\nimport numpy as np\nfor k in globals():\n if hasattr(globals()[k], 'select_dtypes'): globals()[k] = globals()[k].select_dtypes(include=[np.number])"
    if "input x contains nan" in msg:
        return "\n# [AUTO-FIX]\nfrom sklearn.impute import SimpleImputer\nimp=SimpleImputer(strategy='median')\nfor k in ['X','X_train']: \n if k in globals(): globals()[k] = imp.fit_transform(globals()[k])"
    return None

def apply_heuristics(nb, errors):
    changed = 0
    for e in errors:
        idx = int(e["cell_index"])
        patch = _known_fix_snippet(e.get("evalue", ""))
        if patch and idx < len(nb.cells):
            src = nb.cells[idx].source
            if patch not in src:
                nb.cells[idx].source += "\n" + patch
                changed += 1
    return changed

# =============================================================================
# 4. LLM Fix (Enhanced with Hash Check & Centralized LLM)
# =============================================================================

def _llm_fix_once(nb, errors, cfg, timeout):
    # 1. Prepare Prompt
    err_context = [{"index": e["cell_index"], "msg": f"{e['ename']}: {e['evalue']}"} for e in errors]
    failing_src = [{"index": i, "code": nb.cells[i].source[-4000:]} for i in {e["cell_index"] for e in errors}]
    
    messages = [{
        "role": "system", 
        "content": "You are a Python Debugger. Return JSON {edits: [{cell_index: int, source: str}]}. Fix the code."
    }, {
        "role": "user",
        "content": json.dumps({"errors": err_context, "code": failing_src})
    }]

    # 2. Robust Call
    try:
        resp = chat_json(messages, cfg, temperature=0.0)
    except Exception as e:
        print(f"[FIX] LLM Error: {e}")
        return nb, False

    # 3. Apply with HASH CHECK (Preserved)
    edits = resp.get("edits", [])
    changed = False
    for ed in edits:
        try:
            idx = int(ed["cell_index"])
            new_src = ed["source"]
            if idx < len(nb.cells):
                old_hash = hashlib.md5(nb.cells[idx].source.encode('utf-8')).hexdigest()
                new_hash = hashlib.md5(new_src.encode('utf-8')).hexdigest()
                if old_hash != new_hash:
                    nb.cells[idx].source = new_src
                    changed = True
        except: pass
        
    return nb, changed

def execute_with_autofix(ipynb_path, workdir, phase_cfg, timeout_seconds=18000, max_fix_rounds=3, **kwargs):
    nb = nbformat.read(ipynb_path, as_version=4)
    
    # [NEW] Extract CUDA ID
    cuda_id = phase_cfg.get("exec", {}).get("cuda_device_id")
    device_msg = f" (CUDA: {cuda_id})" if cuda_id is not None else ""
    print(f"[EXEC] Running: {ipynb_path}{device_msg}")
    
    # Pass cuda_id to execute_once
    exec_nb, errors = execute_once(nb, workdir, timeout_seconds, cuda_device_id=cuda_id)
    out_path = ipynb_path.replace(".ipynb", "_exec.ipynb")
    
    round_idx = 0
    while errors and round_idx < max_fix_rounds:
        # [RESTORED] Visual Debug Map
        _debug_cell_mapping(exec_nb, errors)
        
        round_idx += 1
        print(f"[FIX] Round {round_idx}/{max_fix_rounds} ({len(errors)} errors)")
        
        # A. Heuristics
        nb_copy = deepcopy(exec_nb)
        if apply_heuristics(nb_copy, errors) > 0:
            print(f"   -> Applied Heuristics")
            # Pass cuda_id to execute_once
            nb_copy, errs_h = execute_once(nb_copy, workdir, timeout_seconds, cuda_device_id=cuda_id)
            if not errs_h:
                exec_nb = nb_copy
                errors = []
                print("   -> Fixed by Heuristics!")
                break
        
        # B. LLM
        if errors:
            print(f"   -> Escalating to LLM")
            nb_llm, changed = _llm_fix_once(deepcopy(exec_nb), errors, phase_cfg, timeout_seconds)
            if not changed:
                print("   -> LLM made no effective changes. Stopping.")
                break
            
            # Pass cuda_id to execute_once
            exec_nb, errors = execute_once(nb_llm, workdir, timeout_seconds, cuda_device_id=cuda_id)
            if not errors: print("   -> Fixed by LLM!")

    nbformat.write(exec_nb, out_path)
    print(f"[EXEC] Done. Saved: {out_path}")
    
    # [RESTORED] Final Check Print
    if errors:
        print(f"[EXEC] Failed with {len(errors)} errors remaining.")
        _debug_cell_mapping(exec_nb, errors)
        
    return out_path