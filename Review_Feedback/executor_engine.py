from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import re
import hashlib
import nbformat
from nbclient import NotebookClient
from copy import deepcopy

# Import centralized LLM utilities
from llm_utils import chat_json

# =============================================================================
# 1. Error Handling & Reporting
# =============================================================================

def collect_cell_errors(nb: nbformat.NotebookNode) -> List[Dict[str, Any]]:
    """Extracts errors from the notebook object."""
    errs = []
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code": continue
        for out in (c.get("outputs") or []):
            if out.get("output_type") == "error":
                # Strip ANSI color codes
                tb_str = "\n".join(out.get("traceback", []))
                tb_clean = re.sub(r'\x1b\[[0-9;]*m', '', tb_str)
                errs.append({
                    "cell_index": i,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback": tb_clean,
                    "source": c.get("source", "")
                })
    return errs

def dump_error_log(workdir: str, errors: List[Dict], round_idx: int = 0) -> str:
    """
    Explicitly dumps errors to console and file.
    Generic version: Reports details for ALL errors found.
    """
    log_path = os.path.join(workdir, f"error_log_round_{round_idx}.txt")
    
    report = f"=== ERROR REPORT (Round {round_idx}) ===\n"
    print(f"\n[ERROR DUMP] Found {len(errors)} errors. Details saved to: {log_path}")
    
    for e in errors:
        idx = e['cell_index']
        
        entry = (
            f"\n{'-'*60}\n"
            f"CELL INDEX: {idx}\n"
            f"ERROR TYPE: {e['ename']}\n"
            f"MESSAGE   : {e['evalue']}\n"
            f"{'-'*20} SOURCE CODE {'-'*20}\n"
            f"{e['source']}\n"
            f"{'-'*20} TRACEBACK {'-'*20}\n"
            f"{e['traceback']}\n"
        )
        report += entry
        
        # Immediate Console Output for Visibility (Generic for all cells)
        print(f"\n>> Cell {idx} Error: {e['ename']} - {e['evalue']}")
        # Print a brief preview for every error to help debugging
        print(f"   [Source Preview]: {e['source'][:100].replace(chr(10), ' ')}...") 
        print(f"   [Traceback Tail]: ...{e['traceback'][-300:].replace(chr(10), ' ')}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    return log_path

# =============================================================================
# 2. Pure Execution (No LLM)
# =============================================================================

def run_notebook_pure(nb: nbformat.NotebookNode, 
                      workdir: str, 
                      timeout: int = 1800, 
                      cuda_device_id: Optional[int] = None) -> Tuple[nbformat.NotebookNode, List[Dict]]:
    """
    Executes the notebook without any auto-fix attempts.
    Returns the executed notebook object and a list of errors.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    os.environ["OUTPUT_DIR"] = workdir
    
    if cuda_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    
    nb_run = deepcopy(nb)
    
    # Inject Guard to prevent sys.exit from killing the kernel
    guard = "# [SYSTEM] Guard\nimport sys\nsys.exit = lambda *x: None\n"
    nb_run.cells.insert(0, nbformat.v4.new_code_cell(guard))

    # Execute using nbclient
    client = NotebookClient(
        nb_run, 
        timeout=timeout, 
        kernel_name="python3", 
        allow_errors=True, 
        resources={"metadata": {"path": workdir}}
    )
    
    try: 
        client.execute()
    except Exception as e: 
        print(f"[EXEC] Kernel Exception (Partial): {e}")
    
    # Remove the guard cell
    if len(client.nb.cells) > 0: 
        client.nb.cells.pop(0) 
        
    errors = collect_cell_errors(client.nb)
    return client.nb, errors

# =============================================================================
# 3. Auto-Fix Logic (LLM + Heuristics)
# =============================================================================

def _apply_heuristics(nb: nbformat.NotebookNode, errors: List[Dict]) -> int:
    """Apply rule-based fixes."""
    changed = 0
    for e in errors:
        idx = int(e["cell_index"])
        msg = (e.get("evalue") or "").lower()
        patch = None
        
        # Common Pandas/Numpy fixes
        if "could not convert string to float" in msg:
            patch = "\n# [AUTO-FIX] Force numeric types\nimport numpy as np\nfor k in list(globals().keys()):\n if hasattr(globals()[k], 'select_dtypes'): globals()[k] = globals()[k].select_dtypes(include=[np.number])"
        elif "input x contains nan" in msg:
            patch = "\n# [AUTO-FIX] Impute NaNs\nfrom sklearn.impute import SimpleImputer\nimp=SimpleImputer(strategy='median')\nfor k in ['X','X_train']: \n if k in globals(): globals()[k] = imp.fit_transform(globals()[k])"
            
        if patch and idx < len(nb.cells):
            src = nb.cells[idx].source
            if patch not in src:
                nb.cells[idx].source += "\n" + patch
                changed += 1
    return changed

def _llm_fix_request(nb: nbformat.NotebookNode, errors: List[Dict], cfg: Dict[str, Any]) -> Tuple[nbformat.NotebookNode, bool]:
    """Request a fix from the LLM."""
    
    # Construct Context
    err_context = [{"index": e["cell_index"], "msg": f"{e['ename']}: {e['evalue']}"} for e in errors]
    failing_src = [{"index": i, "code": nb.cells[i].source[-2000:]} for i in {e["cell_index"] for e in errors}]
    
    messages = [{
        "role": "system", 
        "content": (
            "You are a Python Debugger. Fix the provided code errors.\n"
            "Return ONLY a JSON object: {\"edits\": [{\"cell_index\": <int>, \"source\": \"<full_new_code>\"}]}"
        )
    }, {
        "role": "user",
        "content": json.dumps({"errors": err_context, "code": failing_src})
    }]

    try:
        resp = chat_json(messages, cfg, temperature=0.1)
    except Exception as e:
        print(f"[FIX] LLM Error: {e}")
        return nb, False

    edits = resp.get("edits", [])
    changed = False
    for ed in edits:
        try:
            idx = int(ed["cell_index"])
            new_src = ed["source"]
            if idx < len(nb.cells):
                # Hash check to prevent useless edits
                old_hash = hashlib.md5(nb.cells[idx].source.encode('utf-8')).hexdigest()
                new_hash = hashlib.md5(new_src.encode('utf-8')).hexdigest()
                if old_hash != new_hash:
                    nb.cells[idx].source = new_src
                    changed = True
        except: pass
        
    return nb, changed

def attempt_fix_notebook(nb: nbformat.NotebookNode, 
                         errors: List[Dict], 
                         cfg: Dict[str, Any]) -> Tuple[nbformat.NotebookNode, bool, str]:
    """
    Attempts to fix the notebook. First Heuristics, then LLM.
    Returns: (NewNotebook, Changed(bool), Method(str))
    """
    nb_next = deepcopy(nb)
    
    # 1. Try Heuristics
    h_changes = _apply_heuristics(nb_next, errors)
    if h_changes > 0:
        return nb_next, True, "Heuristics"
        
    # 2. Try LLM
    nb_llm, l_changes = _llm_fix_request(nb_next, errors, cfg)
    if l_changes:
        return nb_llm, True, "LLM"
        
    return nb, False, "None"