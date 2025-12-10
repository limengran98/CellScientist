from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import re
import hashlib
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from copy import deepcopy

# Import centralized LLM utilities
from llm_utils import chat_json

# =============================================================================
# 0. Verbose Client (带输出监控的话痨版)
# =============================================================================

class VerboseNotebookClient(NotebookClient):
    """
    A wrapper that prints real-time progress AND cell outputs (stdout/stderr).
    """
    
    def setup_kernel(self, **kwargs):
        print(f"[EXEC] 🚀 Starting Kernel Manager...", flush=True)
        return super().setup_kernel(**kwargs)

    def _log_start(self, cell, cell_index):
        cell_type = cell.get("cell_type", "unknown")
        snippet = cell.get("source", "").strip().split("\n")[0][:50]
        if cell_type == "code":
            print(f"[EXEC] ⏳ Cell {cell_index} [CODE] Running... | {snippet}...", flush=True)

    def _log_outputs(self, cell, cell_index):
        """Helper to print stdout/stderr from the cell."""
        if cell.get("cell_type") != "code": return
        
        outputs = cell.get("outputs", [])
        for out in outputs:
            # 打印标准输出 (print语句)
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                text = out.get("text", "").strip()
                if text:
                    print(f"       📝 [STDOUT]: {text}", flush=True)
            
            # 打印标准错误 (stderr)
            elif out.get("output_type") == "stream" and out.get("name") == "stderr":
                text = out.get("text", "").strip()
                if text:
                    print(f"       ⚠️ [STDERR]: {text}", flush=True)
            
            # 打印错误回溯 (Traceback)
            elif out.get("output_type") == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "Unknown")
                print(f"       ❌ [ERROR]: {ename} - {evalue}", flush=True)

    def _log_end(self, cell, cell_index, success=True):
        if cell.get("cell_type") == "code":
            icon = "✅" if success else "❌"
            print(f"[EXEC] {icon} Cell {cell_index} Done.", flush=True)

    # --- 拦截同步执行 ---
    def execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        self._log_start(cell, cell_index)
        try:
            result = super().execute_cell(cell, cell_index, execution_count, store_history)
            self._log_outputs(cell, cell_index) # <--- [关键] 打印输出
            self._log_end(cell, cell_index, success=True)
            return result
        except Exception as e:
            self._log_outputs(cell, cell_index) # <--- [关键] 即使挂了也要打印输出
            self._log_end(cell, cell_index, success=False)
            raise e

    # --- 拦截异步执行 ---
    async def async_execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        self._log_start(cell, cell_index)
        try:
            result = await super().async_execute_cell(cell, cell_index, execution_count, store_history)
            self._log_outputs(cell, cell_index) # <--- [关键] 打印输出
            self._log_end(cell, cell_index, success=True)
            return result
        except Exception as e:
            self._log_outputs(cell, cell_index) # <--- [关键] 即使挂了也要打印输出
            self._log_end(cell, cell_index, success=False)
            raise e

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
    log_path = os.path.join(workdir, f"error_log_round_{round_idx}.txt")
    report = f"=== ERROR REPORT (Round {round_idx}) ===\n"
    print(f"\n{'!'*60}")
    print(f"[ERROR DUMP] Found {len(errors)} errors. Details saved to: {log_path}")
    
    for e in errors:
        idx = e['cell_index']
        print(f"\n>> Cell {idx} Error: {e['ename']} - {e['evalue']}")
        # 强制打印 Traceback 的最后几行
        trace_tail = e['traceback'][-500:] if len(e['traceback']) > 500 else e['traceback']
        print(f"   [Traceback]:\n{trace_tail}")
        
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

    print(f"{'!'*60}\n")
    with open(log_path, "w", encoding="utf-8") as f: f.write(report)
    return log_path

# =============================================================================
# 2. Pure Execution (Code Injection + Verbose Output)
# =============================================================================

def run_notebook_pure(nb: nbformat.NotebookNode, 
                      workdir: str, 
                      timeout: int = 1800, 
                      cuda_device_id: Optional[int] = None,
                      extra_env: Optional[Dict[str, str]] = None) -> Tuple[nbformat.NotebookNode, List[Dict]]:
    """
    Executes the notebook using VerboseNotebookClient.
    Uses Code Injection for reliable environment variables.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    
    nb_run = deepcopy(nb)
    
    # --- [Code Injection] ---
    setup_code = [
        "# [SYSTEM] Auto-Injected Configuration",
        "import os",
        f"os.environ['OUTPUT_DIR'] = r'{workdir}'"
    ]
    
    if extra_env:
        print(f"[EXEC] 💉 Injecting Code Variables: {list(extra_env.keys())}", flush=True)
        for k, v in extra_env.items():
            setup_code.append(f"os.environ['{k}'] = {repr(v)}")
            
    # Inject Guard (Keep sys.exit patched for now, but watch logs!)
    setup_code.append("import sys; sys.exit = lambda *x: print('SYS.EXIT TRIGGERED (IGNORED)')")
    
    setup_cell = nbformat.v4.new_code_cell("\n".join(setup_code))
    nb_run.cells.insert(0, setup_cell)
    # ------------------------

    # Env Setup
    env_vars = os.environ.copy()
    if cuda_device_id is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)

    # Use Verbose Client
    client = VerboseNotebookClient(
        nb_run, 
        timeout=timeout, 
        kernel_name="python3", 
        allow_errors=True,
        resources={"metadata": {"path": workdir}},
        env=env_vars
    )
    
    try: 
        client.execute()
    except Exception as e: 
        print(f"[EXEC] Kernel Exception (Partial): {e}")
    
    if len(client.nb.cells) > 0: client.nb.cells.pop(0) 
        
    errors = collect_cell_errors(client.nb)
    return client.nb, errors

# =============================================================================
# 3. Auto-Fix Logic (Standard)
# =============================================================================

def _apply_heuristics(nb: nbformat.NotebookNode, errors: List[Dict]) -> int:
    changed = 0
    for e in errors:
        idx = int(e["cell_index"])
        msg = (e.get("evalue") or "").lower()
        patch = None
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
    err_context = [{"index": e["cell_index"], "msg": f"{e['ename']}: {e['evalue']}"} for e in errors]
    failing_src = [{"index": i, "code": nb.cells[i].source[-2000:]} for i in {e["cell_index"] for e in errors}]
    messages = [{
        "role": "system", 
        "content": "You are a Python Debugger. Fix the provided code errors.\nReturn ONLY a JSON object: {\"edits\": [{\"cell_index\": <int>, \"source\": \"<full_new_code>\"}]}"
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
                old_hash = hashlib.md5(nb.cells[idx].source.encode('utf-8')).hexdigest()
                new_hash = hashlib.md5(new_src.encode('utf-8')).hexdigest()
                if old_hash != new_hash:
                    nb.cells[idx].source = new_src
                    changed = True
        except: pass
    return nb, changed

def attempt_fix_notebook(nb: nbformat.NotebookNode, 
                         errors: List[Dict], 
                         cfg: Dict[str, Any],
                         mutable_indices: Optional[List[int]] = None) -> Tuple[nbformat.NotebookNode, bool, str]:
    nb_next = deepcopy(nb)
    target_errors = errors
    if mutable_indices is not None:
        target_errors = []
        for e in errors:
            idx = int(e["cell_index"])
            if idx in mutable_indices:
                target_errors.append(e)
            else:
                print(f"[FIX] 🚫 Skipping fix for Immutable Cell {idx} (Error: {e.get('ename')})")
        if not target_errors:
            return nb, False, "Aborted:ImmutableErrorsOnly"
    h_changes = _apply_heuristics(nb_next, target_errors)
    if h_changes > 0: return nb_next, True, "Heuristics"
    nb_llm, l_changes = _llm_fix_request(nb_next, target_errors, cfg)
    if l_changes: return nb_llm, True, "LLM"
    return nb, False, "None"

# =============================================================================
# 4. Main Execution Loop
# =============================================================================

def execute_and_recover(nb_path: str, workdir: str, cfg: Dict[str, Any], mutable_indices: Optional[List[int]] = None, extra_env: Optional[Dict[str, str]] = None) -> Tuple[str, bool]:
    timeout = cfg["exec"]["timeout_seconds"]
    max_fixes = cfg["exec"]["max_fix_rounds"]
    current_nb_path = nb_path
    
    nb = nbformat.read(current_nb_path, as_version=4)
    print(f"\n[EXEC] 🟢 Starting Run: {os.path.basename(current_nb_path)}", flush=True)
    
    executed_nb, errors = run_notebook_pure(nb, workdir, timeout, cuda_device_id=None, extra_env=extra_env)
    
    if not errors:
        out_path = current_nb_path.replace(".ipynb", "_exec.ipynb")
        nbformat.write(executed_nb, out_path)
        print(f"[EXEC] 🏁 Run Completed Successfully.", flush=True)
        return out_path, True
    
    fix_round = 0
    current_nb_obj = executed_nb
    
    while errors and fix_round < max_fixes:
        fix_round += 1
        print(f"\n[EXEC] 🔴 Errors Found (Round {fix_round}) - Initiating Recovery...", flush=True)
        dump_error_log(workdir, errors, round_idx=fix_round)
        
        fixed_nb, changed, method = attempt_fix_notebook(current_nb_obj, errors, cfg, mutable_indices)
        if not changed:
            print(f"[EXEC] ⚠️ Could not generate valid fix ({method}). Stopping.", flush=True)
            break
        print(f"[EXEC] 🔧 Applying Fix ({method}) -> Rerunning...", flush=True)
        current_nb_obj, errors = run_notebook_pure(fixed_nb, workdir, timeout, cuda_device_id=None, extra_env=extra_env)
        if not errors:
            print(f"[EXEC] ✅ Fixed successfully!", flush=True)
            break

    final_out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    nbformat.write(current_nb_obj, final_out_path)
    if errors:
        print(f"[EXEC] ❌ Final Execution Failed.", flush=True)
        return final_out_path, False
    return final_out_path, True