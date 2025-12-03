# design_execution/prompt_executor.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, json, re, hashlib
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from copy import deepcopy

# Import centralized LLM tools
from .llm_utils import chat_json

# =============================================================================
# 1. Heuristic Logic (Static Analysis)
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
# 2. LLM Auto-Fix Logic
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
        "task": "Fix specific notebook cells. maintain logic, fix errors.",
        "errors": error_list,
        "failing_code": failing_cells,
        "instruction": "Return FULL corrected source for the failing cells in JSON format."
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

# =============================================================================
# 3. Graph Executor (Adaptive Node-by-Node Execution)
# =============================================================================

class GraphExecutor(NotebookClient):
    """
    Advanced Executor that runs the notebook as a directed graph of cells.
    Allows interrupting execution on error, fixing the specific node (cell),
    and retrying IN-PLACE without restarting the kernel.
    """
    def __init__(self, nb, workdir, llm_config, autofix_prompt, max_fix_rounds=3, **kwargs):
        super().__init__(nb, **kwargs)
        self.workdir = os.path.abspath(workdir)
        self.llm_config = llm_config
        self.autofix_prompt = autofix_prompt
        self.max_fix_rounds = max_fix_rounds
        self.global_errors = [] # Track errors that couldn't be fixed

    def execute_graph(self):
        """Main entry point for graph execution."""
        print(f"[GRAPH] 🚀 Initializing Kernel in {self.workdir}", flush=True)
        
        # 1. Setup Environment & Guard Code
        self._inject_setup_cells()
        
        # 2. Start Persistent Kernel
        self.create_kernel_manager()
        self.start_new_kernel()
        self.start_new_kernel_client()

        try:
            # 3. Iterate Cells (Nodes)
            cell_idx = 0
            while cell_idx < len(self.nb.cells):
                cell = self.nb.cells[cell_idx]
                
                if cell.cell_type != 'code':
                    cell_idx += 1
                    continue

                # Identify Task
                task_meta = cell.metadata.get("subtask", {})
                task_id = task_meta.get("id", f"Cell_{cell_idx}")
                task_name = task_meta.get("name", "Unnamed")
                
                print(f"[GRAPH] ▶️  Running Node {task_id}: {task_name} (Idx: {cell_idx})", flush=True)

                try:
                    # Execute single cell using nbclient's low-level method
                    self.execute_cell(cell, cell_idx)
                    
                    # If we are here, execution was successful
                    cell_idx += 1 
                
                except CellExecutionError:
                    print(f"[GRAPH] ❌ Error in Node {task_id}. Initiating Adaptive Fix...", flush=True)
                    
                    # Try to fix IN-PLACE
                    fixed = self._attempt_node_fix(cell_idx, task_id)
                    
                    if fixed:
                        print(f"[GRAPH] 🔄 Patch applied to Node {task_id}. Retrying immediately...", flush=True)
                        # We do NOT increment cell_idx, so the loop will re-execute the SAME cell index
                        # but with the new source code we just patched into self.nb
                        continue 
                    else:
                        print(f"[GRAPH] 🛑 Failed to fix Node {task_id} after {self.max_fix_rounds} rounds.", flush=True)
                        self.global_errors.append(f"Task {task_id} Failed.")
                        # Stop execution here to preserve partial results or debug
                        break
        
        finally:
            print("[GRAPH] 🛑 Shutting down kernel.", flush=True)
            self._cleanup_kernel()

        return self.nb

    def _inject_setup_cells(self):
        """Inject setup code (Env vars, Guard) at the top."""
        # Sys Exit Guard
        guard_code = (
            "# [AUTO-FIX] Guard Cell & Env Setup\n"
            "import sys, os\n"
            f"os.environ['OUTPUT_DIR'] = r'{self.workdir}'\n"
            "def _guard_exit(*args, **kwargs):\n"
            "    raise RuntimeError(\"SysExitBlocked: Use raise ValueError() instead.\")\n"
            "sys.exit = _guard_exit\n"
            "print(f'[SETUP] Environment Configured. OUTPUT_DIR={os.environ[\"OUTPUT_DIR\"]}')\n"
        )
        # Insert at 0
        self.nb.cells.insert(0, nbformat.v4.new_code_cell(guard_code))

    def _get_cell_errors(self, cell_idx: int) -> List[Dict[str, Any]]:
        """Extract error details from the cell outputs."""
        cell = self.nb.cells[cell_idx]
        errs = []
        for out in (cell.get("outputs") or []):
            if out.get("output_type") == "error":
                tb = out.get("traceback", [])
                tb_str = "\n".join(tb) if isinstance(tb, list) else str(tb)
                tb_clean = re.sub(r'\x1b\[[0-9;]*m', '', tb_str)
                errs.append({
                    "cell_index": cell_idx,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback": tb_clean,
                })
        return errs

    def _attempt_node_fix(self, cell_idx: int, task_id: str) -> bool:
        """
        The Local Fix Loop.
        Returns True if the cell source was modified and should be retried.
        """
        for attempt in range(self.max_fix_rounds):
            errors = self._get_cell_errors(cell_idx)
            if not errors:
                return False # Should not happen inside CellExecutionError catch

            print(f"   [FIX] {task_id} | Round {attempt+1}/{self.max_fix_rounds} | Analyzing...", flush=True)

            # 1. Heuristics (Fast path)
            h_changes, _ = _apply_heuristics(self.nb, errors)
            if h_changes > 0:
                print(f"   [FIX] ⚡ Applied heuristic patch.", flush=True)
                return True

            # 2. LLM (Slow path)
            # Pass a COPY of notebook to LLM func to avoid partial mutations if it fails
            # But we want the result to apply to SELF.NB if successful
            nb_copy = deepcopy(self.nb)
            
            # NOTE: We only send the CURRENT failing cell errors to the LLM
            nb_fixed, patched = _llm_auto_fix_once(
                nb_copy, 
                errors, 
                self.llm_config, 
                self.autofix_prompt
            )

            if patched:
                # Update our live notebook's specific cell
                # We trust _llm_auto_fix_once has modified the cell at cell_idx in nb_copy
                new_source = nb_fixed.cells[cell_idx].source
                if self.nb.cells[cell_idx].source != new_source:
                    self.nb.cells[cell_idx].source = new_source
                    print(f"   [FIX] 🧠 LLM patch generated and applied.", flush=True)
                    return True
                else:
                    print(f"   [FIX] ⚠️ LLM returned identical code.", flush=True)
            
        return False

# =============================================================================
# 4. Main Entry Point
# =============================================================================

def run_notebook_with_autofix(
    nb_path: str,
    workdir: str,
    cfg: Dict[str, Any]
) -> str:
    """
    Executes notebook using the Adaptive Graph Executor.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    
    # Load Notebook
    nb_orig = nbformat.read(nb_path, as_version=4)
    
    # Config
    exec_cfg = cfg.get("exec", {})
    max_rounds = int(exec_cfg.get("max_fix_rounds", 3))
    
    prompts_map = cfg.get("prompts", {})
    autofix_prompt = prompts_map.get("autofix", {}).get("system_prompt", 
        "You are a Python Expert. Fix the code errors provided. Return JSON with 'edits'.")

    # Instantiate Graph Executor
    executor = GraphExecutor(
        nb=nb_orig,
        workdir=workdir,
        llm_config=cfg.get("llm", {}),
        autofix_prompt=autofix_prompt,
        max_fix_rounds=max_rounds,
        # nbclient args
        timeout=int(exec_cfg.get("timeout_seconds", 3600)),
        kernel_name="python3",
        allow_errors=False, # We handle errors manually
        resources={"metadata": {"path": workdir}}
    )
    
    # Run
    print(f"[EXEC] Starting Adaptive Graph Execution: {nb_path}", flush=True)
    try:
        final_nb = executor.execute_graph()
    except Exception as e:
        print(f"[EXEC] ☢️ Critical Framework Error: {e}", flush=True)
        final_nb = executor.nb

    # Save Result
    out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    
    # Remove the injected setup cell (index 0) before saving to keep it clean (optional)
    if len(final_nb.cells) > 0 and "[AUTO-FIX] Guard Cell" in final_nb.cells[0].source:
        final_nb.cells.pop(0)

    nbformat.write(final_nb, out_path)
    print(f"[EXEC] Saved final state: {out_path}", flush=True)
    
    if executor.global_errors:
        print(f"[EXEC] Finished with unresolved errors: {executor.global_errors}", flush=True)
    else:
        print(f"[EXEC] ✅ Execution completed successfully.", flush=True)
        
    return out_path