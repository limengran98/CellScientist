# nb_autofix.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json, re, hashlib
import nbformat
from nbclient import NotebookClient
from pathlib import Path
from copy import deepcopy

# =============================================================================
# [FIX] Robust Import Strategy for Script vs Package Execution
# =============================================================================
try:
    from run_llm_nb import (
        chat_json as _llm_chat_json,
        apply_edits as _llm_apply_edits,
        execute_notebook as _llm_execute_notebook,
    )
except ImportError:
    from .run_llm_nb import (
        chat_json as _llm_chat_json,
        apply_edits as _llm_apply_edits,
        execute_notebook as _llm_execute_notebook,
    )

# =============================================================================
# LLM profile loading
# =============================================================================

def _load_llm_profile(cfg: dict | None, *, verbose: bool = False) -> dict:
    cfg = cfg or {}
    llm_cfg = cfg.get("llm") or {}
    providers = cfg.get("providers") or {}

    provider = llm_cfg.get("provider") or cfg.get("default_provider")
    if not provider and providers:
        provider = next(iter(providers.keys()))

    prov_profile = (providers.get(provider) or {}) if provider else {}
    model = llm_cfg.get("model") or prov_profile.get("model")
    if not model:
        models = prov_profile.get("models") or []
        model = models[0] if models else "gpt-4"

    base_url = (
        llm_cfg.get("base_url")
        or os.environ.get(f"{(provider or 'openai').upper()}_BASE_URL")
        or prov_profile.get("base_url")
    )
    
    api_key = (
        llm_cfg.get("api_key")
        or os.environ.get(f"{(provider or 'openai').upper()}_API_KEY")
        or prov_profile.get("api_key")
    )

    temperature = float(prov_profile.get("temperature", llm_cfg.get("temperature", 0.0)))
    max_tokens = int(prov_profile.get("max_tokens", llm_cfg.get("max_tokens", 20000)))

    if verbose:
        print(f"[AUTO-FIX][LLM] provider={provider} model={model} base_url={base_url}")

    return {
        "provider": provider or "openai_compat",
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

# =============================================================================
# Basic utilities
# =============================================================================

def collect_cell_errors(nb: nbformat.NotebookNode) -> List[Dict[str, Any]]:
    errs: List[Dict[str, Any]] = []
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
    allow_errors: bool = True,
    inject_exit_guard: bool = True,
) -> Tuple[nbformat.NotebookNode, List[Dict[str, Any]]]:
    
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    os.environ["OUTPUT_DIR"] = workdir

    nb_to_run = deepcopy(nb)
    guard_added = False
    if inject_exit_guard:
        guard_code = (
            "# [AUTO-FIX] Guard Cell\n"
            "import sys, os\n"
            "def _guard_exit(*args, **kwargs):\n"
            "    raise RuntimeError(\"SysExitBlocked: Do not use sys.exit(). Use raise ValueError() instead.\")\n"
            "sys.exit = _guard_exit\n"
        )
        nb_to_run.cells.insert(0, nbformat.v4.new_code_cell(guard_code))
        guard_added = True

    client = NotebookClient(
        nb_to_run, timeout=timeout, kernel_name="python3",
        allow_errors=allow_errors, resources={"metadata": {"path": workdir}},
    )
    
    try:
        exec_nb = client.execute()
    except Exception as e:
        print(f"[EXEC] Warning: Notebook execution crashed: {e}")
        exec_nb = client.nb

    if guard_added and len(exec_nb.cells) > 0:
        exec_nb.cells.pop(0)

    errors = collect_cell_errors(exec_nb)
    return exec_nb, errors

def _short(s: str, n: int = 160) -> str:
    if not s: return ""
    s = s.replace("\n", " ").strip()
    return (s[:n] + "...") if len(s) > n else s

# =============================================================================
# HEURISTIC PATCHES
# =============================================================================

def _patch_sanitize_features() -> str:
    return (
        "import numpy as np, pandas as pd\n"
        "def _sanitize_X(X):\n"
        "    if isinstance(X, pd.DataFrame):\n"
        "        print('[AUTO-FIX] Filtering X to numeric-only...')\n"
        "        return X.select_dtypes(include=[np.number])\n"
        "    return X\n"
        "for _nm in ('X', 'X_train', 'X_test', 'X_val', 'X_all'):\n"
        "    if _nm in globals(): globals()[_nm] = _sanitize_X(globals()[_nm])\n"
    )

def _patch_minimal_config() -> str:
    return (
        "import os\n"
        "print('[AUTO-FIX] reconstructing minimal config...')\n"
        "config = {'experiment': {'primary_metric': 'PCC'}, 'training': {'max_epochs': 100}}\n"
    )

def _patch_fix_missing_file() -> str:
    return (
        "import os\n"
        "from pathlib import Path\n"
        "print('[AUTO-FIX] Resolving paths...')\n"
        "def _find_root():\n"
        "    p = Path(os.getcwd()).resolve()\n"
        "    for _ in range(3): \n"
        "        if (p/'data').exists(): return p\n"
        "        p = p.parent\n"
        "    return Path(os.getcwd()).resolve()\n"
        "root=_find_root()\n"
        "for var in ('DATA_FILE','STAGE1_H5_PATH'):\n"
        "    if var in globals() and isinstance(globals()[var], str):\n"
        "        p = Path(globals()[var])\n"
        "        if not p.is_absolute():\n"
        "            globals()[var] = str((root/p).resolve())\n"
    )

def _patch_impute_nans() -> str:
    return (
        "from sklearn.impute import SimpleImputer\n"
        "import numpy as np, pandas as pd\n"
        "print('[AUTO-FIX] Imputing NaNs...')\n"
        "imp = SimpleImputer(strategy='median')\n"
        "for _nm in ('X', 'X_train'):\n"
        "    if _nm in globals():\n"
        "        val = globals()[_nm]\n"
        "        if hasattr(val, 'shape') and len(val.shape)==2:\n"
        "             globals()[_nm] = imp.fit_transform(val)\n"
    )

def _known_fix_snippet(evalue: str) -> Optional[str]:
    msg = (evalue or "").lower()
    if "could not convert string to float" in msg: return _patch_sanitize_features()
    if "filenotfounderror" in msg: return _patch_fix_missing_file()
    if "name 'config' is not defined" in msg: return _patch_minimal_config()
    if "input x contains nan" in msg: return _patch_impute_nans()
    return None

def apply_known_patches(nb: nbformat.NotebookNode, errors: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
    changed = 0
    changed_indices: List[int] = []
    for e in errors:
        idx = int(e["cell_index"])
        patch = _known_fix_snippet(e.get("evalue",""))
        if not patch: continue
        if idx < 0 or idx >= len(nb.cells): continue
        cell = nb.cells[idx]
        if cell.get("cell_type") != "code": continue
        
        src = cell.get("source") or ""
        if patch not in src:
            cell["source"] = src.rstrip() + "\n\n# [AUTO-FIX] Heuristic Patch\n" + patch
            changed += 1
            changed_indices.append(idx)
    return changed, changed_indices

def _restore_sources(exec_nb, original_sources, indices=None):
    if indices is None:
        indices = list(range(min(len(original_sources), len(exec_nb.cells))))
    for i in indices:
        if 0 <= i < len(exec_nb.cells) and i < len(original_sources):
            if exec_nb.cells[i].get("cell_type") == "code":
                exec_nb.cells[i]["source"] = original_sources[i]

# =============================================================================
# LLM Repair Logic (Hash-Verified)
# =============================================================================

def _debug_cell_mapping(nb: nbformat.NotebookNode, errors: List[Dict[str, Any]]):
    """Visually confirm error locations."""
    print("\n[AUTO-FIX] === CELL INDEX MAP (DEBUG) ===")
    err_indices = {e['cell_index'] for e in errors}
    for i, c in enumerate(nb.cells):
        marker = " [ERROR] >>" if i in err_indices else "           "
        ctype = c.get('cell_type', 'unk')[:4].upper()
        src = c.get('source', '').strip().split('\n')[0][:60]
        print(f"{marker} Idx {i:02d} | {ctype} | {src}")
    print("==========================================\n")

def _make_bugfix_messages(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    max_src_chars: int = 8000
) -> List[Dict[str, str]]:
    
    target_indices = sorted({int(e["cell_index"]) for e in errors})
    failing_cells = []
    for i in target_indices:
        if 0 <= i < len(nb.cells):
            src = nb.cells[i].get("source", "")
            if len(src) > max_src_chars: src = src[:max_src_chars] + "\n# ... [TRUNCATED]"
            failing_cells.append({"cell_index": i, "source": src})

    error_list = []
    is_syntax_error = False
    for e in errors:
        ename = e.get("ename", "")
        evalue = e.get("evalue", "")
        if "SyntaxError" in ename or "unterminated string" in evalue:
            is_syntax_error = True
            evalue += " \n[HINT: Check nested quotes in f-strings! Extract variables first.]"
        if "could not convert string to float" in evalue:
            evalue += " \n[HINT: Use df.select_dtypes(include=[np.number]) to filter.]"
        
        error_list.append({
            "cell_index": e['cell_index'],
            "error": f"{ename}: {evalue}",
            "traceback": (e.get("traceback") or "")[-2000:]
        })

    # Load system prompt
    prompt_path = os.path.join(os.getcwd(), "prompts", "autofix.yml")
    base_prompt = "You are an Expert Python Debugger. Fix the code. Return JSON."
    try:
        import yaml
        if os.path.exists(prompt_path):
            with open(prompt_path) as f: base_prompt = yaml.safe_load(f).get("system_prompt")
    except: pass
    
    if is_syntax_error:
        base_prompt = "CRITICAL: SYNTAX ERROR DETECTED. CHECK F-STRINGS.\n" + base_prompt

    user_payload = {
        "task": "Fix specific notebook cells.",
        "errors": error_list,
        "failing_code": failing_cells,
        "instruction": "Return FULL corrected source for the failing cells."
    }

    return [
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2)}
    ]

def _llm_auto_fix_once(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    *,
    phase_cfg: dict | None,
    timeout: int,
    verbose: bool,
    workdir: str
) -> Tuple[nbformat.NotebookNode, List[Dict[str, Any]], List[int]]:
    
    if verbose: _debug_cell_mapping(nb, errors)
    prof = _load_llm_profile(phase_cfg, verbose=verbose)
    
    # 1. Capture Pre-Fix State (Hash)
    cell_hashes_before = {}
    for i, c in enumerate(nb.cells):
        cell_hashes_before[i] = hashlib.md5(c.source.encode('utf-8')).hexdigest()

    # 2. Call LLM
    messages = _make_bugfix_messages(nb, errors)
    try:
        spec = _llm_chat_json(
            messages, api_key=prof["api_key"], base_url=prof["base_url"],
            model=prof["model"], temperature=0.2, max_tokens=prof["max_tokens"]
        )
    except Exception as e:
        print(f"[AUTO-FIX][ERROR] LLM Call Failed: {e}")
        return nb, errors, []

    edits = spec.get("edits") or []
    if not edits:
        print("[AUTO-FIX] LLM returned no edits.")
        return nb, errors, []

    # 3. Apply Edits
    target_indices = {e['cell_index'] for e in errors}
    valid_edits = []
    for ed in edits:
        idx = int(ed.get("cell_index", -1))
        src = ed.get("source")
        if idx in target_indices and src:
            valid_edits.append({"cell_index": idx, "source": src})
        else:
            print(f"[AUTO-FIX] Ignoring edit for non-failing cell {idx}.")

    if not valid_edits:
        print("[AUTO-FIX] No valid edits applied.")
        return nb, errors, []

    _llm_apply_edits(nb, valid_edits)

    # 4. VERIFICATION: Hash Check
    patched_idx = []
    for edit in valid_edits:
        idx = edit["cell_index"]
        new_hash = hashlib.md5(nb.cells[idx].source.encode('utf-8')).hexdigest()
        if new_hash != cell_hashes_before.get(idx):
            print(f"[AUTO-FIX] >>> VERIFIED: Cell {idx} content was successfully updated.")
            patched_idx.append(idx)
        else:
            print(f"[AUTO-FIX] >>> WARNING: Cell {idx} content did NOT change after patch (Duplicate code?).")

    if not patched_idx:
        print("[AUTO-FIX] No effective changes made. Skipping execution.")
        return nb, errors, []

    # 5. Re-execute
    print(f"[AUTO-FIX] Executing patched notebook...")
    nb_new, new_errs = execute_once(nb, workdir, timeout=timeout)
    return nb_new, new_errs, patched_idx

# =============================================================================
# Main Entry
# =============================================================================

def execute_with_autofix(
    nb_path: Optional[str] = None,
    *,
    workdir: str,
    timeout_seconds: int = 18000,
    max_fix_rounds: int = 3,
    verbose: bool = True,
    phase_cfg: dict | None = None,
    save_intermediates: bool = True,
    **kwargs
) -> str:
    
    if nb_path is None: nb_path = kwargs.get("ipynb_path")
    nb_orig = nbformat.read(nb_path, as_version=4)
    
    print(f"[AUTO-FIX] Initial Execution: {nb_path}")
    exec_nb, errors = execute_once(nb_orig, workdir, timeout=timeout_seconds)
    out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    round_idx = 0

    while errors and round_idx < max_fix_rounds:
        round_idx += 1
        print(f"\n[AUTO-FIX] === ROUND {round_idx}/{max_fix_rounds} ===")
        
        # 1. Heuristics
        nb_run = deepcopy(nb_orig)
        changed, _ = apply_known_patches(nb_run, errors)
        if changed > 0:
            print(f"[AUTO-FIX] Applied {changed} heuristic patches...")
            exec_nb, errors = execute_once(nb_run, workdir, timeout=timeout_seconds)
            if not errors:
                print("[AUTO-FIX] Success via heuristics!")
                break
        
        # 2. LLM
        if errors:
            print("[AUTO-FIX] Heuristics failed. Escalating to LLM...")
            nb_fixed, errors_new, patched = _llm_auto_fix_once(
                deepcopy(exec_nb), errors, phase_cfg=phase_cfg, 
                timeout=timeout_seconds, verbose=verbose, workdir=workdir
            )
            if not patched:
                print("[AUTO-FIX] LLM failed to effectively patch. Stopping.")
                break
            exec_nb = nb_fixed
            errors = errors_new
            if not errors:
                print("[AUTO-FIX] Success via LLM!")
                break

    nbformat.write(exec_nb, out_path)
    print(f"[AUTO-FIX] Saved final result: {out_path}")
    if errors:
        _debug_cell_mapping(exec_nb, errors)
        print(f"[AUTO-FIX] Failed with {len(errors)} errors remaining.")
    
    return out_path