# nb_autofix.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json, re
import nbformat
from nbclient import NotebookClient
from pathlib import Path
from copy import deepcopy

# Import utilities used for LLM-based repair I/O.
# Note: chat_json() will also print model/base_url on call.
from .run_llm_nb import (
    build_fix_messages as _llm_build_fix_messages,  # kept for compatibility (not used here)
    chat_json as _llm_chat_json,
    apply_edits as _llm_apply_edits,
    execute_notebook as _llm_execute_notebook,      # kept for compatibility (not used here)
)

# =============================================================================
# LLM profile loading (CFG-ONLY; no llm_providers.json dependency)
# =============================================================================

def _load_llm_profile(cfg: dict | None, *, verbose: bool = False) -> dict:
    """
    Resolve LLM connection profile ONLY from the merged runtime config (cfg).
    Priority:
      1) cfg["llm"].* overrides,
      2) cfg["providers"][provider].* if present,
      3) minimal defaults.
    Returns a dict containing: provider, model, base_url, api_key, temperature, max_tokens.
    """
    cfg = cfg or {}
    llm_cfg = cfg.get("llm") or {}
    providers = cfg.get("providers") or {}

    # provider selection: llm.provider > default_provider > first key
    provider = llm_cfg.get("provider") or cfg.get("default_provider")
    if not provider and providers:
        provider = next(iter(providers.keys()))

    prov_profile = (providers.get(provider) or {}) if provider else {}

    # model override priority: llm.model > provider.model/models[0] > default fallback
    model = llm_cfg.get("model") or prov_profile.get("model")
    if not model:
        models = prov_profile.get("models") or []
        model = models[0] if models else "gpt-5"

    # base_url/api_key: llm.* overrides > ENV > provider profile
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

    temperature = float(
        prov_profile.get("temperature", llm_cfg.get("temperature", 0.0))
    )
    max_tokens = int(
        prov_profile.get("max_tokens", llm_cfg.get("max_tokens", 20000))
    )

    if verbose:
        print(
            f"[AUTO-FIX][LLM] provider={provider or 'openai_compat'} "
            f"model={model} temp={temperature} max_tokens={max_tokens}"
        )

    return {
        "provider": provider or "openai_compat",
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

# =============================================================================
# Basic utilities: notebook execution and error collection
# =============================================================================

def collect_cell_errors(nb: nbformat.NotebookNode) -> List[Dict[str, Any]]:
    """Scan executed notebook and collect error outputs from code cells."""
    errs: List[Dict[str, Any]] = []
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code":
            continue
        for out in (c.get("outputs") or []):
            if out.get("output_type") == "error":
                errs.append({
                    "cell_index": i,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback": "\n".join(out.get("traceback") or []),
                })
    return errs

def execute_once(
    nb: nbformat.NotebookNode,
    workdir: str,
    timeout: int = 1800,
    allow_errors: bool = True,
    inject_exit_guard: bool = True,
) -> Tuple[nbformat.NotebookNode, List[Dict[str, Any]]]:
    """
    Execute a notebook once under `workdir` and return (executed_notebook, errors).

    If `inject_exit_guard` is True, we prepend a temporary guard cell that neutralizes
    sys.exit / builtins.exit / os._exit to keep the kernel alive.
    The guard cell is removed from the returned executed notebook so the structure matches the original.
    """
    os.makedirs(workdir, exist_ok=True)
    os.environ["OUTPUT_DIR"] = workdir

    nb_to_run = deepcopy(nb)

    guard_added = False
    if inject_exit_guard:
        guard_code = (
            "# [AUTO-FIX] execution guard: keep kernel alive and sanitize argv for argparse\n"
            "import sys, builtins, os\n"
            "def _nb_exit_guard(*args, **kwargs):\n"
            "    print('[AUTO-FIX] Intercepted exit(); continuing execution instead of killing kernel.')\n"
            "    raise RuntimeError('NBExitIntercepted')\n"
            "try:\n"
            "    sys.exit = _nb_exit_guard\n"
            "    builtins.exit = _nb_exit_guard\n"
            "    os._exit = _nb_exit_guard\n"
            "except Exception as _e:\n"
            "    print('[AUTO-FIX] exit guard install warning:', _e)\n"
            "try:\n"
            "    # If running under ipykernel, strip its args so argparse won't see -f /tmp/xxx.json\n"
            "    if ('ipykernel' in (sys.argv[0] or '')) or any(a == '-f' or a.startswith('-f=') for a in sys.argv[1:]):\n"
            "        sys.argv = [sys.argv[0]]\n"
            "        print('[AUTO-FIX] sanitized sys.argv for argparse safety')\n"
            "except Exception as _e:\n"
            "    print('[AUTO-FIX] argv sanitize warning:', _e)\n"
        )
        nb_to_run.cells.insert(0, nbformat.v4.new_code_cell(guard_code))
        guard_added = True

    client = NotebookClient(
        nb_to_run,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=allow_errors,
        resources={"metadata": {"path": workdir}},
    )
    exec_nb = client.execute()

    # Remove guard cell; keep indices aligned with the original
    if guard_added and len(exec_nb.cells) > 0:
        exec_nb.cells.pop(0)

    errors = collect_cell_errors(exec_nb)
    return exec_nb, errors

def _short(s: str, n: int = 160) -> str:
    """Shorten a string for concise logging."""
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    return (s[:n] + "...") if len(s) > n else s

# =============================================================================
# Heuristic known-error patches (minimal, data-agnostic)
# =============================================================================

def _patch_sanitize_features() -> str:
    """Sanitize features: convert +/-inf to NaN and coerce non-numeric columns to numeric (invalid → NaN)."""
    return (
        "import numpy as np, pandas as pd\n"
        "def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    out = pd.DataFrame()\n"
        "    for c in df.columns:\n"
        "        try:\n"
        "            out[c] = pd.to_numeric(df[c], errors='coerce')\n"
        "        except Exception:\n"
        "            out[c] = pd.Series([np.nan]*len(df), index=df.index)\n"
        "    return out\n"
        "\n"
        "def _sanitize_X(X):\n"
        "    if isinstance(X, pd.DataFrame):\n"
        "        X = X.replace([np.inf, -np.inf], np.nan)\n"
        "        non_numeric_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]\n"
        "        if non_numeric_cols:\n"
        "            print('[AUTO-FIX] non-numeric columns -> to_numeric(coerce):', non_numeric_cols[:6], '...')\n"
        "            X = _to_numeric_df(X)\n"
        "        return X\n"
        "    import numpy as _np\n"
        "    X = _np.asarray(X, dtype=float)\n"
        "    X[~_np.isfinite(X)] = _np.nan\n"
        "    return X\n"
        "\n"
        "print('[AUTO-FIX] sanitizing features: inf/NaN & non-numeric → NaN before fit...')\n"
        "for _nm in ('X_tr_fit','X_va_fit','X_train','X_valid','X_val','X'):\n"
        "    try:\n"
        "        globals()[_nm] = _sanitize_X(globals()[_nm])\n"
        "    except Exception:\n"
        "        pass\n"
    )

def _patch_minimal_config() -> str:
    """Build a minimal fallback `config` to avoid NameError for missing config."""
    return (
        "import os\n"
        "from pathlib import Path\n"
        "print(\"[AUTO-FIX] building a minimal fallback 'config' ...\")\n"
        "def _guess_repo_root():\n"
        "    p = Path(os.getcwd()).resolve()\n"
        "    while p != p.parent:\n"
        "        if (p / 'prompts').exists() or (p / 'data').exists():\n"
        "            return p\n"
        "        p = p.parent\n"
        "    return Path(os.getcwd()).resolve()\n"
        "repo_root = _guess_repo_root()\n"
        "ds = os.environ.get('dataset_name') or os.environ.get('DATASET_NAME') or 'BBBC036'\n"
        "config = {\n"
        "  'dataset': {'name': ds, 'resources': {'full_file': str(repo_root / 'data' / ds / 'CP_data.csv')}},\n"
        "  'split': {'method': 'leave-smiles-out', 'n_folds': 5},\n"
        "  'metrics': {'primary': 'PCC', 'list': ['PCC','RMSE']},\n"
        "  'runtime': {'seed': int(os.environ.get('SEED','22') or 22), 'save_dir': str(repo_root / 'results' / ds / 'generate_execution' / 'prompt_runs')},\n"
        "}\n"
        "print('[AUTO-FIX] minimal config ready -> dataset:', config['dataset']['name'])\n"
    )

def _patch_recover_config_from_yaml_or_json() -> str:
    """Try to recover `config` from YAML/JSON spec files; expand placeholders using environment variables."""
    return (
        "import os, json\n"
        "from pathlib import Path\n"
        "print('[AUTO-FIX] JSON/YAML parse failed; trying to recover config from known paths ...')\n"
        "try:\n"
        "    import yaml\n"
        "except Exception:\n"
        "    yaml = None\n"
        "def _read_text(path):\n"
        "    try:\n"
        "        return Path(path).read_text(encoding='utf-8')\n"
        "    except Exception:\n"
        "        return ''\n"
        "def _load_any(text):\n"
        "    if not text: return None\n"
        "    t=text.strip()\n"
        "    try:\n"
        "        if t.startswith('{') or t.startswith('['):\n"
        "            return json.loads(t)\n"
        "    except Exception:\n"
        "        pass\n"
        "    if yaml is not None:\n"
        "        try:\n"
        "            return yaml.safe_load(t)\n"
        "        except Exception:\n"
        "            return None\n"
        "    return None\n"
        "def _expand(obj, envmap):\n"
        "    if isinstance(obj, dict):\n"
        "        return {k:_expand(v,envmap) for k,v in obj.items()}\n"
        "    if isinstance(obj, list):\n"
        "        return [_expand(v,envmap) for v in obj]\n"
        "    if isinstance(obj, str):\n"
        "        s=obj\n"
        "        for k,v in envmap.items(): s=s.replace('${'+k+'}', v)\n"
        "        return s\n"
        "    return obj\n"
        "def _guess_repo_root():\n"
        "    p = Path(os.getcwd()).resolve()\n"
        "    while p != p.parent:\n"
        "        if (p/'prompts').exists() or (p/'data').exists(): return p\n"
        "        p = p.parent\n"
        "    return Path(os.getcwd()).resolve()\n"
        "repo_root=_guess_repo_root()\n"
        "ds=os.environ.get('dataset_name') or os.environ.get('DATASET_NAME') or 'BBBC036'\n"
        "envmap={'repo_root': str(repo_root), 'dataset_name': ds}\n"
        "cands=[]\n"
        "for var in ('SPEC_PATH','spec_path','PROMPT_FILE','prompt_file','PROMPT_SPEC','prompt_spec','PROMPT_PATH','prompt_path'):\n"
        "    if var in globals() and isinstance(globals()[var], str): cands.append(globals()[var])\n"
        "cands.append(str(repo_root / 'prompts' / 'pipeline_prompt.yaml'))\n"
        "cfg=None\n"
        "for p in cands:\n"
        "    from pathlib import Path as _P\n"
        "    if p and _P(p).exists():\n"
        "        spec=_load_any(_read_text(p))\n"
        "        if spec:\n"
        "            cfg=_expand(spec, envmap)\n"
        "            print('[AUTO-FIX] loaded spec from:', p)\n"
        "            break\n"
        "if not cfg:\n"
        "    " + _patch_minimal_config().replace("\n", "\n    ") + "\n"
        "else:\n"
        "    config = cfg\n"
        "    print('[AUTO-FIX] config recovered with keys:', list(config.keys()))\n"
    )

def _patch_fix_missing_file() -> str:
    """Resolve relative data paths to absolute paths under repo root."""
    return (
        "import os\n"
        "from pathlib import Path\n"
        "print('[AUTO-FIX] FileNotFoundError: trying to resolve data path to absolute...')\n"
        "def _find_repo_root():\n"
        "    p = Path(os.getcwd()).resolve()\n"
        "    while p != p.parent:\n"
        "        if (p/'data').exists(): return p\n"
        "        p = p.parent\n"
        "    return Path(os.getcwd()).resolve()\n"
        "root=_find_repo_root()\n"
        "try:\n"
        "    for var in ('DATA_FILE','data_file','full_file'):\n"
        "        if var in globals() and isinstance(globals()[var], str):\n"
        "            p = Path(globals()[var])\n"
        "            if not p.is_absolute():\n"
        "                ap=str((root/p).resolve())\n"
        "                globals()[var] = ap\n"
        "                print(f\"[AUTO-FIX] {var} -> {ap}\")\n"
        "except Exception as _e:\n"
        "    print('[AUTO-FIX] resolve path failed:', _e)\n"
    )

def _patch_create_missing_column(colname: str) -> str:
    """Create a placeholder column in DataFrame to bypass KeyError."""
    return (
        "import numpy as np\n"
        f"print('[AUTO-FIX] KeyError: creating placeholder column: {colname}')\n"
        f"try:\n"
        f"    df['{colname}'] = np.nan\n"
        f"except Exception:\n"
        f"    pass\n"
    )

def _patch_impute_nans() -> str:
    """Impute NaNs with median for common matrices."""
    return (
        "print('[AUTO-FIX] imputing NaNs with median for common matrices ...')\n"
        "from sklearn.impute import SimpleImputer\n"
        "import numpy as np, pandas as pd\n"
        "def _impute(X):\n"
        "    if isinstance(X, pd.DataFrame):\n"
        "        im = SimpleImputer(strategy='median')\n"
        "        arr = im.fit_transform(X)\n"
        "        return pd.DataFrame(arr, columns=X.columns, index=X.index)\n"
        "    return SimpleImputer(strategy='median').fit_transform(X)\n"
        "for _nm in ('X_tr_fit','X_va_fit','X_train','X_valid','X_val','X'):\n"
        "    try:\n"
        "        _tmp = globals()[_nm]\n"
        "        import numpy as _np\n"
        "        if hasattr(_tmp, 'isna') and _tmp.isna().any().any():\n"
        "            globals()[_nm] = _impute(_tmp)\n"
        "        elif isinstance(_tmp, _np.ndarray) and _np.isnan(_tmp).any():\n"
        "            globals()[_nm] = _impute(_tmp)\n"
        "    except Exception:\n"
        "        pass\n"
    )

def _known_fix_snippet(evalue: str) -> Optional[str]:
    """Return a generic fix snippet for known error patterns."""
    msg = (evalue or "").lower()

    # Numeric issues
    if ("input x contains infinity" in msg) or ("too large for dtype" in msg):
        return _patch_sanitize_features()
    if ("input x contains nan" in msg) or ("contains nan" in msg and "input x" in msg):
        return _patch_impute_nans() + "\n" + _patch_sanitize_features()

    # Config / parsing issues
    if "jsondecodeerror" in msg or "expecting value" in msg:
        return _patch_recover_config_from_yaml_or_json()
    if "name 'config' is not defined" in msg or ("name" in msg and "config" in msg and "not defined" in msg):
        return _patch_minimal_config()

    # File/path issues
    if "filenotfounderror" in msg or "no such file or directory" in msg:
        return _patch_fix_missing_file()

    # Missing column in DataFrame
    m = re.search(r"keyerror:\s*['\"]([^'\"]+)['\"]", msg)
    if m:
        missing_col = m.group(1)
        return _patch_create_missing_column(missing_col)

    # String-to-float conversion failures
    if ("could not convert string to float" in msg) or ("could not convert" in msg and "float" in msg):
        return _patch_sanitize_features()

    return None

def apply_known_patches(nb: nbformat.NotebookNode, errors: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
    """Inject known fix snippets at the tail of failing code cells. Return (num_changed, changed_indices)."""
    changed = 0
    changed_indices: List[int] = []
    for e in errors:
        idx = int(e["cell_index"])
        patch = _known_fix_snippet(e.get("evalue",""))
        if not patch:
            continue
        if idx < 0 or idx >= len(nb.cells):
            continue
        cell = nb.cells[idx]
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        if patch not in src:
            cell["source"] = src.rstrip() + "\n\n# --- auto-fix patch injected ---\n" + patch
            changed += 1
            changed_indices.append(idx)
    return changed, changed_indices

def _restore_sources(exec_nb: nbformat.NotebookNode, original_sources: List[str], indices: Optional[List[int]] = None):
    """
    Restore cell `source` from original notebook for selected indices.
    If indices is None, restore all code cells that have a matching position.
    """
    if indices is None:
        indices = list(range(min(len(original_sources), len(exec_nb.cells))))
    for i in indices:
        if 0 <= i < len(exec_nb.cells) and i < len(original_sources):
            if exec_nb.cells[i].get("cell_type") == "code":
                exec_nb.cells[i]["source"] = original_sources[i]

# =============================================================================
# LLM-based minimal repair
# =============================================================================

def _make_bugfix_messages(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    language: str = "English",
    max_src_chars: int = 4000
) -> List[Dict[str, str]]:
    """Build strict guardrail messages to request minimal edits on failing cells."""
    idx_set = {int(e["cell_index"]) for e in errors}
    failing_cells_dump = []
    for i in sorted(idx_set):
        if 0 <= i < len(nb.cells) and nb.cells[i].get("cell_type") == "code":
            src = nb.cells[i].get("source") or ""
            if len(src) > max_src_chars:
                src = src[:max_src_chars] + "\n# ... [TRUNCATED] ..."
            failing_cells_dump.append(f"### Cell {i}\n```python\n{src}\n```")

    err_dump = []
    for e in errors:
        err_dump.append(
            f"Cell {e['cell_index']} :: {e.get('ename','')} :: {e.get('evalue','')}\n{e.get('traceback','')}\n"
        )

    system_msg = {
        "role": "system",
        "content": (
            "You are a senior ML engineer. FIX RUNTIME BUGS with MINIMAL edits.\n"
            "HARD RULES:\n"
            "1) Do NOT modify CV protocol, seeds, folds, grouping, metrics, hyperparams, or file paths.\n"
            "2) Do NOT change labels/targets; do NOT hide exceptions.\n"
            "3) Only edit the failing cells. Keep algorithmic behavior intact.\n"
            "Return STRICT JSON only: {\"edits\":[{\"cell_index\": int, \"source\": \"<full new cell source>\"}]}\n"
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Language: {language}\n"
            "Errors:\n" + "\n".join(err_dump) + "\n\n"
            "Failing cell sources:\n" + "\n\n".join(failing_cells_dump)
        )
    }
    return [system_msg, user_msg]

def _llm_auto_fix_once(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    *,
    phase_cfg: dict | None,
    timeout: int,
    verbose: bool
) -> Tuple[nbformat.NotebookNode, List[Dict[str, Any]], List[int]]:
    """
    Call the LLM once to propose minimal edits. Only apply edits to failing cells.
    Returns (new_notebook, new_errors, patched_indices).
    """
    # Resolve profile from cfg only (no llm_providers.json)
    prof = _load_llm_profile(phase_cfg, verbose=verbose)

    api_key   = prof["api_key"]
    base_url  = prof["base_url"]
    model     = prof["model"]
    temp      = prof["temperature"]
    max_tok   = prof["max_tokens"]

    if verbose:
        print(f"[LLM] provider={prof['provider']} model={model} base_url={base_url}")

    if not api_key:
        print("[AUTO-FIX][LLM][WARN] API key is missing (neither env nor cfg.providers). LLM call will likely fail.")

    messages = _make_bugfix_messages(nb, errors, language="English")

    # Call LLM. chat_json() itself prints model/base_url too.
    spec = _llm_chat_json(
        messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temp,
        max_tokens=max_tok,
    )

    edits = spec.get("edits") or []
    if not edits:
        if verbose:
            print("[AUTO-FIX][LLM] no edits suggested by model.")
        return nb, errors, []

    # Keep only edits for failing cells and valid code cells
    target = {int(e["cell_index"]) for e in errors}
    valid_edits: List[Dict[str, Any]] = []
    patched_idx: List[int] = []
    for ed in edits:
        try:
            idx = int(ed.get("cell_index"))
            if idx in target and 0 <= idx < len(nb["cells"]) and nb["cells"][idx].get("cell_type") == "code":
                new_src = ed.get("source") or ""
                if isinstance(new_src, str) and new_src.strip():
                    valid_edits.append({"cell_index": idx, "source": new_src})
                    patched_idx.append(idx)
        except Exception:
            continue

    if not valid_edits:
        if verbose:
            print("[AUTO-FIX][LLM] no valid edits after filtering.")
        return nb, errors, []

    changed = _llm_apply_edits(nb, valid_edits)
    if verbose:
        print(f"[AUTO-FIX][LLM] patched cells={sorted(patched_idx)} | changed={changed}")
    if changed == 0:
        return nb, errors, []

    # Re-exec once (allowing errors); caller loop decides next round
    nb, _ = _llm_execute_notebook(nb, timeout=timeout, allow_errors=True)
    new_errs = collect_cell_errors(nb)
    if verbose:
        if new_errs:
            print(f"[AUTO-FIX][LLM] post-exec errors={len(new_errs)}")
        else:
            print("[AUTO-FIX][LLM] post-exec success (no errors).")
    return nb, new_errs, patched_idx

# =============================================================================
# Main entry: execute + (known/LLM) auto-fix
# Compatible with both call styles:
#   - New: execute_with_autofix(nb_path=..., workdir=..., timeout_seconds=..., ...)
#   - Legacy: execute_with_autofix(ipynb_path=..., out_exec_path=..., workdir=..., timeout=..., ...)
# =============================================================================

def execute_with_autofix(
    nb_path: Optional[str] = None,
    *,
    workdir: str,
    timeout_seconds: Optional[int] = None,
    max_fix_rounds: int = 3,
    max_cell_retries: int = 2,  # accepted for compatibility; not used directly
    preserve_source_in_exec: bool = True,
    phase_cfg: dict | None = None,
    verbose: bool = True,
    save_intermediates: bool = False,
    **kwargs: Any,
) -> str:
    """
    Execute the notebook and try to repair errors using known patches and a few LLM rounds.
    Returns the final executed notebook path (suffix: "_exec.ipynb").

    This function supports both the new and legacy call signatures:
      - New style:
          execute_with_autofix(nb_path=..., workdir=..., timeout_seconds=..., ...)
      - Legacy style:
          execute_with_autofix(ipynb_path=..., out_exec_path=..., workdir=..., timeout=..., ...)
    """
    # ---- Compatibility shim for legacy kwargs ----
    if nb_path is None and "ipynb_path" in kwargs:
        nb_path = kwargs.pop("ipynb_path")

    # Map legacy 'timeout' to 'timeout_seconds' if provided
    if timeout_seconds is None:
        if "timeout" in kwargs:
            timeout_seconds = int(kwargs.pop("timeout"))
        else:
            timeout_seconds = 1800  # default

    # Legacy optional explicit output path
    out_exec_path: Optional[str] = kwargs.pop("out_exec_path", None)

    # Sanity checks
    if nb_path is None:
        raise TypeError("execute_with_autofix() requires 'nb_path' (or legacy 'ipynb_path').")

    if verbose:
        print(f"[AUTO-FIX] start execute: {nb_path}")
        print(f"[AUTO-FIX] workdir={workdir}, timeout={timeout_seconds}s, max_fix_rounds={max_fix_rounds}")
        if preserve_source_in_exec:
            print("[AUTO-FIX] preserve_source_in_exec=True (exec keeps original code)")

    # Load original notebook; keep sources for later restoration
    nb_orig = nbformat.read(nb_path, as_version=4)
    original_sources = [c.get("source", "") for c in nb_orig.cells]

    # Work on a copy for patching/execution
    nb_run = deepcopy(nb_orig)

    # Initial run
    exec_nb, errors = execute_once(nb_run, workdir, timeout=timeout_seconds, allow_errors=True)

    if verbose:
        if errors:
            print(f"[AUTO-FIX] initial run: {len(errors)} error(s)")
            for e in errors:
                print(f"  - cell {e['cell_index']}: {e.get('ename','')} | {_short(e.get('evalue',''))}")
        else:
            print("[AUTO-FIX] initial run: success (no errors)")

    # Compute final output path
    if out_exec_path is None:
        out_exec_path = nb_path.replace(".ipynb", "_exec.ipynb")

    # Early exit if no errors on first run
    if not errors:
        final_nb = exec_nb
        if preserve_source_in_exec and len(exec_nb.cells) == len(original_sources):
            final_nb = deepcopy(exec_nb)
            # Nothing to restore, kept for symmetry
        nbformat.write(final_nb, out_exec_path)
        if verbose:
            print(f"[AUTO-FIX] early exit: no errors on initial run. -> {out_exec_path}")
        return out_exec_path

    # Otherwise, try fixes up to max_fix_rounds
    all_changed_indices: set[int] = set()
    round_idx = 0

    while errors and round_idx < max_fix_rounds:
        round_idx += 1
        if verbose:
            err_cells = [e["cell_index"] for e in errors]
            print(f"[AUTO-FIX] round {round_idx}/{max_fix_rounds} | error cells={err_cells}")

        # A) Known patches on a fresh copy of ORIGINAL
        nb_run = deepcopy(nb_orig)
        changed_known, changed_idxs = apply_known_patches(nb_run, errors)
        all_changed_indices.update(changed_idxs)

        if changed_known > 0 and save_intermediates:
            patched_src = os.path.join(
                os.path.dirname(out_exec_path),
                f"{os.path.splitext(os.path.basename(out_exec_path))[0]}_patched_r{round_idx}.ipynb"
            )
            nbformat.write(nb_run, patched_src)
            if verbose:
                print(f"[AUTO-FIX] patched source -> {patched_src} (cells changed={changed_known})")

        # Re-execute after known patches
        exec_nb, errors = execute_once(nb_run, workdir, timeout=timeout_seconds, allow_errors=True)

        # Early exit if fixed
        if not errors:
            final_nb = exec_nb
            if preserve_source_in_exec and len(exec_nb.cells) == len(original_sources):
                final_nb = deepcopy(exec_nb)
                _restore_sources(final_nb, original_sources, list(all_changed_indices) or None)
            nbformat.write(final_nb, out_exec_path)
            if verbose:
                print(f"[AUTO-FIX] success after known patches in round {round_idx}. -> {out_exec_path}")
            return out_exec_path

        # B) LLM-based minimal patch
        if verbose:
            print("[AUTO-FIX] escalating to LLM-based fix...")
        nb_after, errors_after, patched_cells = _llm_auto_fix_once(
            deepcopy(nb_orig), errors, phase_cfg=phase_cfg, timeout=timeout_seconds, verbose=verbose
        )
        exec_nb = nb_after
        errors = errors_after
        for idx in patched_cells or []:
            all_changed_indices.add(int(idx))

        # Early exit if fixed
        if not errors:
            final_nb = exec_nb
            if preserve_source_in_exec and len(exec_nb.cells) == len(original_sources):
                final_nb = deepcopy(exec_nb)
                _restore_sources(final_nb, original_sources, list(all_changed_indices) or None)
            nbformat.write(final_nb, out_exec_path)
            if verbose:
                print(f"[AUTO-FIX] success after LLM patches in round {round_idx}. -> {out_exec_path}")
            return out_exec_path

    # If here, still have errors or exhausted rounds. Save whatever we have.
    final_nb = exec_nb
    if preserve_source_in_exec and len(exec_nb.cells) == len(original_sources):
        final_nb = deepcopy(exec_nb)
        _restore_sources(final_nb, original_sources, list(all_changed_indices) or None)

    nbformat.write(final_nb, out_exec_path)
    if verbose:
        print(f"[AUTO-FIX] done with remaining errors={len(errors)}; final exec -> {out_exec_path}")
    return out_exec_path
