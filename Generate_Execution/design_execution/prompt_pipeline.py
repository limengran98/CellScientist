# -*- coding: utf-8 -*-
"""
Prompt-defined pipeline orchestrator (Notebook mode).

Guarantees:
- generate_notebook_from_prompt(...) ALWAYS returns (nb, user_prompt) or raises.
- Provides run_prompt_pipeline(cfg, spec_path) for callers that expect a single entrypoint.
- Supports {"cells": [...], "hypergraph": {...}} → multi-cell notebook with cellwise execution.
- Backward compatible with {"notebook": {...}} and {"code": "..."}. 

Only execution logic is changed/enhanced.
"""
from __future__ import annotations

import os, json, re, glob, time, shutil, datetime as _dt
from typing import Any, Dict, List, Tuple, Optional
from .llm_client import resolve_llm_from_cfg, LLMClient
from pathlib import Path

import nbformat

# Optional YAML (for nicer prompts); if missing, we still work.
try:
    import yaml
except Exception:
    yaml = None

# Import local helpers
try:
    from .nb_autofix import execute_with_autofix
except Exception:
    from nb_autofix import execute_with_autofix  # type: ignore

try:
    from .llm_client import LLMClient
except Exception:
    LLMClient = None  # type: ignore


# ========== FS utils ==========

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _read_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the prompt spec.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def _expand_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj

def _summarize_phase1(reference_dir: str, max_chars: int = 1600) -> str:
    """Light summary from the first small .ipynb under reference_dir (best-effort)."""
    if not reference_dir or not os.path.isdir(reference_dir):
        return ""
    candidates = sorted(glob.glob(os.path.join(reference_dir, "*.ipynb")))
    if not candidates:
        return ""
    try:
        nb = nbformat.read(candidates[0], as_version=4)
        lines = []
        for cell in nb.cells[:20]:
            if cell.cell_type == "markdown":
                src = (cell.source or "").strip()
                if src:
                    lines.append(src[:200])
            elif cell.cell_type == "code":
                src = (cell.source or "").strip()
                if src:
                    keep = [ln for ln in src.splitlines() if ln.strip().startswith(("import", "from ", "def "))]
                    if keep:
                        lines.append("\n".join(keep)[:200])
        return "\n".join(lines)[:max_chars]
    except Exception:
        return ""


def _build_user_prompt(spec_obj: Dict[str, Any], phase1_hint: str) -> str:
    """Convert dict spec to user YAML while injecting a phase-1 summary hint."""
    d = dict(spec_obj or {})
    for k in ("system", "developer", "user"):
        d.pop(k, None)
    if phase1_hint:
        d.setdefault("reference", {})["phase1_summary"] = phase1_hint
    if yaml is None:
        return json.dumps(d, ensure_ascii=False, indent=2)
    return yaml.safe_dump(d, allow_unicode=True, sort_keys=False)


# ========== LLM helper ==========

def _chat_text_stable(messages: List[Dict[str, str]], *, api_key: str, base_url: str,
                      model: str, temperature: float, max_tokens: int, retries: int,
                      timeout: int, debug_dir: Optional[str]) -> str:
    """OpenAI-compatible chat call via LLMClient (if available) else HTTP fallback."""
    os.makedirs(debug_dir or ".", exist_ok=True)
    if LLMClient is not None:
        client = LLMClient(model=model, base_url=base_url, api_key=api_key, timeout=timeout)
        return client.chat(messages, temperature=temperature, max_tokens=max_tokens,
                           enforce_json=False, retries=max(retries, 1), debug_dir=debug_dir)
    import requests
    url = base_url.rstrip("/") + "/chat/completions"
    last_err = None
    for i in range(max(1, retries)):
        try:
            resp = requests.post(url, json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }, timeout=timeout)
            data = resp.json()
            # debug dump
            try:
                with open(os.path.join(debug_dir or ".", f"raw_response_try{i+1}.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
            if content.strip():
                return content.strip()
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    raise RuntimeError(f"LLM call failed after {retries} attempt(s): {last_err}")


# ========== Notebook build helpers ==========

def _wrap_code_to_notebook(code_text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(code_text)]
    return nb

def _topo_from_hypergraph(cells_spec: List[Dict[str, Any]], hypergraph: Dict[str, Any] | None) -> List[str]:
    """Topological order from hyperedges; fallback to cells order on failure."""
    try:
        ids = [c.get("id") for c in (cells_spec or []) if c.get("id")]
        edges: List[Tuple[str, str]] = []
        for e in (hypergraph or {}).get("hyperedges", []) or []:
            head = e.get("head")
            tails = e.get("tail") or []
            for t in tails:
                if head in ids and t in ids:
                    edges.append((t, head))
        from collections import defaultdict, deque
        indeg = defaultdict(int); graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v); indeg[v] += 1
            indeg.setdefault(u, 0)
        for i in ids:
            indeg.setdefault(i, 0)
        q = deque([i for i in ids if indeg[i] == 0])
        order: List[str] = []
        while q:
            u = q.popleft(); order.append(u)
            for v in graph[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return order if len(order) == len(ids) else ids
    except Exception:
        return [c.get("id") for c in (cells_spec or []) if c.get("id")]

def _wrap_cells_to_notebook(cells_spec: List[Dict[str, Any]], hypergraph: Dict[str, Any] | None = None) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata.setdefault("execution", {})["mode"] = "cellwise"
    if hypergraph:
        nb.metadata["execution"]["hypergraph"] = hypergraph
    # intro markdown
    md = ["# Generated Notebook (cell-wise)", "", "Subtasks:"]
    for i, c in enumerate(cells_spec or []):
        md.append(f"- **{c.get('id', 'T'+str(i))} — {c.get('name','')}:** {c.get('purpose','')}")
    nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(md)))
    # cell order
    order = _topo_from_hypergraph(cells_spec, hypergraph)
    id2cell = {(c.get("id") or f"T{i}"): c for i, c in enumerate(cells_spec or [])}
    for cid in order:
        c = id2cell.get(cid) or {}
        code = c.get("code") or ""
        cell = nbformat.v4.new_code_cell(code)
        cell.metadata["subtask"] = {
            "id": cid,
            "name": c.get("name"),
            "purpose": c.get("purpose"),
            "deps": [t for t in (hypergraph or {}).get("hyperedges", []) if t.get("head") == cid]
        }
        nb.cells.append(cell)
    return nb


# ========== Prompt → Notebook ==========

def generate_notebook_from_prompt(cfg: Dict[str, Any], spec_path: str, debug_dir: str) -> Tuple[nbformat.NotebookNode, str]:
    """
    Returns: (nb, user_prompt_str). Never returns None; raises on hard failure.
    """
    os.makedirs(debug_dir, exist_ok=True)

    # dataset → env
    ds_name = (cfg.get("dataset_name")
            or os.environ.get("dataset_name")
            or os.environ.get("DATASET_NAME")
            or "").strip()
    if ds_name:
        os.environ["dataset_name"] = ds_name
        os.environ["DATASET_NAME"] = ds_name

    # repo_root for ${...} expansions
    repo_root = Path(__file__).resolve()
    while not (repo_root / "data").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent
    os.environ.setdefault("repo_root", str(repo_root))

    # Read spec and expand vars
    spec = _expand_vars(_read_yaml(spec_path))

    # Build messages
    sys_txt = (spec.get("system") or "").strip() if isinstance(spec, dict) else ""
    dev_txt = (spec.get("developer") or "").strip() if isinstance(spec, dict) else ""
    if isinstance(spec, dict):
        core = {k: v for k, v in spec.items() if k not in ("system", "developer", "user")}
    else:
        core = {}

    # Stage 1 Analysis (if available)
    phase1_dir = cfg.get("paths", {}).get("stage1_analysis_dir", "")
    phase1_hint = _summarize_phase1(phase1_dir)

    # Integrate the phase1_hint into the user prompt
    if yaml is not None:
        usr_txt = yaml.safe_dump(core, allow_unicode=True, sort_keys=False).strip()
        if not usr_txt:
            usr_txt = "{}"
    else:
        usr_txt = _build_user_prompt(core, phase1_hint)

    messages = [{"role": "system", "content": (sys_txt or "You are a principal ML scientist. Return STRICT JSON only.")}]
    if dev_txt:
        messages.append({"role": "system", "content": "Developer instructions:\n" + dev_txt})
    messages.append({"role": "user", "content": usr_txt})

    # LLM config
    resolved = resolve_llm_from_cfg(cfg)
    base_url = resolved['base_url']
    api_key = resolved['api_key']
    model = resolved['model']
    timeout = int(resolved.get('timeout', 1200))
    pb = cfg.get("prompt_branch", {}) or {}

    def _strip_code_fences(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
        return s

    # Round 1
    raw_text_1 = _chat_text_stable(
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=float(pb.get("temperature", 0.2)),
        max_tokens=int(pb.get("max_tokens", 20000)),
        retries=int(pb.get("retries", 2)),
        timeout=timeout,
        debug_dir=debug_dir
    )
    open(os.path.join(debug_dir, "llm_raw_text_r1.txt"), "w", encoding="utf-8").write(raw_text_1 or "")
    cleaned = _strip_code_fences(raw_text_1)

    # Parse
    nb_json = None
    try:
        if cleaned:
            nb_json = json.loads(cleaned)
        else:
            raise ValueError("empty response")
    except Exception:
        if cleaned and not cleaned.lstrip().startswith("{"):
            return _wrap_code_to_notebook(cleaned), usr_txt
        # Round 2 (strict JSON)
        hard_rule = (
            "Return STRICT JSON only with one of these keys:\n"
            "  1) {\"notebook\": <nbformat v4 JSON>}\n"
            "  2) {\"code\": \"<python script>\"}\n"
            "  3) {\"cells\": [...], \"hypergraph\": {...}}\n"
            "No markdown, no backticks, no extra commentary."
        )
        messages2 = list(messages) + [{"role": "system", "content": hard_rule}]
        print(f"[LLM] model={model} base_url={base_url}")
        raw_text_2 = _chat_text_stable(
            messages=messages2,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=float(pb.get("temperature", 0.2)),
            max_tokens=int(pb.get("max_tokens", 20000)),
            retries=max(1, int(pb.get("retries", 2)) - 1),
            timeout=timeout,
            debug_dir=debug_dir
        )
        open(os.path.join(debug_dir, "llm_raw_text_r2.txt"), "w", encoding="utf-8").write(raw_text_2 or "")
        cleaned2 = _strip_code_fences(raw_text_2)
        try:
            nb_json = json.loads(cleaned2)
        except Exception as e2:
            if cleaned2 and not cleaned2.lstrip().startswith("{"):
                return _wrap_code_to_notebook(cleaned2), usr_txt
            with open(os.path.join(debug_dir, "raw_unparsed.txt"), "w", encoding="utf-8") as fdbg:
                fdbg.write(raw_text_1 or ""); fdbg.write("\n\n===== SECOND ROUND =====\n\n"); fdbg.write(raw_text_2 or "")
            raise RuntimeError(f"Notebook JSON parse failed: {e2}")

    # JSON decoded
    if isinstance(nb_json, dict):
        if "notebook" in nb_json and isinstance(nb_json["notebook"], dict):
            try:
                return nbformat.from_dict(nb_json["notebook"]), usr_txt
            except Exception:
                nb = nbformat.v4.new_notebook()
                nb.cells = [nbformat.v4.new_code_cell("# Failed to parse provided notebook")]
                return nb, usr_txt
        if "cells" in nb_json and isinstance(nb_json["cells"], list):
            return _wrap_cells_to_notebook(nb_json["cells"], nb_json.get("hypergraph")), usr_txt
        if "code" in nb_json and isinstance(nb_json["code"], str):
            return _wrap_code_to_notebook(nb_json["code"]), usr_txt

    # Fallback: try treat as full nbformat dict
    try:
        nb = nbformat.from_dict(nb_json)
        if getattr(nb, "nbformat", 4) != 4:
            raise RuntimeError(f"nbformat must be 4, got {getattr(nb, 'nbformat', None)}")
        return nb, usr_txt
    except Exception as e:
        # Final guard: never return None
        raise RuntimeError(f"Invalid notebook format: {e}")

# ========== Output & Viz ==========

def _prompt_out_root(cfg: Dict[str, Any]) -> str:
    return cfg.get("prompt_branch", {}).get("save_root", cfg["paths"]["design_execution_root"])

def _latest_prompt_dir(cfg: Dict[str, Any]) -> Optional[str]:
    root = os.path.join(_prompt_out_root(cfg), "prompt")
    subs = sorted([p for p in glob.glob(os.path.join(root, "prompt_run_*")) if os.path.isdir(p)])
    return subs[-1] if subs else None

def _write_hypergraph_viz(trial_dir: str, nb_path: str, fmt: str = "mermaid") -> Dict[str, str]:
    """Emit hypergraph.json and hypergraph.md (mermaid) if notebook has metadata.execution.hypergraph."""
    out: Dict[str, str] = {}
    try:
        nb = nbformat.read(nb_path, as_version=4)
        ex = ((nb.metadata or {}).get("execution") or {})
        hg = ex.get("hypergraph") or {}
        cells = []
        for c in nb.cells:
            if c.get("cell_type") == "code" and isinstance(c.get("metadata"), dict):
                st = (c.get("metadata").get("subtask") or {})
                if st.get("id"):
                    cells.append({"id": st.get("id"), "name": st.get("name") or "", "purpose": st.get("purpose") or ""})
        # dedup
        seen = set(); uniq = []
        for c in cells:
            if c["id"] not in seen:
                uniq.append(c); seen.add(c["id"])
        os.makedirs(trial_dir, exist_ok=True)
        # JSON
        jpath = os.path.join(trial_dir, "hypergraph.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump({"cells": uniq, "hypergraph": hg}, f, ensure_ascii=False, indent=2)
        out["json"] = jpath
        # Mermaid
        if fmt == "mermaid":
            lines = ["```mermaid", "flowchart TD"]
            for c in uniq:
                purpose = (c.get("purpose") or "").replace("\n", " ")
                if len(purpose) > 60: purpose = purpose[:57] + "..."
                label = (c.get("name") or c["id"]).replace('"', "'")
                lines.append(f'    {c["id"]}["{c["id"]}: {label}\\n{purpose}"]')
            for e in (hg.get("hyperedges") or []):
                head = e.get("head")
                for t in (e.get("tail") or []):
                    if head and t:
                        lines.append(f"    {t} --> {head}")
            lines.append("```")
            mpath = os.path.join(trial_dir, "hypergraph.md")
            with open(mpath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            out["mermaid"] = mpath
    except Exception as e:
        try:
            with open(os.path.join(trial_dir, "hypergraph_viz_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass
    return out


# ========== Public API (single steps) ==========

def prompt_generate(cfg: Dict[str, Any], spec_path: str) -> Dict[str, Any]:
    """Only generate the prompt-defined artifact (notebook), do not execute."""
    debug_dir = os.path.join(cfg["paths"]["design_execution_root"], "debug_prompt")
    out_root = _prompt_out_root(cfg)
    os.makedirs(os.path.join(out_root, "prompt"), exist_ok=True)

    nb, _user_prompt = generate_notebook_from_prompt(cfg, spec_path, debug_dir)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tdir = os.path.join(out_root, "prompt", f"prompt_run_{ts}")
    os.makedirs(tdir, exist_ok=True)

    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return {"trial_dir": tdir, "artifact": nb_path}

def prompt_execute(cfg: Dict[str, Any], trial_dir: Optional[str] = None) -> Dict[str, Any]:
    """Execute the latest prompt artifact with auto-fix (known patches + optional LLM)."""
    tdir = trial_dir or _latest_prompt_dir(cfg)
    if not tdir:
        raise RuntimeError("No prompt trial found. Run 'generate' first.")

    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    if not os.path.exists(nb_path):
        raise RuntimeError(f"notebook_prompt.ipynb not found in {tdir}")

    exec_cfg = (cfg.get("exec") or {})
    timeout = int(exec_cfg.get("timeout_seconds", 1800))
    max_fix_rounds = int(exec_cfg.get("max_fix_rounds", 1))
    max_cell_retries = int(exec_cfg.get("max_cell_retries", 2))

    print(f"[PROMPT] exec config -> timeout_seconds={timeout}, max_fix_rounds={max_fix_rounds}, max_cell_retries={max_cell_retries}")
    print(f"[PROMPT] trial_dir={tdir}")
    print(f"[PROMPT] notebook={nb_path}")


    out_exec = os.path.join(tdir, "notebook_prompt_exec.ipynb")

    final_exec_path = execute_with_autofix(
        ipynb_path=nb_path,
        out_exec_path=out_exec,
        workdir=tdir,
        timeout=timeout,
        max_fix_rounds=max_fix_rounds,
        verbose=True,
        phase_cfg=cfg,
        preserve_source_in_exec=True,
        save_intermediates=True,
    )
    print(f"[PROMPT] executed notebook -> {final_exec_path}")

    metrics_path = os.path.join(tdir, "metrics.json")
    metrics: Dict[str, Any] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[PROMPT][WARN] failed to read metrics.json: {e}")
    else:
        print(f"[PROMPT][WARN] metrics.json not found in {tdir}")

    # viz
    try:
        if bool(exec_cfg.get("enable_hypergraph_viz", True)):
            viz = _write_hypergraph_viz(tdir, nb_path, fmt=str(exec_cfg.get("viz_format","mermaid")))
            if viz:
                print(f"[PROMPT] hypergraph viz written: {viz}")
    except Exception as e:
        print(f"[PROMPT][WARN] hypergraph viz generation failed: {e}")

    return {"trial_dir": tdir, "metrics": metrics, "exec_notebook": final_exec_path}

def prompt_analyze(cfg: Dict[str, Any], trial_dir: str) -> Dict[str, Any]:
    """
    Read-only analyze phase:
      - Do NOT execute or auto-fix here.
      - If metrics.json is missing / invalid / empty, print the reason and return a report without metrics.
    """
    import os, json

    metrics_path = os.path.join(trial_dir, "metrics.json")

    report: Dict[str, Any] = {
        "trial_dir": trial_dir,
        "has_metrics": False,
        "metrics_keys": [],
        "reason": None,
        "metrics": {}
    }

    if not os.path.exists(metrics_path):
        reason = f"metrics.json not found in {trial_dir}; analyze is read-only and will NOT re-execute."
        print(f"[WARN] {reason}")
        report["reason"] = "missing_metrics_json"
        try:
            with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return report

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_obj = json.load(f)
    except Exception as e:
        reason = f"failed to read/parse metrics.json: {e}"
        print(f"[WARN] {reason}")
        report["reason"] = "invalid_metrics_json"
        try:
            with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return report

    if not isinstance(metrics_obj, dict) or not metrics_obj:
        reason = "metrics.json loaded but empty or not a dict; analyze will NOT re-execute."
        print(f"[WARN] {reason}")
        report["reason"] = "empty_or_invalid_metrics"
        try:
            with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return report
    
    report["has_metrics"] = True
    report["metrics_keys"] = list(metrics_obj.keys())
    report["metrics"] = metrics_obj
    report["reason"] = "ok"
    print(f"[INFO] Loaded metrics from {metrics_path}: keys={report['metrics_keys']}")

    primary = ((cfg.get("experiment") or {}).get("primary_metric")
               or (cfg.get("improve") or {}).get("primary_metric"))
    if primary and primary not in metrics_obj:
        print(f"[WARN] primary metric '{primary}' not found in metrics.json keys.")
    try:
        with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return report


# ========== Public API (pipeline entrypoint) ==========

from typing import Dict, Any

def run_prompt_pipeline(cfg: Dict[str, Any], spec_path: str) -> Dict[str, Any]:
    """
    Orchestrate the prompt pipeline with 3 phases (read-only analyze):
      1) prompt_generate -> materialize notebook
      2) prompt_execute  -> execute notebook (with auto-fix if enabled)
      3) prompt_analyze  -> read-only analysis (no re-exec)
    Returns dict with keys: trial_dir, exec_notebook, metrics, report
    """
    print("[INFO] === Generating prompt notebook ===")
    g = prompt_generate(cfg, spec_path)

    print("[INFO] === Executing prompt notebook ===")
    e = prompt_execute(cfg, g["trial_dir"])

    print("[INFO] === Analyzing results ===")
    trial_for_analyze = e.get("trial_dir", g["trial_dir"])
    a = prompt_analyze(cfg, trial_for_analyze)

    merged_metrics = a.get("metrics") or e.get("metrics") or {}

    return {
        "trial_dir": trial_for_analyze,
        "exec_notebook": e.get("exec_notebook"),
        "metrics": merged_metrics,
        "report": a,
    }