# -*- coding: utf-8 -*-
"""
Utilities for building the prompt-defined notebook:
- YAML reading & env expansion
- Stage-1 reference summarization (markdown)
- LLM chat wrapper
- Converting LLM JSON into a valid nbformat notebook
"""

from __future__ import annotations
import os, json, re, glob, time
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import nbformat

try:
    import yaml
except Exception:
    yaml = None

try:
    from .llm_client import resolve_llm_from_cfg, LLMClient
except Exception:
    resolve_llm_from_cfg = None  # type: ignore
    LLMClient = None  # type: ignore


# ---------- YAML & Env ----------

def read_yaml(path: str) -> Dict[str, Any]:
    """Read YAML; requires PyYAML."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the prompt spec.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} using current environment variables."""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


# ---------- Stage-1 reference (markdown summary) ----------

def summarize_stage1_ipynb(reference_dir: str, max_cells: int = 20, max_chars: int = 2000) -> str:
    """
    Make a lightweight text summary from the first .ipynb under reference_dir.
    Returns a markdown string (no code execution).
    """
    if not reference_dir or not os.path.isdir(reference_dir):
        return ""
    cands = sorted(glob.glob(os.path.join(reference_dir, "*.ipynb")))
    if not cands:
        return ""
    try:
        nb = nbformat.read(cands[0], as_version=4)
        lines: List[str] = []
        lines.append("**Light summary extracted from Stage-1 notebook**  \n")
        for cell in nb.cells[:max_cells]:
            if cell.cell_type == "markdown":
                src = (cell.source or "").strip()
                if src:
                    lines.append(src[:300])
            elif cell.cell_type == "code":
                src = (cell.source or "").strip()
                if not src:
                    continue
                # keep only imports/defs to avoid dumping big code
                keep = [ln for ln in src.splitlines()
                        if ln.strip().startswith(("import", "from ", "def ", "class "))]
                if keep:
                    lines.append("```python\n" + "\n".join(keep[:20]) + "\n```")
        text = "\n\n".join(lines)
        return text[:max_chars]
    except Exception:
        return ""


def build_stage1_markdown(cfg: Dict[str, Any]) -> str:
    """
    Build the markdown block for Stage-1 reference if enabled.

    Toggle precedence:
      1) cfg['prompt_branch']['use_stage1_ref'] (bool)
      2) default: True
    """
    pb = cfg.get("prompt_branch", {}) or {}
    use_ref = bool(pb.get("use_stage1_ref", True))
    if not use_ref:
        return ""
    ref_dir = (cfg.get("paths", {}) or {}).get("stage1_analysis_dir", "")
    md = summarize_stage1_ipynb(ref_dir)
    if not md:
        return ""
    header = "## Experiment & Validation Suggestions (from Stage-1)\n\n"
    note = "> Note: The following summary is only a **reference** extracted from Stage-1. The pipeline remains driven by `pipeline_prompt.yaml`.\n\n"
    return header + note + md + "\n"


# ---------- LLM wrappers ----------

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
    return s

def chat_text(messages: List[Dict[str, str]],
              *,
              cfg: Dict[str, Any],
              timeout: int,
              debug_dir: Optional[str] = None,
              temperature: float = 0.2,
              max_tokens: int = 20000,
              retries: int = 2) -> str:
    """
    OpenAI-compatible chat using LLMClient if available; otherwise raw HTTP POST.
    Prints provider/model for easy debugging.
    """
    os.makedirs(debug_dir or ".", exist_ok=True)
    if resolve_llm_from_cfg is None:
        raise RuntimeError("resolve_llm_from_cfg is unavailable.")

    resolved = resolve_llm_from_cfg(cfg)
    base_url = resolved["base_url"]
    api_key = resolved["api_key"]
    model = resolved["model"]

    print(f"[LLM] provider={resolved.get('provider','?')} model={model} base_url={base_url}")

    if LLMClient is not None:
        client = LLMClient(model=model, base_url=base_url, api_key=api_key, timeout=timeout)
        return client.chat(messages,
                           temperature=float(temperature),
                           max_tokens=int(max_tokens),
                           enforce_json=False,
                           retries=max(int(retries), 1),
                           debug_dir=debug_dir)

    # Fallback HTTP
    import requests
    url = base_url.rstrip("/") + "/chat/completions"
    last_err = None
    for i in range(max(1, retries)):
        try:
            resp = requests.post(url, json={
                "model": model,
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }, timeout=timeout)
            data = resp.json()
            try:
                with open(os.path.join(debug_dir or ".", f"raw_response_try{i+1}.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
            if content.strip():
                return content.strip()
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"LLM call failed after {retries} attempt(s): {last_err}")


# ---------- Notebook assembly ----------

def _topo_from_hypergraph(cells_spec: List[Dict[str, Any]],
                          hypergraph: Dict[str, Any] | None) -> List[str]:
    """
    Topological order from hyperedges; fallback to listed order.
    """
    try:
        ids = [c.get("id") for c in (cells_spec or []) if c.get("id")]
        edges: List[Tuple[str, str]] = []
        for e in (hypergraph or {}).get("hyperedges", []) or []:
            head = e.get("head")
            for t in (e.get("tail") or []):
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

def _wrap_code_to_notebook(code_text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(code_text)]
    return nb

def _wrap_cells_to_notebook(cells_spec: List[Dict[str, Any]],
                            hypergraph: Dict[str, Any] | None = None) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata.setdefault("execution", {})["mode"] = "cellwise"
    if hypergraph:
        nb.metadata["execution"]["hypergraph"] = hypergraph

    md = ["# Generated Notebook (cell-wise)", "", "Subtasks:"]
    for i, c in enumerate(cells_spec or []):
        md.append(f"- **{c.get('id','T'+str(i))} — {c.get('name','')}:** {c.get('purpose','')}")
    nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(md)))

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
            "deps": [t for t in (hypergraph or {}).get("hyperedges", []) if t.get("head") == cid],
        }
        nb.cells.append(cell)
    return nb


def prepend_stage1_markdown(nb: nbformat.NotebookNode, stage1_md: str) -> nbformat.NotebookNode:
    """
    Prepend a markdown cell with Stage-1 suggestions. No changes if empty.
    """
    if not stage1_md:
        return nb
    md_cell = nbformat.v4.new_markdown_cell(stage1_md)
    nb.cells.insert(0, md_cell)
    return nb


def generate_notebook_from_prompt(cfg: Dict[str, Any],
                                  spec_path: str,
                                  debug_dir: str) -> Tuple[nbformat.NotebookNode, str]:
    """
    Main builder entrypoint. Always returns (notebook, user_prompt) or raises.
    Behavior:
      - Read YAML spec (pipeline_prompt.yaml).
      - Call LLM to produce STRICT JSON for {notebook}|{code}|{cells}+{hypergraph}.
      - Materialize notebook.
      - If Stage-1 ref is enabled and available, prepend markdown as the FIRST cell.
    """
    os.makedirs(debug_dir, exist_ok=True)

    # Expose dataset_name/repo_root for ${...} expansions in YAML
    ds_name = (cfg.get("dataset_name")
               or os.environ.get("dataset_name")
               or os.environ.get("DATASET_NAME")
               or "").strip()
    if ds_name:
        os.environ["dataset_name"] = ds_name
        os.environ["DATASET_NAME"] = ds_name

    repo_root = Path(__file__).resolve()
    while not (repo_root / "data").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent
    os.environ.setdefault("repo_root", str(repo_root))

    # Read & expand spec
    spec = expand_env_vars(read_yaml(spec_path))

    # Build messages
    sys_txt = (spec.get("system") or "").strip() if isinstance(spec, dict) else ""
    dev_txt = (spec.get("developer") or "").strip() if isinstance(spec, dict) else ""
    core = {k: v for k, v in spec.items() if k not in ("system", "developer", "user")} if isinstance(spec, dict) else {}

    # User payload (pure YAML dump; Stage-1 is NOT mixed into prompt text)
    if yaml is not None:
        usr_txt = yaml.safe_dump(core, allow_unicode=True, sort_keys=False).strip() or "{}"
    else:
        usr_txt = json.dumps(core, ensure_ascii=False, indent=2)

    messages = [{"role": "system", "content": (sys_txt or "You are a principal ML scientist. Return STRICT JSON only.")}]
    if dev_txt:
        messages.append({"role": "system", "content": "Developer instructions:\n" + dev_txt})
    messages.append({"role": "user", "content": usr_txt})

    # LLM params
    pb = cfg.get("prompt_branch", {}) or {}
    timeout = int(pb.get("timeout", cfg.get("llm", {}).get("timeout", 1200)))
    raw_text_1 = chat_text(messages,
                           cfg=cfg,
                           timeout=timeout,
                           debug_dir=debug_dir,
                           temperature=float(pb.get("temperature", 0.2)),
                           max_tokens=int(pb.get("max_tokens", 20000)),
                           retries=int(pb.get("retries", 2)))
    with open(os.path.join(debug_dir, "llm_raw_text_r1.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text_1 or "")
    cleaned = _strip_code_fences(raw_text_1)

    # Parse first attempt
    nb_json = None
    try:
        nb_json = json.loads(cleaned) if cleaned else None
    except Exception:
        nb_json = None

    # If missing/invalid JSON, force STRICT-JSON retry
    if not nb_json:
        hard_rule = (
            "Return STRICT JSON only with one of these keys:\n"
            "  1) {\"notebook\": <nbformat v4 JSON>}\n"
            "  2) {\"code\": \"<python script>\"}\n"
            "  3) {\"cells\": [...], \"hypergraph\": {...}}\n"
            "No markdown, no backticks, no extra commentary."
        )
        messages2 = list(messages) + [{"role": "system", "content": hard_rule}]
        raw_text_2 = chat_text(messages2,
                               cfg=cfg,
                               timeout=timeout,
                               debug_dir=debug_dir,
                               temperature=float(pb.get("temperature", 0.2)),
                               max_tokens=int(pb.get("max_tokens", 20000)),
                               retries=max(1, int(pb.get("retries", 2)) - 1))
        with open(os.path.join(debug_dir, "llm_raw_text_r2.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text_2 or "")
        cleaned2 = _strip_code_fences(raw_text_2)
        try:
            nb_json = json.loads(cleaned2)
        except Exception as e2:
            # As a last resort, if it's plain code, wrap it
            if cleaned2 and not cleaned2.lstrip().startswith("{"):
                nb = _wrap_code_to_notebook(cleaned2)
                stage1_md = build_stage1_markdown(cfg)
                return prepend_stage1_markdown(nb, stage1_md), usr_txt
            with open(os.path.join(debug_dir, "raw_unparsed.txt"), "w", encoding="utf-8") as fdbg:
                fdbg.write(raw_text_1 or ""); fdbg.write("\n\n===== SECOND ROUND =====\n\n"); fdbg.write(raw_text_2 or "")
            raise RuntimeError(f"Notebook JSON parse failed: {e2}")

    # Build notebook from decoded JSON
    nb: Optional[nbformat.NotebookNode] = None
    if isinstance(nb_json, dict):
        if "notebook" in nb_json and isinstance(nb_json["notebook"], dict):
            try:
                nb = nbformat.from_dict(nb_json["notebook"])
            except Exception:
                nb = nbformat.v4.new_notebook()
                nb.cells = [nbformat.v4.new_code_cell("# Failed to parse provided notebook")]
        elif "cells" in nb_json and isinstance(nb_json["cells"], list):
            nb = _wrap_cells_to_notebook(nb_json["cells"], nb_json.get("hypergraph"))
        elif "code" in nb_json and isinstance(nb_json["code"], str):
            nb = _wrap_code_to_notebook(nb_json["code"])

    if nb is None:
        # Try full nbformat dict
        try:
            nb = nbformat.from_dict(nb_json)
            if getattr(nb, "nbformat", 4) != 4:
                raise RuntimeError(f"nbformat must be 4, got {getattr(nb, 'nbformat', None)}")
        except Exception as e:
            raise RuntimeError(f"Invalid notebook format: {e}")

    # Prepend Stage-1 markdown (if toggled & available)
    stage1_md = build_stage1_markdown(cfg)
    nb = prepend_stage1_markdown(nb, stage1_md)
    return nb, usr_txt
