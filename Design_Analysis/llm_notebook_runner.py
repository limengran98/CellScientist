from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat as nbf

# [NEW] Use the centralized config loader
from config_loader import load_app_config

# ---------------- Utilities ----------------

def safe_read_csv_preview(csv_path: str, max_rows: int = 8, max_cols: int = 20) -> Dict[str, Any]:
    """Read a small preview of a CSV (if available) to provide schema/context to the LLM."""
    import pandas as pd
    p = Path(csv_path)
    if not p.exists():
        return {"exists": False, "error": f"Data not found: {csv_path}"}
    try:
        df = pd.read_csv(p, nrows=500)  # shallow preview
        cols = list(df.columns)[:max_cols]
        head = df[cols].head(max_rows).to_dict(orient="records")
        meta_cols = [c for c in df.columns if str(c).lower().startswith("metadata_")]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        return {
            "exists": True,
            "n_rows_preview": len(df),
            "n_cols": len(df.columns),
            "meta_cols": meta_cols[:30],
            "feature_cols_head": feat_cols[:30],
            "head_rows": head,
        }
    except Exception as e:
        return {"exists": True, "error": f"Preview failed: {e}"}

def read_pdf_excerpt(pdf_path: Optional[str], max_pages: int = 3, max_chars: int = 4000) -> str:
    """Extract a short excerpt from the first pages of a PDF if PyPDF2 is available."""
    if not pdf_path:
        return ""
    p = Path(pdf_path)
    if not p.exists():
        return ""
    try:
        import PyPDF2
        text = []
        with open(p, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i in range(min(len(reader.pages), max_pages)):
                try:
                    text.append(reader.pages[i].extract_text() or "")
                except Exception:
                    pass
        return ("\n".join(text)).strip()[:max_chars]
    except Exception:
        return ""

def extract_json_block(text: str) -> Dict[str, Any]:
    """Extract a JSON object from raw LLM output.

    [PATCH] More robust against:
      - <think> blocks / prefixed reasoning text
      - fenced markdown
      - invalid control characters (e.g., raw newlines inside JSON strings)
      - Python-dict-like outputs (single quotes / True/False)
    """
    if not text:
        raise ValueError("Empty LLM response (no JSON).")

    # 1) strip <think> blocks if present
    try:
        text2 = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    except Exception:
        text2 = (text or "").strip()

    def _escape_control_chars_in_strings(s: str) -> str:
        if not s:
            return s
        out = []
        in_str = False
        esc = False
        for ch in s:
            o = ord(ch)
            if in_str:
                if esc:
                    out.append(ch)
                    esc = False
                    continue
                if ch == "\\":  # begin escape
                    out.append(ch)
                    esc = True
                    continue
                if ch == '"':
                    out.append(ch)
                    in_str = False
                    continue
                if ch == "\n":
                    out.append("\\n")
                    continue
                if ch == "\r":
                    out.append("\\r")
                    continue
                if ch == "\t":
                    out.append("\\t")
                    continue
                if o < 32:
                    out.append(f"\\u{o:04x}")
                    continue
                out.append(ch)
                continue

            # not in string
            if ch == '"':
                out.append(ch)
                in_str = True
                esc = False
                continue

            if o < 32 and ch not in ("\n", "\r", "\t"):
                # drop other control chars outside strings
                continue
            out.append(ch)
        return "".join(out)

    def _extract_balanced_object(s: str) -> str:
        if not s:
            return ""
        start = s.find("{")
        if start < 0:
            return ""
        in_str = False
        esc = False
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":  # escape
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                esc = False
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return ""

    candidates = []

    # 2) fenced blocks
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text2, flags=re.IGNORECASE)
    if m:
        candidates.append(m.group(1).strip())

    # 3) balanced object (preferred over greedy regex)
    obj = _extract_balanced_object(text2)
    if obj:
        candidates.append(obj)

    # 4) raw text as last resort
    candidates.append(text2)

    last_err = None
    for cand in candidates:
        for attempt in (cand, _escape_control_chars_in_strings(cand)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as e:
                last_err = e

        # python literal fallback
        try:
            import ast
            py = cand
            py = re.sub(r"\btrue\b", "True", py, flags=re.IGNORECASE)
            py = re.sub(r"\bfalse\b", "False", py, flags=re.IGNORECASE)
            py = re.sub(r"\bnull\b", "None", py, flags=re.IGNORECASE)
            parsed = ast.literal_eval(py)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            last_err = e

    preview = text2[:800] + ("..." if len(text2) > 800 else "")
    raise ValueError(f"Could not parse JSON notebook spec from LLM response. Last error: {last_err}. Preview:\n{preview}")
def nb_from_spec(spec: Dict[str, Any]) -> nbf.NotebookNode:
    """Convert the LLM JSON spec to a Notebook object."""
    nb = nbf.v4.new_notebook()
    cells = []
    title = spec.get("title") or "LLM Notebook"
    cells.append(nbf.v4.new_markdown_cell(f"# {title}"))
    for cell in spec.get("cells", []):
        ctype = (cell.get("type") or "").lower()
        src = cell.get("source") or ""
        if ctype == "markdown":
            cells.append(nbf.v4.new_markdown_cell(src))
        elif ctype == "code":
            cells.append(nbf.v4.new_code_cell(src))
    nb["cells"] = cells
    return nb

# ---------------- Config helpers (phase-scoped) ----------------

def _phase_enabled(config: Dict[str, Any], phase_name: str = "task_analysis") -> bool:
    phases = config.get("workflow_phases") or []
    phase_cfg = (config.get("phases") or {}).get(phase_name) or {}
    return (phase_name in phases) and bool(phase_cfg.get("enabled", False))

def _get_phase_llm_nb_cfg(config: Dict[str, Any], phase_name: str = "task_analysis") -> Dict[str, Any]:
    return ((config.get("phases") or {}).get(phase_name) or {}).get("llm_notebook") or {}

def _resolve_paths(cfg: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    paths = cfg.get("paths") or {}
    data = paths.get("data") or "/mnt/data/CP_data.csv"
    paper = paths.get("paper") or "/mnt/data/BBBC036.pdf"
    preprocess = paths.get("preprocess") or "/mnt/data/BBBC036_data_process.ipynb"
    out = paths.get("out") or "/mnt/data/CP_llm.ipynb"
    out_exec = paths.get("out_exec") or paths.get("out-exec") or "/mnt/data/CP_llm_executed.ipynb"
    h5_out = paths.get("h5_out") or (str(Path(out).parent / "preprocessed_data.h5"))
    return data, paper, preprocess, out, out_exec, h5_out

def _resolve_exec_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg.get("exec") or {}
    return {
        "timeout_seconds": int(d.get("timeout_seconds", 1800)),
        "max_preview_rows": int(d.get("max_preview_rows", 8)),
        "max_preview_cols": int(d.get("max_preview_cols", 20)),
        "pdf_max_pages": int(d.get("pdf_max_pages", 3)),
        "force_json_mode": bool(d.get("force_json_mode", True)),
        "save_intermediate": bool(d.get("save_intermediate", True)),
        "allow_errors": bool(d.get("allow_errors", True)),
        "cuda_device_id": d.get("cuda_device_id", None) # [NEW]
    }

def _resolve_llm_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve LLM connection settings with sensible precedence:
    """
    llm = cfg.get("llm") or {}
    base_url = llm.get("base_url") or None
    model = llm.get("model") or None

    try:
        base_dir = Path(__file__).resolve().parents[1]
        prov_file = base_dir / "llm_providers.json"
        prov_name = (llm.get("provider") or "").strip() or None
        if (not base_url or not model) and prov_name and prov_file.exists():
            data = json.loads(prov_file.read_text(encoding="utf-8"))
            prov = (data.get("providers") or {}).get(prov_name) or {}
            if not base_url:
                base_url = prov.get("base_url") or None
            if not model:
                models = prov.get("models") or []
                model = models[0] if models else None
    except Exception:
        pass

    if not base_url:
        base_url_env = llm.get("base_url_env")
        if base_url_env and os.environ.get(base_url_env):
            base_url = os.environ.get(base_url_env)
        else:
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://vip.yi-zhan.top/v1"

    if not model:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    api_key = llm.get("api_key") or None
    if not api_key:
        api_key_env = llm.get("api_key_env") or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)

    temperature = float(llm.get("temperature", 0.2))
    max_tokens = int(llm.get("max_tokens", 8192))
    top_p = float(llm.get("top_p", 1))
    freq_pen = float(llm.get("frequency_penalty", 0))
    pres_pen = float(llm.get("presence_penalty", 0))
    force_json = bool(llm.get("force_json_mode", True))

    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": freq_pen,
        "presence_penalty": pres_pen,
        "force_json_mode": force_json,
    }


def _resolve_language_and_headings(cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    language = (cfg.get("language") or "en").lower()
    headings_cfg = ((cfg.get("headings") or {}).get("sections")) or [
        "## Data Loading & Initial Exploration",
        "## Data Patterns",
        "## Hidden Information",
        "## Innovation Motivation",
        "## Experiment & Validation Suggestions",
    ]
    return language, headings_cfg

# ---------------- LLM client (OpenAI-compatible) ----------------

@dataclass
class LLMConfig:
    model: str
    api_key: Optional[str]
    base_url: Optional[str]

class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if not cfg.api_key:
            raise RuntimeError("OPENAI_API_KEY (or configured api_key) is required.")
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url) if cfg.base_url else OpenAI(api_key=cfg.api_key)
        except Exception as e:
            raise RuntimeError("Install openai: `pip install openai`.") from e

    def chat_json(self, messages, force_json: bool = True):
        print(f"👉 Sending LLM request | model={self.cfg.model} | base_url={self.cfg.base_url or 'default OpenAI'}")

        # [PATCH] Retry once with a stricter system nudge if parsing/empty-content issues occur.
        base_messages = list(messages) if isinstance(messages, list) else messages
        last_err = None

        for attempt in range(2):
            msgs = base_messages
            if attempt == 1:
                try:
                    msgs = list(base_messages) + [
                        {"role": "system", "content": "Return ONLY a valid JSON object. No markdown, no prose, no <think>."}
                    ]
                except Exception:
                    msgs = base_messages

            # Prefer provider-side JSON mode if available
            if force_json:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.cfg.model,
                        messages=msgs,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    print(f"✅ LLM responded | model={resp.model}")
                    content = (resp.choices[0].message.content or "").strip()
                    if content:
                        try:
                            return json.loads(content)
                        except Exception:
                            # fall back to robust extractor (handles control chars / think text)
                            return extract_json_block(content)
                except Exception as e:
                    last_err = e

            # Fallback: normal completion then robust extract
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=msgs,
                    temperature=0.2,
                )
                print(f"✅ LLM responded | model={resp.model}")
                content = resp.choices[0].message.content or "{}"
                return extract_json_block(content)
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"LLM JSON generation failed after retries. Last error: {last_err}")


# ---------------- High-level Runner ----------------

def _system_prompt(language: str, headings: List[str], base_prompt_str: str, data_path: str, h5_output_path: str, split_strategy: str) -> str:
    if not base_prompt_str:
        raise ValueError("Base system prompt is empty. Check prompts/notebook_generation.yml")
    lang_label = "English" if language.startswith("en") else language
    headings_bulleted = "\n".join([f"  {h}" for h in headings])
    
    return base_prompt_str.format(
        language_label=lang_label, 
        headings_bulleted=headings_bulleted,
        data_path=data_path,
        h5_output_path=h5_output_path,
        split_strategy=split_strategy
    )

def make_messages(
    user_prompt: str,
    data_path: str,
    paper_excerpt: str,
    csv_preview: Dict[str, Any],
    language: str,
    headings: List[str],
    base_system_prompt_str: str,
    h5_output_path: str,
    split_strategy: str = "random",
) -> List[Dict[str, str]]:
    """Compose the system+user messages for the LLM."""
    context = {
        "user_prompt": user_prompt,
        "language": language,
        "data_path": data_path,
        "h5_output_path": h5_output_path,
        "paper_excerpt": (paper_excerpt or "")[:2000],
        "csv_preview": csv_preview,
        "constraints": {
            "alternate_markdown_code": True,
            "no_internet": True,
            "allowed_libs": ["pandas","numpy","matplotlib","statsmodels","scikit-learn"],
            "required_headings": headings,
        }
    }
    return [
        {"role": "system", "content": _system_prompt(
            language, headings, base_system_prompt_str, data_path, h5_output_path, split_strategy
        )},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, indent=2)},
    ]

def run_llm_notebook(
    data_path: str,
    paper_pdf: Optional[str],
    preprocess_nb: Optional[str],
    user_prompt: str,
    out_path: str,
    base_system_prompt_str: str, 
    h5_out_path: str, 
    split_strategy: str = "random", 
    executed_path: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: int = 1800,
    force_json_mode: bool = True,
    max_preview_rows: int = 8,
    max_preview_cols: int = 20,
    pdf_max_pages: int = 3,
    language: str = "en",
    headings: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    save_intermediate: bool = True,
    allow_errors: bool = True,
    cuda_device_id: Optional[str] = None, # [NEW]
) -> str:
    """
    Core runner.
    """
    if headings is None:
        headings = [
            "## Data Loading & Initial Exploration",
            "## Data Patterns",
            "## Hidden Information",
            "## Innovation Motivation",
            "## Experiment & Validation Suggestions",
        ]

    # Build context
    csv_preview = safe_read_csv_preview(data_path, max_rows=max_preview_rows, max_cols=max_preview_cols)
    paper_excerpt = read_pdf_excerpt(paper_pdf, max_pages=pdf_max_pages)

    # LLM messages
    messages = make_messages(
        user_prompt, data_path, paper_excerpt, csv_preview, 
        language=language, headings=headings, 
        base_system_prompt_str=base_system_prompt_str,
        h5_output_path=h5_out_path,
        split_strategy=split_strategy
    )

    # LLM config
    model_final = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_key_final = api_key or os.environ.get("OPENAI_API_KEY")
    base_url_final = base_url or os.environ.get("OPENAI_BASE_URL")
    client = OpenAICompatClient(LLMConfig(model=model_final, api_key=api_key_final, base_url=base_url_final))

    # Generate JSON notebook spec
    spec = client.chat_json(messages, force_json=force_json_mode)

    # Build and write notebook
    nb = nb_from_spec(spec)
    out_p = Path(out_path); out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(nbf.writes(nb), encoding="utf-8")

    if save_intermediate:
        out_p.write_text(nbf.writes(nb), encoding="utf-8")

    # [NEW] Initial Execution with CUDA
    # Note: This is the initial run. The Adaptive Fix loop in run_llm_nb.py will also need this info.
    from nbclient import NotebookClient
    
    # Inject CUDA device into environment
    env_vars = os.environ.copy()
    if cuda_device_id is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
        print(f"🖥️  CUDA Device set to: {cuda_device_id}", flush=True)

    # We must start the kernel with this env
    # nbclient passes 'env' to the kernel manager
    
    nb_client = NotebookClient(
        nb,
        timeout=timeout_seconds,
        kernel_name="python3",
        allow_errors=allow_errors,
        resources={'env': env_vars} # [NEW] Pass Env
    )

    errors_summary = []
    try:
        nb_client.execute()
    except Exception as e:
        errors_summary.append(f"Notebook execution raised: {type(e).__name__}: {e}")

    if errors_summary:
        nb.metadata["execution_errors"] = errors_summary

    exec_p = Path(executed_path or out_p.with_name(out_p.stem + "_executed.ipynb"))
    exec_p.write_text(nbf.writes(nb), encoding="utf-8")
    return str(exec_p)


# ---------------- Phase-scoped wrappers ----------------

def run_llm_notebook_with_config(config: Dict[str, Any], phase_name: str = "task_analysis") -> str:
    """Run only if the `phase_name` is active; read all params from phases.task_analysis.llm_notebook."""
    if not _phase_enabled(config, phase_name=phase_name):
        raise RuntimeError(f"Phase '{phase_name}' is not active (workflow_phases + phases.{phase_name}.enabled).")

    nb_cfg = _get_phase_llm_nb_cfg(config, phase_name=phase_name)

    prompt = (config.get('prompts', {}).get('notebook_generation', {}).get('user_prompt'))
    base_system_prompt_str = (config.get('prompts', {}).get('notebook_generation', {}).get('system_prompt'))
    
    if not prompt or not base_system_prompt_str:
        print("Warning: Prompts not found in config. Check config_loader and prompts/ dir.", flush=True)
        if not prompt:
            prompt = "Please generate an English analysis notebook following the required structure."
        if not base_system_prompt_str:
            base_system_prompt_str = "You are an expert computational biologist. Return ONLY a JSON object..."


    language, headings = _resolve_language_and_headings(nb_cfg)

    data, paper, preprocess, out, out_exec, h5_out = _resolve_paths(nb_cfg)
    exec_cfg = _resolve_exec_cfg(nb_cfg)
    llm_basic = _resolve_llm_cfg(nb_cfg)
    split_strategy = nb_cfg.get("split_strategy", "random")

    return run_llm_notebook(
        data_path=data,
        paper_pdf=paper,
        preprocess_nb=preprocess,
        user_prompt=prompt,
        out_path=out,
        executed_path=out_exec,
        base_system_prompt_str=base_system_prompt_str,
        h5_out_path=h5_out,
        split_strategy=split_strategy, 
        model=llm_basic["model"],
        timeout_seconds=exec_cfg["timeout_seconds"],
        force_json_mode=exec_cfg["force_json_mode"],
        max_preview_rows=exec_cfg["max_preview_rows"],
        max_preview_cols=exec_cfg["max_preview_cols"],
        pdf_max_pages=exec_cfg["pdf_max_pages"],
        language=language,
        headings=headings,
        api_key=llm_basic["api_key"],
        base_url=llm_basic["base_url"],
        save_intermediate=bool(exec_cfg.get("save_intermediate", True)),
        allow_errors=bool(exec_cfg.get("allow_errors", True)),
        cuda_device_id=exec_cfg.get("cuda_device_id") # [NEW]
    )

def run_llm_notebook_from_file(
    config_path: str, 
    prompts_dir_path: str,
    phase_name: str = "task_analysis"
) -> str:
    cfg = load_app_config(config_path, prompts_dir_path)
    return run_llm_notebook_with_config(cfg, phase_name=phase_name)