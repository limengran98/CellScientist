# -*- coding: utf-8 -*-
"""
Prompt-defined pipeline orchestrator (Notebook mode).

Generates a NEW experiment-focused Jupyter notebook (nbformat v4) from a prompt:
- Does NOT copy or depend on baseline code
- May use Phase-1 notebook(s) only as high-level reference (style/steps), via a short summary
- Must implement: data loading -> fair 5-fold CV (leave-smiles-out or leave-plates-out) -> train -> compute metrics
- Must write metrics.json to the output directory
- Avoids heavy EDA; focus on running experiments and reporting metrics

Outputs are saved under:
results/${dataset_name}/generate_execution/prompt/prompt_run_YYYYMMDD_HHMMSS/
  - notebook_prompt.ipynb
  - notebook_prompt_exec.ipynb
  - metrics.json
  - exec_stdout.log
"""
import os, io, re, json, time, datetime, subprocess, textwrap, pathlib, sys, glob
from typing import Dict, Any, Tuple, List, Optional

try:
    import yaml
except Exception:
    yaml = None  # Requires pyyaml

import nbformat
from nbclient import NotebookClient
import requests


PROMPT_SYSTEM = """\
You are an elite ML engineer. Produce a SINGLE Jupyter notebook (nbformat v4 JSON).
The notebook must be EXPERIMENT-FOCUSED, not exploratory analysis:
- Load dataset from the USER spec
- Build 5-fold cross validation with fair grouping (GroupKFold):
  - leave-smiles-out: group by "smiles" column
  - leave-plates-out: group by "plate" column
  - If group_column_override is provided, group by that
- Train reasonable models for numeric targets (regression). Add minimal feature processing if needed
- Compute metrics:
  * PCC, RMSE
  * DEG_PCC, DEG_RMSE (subset deg_flag==True/1; null if column missing)
  * Direction_ACC (sign vs. direction_threshold; average if multi-target)
  * Systema_PCC, Systema_RMSE (subset systema_flag==True/1; null if column missing)
- Aggregate per-fold & overall metrics (dict: per_fold, aggregate, metadata)
- Save metrics.json under the output directory
- Avoid external network calls; only common libs: numpy, pandas, scikit-learn, scipy, rdkit (if present)
- The notebook should be minimal, organized as:
  1) Config cell (inline JSON of the spec; read env var OUTPUT_DIR)
  2) Imports + helper functions (metrics, CV)
  3) Data loading & validation
  4) 5-fold CV training/eval
  5) Save metrics.json & print final aggregate metrics
- The notebook must be valid nbformat v4 JSON (no markdown fencing). Return ONLY the notebook JSON.
"""


# ---------------------------
# Helpers
# ---------------------------
def _read_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required by the orchestrator. Please install pyyaml.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _expand_vars(obj):
    if isinstance(obj, dict):
        return {k: _expand_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_vars(v) for v in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj

def _summarize_phase1(reference_dir: str, max_chars: int = 1600) -> str:
    """Lightly summarize the first small .ipynb under reference_dir (high-level hint only)."""
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
                if 10 <= len(src) <= 500:
                    lines.append(src[:500])
            elif cell.cell_type == "code":
                src = (cell.source or "").strip().split("\n")
                keep = [ln for ln in src if len(ln) < 120 and not ln.strip().startswith("#")]
                if keep:
                    lines.append("\n".join(keep[:5]))
        return "\n".join(lines)[:max_chars]
    except Exception:
        return ""

def _build_user_prompt(spec: Dict[str, Any], phase1_hint: str) -> str:
    spec_text = json.dumps(spec, ensure_ascii=False, indent=2)
    ref_hint = (phase1_hint or "").strip()
    return f"""\
# Dataset & Pipeline Spec (authoritative; copy into the first cell as CONFIG_JSON)
{spec_text}

# Additional constraints
- Five-fold GroupKFold with fair grouping (no leakage).
- Save metrics.json into OUTPUT_DIR (env or default path).
- Keep the notebook concise and deterministic (set random seeds).
- If optional columns missing (deg_flag/systema_flag), set corresponding metrics to null.

# Phase-1 notebook hint (high-level only; DO NOT copy code verbatim)
{ref_hint if ref_hint else "(no hint available)"}
"""


# ---------------------------
# Robust LLM call (TEXT)
# ---------------------------
def _chat_text_stable(messages, api_key: str, base_url: str, model: str,
                      temperature: float = 0.2, max_tokens: int = 6000,
                      retries: int = 4, timeout: int = 120,
                      debug_dir: Optional[str] = None) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    last_err = None
    for i in range(max(1, retries)):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or choice.get("text") or ""
            if isinstance(content, list):
                content = "".join(seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in content)
            content = (content or "").strip()
            if (not content) and message.get("tool_calls"):
                try:
                    content = (message["tool_calls"][0]["function"]["arguments"] or "").strip()
                except Exception:
                    pass

            if debug_dir:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    with open(os.path.join(debug_dir, f"request_try{i+1}.json"), "w", encoding="utf-8") as fq:
                        json.dump({"url": url, "payload": payload}, fq, ensure_ascii=False, indent=2)
                    with open(os.path.join(debug_dir, f"raw_response_try{i+1}.json"), "w", encoding="utf-8") as fr:
                        json.dump(data, fr, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            if not content:
                raise RuntimeError("empty content in LLM response")

            return content
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    raise RuntimeError(f"LLM call failed after {retries} attempt(s): {last_err}")


# ---------------------------
# Core generation (Notebook JSON)
# ---------------------------
def generate_notebook_from_prompt(cfg: Dict[str, Any], spec_path: str, debug_dir: str) -> Tuple[nbformat.NotebookNode, str]:
    os.makedirs(debug_dir, exist_ok=True)
    spec = _expand_vars(_read_yaml(spec_path))

    # optional Phase-1 hint (high-level only)
    phase1_dir = cfg.get("paths", {}).get("stage1_analysis_dir", "")
    phase1_hint = _summarize_phase1(phase1_dir)

    system = PROMPT_SYSTEM
    user = _build_user_prompt(spec, phase1_hint)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    llm_cfg = cfg.get("llm", {})
    base_url = llm_cfg.get("base_url", os.getenv("OPENAI_BASE_URL", "https://vip.yi-zhan.top/v1"))
    api_key  = llm_cfg.get("api_key",  os.getenv("OPENAI_API_KEY", "any_string_if_required"))
    model    = llm_cfg.get("model",    "gpt-5")
    timeout  = int(llm_cfg.get("timeout", 1200))
    pb = cfg.get("prompt_branch", {}) or {}

    raw_text = _chat_text_stable(
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=pb.get("temperature", 0.2),
        max_tokens=pb.get("max_tokens", 80000),
        retries=pb.get("retries", 4),
        timeout=timeout,
        debug_dir=debug_dir
    )

    # Expect a pure JSON for nbformat v4; try to extract if fenced
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    try:
        nb_json = json.loads(cleaned)
    except Exception as e:
        with open(os.path.join(debug_dir, "raw_unparsed.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
        raise RuntimeError(f"Notebook JSON parse failed: {e}")

    try:
        nb = nbformat.from_dict(nb_json)
        if nb.nbformat != 4:
            raise RuntimeError(f"nbformat must be 4, got {nb.nbformat}")
    except Exception as e:
        raise RuntimeError(f"Invalid notebook format: {e}")

    return nb, user


# ---------------------------
# Save & execute notebook
# ---------------------------
def save_and_execute_notebook(nb: nbformat.NotebookNode, out_root: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tdir = os.path.join(out_root, "prompt", f"prompt_run_{ts}")
    os.makedirs(tdir, exist_ok=True)

    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    nb_exec_path = os.path.join(tdir, "notebook_prompt_exec.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    # Execute notebook with OUTPUT_DIR as env
    os.environ["OUTPUT_DIR"] = tdir
    try:
        client = NotebookClient(
            nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": tdir}}
        )
        exec_nb = client.execute()
        with open(nb_exec_path, "w", encoding="utf-8") as f:
            nbformat.write(exec_nb, f)
    except Exception as e:
        import traceback
        log_path = os.path.join(tdir, "exec_stdout.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write("Notebook execution failed:\n")
            logf.write(traceback.format_exc())
        raise RuntimeError(f"Generated notebook execution failed. Check {log_path}")

    return tdir


# ---------------------------
# 3-phase APIs for prompt branch
# ---------------------------
def _prompt_out_root(cfg):
    return cfg.get("prompt_branch", {}).get("save_root", cfg["paths"]["design_execution_root"])

def _latest_prompt_dir(cfg):
    root = os.path.join(_prompt_out_root(cfg), "prompt")
    subs = sorted([p for p in glob.glob(os.path.join(root, "prompt_run_*")) if os.path.isdir(p)])
    return subs[-1] if subs else None

def prompt_generate(cfg: dict, spec_path: str) -> dict:
    """Only generate the prompt-defined artifact (notebook), do not execute."""
    debug_dir = os.path.join(cfg["paths"]["design_execution_root"], "debug_prompt")
    out_root = _prompt_out_root(cfg)
    os.makedirs(os.path.join(out_root, "prompt"), exist_ok=True)

    nb, _user_prompt = generate_notebook_from_prompt(cfg, spec_path, debug_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tdir = os.path.join(out_root, "prompt", f"prompt_run_{ts}")
    os.makedirs(tdir, exist_ok=True)

    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return {"trial_dir": tdir, "artifact": nb_path}

def prompt_execute(cfg: dict, trial_dir: str = None) -> dict:
    """Execute an existing prompt artifact (default: the latest one)."""
    tdir = trial_dir or _latest_prompt_dir(cfg)
    if not tdir:
        raise RuntimeError("No prompt trial found. Run 'generate' first.")

    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    if not os.path.exists(nb_path):
        raise RuntimeError(f"notebook_prompt.ipynb not found in {tdir}")

    nb = nbformat.read(nb_path, as_version=4)
    os.environ["OUTPUT_DIR"] = tdir
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": tdir}})
    exec_nb = client.execute()
    with open(os.path.join(tdir, "notebook_prompt_exec.ipynb"), "w", encoding="utf-8") as f:
        nbformat.write(exec_nb, f)

    metrics_path = os.path.join(tdir, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"trial_dir": tdir, "metrics": metrics}

def prompt_analyze(cfg: dict, trial_dir: str = None) -> str:
    """Read metrics.json and write a short markdown report under reports/."""
    tdir = trial_dir or _latest_prompt_dir(cfg)
    if not tdir:
        raise RuntimeError("No prompt trial found. Run 'generate'/'execute' first.")
    metrics_path = os.path.join(tdir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise RuntimeError(f"metrics.json not found in {tdir}")
    metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
    # write a minimal report
    reports_dir = os.path.join(cfg["paths"]["design_execution_root"], "reports")
    os.makedirs(reports_dir, exist_ok=True)
    name = os.path.basename(tdir)
    rpt = os.path.join(reports_dir, f"Report_{name}.md")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"# Prompt Trial Report: {name}\n\n")
        f.write("## Aggregate metrics\n\n```json\n")
        f.write(json.dumps(metrics.get("aggregate", {}), indent=2, ensure_ascii=False))
        f.write("\n```\n")
    return rpt


# ---------------------------
# One-click
# ---------------------------
def run_prompt_pipeline(cfg: Dict[str, Any], spec_path: str) -> Dict[str, Any]:
    """One-click: generate → execute → analyze."""
    gen = prompt_generate(cfg, spec_path)
    exe = prompt_execute(cfg, gen["trial_dir"])
    rpt = prompt_analyze(cfg, gen["trial_dir"])
    ret = {"trial_dir": gen["trial_dir"], "metrics": exe.get("metrics", {})}
    ret["notebook_path"] = os.path.join(gen["trial_dir"], "notebook_prompt.ipynb")
    ret["report_path"] = rpt
    return ret
