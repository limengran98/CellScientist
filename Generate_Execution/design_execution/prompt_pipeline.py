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

from pathlib import Path


try:
    import yaml
except Exception:
    yaml = None  # Requires pyyaml

import nbformat
from nbclient import NotebookClient
import requests

from pathlib import Path
from design_execution.nb_autofix import execute_with_autofix 

def _find_llm_providers_path_for_gen(cfg: Optional[Dict[str, Any]]) -> Optional[Path]:
    # 1) cfg.paths.llm_providers
    if cfg:
        p = (cfg.get("paths") or {}).get("llm_providers")
        if p:
            pp = Path(p).expanduser().resolve()
            if pp.exists():
                return pp
    # 2) env override
    env_p = os.getenv("LLM_PROVIDERS_PATH")
    if env_p:
        ep = Path(env_p).expanduser().resolve()
        if ep.exists():
            return ep
    # 3) walk upwards to find llm_providers.json
    here = Path(__file__).resolve()
    for anc in [here] + list(here.parents):
        cand = anc / "llm_providers.json"
        if cand.exists():
            return cand
    return None

def _load_llm_profile_for_generation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    fp = _find_llm_providers_path_for_gen(cfg)
    if not fp:
        raise RuntimeError("llm_providers.json not found. Set cfg.paths.llm_providers or LLM_PROVIDERS_PATH.")
    mp = json.loads(fp.read_text(encoding="utf-8"))
    providers = (mp.get("providers") or {})
    want = ((cfg.get("llm") or {}).get("provider")) or mp.get("default_provider")
    if not want and providers:
        want = next(iter(providers.keys()))
    if not want or want not in providers:
        raise RuntimeError(f"Provider '{want}' not found in {fp}")
    prof = providers[want] or {}

    # pick model
    model = prof.get("model")
    if not model:
        models = prof.get("models") or []
        model = models[0] if models else "gpt-4o-mini"

    # env override for secrets
    base_url = os.getenv(f"{want.upper()}_BASE_URL", prof.get("base_url"))
    api_key  = os.getenv(f"{want. upper()}_API_KEY",  prof.get("api_key"))

    # temperature / max_tokens: 优先 prompt_branch（生成阶段常用），否则 provider 默认
    pb = (cfg.get("prompt_branch") or {})
    temperature = pb.get("temperature", prof.get("temperature", 0.0))
    max_tokens  = pb.get("max_tokens",  prof.get("max_tokens", 20000))

    return {
        "provider": want,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "providers_path": str(fp),
    }

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

    # Read and expand variables (original logic)
    spec = _expand_vars(_read_yaml(spec_path))

    # ==== (1) Build messages: support YAMLs without a 'user:' field ====
    import yaml
    sys_txt = ""
    dev_txt = ""
    usr_txt = ""

    if isinstance(spec, dict):
        sys_txt = (spec.get("system") or "").strip()
        dev_txt = (spec.get("developer") or "").strip()
        # When there's no 'user:' key, treat all top-level keys except system/developer/user as SPEC and serialize them
        core = {k: v for k, v in spec.items() if k not in ("system", "developer", "user")}
        if core:
            usr_txt = yaml.safe_dump(core, allow_unicode=True, sort_keys=False).strip()

    if not usr_txt:
        # Fallback to phase-1 logic
        phase1_dir = cfg.get("paths", {}).get("stage1_analysis_dir", "")
        phase1_hint = _summarize_phase1(phase1_dir)
        usr_txt = _build_user_prompt(spec, phase1_hint)

    system = sys_txt if sys_txt else PROMPT_SYSTEM

    messages = [{"role": "system", "content": system}]
    if dev_txt:
        messages.append({"role": "system", "content": "Developer instructions:\n" + dev_txt})
    messages.append({"role": "user", "content": usr_txt})

    # ==== (2) Call the LLM (preserve original parameters) ====
    llm_cfg = cfg.get("llm", {})
    base_url = llm_cfg.get("base_url", os.getenv("OPENAI_BASE_URL", "https://vip.yi-zhan.top/v1"))
    api_key  = llm_cfg.get("api_key",  os.getenv("OPENAI_API_KEY", "any_string_if_required"))
    model    = llm_cfg.get("model",    "gpt-5")
    timeout  = int(llm_cfg.get("timeout", 1200))
    pb = cfg.get("prompt_branch", {}) or {}

    def _strip_code_fences(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        if s.startswith("```"):
            # Remove ```json / ```python code fences
            s = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
        return s

    def _wrap_code_to_notebook(code_text: str) -> nbformat.NotebookNode:
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell(code_text)]
        return nb

    # Round 1: original prompt
    raw_text_1 = _chat_text_stable(
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
    # Save snapshot
    open(os.path.join(debug_dir, "llm_raw_text_r1.txt"), "w", encoding="utf-8").write(raw_text_1 or "")

    cleaned = _strip_code_fences(raw_text_1)

    # Try parsing as JSON
    try:
        if cleaned:
            nb_json = json.loads(cleaned)
        else:
            raise ValueError("empty response")
    except Exception:
        # If not valid JSON, try to parse as code and wrap into a notebook (non-fatal)
        if cleaned and not cleaned.lstrip().startswith("{"):
            # Many models return raw scripts or ```python blocks
            nb = _wrap_code_to_notebook(cleaned)
            return nb, usr_txt

        # Round 2: enforce "return JSON only"
        hard_rule = (
            "Return STRICT JSON only with one of these keys:\n"
            "  1) {\"notebook\": <nbformat v4 JSON>}\n"
            "  2) {\"code\": \"<python script>\"}\n"
            "No markdown, no backticks, no extra commentary."
        )
        messages2 = list(messages) + [{"role": "system", "content": hard_rule}]
        raw_text_2 = _chat_text_stable(
            messages=messages2,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=pb.get("temperature", 0.2),
            max_tokens=pb.get("max_tokens", 80000),
            retries=max(1, pb.get("retries", 4)-1),
            timeout=timeout,
            debug_dir=debug_dir
        )
        open(os.path.join(debug_dir, "llm_raw_text_r2.txt"), "w", encoding="utf-8").write(raw_text_2 or "")
        cleaned2 = _strip_code_fences(raw_text_2)

        # Second JSON parsing attempt
        try:
            nb_json = json.loads(cleaned2)
        except Exception as e2:
            # Final fallback: if it looks like code, wrap as notebook
            if cleaned2 and not cleaned2.lstrip().startswith("{"):
                nb = _wrap_code_to_notebook(cleaned2)
                return nb, usr_txt
            # Save raw text and raise error
            with open(os.path.join(debug_dir, "raw_unparsed.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text_1 or "")
                f.write("\n\n===== SECOND ROUND =====\n\n")
                f.write(raw_text_2 or "")
            raise RuntimeError(f"Notebook JSON parse failed: {e2}")

    # At this point: JSON obtained, could be {"notebook": {...}} or a direct nbformat dict
    if isinstance(nb_json, dict) and "notebook" in nb_json and isinstance(nb_json["notebook"], dict):
        nb_payload = nb_json["notebook"]
    else:
        nb_payload = nb_json

    try:
        nb = nbformat.from_dict(nb_payload)
        if nb.nbformat != 4:
            raise RuntimeError(f"nbformat must be 4, got {nb.nbformat}")
    except Exception as e:
        # If JSON is actually {"code": "..."} handle it as well
        code_fallback = None
        if isinstance(nb_json, dict) and isinstance(nb_json.get("code"), str):
            code_fallback = nb_json["code"]
        if code_fallback:
            nb = _wrap_code_to_notebook(code_fallback)
            return nb, usr_txt
        raise RuntimeError(f"Invalid notebook format: {e}")

    return nb, usr_txt


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
    """Execute the latest prompt artifact with auto-fix (known patches + optional LLM)."""
    tdir = trial_dir or _latest_prompt_dir(cfg)
    if not tdir:
        raise RuntimeError("No prompt trial found. Run 'generate' first.")

    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    if not os.path.exists(nb_path):
        raise RuntimeError(f"notebook_prompt.ipynb not found in {tdir}")

    # --- read exec config & LLM switch ---
    exec_cfg = (cfg.get("exec") or {})
    timeout = int(exec_cfg.get("timeout_seconds", 1800))
    max_fix_rounds = int(exec_cfg.get("max_fix_rounds", 1))
    use_llm = bool(exec_cfg.get("enable_llm_autofix", True))   # <— the switch

    print(f"[PROMPT] exec config -> timeout_seconds={timeout}, max_fix_rounds={max_fix_rounds}, "
          f"enable_llm_autofix={use_llm}")
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
        phase_cfg=cfg,                 # pass full cfg so LLM can read provider/env
        preserve_source_in_exec=True,
        save_intermediates=True,
    )
    print(f"[PROMPT] executed notebook -> {final_exec_path}")

    # read metrics if present
    metrics_path = os.path.join(tdir, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            print(f"[PROMPT] metrics loaded: keys={list(metrics.keys())}")
        except Exception as e:
            print(f"[PROMPT][WARN] failed to read metrics.json: {e}")
    else:
        print(f"[PROMPT][WARN] metrics.json not found in {tdir}")

    return {"trial_dir": tdir, "metrics": metrics, "exec_notebook": final_exec_path}



def prompt_analyze(cfg: dict, trial_dir: str) -> dict:
    """
    Analyze the executed prompt run. If metrics.json is missing, attempt to
    execute once, then degrade gracefully instead of raising.
    """
    metrics_path = os.path.join(trial_dir, "metrics.json")

    if not os.path.exists(metrics_path):
        print(f"[WARN] metrics.json not found in {trial_dir}; attempting to execute once...")
        try:
            # 兜底：跑一遍执行（如果上游没跑或失败），不抛错影响后续
            _ = prompt_execute(cfg, trial_dir)
        except Exception as e:
            print(f"[WARN] execution attempt during analyze failed: {e}")

    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to read metrics.json: {e}")
    else:
        # 仍然没有，就降级而不是 raise，避免像你现在的栈那样中断
        print(f"[WARN] metrics.json still missing after execution attempt; "
              f"continue analysis with limited info.")

    # === 这里保持你原有的分析逻辑 ===
    # 例：生成一份简要报告占位；如果你已有更完整的报告逻辑，直接用你现有的
    report = {
        "trial_dir": trial_dir,
        "has_metrics": bool(metrics),
        "metrics_keys": list(metrics.keys()) if metrics else [],
        "notes": "metrics unavailable" if not metrics else "ok"
    }

    # （可选）把分析结果落盘
    try:
        with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return report



# ---------------------------
# One-click
# ---------------------------
def run_prompt_pipeline(cfg: dict, prompt_path: str) -> dict:
    """
    End-to-end pipeline for prompt-defined run:
      1) generate notebook
      2) execute with auto-fix (produces metrics.json)
      3) analyze (consumes metrics.json if present)
    """
    print("[INFO] === Generating prompt notebook ===")
    gen = prompt_generate(cfg, prompt_path)   
    tdir = gen["trial_dir"]

    print("[INFO] === Executing prompt notebook (with auto-fix) ===")
    exec_ret = prompt_execute(cfg, tdir)     
    print("[INFO] === Analyzing results ===")
    rpt = prompt_analyze(cfg, exec_ret["trial_dir"]) 

    return {
        "trial_dir": exec_ret["trial_dir"],
        "exec_notebook": exec_ret.get("exec_notebook"),
        "metrics": exec_ret.get("metrics", {}),
        "report": rpt
    }



