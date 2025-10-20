# -*- coding: utf-8 -*-
import os
import json
import time
import re
import glob
import shutil
import datetime
import nbformat
import requests
from typing import Optional

from .prompts import NOTEBOOK_IMPROVEMENT_SYSTEM, NOTEBOOK_IMPROVEMENT_USER_TEMPLATE


# =============== Baseline utilities ===============
def _latest_baseline_folder(root: str) -> str:
    """Pick the most recent baseline date folder under <root>/baselines/."""
    bdir = os.path.join(root, "baselines")
    subs = sorted([p for p in glob.glob(os.path.join(bdir, "*")) if os.path.isdir(p)])
    if not subs:
        raise RuntimeError("No baselines found. Run migration first.")
    return subs[-1]


def create_trial_from_baseline(root: str, tag: str, baseline_id: int = 0, seed: Optional[int] = None) -> str:
    """Create a trial directory from a chosen baseline notebook."""
    date = datetime.date.today().isoformat()
    name = f"T{date.replace('-', '')}-{tag}-s{seed or 0}"
    tdir = os.path.join(root, "trials", name)
    os.makedirs(os.path.join(tdir, "figs"), exist_ok=True)

    bfolder = _latest_baseline_folder(root)
    src_nb = os.path.join(bfolder, f"baseline_{baseline_id:02d}.ipynb")
    if not os.path.exists(src_nb):
        raise FileNotFoundError(f"Baseline notebook not found: {src_nb}")

    shutil.copy2(src_nb, os.path.join(tdir, "notebook_unexec.ipynb"))
    with open(os.path.join(tdir, "config.run.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"tag": tag, "seed": seed, "baseline_id": baseline_id, "baseline_folder": bfolder},
            f, indent=2, ensure_ascii=False,
        )
    return tdir


# =============== Notebook summary (context for LLM) ===============
def _summarize_notebook(nb_path: str, max_chars: int = 1200) -> str:
    """Lightweight textual summary for an .ipynb to provide LLM context."""
    if not os.path.exists(nb_path):
        return ""
    nb = nbformat.read(nb_path, as_version=4)
    lines = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            src = (cell.source or "").strip()
            if len(src) < 300:
                lines.append(src)
        elif cell.cell_type == "code":
            src = (cell.source or "").strip()
            if any(k in src for k in ["torch", "sklearn", "xgboost", "keras", "DataLoader"]):
                lines.append(src[:200])
    return "\n".join(lines)[:max_chars]


def _dump_llm_io(tdir: str, messages, raw_text: str):
    """Save messages and raw LLM response for debugging."""
    dbg_dir = os.path.join(tdir, "_llm_debug")
    os.makedirs(dbg_dir, exist_ok=True)
    with open(os.path.join(dbg_dir, "messages.json"), "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    with open(os.path.join(dbg_dir, "raw_response.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text or "")


# =============== JSON parse helper (loose) ===============
def _parse_json_loose(text: str):
    """Extract a JSON object from possibly fenced or slightly invalid JSON."""
    if not text:
        raise ValueError("empty response")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("no braces found")
    sub = t[s:e + 1]
    try:
        return json.loads(sub)
    except Exception:
        sub = re.sub(r",(\s*[}\]])", r"\1", sub)  # drop trailing commas
        return json.loads(sub)


# =============== Stable LLM call (JSON preferred) ===============
def _chat_json_stable(messages, api_key: str, base_url: str, model: str,
                      temperature: float = 0.2, max_tokens: int = 1800,
                      retries: int = 2, debug_dir: str = None):
    """
    Call an OpenAI-compatible /chat/completions endpoint and return parsed JSON.
    Tries response_format=json first, then falls back to plain text if needed.
    Also handles providers that put content in tool_calls.arguments.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for i in range(max(1, retries)):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
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

            # Fallback: remove response_format and retry once in plain text mode
            if not content:
                payload.pop("response_format", None)
                resp2 = requests.post(url, headers=headers, json=payload, timeout=90)
                data = resp2.json()
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
                    with open(os.path.join(debug_dir, f"raw_response_try{i+1}.json"), "w", encoding="utf-8") as fdbg:
                        json.dump(data, fdbg, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            if not content:
                raise RuntimeError("empty content in LLM response")

            # Try strict JSON first, then loose parsing
            try:
                return json.loads(content)
            except Exception:
                return _parse_json_loose(content)

        except Exception as e:
            last_err = e
            time.sleep(1.5)

    raise RuntimeError(f"LLM call failed after {retries} attempt(s): {last_err}")


# =============== Main: propose improvements for the baseline notebook ===============
def propose_notebook_improvements(
    cfg: dict,
    tdir: str,
    related_work_bullets: str,
    seed: Optional[int],
    llm,
    reference_summary: str,
    baseline_summary: str,
    require_llm: bool = True,
    llm_retries: int = 2,
):
    """
    Generate a minimal patch for the baseline notebook (only data preprocessing & model cells).
    The branch-level LLM knobs are read from cfg["baseline_branch"], with defaults:
      - temperature: 0.8
      - max_tokens: 20000
      - retries: 2
    """
    baseline_path = os.path.join(tdir, "notebook_unexec.ipynb")
    auto_summary = _summarize_notebook(baseline_path, 1200)

    user = NOTEBOOK_IMPROVEMENT_USER_TEMPLATE.format(
        baseline_path=baseline_path,
        related_work_bullets=(related_work_bullets or "(empty)"),
        reference_summary=(reference_summary or "(empty)"),
        baseline_summary=(baseline_summary or auto_summary or "(empty)"),
        seed=seed,
    )

    messages = [
        {"role": "system", "content": NOTEBOOK_IMPROVEMENT_SYSTEM},
        {"role": "user", "content": user + "\n\nReturn ONLY a JSON with keys: cells_to_add, cells_to_replace, rationale."}
    ]

    # LLM connection config
    if isinstance(llm, dict):
        model = llm.get("model", "gpt-5")
        api_key = llm.get("api_key")
        base_url = llm.get("base_url", "https://vip.yi-zhan.top/v1")
    else:
        model = getattr(llm, "model", "gpt-5")
        api_key = getattr(llm, "api_key", None)
        base_url = getattr(llm, "base_url", "https://vip.yi-zhan.top/v1")

    # Branch-level overrides (from config)
    branch_cfg = (cfg or {}).get("baseline_branch", {}) or {}
    temperature = branch_cfg.get("temperature", 0.8)
    max_tokens = branch_cfg.get("max_tokens", 20000)
    retries = branch_cfg.get("retries", 2)

    out_path = os.path.join(tdir, "llm_patch.json")
    last_err = None

    for _ in range(max(1, llm_retries)):
        try:
            spec = _chat_json_stable(
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
                debug_dir=os.path.join(tdir, "_llm_debug"),
            )

            _dump_llm_io(tdir, messages, json.dumps(spec, ensure_ascii=False, indent=2))

            if not isinstance(spec, dict):
                raise RuntimeError("LLM did not return a dict")
            if not (spec.get("cells_to_add") or spec.get("cells_to_replace")):
                raise RuntimeError("empty patch content")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)
            return out_path

        except Exception as e:
            last_err = e
            time.sleep(1.2)

    if require_llm:
        raise RuntimeError(f"LLM patch generation failed after {llm_retries} attempt(s): {last_err}")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"cells_to_add": [], "cells_to_replace": [], "rationale": "LLM disabled"}, f, indent=2)
        return out_path
