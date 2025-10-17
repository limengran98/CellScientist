# -*- coding: utf-8 -*-
import os
import json
import re
import time
import glob
import shutil
import datetime
from typing import Optional, List
import nbformat
import requests

from .prompts import NOTEBOOK_IMPROVEMENT_SYSTEM, NOTEBOOK_IMPROVEMENT_USER_TEMPLATE


# =========================
# Baseline selection
# =========================
def _latest_baseline_folder(root: str) -> str:
    """Return the latest-dated folder under <root>/baselines."""
    bdir = os.path.join(root, "baselines")
    subs = sorted([p for p in glob.glob(os.path.join(bdir, "*")) if os.path.isdir(p)])
    if not subs:
        raise RuntimeError("No baselines found. Run migration first.")
    return subs[-1]


def create_trial_from_baseline(root: str, tag: str, baseline_id: int = 0, seed: Optional[int] = None) -> str:
    """
    Create a new trial directory from baseline_<id>.ipynb.
    Keep signature as pipeline expects: (root, tag, baseline_id, seed)
    """
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


# =========================
# Notebook summarization
# =========================
def _summarize_notebook(nb_path: str, max_chars: int = 1200) -> str:
    """Extract headings and short code cues from baseline notebook."""
    if not os.path.exists(nb_path):
        return ""
    nb = nbformat.read(nb_path, as_version=4)
    md_lines, code_cues = [], []
    for c in nb.cells:
        if c.cell_type == "markdown":
            src = (c.source or "").strip()
            for line in src.splitlines():
                t = line.strip()
                if t.startswith(("#", "-", "*")) and len(t) <= 200:
                    md_lines.append(t)
        elif c.cell_type == "code":
            src = (c.source or "").strip()
            if any(k in src for k in ["torch", "tensorflow", "sklearn", "xgboost", "keras", "MLP", "RandomForest", "LogisticRegression"]):
                code_cues.append(" ".join(src.split())[:220])
    text = []
    if md_lines:
        text += ["## Baseline headings", "\n".join(md_lines[:50])]
    if code_cues:
        text += ["\n## Baseline code cues", "\n".join(code_cues[:40])]
    out = "\n".join(text)
    return out[:max(100, max_chars)]


# =========================
# Debug dump
# =========================
def _dump_llm_io(tdir: str, messages, raw_text: str):
    dbg_dir = os.path.join(tdir, "_llm_debug")
    os.makedirs(dbg_dir, exist_ok=True)
    with open(os.path.join(dbg_dir, "messages.json"), "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    with open(os.path.join(dbg_dir, "raw_response.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text or "")


# =========================
# JSON loose parsing
# =========================
def _parse_json_loose(text: str):
    """Extract a JSON object from possibly fenced/verbose text."""
    if not text:
        raise ValueError("empty response")
    t = text.strip()
    # strip markdown fences
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.DOTALL).strip()
    s = t.find("{"); e = t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("no json object braces found")
    cand = t[s:e+1].strip()
    try:
        return json.loads(cand)
    except Exception:
        # remove trailing commas
        cand2 = re.sub(r",(\s*[}\]])", r"\1", cand)
        return json.loads(cand2)


# =========================
# Phase-1 style stable chat (embedded)
# =========================
def _chat_json_local(messages,
                     api_key: Optional[str],
                     base_url: str = "https://vip.yi-zhan.top/v1",
                     model: str = "gpt-5",
                     temperature: float = 0.2,
                     max_tokens: int = 1800,
                     retries: int = 2,
                     enforce_json: bool = True):
    """
    Phase-1 proven-stable direct HTTP call to OpenAI-compatible endpoint.
    - Tries response_format=json_object when supported.
    - Falls back to plain text and loose JSON parsing.
    - Tries multiple header conventions for maximum compatibility.
    Returns a dict (parsed JSON).
    """
    if not api_key:
        raise RuntimeError("LLM api_key is missing. Set in config.llm.api_key or env OPENAI_API_KEY.")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if enforce_json:
        payload["response_format"] = {"type": "json_object"}

    def _do_post(headers):
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        return r

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # 1) standard Authorization: Bearer <key>
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            r = _do_post(headers)
            if r.status_code == 401:
                # 2) Some gateways require X-API-Key
                headers2 = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "X-API-Key": api_key,
                }
                r = _do_post(headers2)
            if r.status_code == 401:
                # 3) Rare cases: Authorization without Bearer + openai-api-key
                headers3 = {
                    "Content-Type": "application/json",
                    "Authorization": api_key,
                    "openai-api-key": api_key,
                }
                r = _do_post(headers3)

            if r.status_code != 200:
                raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:300]}")

            data = r.json()
            ch = (data.get("choices") or [{}])[0]
            # Prefer message.content, fallback to text
            msg = ch.get("message") or {}
            content = (msg.get("content") or ch.get("text") or "").strip()
            if not content:
                raise RuntimeError("empty content in LLM response")

            # Try parse as JSON dict; if fails, use loose parser then
            try:
                return json.loads(content)
            except Exception:
                return _parse_json_loose(content)

        except Exception as e:
            last_err = e
            time.sleep(1.2)

    raise RuntimeError(f"LLM call failed after {retries} attempt(s): {last_err}")


# =========================
# Patch generation (LLM)
# =========================
def propose_notebook_improvements(
    tdir: str,
    related_work_bullets: str,
    seed: Optional[int],
    llm,  # can be dict or an object with attributes api_key/base_url/model
    reference_summary: str,
    baseline_summary: str,
    require_llm: bool = True,
    llm_retries: int = 2,
) -> str:
    """
    Ask LLM for a JSON patch that ONLY changes data/model parts.
    - Uses embedded Phase-1 style _chat_json_local (no external imports).
    - Strictly keeps evaluation/splitting/metrics untouched.
    """
    baseline_path = os.path.join(tdir, "notebook_unexec.ipynb")
    baseline_auto = _summarize_notebook(baseline_path, max_chars=1200)

    user = NOTEBOOK_IMPROVEMENT_USER_TEMPLATE.format(
        baseline_path=baseline_path,
        related_work_bullets=(related_work_bullets or "(empty)"),
        reference_summary=(reference_summary or "(empty)"),
        baseline_summary=(baseline_summary or baseline_auto or "(empty)"),
        seed=seed,
    )
    # Strong instruction: return ONLY JSON object
    messages = [
        {"role": "system", "content": NOTEBOOK_IMPROVEMENT_SYSTEM},
        {"role": "user", "content": user + "\n\nReturn ONLY a JSON object with keys: cells_to_add, cells_to_replace, rationale, dependencies."},
    ]

    # pick llm config
    if isinstance(llm, dict):
        model = llm.get("model") or os.environ.get("OPENAI_MODEL", "gpt-5")
        api_key = llm.get("api_key") or os.environ.get("OPENAI_API_KEY")
        base_url = llm.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://vip.yi-zhan.top/v1")
    else:
        model = getattr(llm, "model", None) or os.environ.get("OPENAI_MODEL", "gpt-5")
        api_key = getattr(llm, "api_key", None) or os.environ.get("OPENAI_API_KEY")
        base_url = getattr(llm, "base_url", None) or os.environ.get("OPENAI_BASE_URL", "https://vip.yi-zhan.top/v1")

    last_exc = None
    out_path = os.path.join(tdir, "llm_patch.json")

    for _ in range(max(1, llm_retries)):
        try:
            spec = _chat_json_local(
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.2,
                max_tokens=2000,
                retries=2,
                enforce_json=True,
            )
            # Save raw spec for debug
            _dump_llm_io(tdir, messages, json.dumps(spec, ensure_ascii=False, indent=2))

            if not isinstance(spec, dict):
                raise RuntimeError("LLM did not return a JSON object.")

            # normalize fields
            cells_add = spec.get("cells_to_add") or []
            cells_rep = spec.get("cells_to_replace") or []
            # allow alternative field 'edits' -> convert to replace list
            if not cells_add and not cells_rep and spec.get("edits"):
                tmp = []
                for ed in spec["edits"]:
                    idx = ed.get("cell_index")
                    src = ed.get("source")
                    if isinstance(idx, int) and isinstance(src, str):
                        tmp.append({"index": idx, "new_source": src})
                spec = {"cells_to_add": [], "cells_to_replace": tmp, "rationale": spec.get("rationale", "")}
                cells_add, cells_rep = [], tmp

            if not cells_add and not cells_rep:
                raise RuntimeError("LLM returned an empty patch (no cells_to_add/replace).")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)
            return out_path

        except Exception as e:
            last_exc = e
            time.sleep(1.2)

    if require_llm:
        raise RuntimeError(f"LLM patch generation failed after {llm_retries} attempt(s): {last_exc}")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"cells_to_add": [], "cells_to_replace": [], "rationale": "empty (LLM disabled)"}, f, indent=2, ensure_ascii=False)
        return out_path
