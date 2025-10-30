# -*- coding: utf-8 -*-
"""
Context extractor (text-only for Stage-1).
- Strictly reads text-like artifacts from Stage-1 outputs and returns a compact
  summary string for prompt context.
- Code cells and notebooks are not parsed here; we only consume .md/.txt/.json summaries.
"""

from __future__ import annotations
import os, json, glob
from typing import List, Tuple


TEXT_EXTS = {".md", ".txt"}
JSON_FILES_HINTS = (
    "analysis_summary.json",
    "summary.json",
    "report.json",
    "metrics.json",
)


def _safe_read(path: str, limit: int = 20000) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        if len(s) > limit:
            s = s[:limit] + "\n\n...[TRUNCATED]..."
        return s
    except Exception:
        return ""


def _collect_text_files(reference_dir: str) -> List[str]:
    if not reference_dir or not os.path.isdir(reference_dir):
        return []
    files: List[str] = []
    for pat in ("*.md", "*.txt"):
        files.extend(glob.glob(os.path.join(reference_dir, pat)))
    # prefer auto_sections.md if exists
    auto_md = os.path.join(reference_dir, "auto_sections.md")
    if os.path.exists(auto_md):
        files = [auto_md] + [f for f in files if f != auto_md]
    return files


def _collect_json_summaries(reference_dir: str) -> List[str]:
    if not reference_dir or not os.path.isdir(reference_dir):
        return []
    out: List[str] = []
    for name in JSON_FILES_HINTS:
        p = os.path.join(reference_dir, name)
        if os.path.exists(p):
            out.append(p)
    # also any *.json at top level (best-effort)
    for p in glob.glob(os.path.join(reference_dir, "*.json")):
        if p not in out:
            out.append(p)
    return out


def _json_brief(path: str, limit: int = 2000) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return ""
    # try common keys in stage-1 summaries
    for k in ("summary", "highlights", "bullets", "notes"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            txt = v.strip()
            return txt if len(txt) <= limit else (txt[:limit] + " ...")
        if isinstance(v, list):
            items = [str(x) for x in v if isinstance(x, (str, int, float))]
            if items:
                txt = "- " + "\n- ".join(items)
                return txt if len(txt) <= limit else (txt[:limit] + " ...")
    # fallback: first-level keys dump (trimmed)
    try:
        dump = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return ""
    return dump if len(dump) <= limit else (dump[:limit] + " ...")


def summarize_stage1_text(reference_dir: str, max_chars: int = 1600) -> str:
    """
    Returns a text-only Stage-1 summary suitable for a Markdown cell.
    Order: auto_sections.md > *.md/*.txt (first N) > selected JSON summaries.
    """
    if not reference_dir or not os.path.isdir(reference_dir):
        return ""

    parts: List[str] = []

    # Prefer authored/LLM-synthesized markdown
    text_files = _collect_text_files(reference_dir)
    for i, p in enumerate(text_files):
        s = _safe_read(p, limit=max_chars // 2)
        if s:
            parts.append(f"### From {os.path.basename(p)}\n{s}")
        if sum(len(x) for x in parts) >= max_chars:
            break

    # Add JSON brief summaries (selected keys)
    if sum(len(x) for x in parts) < max_chars:
        for p in _collect_json_summaries(reference_dir):
            brief = _json_brief(p, limit=max_chars // 3)
            if brief:
                parts.append(f"### From {os.path.basename(p)} (JSON)\n{brief}")
            if sum(len(x) for x in parts) >= max_chars:
                break

    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[TRUNCATED]..."
    return text
