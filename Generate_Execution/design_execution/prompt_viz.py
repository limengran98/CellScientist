# -*- coding: utf-8 -*-
"""
Hypergraph visualization helpers for prompt-defined notebooks.
"""

from __future__ import annotations
import os, json
from typing import Dict
import nbformat


def write_hypergraph_viz(trial_dir: str, nb_path: str, fmt: str = "mermaid") -> Dict[str, str]:
    """
    Emit hypergraph.json and hypergraph.md (mermaid) if notebook has metadata.execution.hypergraph.
    Returns a dict of written paths {'json': ..., 'mermaid': ...} (best-effort).
    """
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
                if len(purpose) > 60:
                    purpose = purpose[:57] + "..."
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
