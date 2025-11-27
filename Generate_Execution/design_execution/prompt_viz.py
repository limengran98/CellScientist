# design_execution/prompt_viz.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import Dict
import nbformat

def write_hypergraph_viz(trial_dir: str, nb_path: str, fmt: str = "mermaid") -> Dict[str, str]:
    """
    Reads notebook metadata to generate hypergraph visualization artifacts.
    Produces:
      - hypergraph.json: Raw structure
      - hypergraph.md: Mermaid flowchart
    """
    out: Dict[str, str] = {}
    try:
        nb = nbformat.read(nb_path, as_version=4)
        
        # 1. Extract Global Metadata
        ex = ((nb.metadata or {}).get("execution") or {})
        hg = ex.get("hypergraph") or {}
        
        # 2. Extract Cell Metadata
        cells = []
        for c in nb.cells:
            # We look for 'subtask' metadata injected by prompt_generator
            meta = c.get("metadata", {})
            st = meta.get("subtask")
            if st and isinstance(st, dict) and st.get("id"):
                cells.append({
                    "id": st.get("id"), 
                    "name": st.get("name") or "", 
                    "purpose": st.get("purpose") or ""
                })

        # Dedup cells by ID
        seen = set()
        uniq_cells = []
        for c in cells:
            if c["id"] not in seen:
                uniq_cells.append(c)
                seen.add(c["id"])

        os.makedirs(trial_dir, exist_ok=True)

        # 3. Write JSON
        jpath = os.path.join(trial_dir, "hypergraph.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump({"cells": uniq_cells, "hypergraph": hg}, f, ensure_ascii=False, indent=2)
        out["json"] = jpath

        # 4. Write Mermaid
        if fmt == "mermaid":
            lines = ["```mermaid", "flowchart TD"]
            
            # Nodes
            for c in uniq_cells:
                # Sanitize text
                purpose = (c.get("purpose") or "").replace("\n", " ").replace('"', "'")
                if len(purpose) > 60: purpose = purpose[:57] + "..."
                label = (c.get("name") or c["id"]).replace('"', "'")
                
                lines.append(f'    {c["id"]}["{c["id"]}: {label}\\n{purpose}"]')
            
            # Edges
            edges = hg.get("hyperedges") or []
            for e in edges:
                head = e.get("head")
                tails = e.get("tail")
                # Handle both string and list formats for tail
                if isinstance(tails, str): tails = [tails]
                
                if head and tails:
                    for t in tails:
                        lines.append(f"    {t} --> {head}")
            
            lines.append("```")
            mpath = os.path.join(trial_dir, "hypergraph.md")
            with open(mpath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            out["mermaid"] = mpath
            
    except Exception as e:
        # Robustness: Write error log but don't crash pipeline
        err_path = os.path.join(trial_dir, "hypergraph_viz_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(str(e))
            
    return out