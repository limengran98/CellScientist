
# -*- coding: utf-8 -*-
"""
Resolve and summarize Stage-1 artifacts for Phase-2.
- Finds REFERENCE_* files under cfg["paths"]["stage1_analysis_dir"]
- Exposes prepare_stage1_context(cfg, enable=True) -> {"h5_path": str, "markdown": str}
- Also sets os.environ["STAGE1_H5_PATH"] when h5 exists.
"""
from __future__ import annotations
import os, glob, json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import h5py  # type: ignore
except Exception:
    h5py = None

def _find_reference(dirpath: str) -> Dict[str, Optional[str]]:
    d = Path(dirpath).resolve()
    nb = None; h5 = None
    if d.exists():
        nbs = sorted(d.glob('REFERENCE_*.ipynb'))
        if nbs: nb = str(nbs[-1])
        h5s = [p for p in d.glob('REFERENCE_DATA.h5')] + [p for p in d.glob('*.h5') if 'REFERENCE' in p.name.upper()]
        if h5s: h5 = str(sorted(h5s)[-1])
    return {"notebook": nb, "h5": h5}

def _summarize_h5(h5_path: str) -> Dict[str, Any]:
    if not h5py or not h5_path or not Path(h5_path).exists():
        return {}
    info = {}
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        info['keys'] = keys
        info['dsets'] = {k: {"shape": tuple(f[k].shape), "dtype": str(f[k].dtype)} for k in keys}
    return info

def _mk_markdown(stage1_dir: str, nb: Optional[str], h5: Optional[str], h5_info: Dict[str, Any]) -> str:
    lines = []
    lines.append("## Stage-1 Reference Context")
    lines.append(f"- Reference dir: `{stage1_dir}`")
    if nb: lines.append(f"- Notebook: `{nb}`")
    if h5:
        lines.append(f"- H5: `{h5}`")
        if h5_info.get('keys'):
            lines.append("- H5 keys: " + ", ".join(h5_info['keys'][:12]) + (" ..." if len(h5_info['keys'])>12 else ""))
    else:
        lines.append("- H5: (not found, pipeline should fallback to raw CSV)")
    return "\n".join(lines)

def prepare_stage1_context(cfg: dict, enable: bool = True) -> Optional[Dict[str, Any]]:
    if not enable: 
        return None
    stage1_dir = ((cfg.get('paths') or {}).get('stage1_analysis_dir') or '').strip()
    if not stage1_dir:
        print('[STAGE1][WARN] paths.stage1_analysis_dir not configured.')
        return None
    refs = _find_reference(stage1_dir)
    h5_info = _summarize_h5(refs.get('h5') or '')
    md = _mk_markdown(stage1_dir, refs.get('notebook'), refs.get('h5'), h5_info)
    if refs.get('h5') and Path(refs['h5']).exists():
        os.environ['STAGE1_H5_PATH'] = refs['h5']
    return {'h5_path': refs.get('h5'), 'notebook': refs.get('notebook'), 'markdown': md, 'h5_info': h5_info}
