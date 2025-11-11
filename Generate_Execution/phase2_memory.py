
#!/usr/bin/env python3
# Simplified Phase-2 memory API bridging to design_execution.stage1_adapter
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Dict, Optional
try:
    from design_execution.stage1_adapter import prepare_stage1_context
except Exception as e:
    raise RuntimeError("Missing design_execution.stage1_adapter; run patched Phase-2 tree.") from e

def get_phase1_memory(cfg: dict) -> Dict[str, Any]:
    s1 = prepare_stage1_context(cfg, enable=True) or {}
    return s1

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mem = get_phase1_memory(cfg)
    print(json.dumps(mem, ensure_ascii=False, indent=2))
