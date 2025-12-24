#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for run_cellscientist.

This module is intentionally dependency-light and contains:

- Project-root helpers
- JSON/text IO helpers
- Streamed subprocess runner + tee logging
- Small file-copy/export helpers

It is used by the refactored run_cellscientist.py to keep that file small.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Project root / CWD
# =============================================================================


def project_root() -> str:
    """Return the directory containing run_cellscientist.py (repo root)."""
    return os.path.dirname(os.path.abspath(__file__))


def ensure_project_cwd() -> None:
    """Force CWD to repo root so relative paths behave consistently."""
    root = project_root()
    marker = os.path.join(root, "Design_Analysis")
    if not os.path.exists(marker):
        print(
            f"[WARN] Script location '{root}' may not be project root (missing 'Design_Analysis').\n"
            "       Continuing anyway and forcing CWD to script directory."
        )
    os.chdir(root)


# =============================================================================
# IO helpers
# =============================================================================


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"❌ Error: Config file not found: {path}")
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_text(path: str) -> str:
    try:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def read_text_limited(path: str, *, max_chars: int = 120_000) -> str:
    raw = read_text(path)
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw
    head = raw[: max_chars // 2]
    tail = raw[-max_chars // 2 :]
    skipped = max(0, len(raw) - len(head) - len(tail))
    return head + f"\n\n... [TRUNCATED: {skipped} chars] ...\n\n" + tail


def read_head_tail_lines(
    path: str, *, head: int = 200, tail: int = 400, max_chars: int = 140_000
) -> str:
    raw = read_text(path)
    if not raw:
        return ""
    lines = raw.splitlines()
    if len(lines) <= head + tail:
        out = raw
    else:
        out = "\n".join(lines[:head]) + "\n\n... [SNIP] ...\n\n" + "\n".join(lines[-tail:])
    if len(out) > max_chars:
        out = out[:max_chars] + f"\n\n... [TRUNCATED to {max_chars} chars] ..."
    return out


# =============================================================================
# Tee logger
# =============================================================================


class TeeStream:
    """Write-through stream to multiple underlying streams."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]
        self.encoding = getattr(self.streams[0], "encoding", "utf-8") if self.streams else "utf-8"

    def write(self, data: str):
        if data is None:
            return 0
        n = 0
        for s in self.streams:
            try:
                n = s.write(data)
            except Exception:
                pass
        return n

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        for s in self.streams:
            try:
                if hasattr(s, "isatty") and s.isatty():
                    return True
            except Exception:
                continue
        return False

    def fileno(self):
        for s in self.streams:
            if hasattr(s, "fileno"):
                try:
                    return s.fileno()
                except Exception:
                    continue
        raise OSError("No underlying fileno")


def setup_logging(results_root: str) -> Tuple[str, str, Any]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(results_root, f"logs_{ts}")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"pipeline_{ts}.log")

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    log_fp = open(log_path, "a", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_fp)  # type: ignore
    sys.stderr = TeeStream(sys.__stderr__, log_fp)  # type: ignore
    print(f"📝 Logging console output to: {log_path}")
    return logs_dir, log_path, log_fp


def append_phase_header(
    phase_fp, dataset: str, phase_name: str, cmd: List[str], cwd: str
) -> None:
    if not phase_fp:
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phase_fp.write("\n" + "=" * 88 + "\n")
    phase_fp.write(f"[{ts}] dataset={dataset} | {phase_name}\n")
    phase_fp.write(f"cwd={cwd}\n")
    phase_fp.write(f"cmd={' '.join(cmd)}\n")
    phase_fp.write("=" * 88 + "\n")
    phase_fp.flush()


def run_cmd_streamed(
    cmd: List[str],
    *,
    cwd: str,
    phase_fp=None,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    env = os.environ.copy()
    if extra_env:
        for k, v in extra_env.items():
            if v is None:
                continue
            env[str(k)] = str(v)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except Exception:
            pass
        if phase_fp:
            try:
                phase_fp.write(line)
                phase_fp.flush()
            except Exception:
                pass
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


# =============================================================================
# Misc filesystem helpers
# =============================================================================


def results_root_for_dataset(dataset_name: str) -> str:
    return os.path.abspath(os.path.join(project_root(), "results", dataset_name))


def safe_copy(src: str, dst_dir: str, dst_name: Optional[str] = None) -> Optional[str]:
    try:
        if not src or not os.path.exists(src):
            return None
        os.makedirs(dst_dir, exist_ok=True)
        name = dst_name or os.path.basename(src)
        dst = os.path.join(dst_dir, name)
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def export_notebook_as_py(nb_path: str, out_py_path: str) -> bool:
    """Extract code cells into a .py for convenience."""
    try:
        import nbformat

        nb = nbformat.read(nb_path, as_version=4)
        parts: List[str] = []
        for cell in nb.cells:
            if cell.get("cell_type") == "code":
                src = cell.get("source") or ""
                if src.strip():
                    parts.append(src.rstrip() + "\n")
        code = "\n\n# ---- cell ----\n\n".join(parts)
        with open(out_py_path, "w", encoding="utf-8") as f:
            f.write(code)
        return True
    except Exception:
        return False
