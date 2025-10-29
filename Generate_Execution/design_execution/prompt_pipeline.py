# -*- coding: utf-8 -*-
"""
Prompt-defined pipeline orchestrator (Notebook mode).

Phases:
  1) prompt_generate -> build notebook from pipeline_prompt.yaml (Stage-1 ref prepended as markdown if enabled)
  2) prompt_execute  -> run notebook with auto-fix
  3) prompt_analyze  -> read-only analysis (no re-execution)

This keeps previous behavior and only adds Stage-1 markdown as the first cell (when enabled).
"""

from __future__ import annotations
import os, json, glob, datetime as _dt
from typing import Any, Dict, Optional
import nbformat

from pathlib import Path

# Local helpers
try:
    from .prompt_builder import generate_notebook_from_prompt
    from .prompt_viz import write_hypergraph_viz
    from .nb_autofix import execute_with_autofix
except Exception:
    # Fallback for relative runs
    from prompt_builder import generate_notebook_from_prompt  # type: ignore
    from prompt_viz import write_hypergraph_viz  # type: ignore
    from nb_autofix import execute_with_autofix  # type: ignore


# ---------- Output roots ----------

def _prompt_out_root(cfg: Dict[str, Any]) -> str:
    return cfg.get("prompt_branch", {}).get("save_root", cfg["paths"]["design_execution_root"])

def _latest_prompt_dir(cfg: Dict[str, Any]) -> Optional[str]:
    root = os.path.join(_prompt_out_root(cfg), "prompt")
    subs = sorted([p for p in glob.glob(os.path.join(root, "prompt_run_*")) if os.path.isdir(p)])
    return subs[-1] if subs else None


# ---------- Phases ----------

def prompt_generate(cfg: Dict[str, Any], spec_path: str) -> Dict[str, Any]:
    """
    Only generate the prompt-defined artifact (notebook), do not execute.
    """
    debug_dir = os.path.join(cfg["paths"]["design_execution_root"], "debug_prompt")
    out_root = _prompt_out_root(cfg)
    os.makedirs(os.path.join(out_root, "prompt"), exist_ok=True)

    nb, _user_prompt = generate_notebook_from_prompt(cfg, spec_path, debug_dir)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tdir = os.path.join(out_root, "prompt", f"prompt_run_{ts}")
    os.makedirs(tdir, exist_ok=True)

    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return {"trial_dir": tdir, "artifact": nb_path}


def prompt_execute(cfg: Dict[str, Any], trial_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the latest prompt artifact with auto-fix (known patches + optional LLM).
    """
    tdir = trial_dir or _latest_prompt_dir(cfg)
    if not tdir:
        raise RuntimeError("No prompt trial found. Run 'generate' first.")

    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    if not os.path.exists(nb_path):
        raise RuntimeError(f"notebook_prompt.ipynb not found in {tdir}")

    exec_cfg = (cfg.get("exec") or {})
    timeout = int(exec_cfg.get("timeout_seconds", 1800))
    max_fix_rounds = int(exec_cfg.get("max_fix_rounds", 1))
    max_cell_retries = int(exec_cfg.get("max_cell_retries", 2))

    print(f"[PROMPT] exec config -> timeout_seconds={timeout}, max_fix_rounds={max_fix_rounds}, max_cell_retries={max_cell_retries}")
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
        phase_cfg=cfg,
        preserve_source_in_exec=True,
        save_intermediates=True,
    )
    print(f"[PROMPT] executed notebook -> {final_exec_path}")

    # Read metrics if available
    metrics_path = os.path.join(tdir, "metrics.json")
    metrics: Dict[str, Any] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[PROMPT][WARN] failed to read metrics.json: {e}")
    else:
        print(f"[PROMPT][WARN] metrics.json not found in {tdir}")

    # Hypergraph viz
    try:
        if bool(exec_cfg.get("enable_hypergraph_viz", True)):
            viz = write_hypergraph_viz(tdir, nb_path, fmt=str(exec_cfg.get("viz_format", "mermaid")))
            if viz:
                print(f"[PROMPT] hypergraph viz written: {viz}")
    except Exception as e:
        print(f"[PROMPT][WARN] hypergraph viz generation failed: {e}")

    return {"trial_dir": tdir, "metrics": metrics, "exec_notebook": final_exec_path}


def prompt_analyze(cfg: Dict[str, Any], trial_dir: str) -> Dict[str, Any]:
    """
    Read-only analyze phase:
      - Do NOT execute or auto-fix here.
      - If metrics.json is missing / invalid / empty, print the reason and return a report without metrics.
    """
    print("\n[INFO] === Analyzing results (read-only) ===")
    metrics_path = os.path.join(trial_dir, "metrics.json")

    report: Dict[str, Any] = {
        "trial_dir": trial_dir,
        "has_metrics": False,
        "metrics_keys": [],
        "reason": None,
        "metrics": {}
    }

    if not os.path.exists(metrics_path):
        reason = f"metrics.json not found in {trial_dir}; analyze is read-only and will NOT re-execute."
        print(f"[WARN] {reason}")
        report["reason"] = "missing_metrics_json"
        try:
            with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return report

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_obj = json.load(f)
    except Exception as e:
        reason = f"failed to read/parse metrics.json: {e}"
        print(f"[WARN] {reason}")
        report["reason"] = "invalid_metrics_json"
        try:
            with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return report

    if not isinstance(metrics_obj, dict) or not metrics_obj:
        reason = "metrics.json loaded but empty or not a dict; analyze will NOT re-execute."
        print(f"[WARN] {reason}")
        report["reason"] = "empty_or_invalid_metrics"
        try:
            with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return report

    report["has_metrics"] = True
    report["metrics_keys"] = list(metrics_obj.keys())
    report["metrics"] = metrics_obj
    report["reason"] = "ok"
    print(f"[INFO] Loaded metrics from {metrics_path}: keys={report['metrics_keys']}")

    primary = ((cfg.get("experiment") or {}).get("primary_metric")
               or (cfg.get("improve") or {}).get("primary_metric"))
    if primary and primary not in metrics_obj:
        print(f"[WARN] primary metric '{primary}' not found in metrics.json keys.")

    try:
        with open(os.path.join(trial_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return report


# ---------- Unified entrypoint ----------

def run_prompt_pipeline(cfg: Dict[str, Any], spec_path: str) -> Dict[str, Any]:
    """
    Orchestrate the 3 phases and return outputs:
      { "trial_dir": ..., "exec_notebook": ..., "metrics": ..., "report": {...} }
    """
    print("[INFO] === Generating prompt notebook ===")
    g = prompt_generate(cfg, spec_path)

    print("[INFO] === Executing prompt notebook ===")
    e = prompt_execute(cfg, g["trial_dir"])

    print("[INFO] === Analyzing results ===")
    trial_for_analyze = e.get("trial_dir", g["trial_dir"])
    a = prompt_analyze(cfg, trial_for_analyze)

    merged_metrics = a.get("metrics") or e.get("metrics") or {}
    return {
        "trial_dir": trial_for_analyze,
        "exec_notebook": e.get("exec_notebook"),
        "metrics": merged_metrics,
        "report": a,
    }
