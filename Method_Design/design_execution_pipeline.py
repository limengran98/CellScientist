#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, glob, shutil, argparse, datetime
import nbformat
from nbclient import NotebookClient
from design_execution.baseline_migrator import migrate_reference_ipynb
from design_execution.literature_manager import (init_scaffold, openalex_search, save_rows_to_csv, csv_to_md, llm_summarize_literature)
from design_execution.trial_manager import create_trial_from_baseline, propose_notebook_improvements
from design_execution.llm_client import LLMClient
from design_execution.patch_applier import apply_patch
from design_execution.report_builder import build_markdown_report
from design_execution.context_extractor import summarize_folder_ipynb, summarize_notebook

def log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)

def _latest(path: str) -> str:
    subs = [p for p in glob.glob(os.path.join(path, "*")) if os.path.isdir(p)]
    return sorted(subs)[-1] if subs else None

def _copy_stage1_to_reference_pool(cfg: dict):
    src = cfg["paths"].get("stage1_analysis_dir")
    root = cfg["paths"]["design_execution_root"]
    dst = os.path.join(root, "reference_pool")
    if not src or not os.path.isdir(src):
        return 0, dst
    os.makedirs(dst, exist_ok=True)
    count = 0
    for p in sorted(glob.glob(os.path.join(src, "*.ipynb"))):
        shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
        count += 1
    return count, dst

def _run_literature(cfg: dict, enable: bool):
    root = cfg["paths"]["design_execution_root"]
    lit_root = os.path.join(root, "literature")
    os.makedirs(lit_root, exist_ok=True)
    if not enable:
        log("LIT", "Disabled by flag; skipped search.")
        open(os.path.join(lit_root, "auto_sections.md"), "w", encoding="utf-8").write("")
        return lit_root, ""
    lit = cfg.get("literature", {}) or {}
    query = (lit.get("query") or "").strip()
    n = int(lit.get("n", 10))
    llm_sum = bool(lit.get("llm_summarize", True))
    if not query:
        log("LIT", "No query configured; skipped search.")
        open(os.path.join(lit_root, "auto_sections.md"), "w", encoding="utf-8").write("")
        return lit_root, ""
    rows = openalex_search(query, per_page=n)
    if rows:
        csv_path = os.path.join(lit_root, "papers.csv")
        save_rows_to_csv(rows, csv_path)
        csv_to_md(csv_path, os.path.join(lit_root, "auto_sections.md"))
        log("LIT", f"Fetched {len(rows)} items from OpenAlex.")
    else:
        open(os.path.join(lit_root, "auto_sections.md"), "w", encoding="utf-8").write("")
        log("LIT", "No results from OpenAlex; wrote empty auto_sections.md")
    bullets = ""
    if llm_sum:
        try:
            syn = llm_summarize_literature(lit_root, query, cfg.get("llm", {}))
            bullets = open(syn, "r", encoding="utf-8").read()
            log("LIT", f"Wrote synthesis to {syn}")
        except Exception as e:
            log("LIT", f"Synthesis failed: {e}")
    return lit_root, bullets

def _prepare_context(cfg: dict, baseline_dir: str, baseline_id: int):
    root = cfg["paths"]["design_execution_root"]
    ref_pool = os.path.join(root, "reference_pool")
    reference_summary = summarize_folder_ipynb(ref_pool, max_chars=1200) if os.path.isdir(ref_pool) else ""
    baseline_nb = os.path.join(baseline_dir, f"baseline_{baseline_id:02d}.ipynb")
    baseline_summary = summarize_notebook(baseline_nb, max_chars=1000) if os.path.exists(baseline_nb) else ""
    return {"reference_summary": reference_summary, "baseline_summary": baseline_summary, "baseline_nb": baseline_nb}

def _exec_nb(in_nb: str, out_nb: str):
    nb = nbformat.read(in_nb, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3", allow_errors=True).execute()
    nbformat.write(nb, out_nb)

def cmd_generate(cfg: dict, baseline_id: int, with_lit: bool):
    paths = cfg["paths"]
    root = paths["design_execution_root"]
    moved, ref_dir = _copy_stage1_to_reference_pool(cfg)
    if moved:
        log("INIT", f"Copied {moved} Stage-1 notebooks into reference_pool (reference only): {ref_dir}")
    bdir = migrate_reference_ipynb(paths["baseline_source_dir"], root, include_globs=["Baseline_*.ipynb"])
    log("MIGRATE", f"Baselines copied -> {bdir}")
    lit_root, bullets = _run_literature(cfg, enable=with_lit)
    ctx = _prepare_context(cfg, bdir, baseline_id)
    tag = (cfg.get("trial", {}) or {}).get("tag", "improve-data-model")
    seed = (cfg.get("trial", {}) or {}).get("seed", 22)
    tdir = create_trial_from_baseline(root, tag, baseline_id, seed)
    log("TRIAL", f"Trial created -> {tdir}")
    llm_conf = cfg.get("llm", {}) or {}
    llm = LLMClient(**llm_conf)
    patch_path = propose_notebook_improvements(
        tdir,
        related_work_bullets=bullets,
        seed=seed,
        llm=llm,
        reference_summary=ctx["reference_summary"],
        baseline_summary=ctx["baseline_summary"],
        require_llm=True,
        llm_retries=int(llm_conf.get("retries", 2)),
    )
    log("IMPROVE", f"LLM patch -> {patch_path}")
    patched = apply_patch(os.path.join(tdir, "notebook_unexec.ipynb"), patch_path)
    log("PATCH", f"Patched notebook -> {patched}")
    log("DONE", "Generate phase completed (no execution).")
    return {"baseline_dir": bdir, "trial_dir": tdir, "patched_nb": patched}

def cmd_execute(cfg: dict, baseline_id: int, which: str, trial_dir: str = None):
    paths = cfg["paths"]
    root = paths["design_execution_root"]
    bdir = _latest(os.path.join(root, "baselines"))
    if not bdir:
        raise RuntimeError("No baselines found. Run `generate` first.")
    baseline_nb = os.path.join(bdir, f"baseline_{baseline_id:02d}.ipynb")
    if not os.path.exists(baseline_nb):
        raise FileNotFoundError(f"baseline not found: {baseline_nb}")
    if which in ("baseline", "both"):
        b_exec = baseline_nb.replace(".ipynb", "_exec.ipynb")
        _exec_nb(baseline_nb, b_exec)
        log("EXEC_BASE", f"Baseline executed -> {b_exec}")
    if which in ("patched", "both"):
        if not trial_dir:
            tdir = _latest(os.path.join(root, "trials"))
            if not tdir:
                raise RuntimeError("No trial found. Run `generate` first.")
        else:
            tdir = trial_dir
        patched_nb = os.path.join(tdir, "notebook_unexec_patched.ipynb")
        if not os.path.exists(patched_nb):
            patched_nb = os.path.join(tdir, "notebook_unexec.ipynb")
        patched_exec = patched_nb.replace(".ipynb", "_exec.ipynb")
        _exec_nb(patched_nb, patched_exec)
        log("EXEC_TRIAL", f"Patched executed -> {patched_exec}")
    log("DONE", f"Execute phase completed ({which}).")

def cmd_analyze(cfg: dict, trial_dir: str = None, baseline_id: int = 0):
    paths = cfg["paths"]
    root = paths["design_execution_root"]
    bdir = _latest(os.path.join(root, "baselines"))
    if not bdir:
        raise RuntimeError("No baselines found.")
    baseline_date = os.path.basename(bdir)
    if not trial_dir:
        tdir = _latest(os.path.join(root, "trials"))
        if not tdir:
            raise RuntimeError("No trial found.")
    else:
        tdir = trial_dir
    trial_name = os.path.basename(tdir)
    try:
        rpt = build_markdown_report(root, trial_name, None)
    except TypeError:
        rpt = build_markdown_report(root, trial_name, None, None)
    log("REPORT", f"Report -> {rpt}")
    log("DONE", "Analyze phase completed.")
    return rpt

def main():
    ap = argparse.ArgumentParser(description="CellScientist design_execution phased pipeline")
    ap.add_argument("--config", required=True, help="path to config JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_g = sub.add_parser("generate", help="Phase-1: LLM patch generation only (no execution)")
    ap_g.add_argument("--baseline-id", type=int, default=0)
    ap_g.add_argument("--with-lit", action="store_true", help="enable literature search & synthesis")
    ap_e = sub.add_parser("execute", help="Phase-2: execute baseline and/or patched")
    ap_e.add_argument("--baseline-id", type=int, default=0)
    ap_e.add_argument("--which", choices=["baseline", "patched", "both"], default="both")
    ap_e.add_argument("--trial-dir", type=str, default=None)
    ap_a = sub.add_parser("analyze", help="Phase-3: analyze & report (no execution)")
    ap_a.add_argument("--baseline-id", type=int, default=0)
    ap_a.add_argument("--trial-dir", type=str, default=None)
    ap_r = sub.add_parser("run", help="One-command: generate → execute → analyze")
    ap_r.add_argument("--baseline-id", type=int, default=0)
    ap_r.add_argument("--with-lit", action="store_true")
    ap_r.add_argument("--which", choices=["baseline", "patched", "both"], default="both")
    args = ap.parse_args()
    cfg = json.loads(open(args.config, "r", encoding="utf-8").read())
    if args.cmd == "generate":
        cmd_generate(cfg, baseline_id=args.baseline_id, with_lit=args.with_lit)
    elif args.cmd == "execute":
        cmd_execute(cfg, baseline_id=args.baseline_id, which=args.which, trial_dir=args.trial_dir)
    elif args.cmd == "analyze":
        cmd_analyze(cfg, baseline_id=args.baseline_id, trial_dir=args.trial_dir)
    elif args.cmd == "run":
        ret = cmd_generate(cfg, baseline_id=args.baseline_id, with_lit=args.with_lit)
        cmd_execute(cfg, baseline_id=args.baseline_id, which=args.which, trial_dir=ret["trial_dir"])
        cmd_analyze(cfg, baseline_id=args.baseline_id, trial_dir=ret["trial_dir"])
if __name__ == "__main__":
    main()
