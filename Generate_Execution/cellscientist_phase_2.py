#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os, sys, json, glob, shutil, argparse, datetime, re
import nbformat
from nbclient import NotebookClient

from design_execution.baseline_migrator import migrate_reference_ipynb
from design_execution.literature_manager import (
    openalex_search, save_rows_to_csv, csv_to_md, llm_summarize_literature
)
from design_execution.trial_manager import create_trial_from_baseline, propose_notebook_improvements
from design_execution.llm_client import LLMClient, resolve_llm_from_cfg
from design_execution.patch_applier import apply_patch
from design_execution.report_builder import build_markdown_report
from design_execution.prompt_pipeline import run_prompt_pipeline as _prompt_run
from design_execution.context_extractor import summarize_folder_ipynb, summarize_notebook
from design_execution.unified_pipeline import run_unified_pipeline

try:
    from design_execution.prompt_pipeline import prompt_generate as _prompt_generate
    from design_execution.prompt_pipeline import prompt_execute as _prompt_execute
    from design_execution.prompt_pipeline import prompt_analyze as _prompt_analyze
except Exception:
    _prompt_generate = _prompt_execute = _prompt_analyze = None


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
    print(f"[INFO] Copied {count} Stage-1 notebooks to reference pool: {dst}")
    return count, dst


def _run_literature(cfg: dict, enable: bool):
    root = cfg["paths"]["design_execution_root"]
    lit_root = os.path.join(root, "literature")
    os.makedirs(lit_root, exist_ok=True)
    if not enable:
        print("[INFO] Literature search disabled. Skipping.")
        open(os.path.join(lit_root, "auto_sections.md"), "w", encoding="utf-8").write("")
        return lit_root, ""
    lit = cfg.get("literature", {}) or {}
    query = (lit.get("query") or "").strip()
    n = int(lit.get("n", 10))
    llm_sum = bool(lit.get("llm_summarize", True))
    if not query:
        print("[INFO] No literature query found. Skipping search.")
        open(os.path.join(lit_root, "auto_sections.md"), "w", encoding="utf-8").write("")
        return lit_root, ""
    rows = openalex_search(query, per_page=n)
    if rows:
        csv_path = os.path.join(lit_root, "papers.csv")
        save_rows_to_csv(rows, csv_path)
        md_path = os.path.join(lit_root, "auto_sections.md")
        csv_to_md(csv_path, md_path)
        print(f"[INFO] Literature search results saved to: {csv_path}")
        print(f"[INFO] Markdown summary created at: {md_path}")
    else:
        open(os.path.join(lit_root, "auto_sections.md"), "w", encoding="utf-8").write("")
        print("[INFO] No literature results found. Empty file created.")
    bullets = ""
    if llm_sum:
        try:
            syn = llm_summarize_literature(lit_root, query, cfg)
            bullets = open(syn, "r", encoding="utf-8").read()
            print(f"[INFO] LLM literature synthesis file created at: {syn}")
        except Exception as e:
            print(f"[INFO] LLM synthesis failed: {e}")
    return lit_root, bullets


def _prepare_context(cfg: dict, baseline_dir: str, baseline_id: int):
    root = cfg["paths"]["design_execution_root"]
    ref_pool = os.path.join(root, "reference_pool")
    reference_summary = summarize_folder_ipynb(ref_pool, max_chars=1200) if os.path.isdir(ref_pool) else ""
    baseline_nb = os.path.join(baseline_dir, f"baseline_{baseline_id:02d}.ipynb")
    baseline_summary = summarize_notebook(baseline_nb, max_chars=1000) if os.path.exists(baseline_nb) else ""
    print(f"[INFO] Prepared baseline notebook summary from: {baseline_nb}")
    return {"reference_summary": reference_summary, "baseline_summary": baseline_summary, "baseline_nb": baseline_nb}


def _exec_nb(in_nb: str, out_nb: str):
    nb = nbformat.read(in_nb, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3", allow_errors=True).execute()
    nbformat.write(nb, out_nb)
    print(f"[INFO] Executed notebook saved at: {out_nb}")


# ---------------- Baseline branch ----------------
def cmd_generate(cfg: dict, baseline_id: int, with_lit: bool):
    print("\n[INFO] === Starting generate phase ===")
    paths = cfg["paths"]
    root = paths["design_execution_root"]

    moved, ref_dir = _copy_stage1_to_reference_pool(cfg)
    bdir = migrate_reference_ipynb(paths["baseline_source_dir"], root, include_globs=["*.ipynb"])
    print(f"[INFO] Baseline notebooks migrated to: {bdir}")

    _lit_root, bullets = _run_literature(cfg, enable=with_lit)
    ctx = _prepare_context(cfg, bdir, baseline_id)

    tag = (cfg.get("trial", {}) or {}).get("tag", "improve-data-model")
    seed = (cfg.get("trial", {}) or {}).get("seed", 22)
    tdir = create_trial_from_baseline(root, tag, baseline_id, seed)
    print(f"[INFO] Trial directory created at: {tdir}")

    llm_conf = resolve_llm_from_cfg(cfg)
    print(f"[LLM] provider={llm_conf['provider']} model={llm_conf['model']} base_url={llm_conf['base_url']}")
    llm = LLMClient(provider=llm_conf['provider'], model=llm_conf['model'], base_url=llm_conf['base_url'], api_key=llm_conf['api_key'], timeout=llm_conf.get('timeout',1200))

    patch_path = propose_notebook_improvements(
        cfg,
        tdir,
        related_work_bullets=bullets,
        seed=seed,
        llm=llm,
        reference_summary=ctx["reference_summary"],
        baseline_summary=ctx["baseline_summary"],
        require_llm=True,
        llm_retries=int(llm_conf.get("retries", 2)),
    )
    print(f"[INFO] LLM-generated patch file created at: {patch_path}")

    patched = apply_patch(os.path.join(tdir, "notebook_unexec.ipynb"), patch_path)
    print(f"[INFO] Patched notebook generated at: {patched}")
    print("[INFO] === Generate phase completed ===\n")

    return {"baseline_dir": bdir, "trial_dir": tdir, "patched_nb": patched}


def cmd_execute(cfg: dict, baseline_id: int, which: str, trial_dir: str = None):
    print("\n[INFO] === Starting execute phase ===")
    paths = cfg["paths"]
    root = paths["design_execution_root"]
    bdir = _latest(os.path.join(root, "baselines"))
    if not bdir:
        raise RuntimeError("No baselines found. Run `generate` first.")
    baseline_nb = os.path.join(bdir, f"baseline_{baseline_id:02d}.ipynb")
    if not os.path.exists(baseline_nb):
        raise FileNotFoundError(f"Baseline not found: {baseline_nb}")

    if which in ("baseline", "both"):
        b_exec = baseline_nb.replace(".ipynb", "_exec.ipynb")
        _exec_nb(baseline_nb, b_exec)

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

    print("[INFO] === Execute phase completed ===\n")


def cmd_analyze(cfg: dict, trial_dir: str = None, baseline_id: int = 0):
    print("\n[INFO] === Starting analyze phase ===")
    paths = cfg["paths"]
    root = paths["design_execution_root"]
    bdir = _latest(os.path.join(root, "baselines"))
    if not bdir:
        raise RuntimeError("No baselines found.")
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
    print(f"[INFO] Markdown report file generated at: {rpt}")
    print("[INFO] === Analyze phase completed ===\n")
    return rpt


def cmd_prompt_defined(cfg: dict, prompt_path: str = None, subcmd: str = "run"):
    print(f"\n[INFO] === Starting prompt phase: {subcmd} ===")
    prompt_path = prompt_path or cfg.get("prompt_branch", {}).get("prompt_file") or "prompts/pipeline_prompt.yaml"
    print(f"[INFO] Using prompt file: {prompt_path}")

    if subcmd == "run":
        ret = _prompt_run(cfg, prompt_path)
        try:
            from design_execution.evaluator import record_metrics
            record_metrics(ret["trial_dir"], ret.get("metrics", {}))
            print(f"[INFO] Metrics recorded for trial: {ret['trial_dir']}")
        except Exception:
            print("[INFO] Metrics recording skipped (not available).")
        try:
            report_path = build_markdown_report(
                ret["trial_dir"],
                bdir=cfg["paths"].get("baseline_source_dir", ""),
                report_dir=os.path.join(cfg["paths"]["design_execution_root"], "reports"),
                baseline_metrics_path=None
            )
            print(f"[INFO] Prompt report file created at: {report_path}")
        except Exception:
            print("[INFO] Report generation skipped (not available).")
        print(f"[INFO] Trial directory: {ret['trial_dir']}")
        print("[INFO] === Prompt phase completed ===\n")
        return ret

    if subcmd == "generate" and _prompt_generate:
        return _prompt_generate(cfg, prompt_path)
    elif subcmd == "execute" and _prompt_execute:
        return _prompt_execute(cfg)
    elif subcmd == "analyze" and _prompt_analyze:
        return _prompt_analyze(cfg)
    else:
        raise ValueError(f"Unknown or unavailable prompt subcmd: {subcmd}")


def _expand_vars(obj, env):
    if isinstance(obj, dict):
        return {k: _expand_vars(v, env) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_vars(v, env) for v in obj]
    elif isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), obj)
    return obj

def main():
    print("\n[INFO] === Pipeline started ===")
    ap = argparse.ArgumentParser(description="CellScientist unified pipeline")
    ap.add_argument("--config", required=True, help="path to config JSON")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # Unified entry: text prompt + optional baseline code
    ap_run = sub.add_parser("run", help="Unified run: text prompt + optional baseline code")
    ap_run.add_argument("--baseline-id", type=int, default=0, help="which baseline ipynb to use if available")
    ap_run.add_argument("--with-lit", action="store_true", help="enable literature search & synthesis")
    ap_run.add_argument("--prompt-file", type=str, default=None, help="path to a plain text/yaml prompt file")
    ap_run.add_argument("--prompt-text", type=str, default=None, help="inline text prompt (overrides file if given)")
    ap_run.add_argument(
        "--which", choices=["baseline", "patched", "both"], default="both",
        help="which notebook(s) to execute"
    )
    group = ap_run.add_mutually_exclusive_group()
    group.add_argument("--use-baseline", dest="include_baseline", action="store_true",
                       help="use baseline notebook as prior context (default)")
    group.add_argument("--no-baseline", dest="include_baseline", action="store_false",
                       help="do NOT use baseline notebook; text prompt only")
    ap_run.set_defaults(include_baseline=False)

    args = ap.parse_args()

    cfg_text = open(args.config, "r", encoding="utf-8").read()
    cfg = json.loads(cfg_text)
    env = dict(os.environ); env.update(cfg)
    cfg = _expand_vars(cfg, env)

    print("[INFO] Pipeline mode: unified")
    if args.cmd == "run":
        print(f"[INFO] use_baseline: {args.include_baseline}")

        ret = run_unified_pipeline(
            cfg=cfg,
            baseline_id=int(args.baseline_id or 0),
            with_lit=bool(args.with_lit),
            prompt_file=args.prompt_file,
            prompt_text=args.prompt_text,
            include_baseline=bool(args.include_baseline),
            which=args.which
        )
        print("[INFO] Trial dir:", ret.get("trial_dir", ""))
        print("[INFO] Patched notebook:", ret.get("patched_nb", ""))
        print("[INFO] Executed notebook:", ret.get("executed_nb", ""))
        if ret.get("report_path"):
            print("[INFO] Report:", ret["report_path"])

    print("[INFO] === Pipeline finished successfully ===\n")

if __name__ == "__main__":
    main()