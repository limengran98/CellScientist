#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os, json, argparse, re
from design_execution.report_builder import build_markdown_report

from design_execution.prompt_pipeline import run_prompt_pipeline as _prompt_run
try:
    from design_execution.prompt_pipeline import prompt_generate as _prompt_generate
    from design_execution.prompt_pipeline import prompt_execute as _prompt_execute
    from design_execution.prompt_pipeline import prompt_analyze as _prompt_analyze
except Exception:
    _prompt_generate = _prompt_execute = _prompt_analyze = None


def _expand_vars(obj, env):
    """Expand ${VAR} placeholders in nested dict/list/str using env."""
    if isinstance(obj, dict):
        return {k: _expand_vars(v, env) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_vars(v, env) for v in obj]
    elif isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), obj)
    return obj


def cmd_prompt_defined(
    cfg: dict,
    *,
    prompt_path: str | None = None,
    subcmd: str = "run",
    use_baseline: bool = False,
    use_stage1_ref: bool = False,
    baseline_id: int = 0,
    inline_prompt_text: str | None = None,
):
    print(f"\n[INFO] === Starting prompt phase: {subcmd} ===")
    prompt_path = (
        prompt_path
        or (cfg.get("prompt_branch") or {}).get("prompt_file")
        or "prompts/pipeline_prompt.yaml"
    )
    print(f"[INFO] Using prompt file: {prompt_path}")

    cfg.setdefault("prompt_branch", {})
    cfg["prompt_branch"]["use_baseline"] = bool(use_baseline)
    cfg["prompt_branch"]["use_stage1_ref"] = bool(use_stage1_ref)
    cfg["prompt_branch"]["baseline_id"] = int(baseline_id)
    if inline_prompt_text:
        cfg["prompt_branch"]["inline_text"] = inline_prompt_text
    # --- Stage-1 memory inheritance ---
    try:
        from design_execution.stage1_adapter import prepare_stage1_context
        s1 = prepare_stage1_context(cfg, enable=bool(cfg["prompt_branch"]["use_stage1_ref"]))
        if s1:
            # store markdown for later insertion into the generated notebook
            cfg["prompt_branch"]["stage1_markdown"] = s1.get("markdown", "")
            # Ensure env var so prompt YAML can reference ${STAGE1_H5_PATH}
            if s1.get("h5_path"):
                os.environ["STAGE1_H5_PATH"] = s1["h5_path"]
                print(f"[STAGE1] H5 override set to: {s1['h5_path']}")
    except Exception as _e:
        print(f"[STAGE1][WARN] Failed to prepare Stage-1 context: {_e}")
    # --- End Stage-1 memory ---


    # --- Stage-1 memory inheritance ---
    try:
        from design_execution.stage1_adapter import prepare_stage1_context
        s1 = prepare_stage1_context(cfg, enable=bool(cfg["prompt_branch"]["use_stage1_ref"]))
        if s1:
            # store markdown for later insertion into the generated notebook
            cfg["prompt_branch"]["stage1_markdown"] = s1.get("markdown", "")
            # Ensure env var so prompt YAML can reference ${STAGE1_H5_PATH}
            if s1.get("h5_path"):
                os.environ["STAGE1_H5_PATH"] = s1["h5_path"]
                print(f"[STAGE1] H5 override set to: {s1['h5_path']}")
    except Exception as _e:
        print(f"[STAGE1][WARN] Failed to prepare Stage-1 context: {_e}")
    # --- End Stage-1 memory ---

        cfg["prompt_branch"]["inline_text"] = inline_prompt_text

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


def main():
    print("\n[INFO] === Pipeline started ===")
    ap = argparse.ArgumentParser(description="CellScientist prompt-only pipeline")
    ap.add_argument("--config", required=True, help="path to config JSON")
    ap.add_argument("--pipeline-mode", choices=["prompt"], default="prompt", help="compat only; always 'prompt'")

    sub = ap.add_subparsers(dest="cmd", required=True)


    ap_r = sub.add_parser("run", help="One-command: generate → execute → analyze (prompt pipeline)")
    ap_r.add_argument("--baseline-id", type=int, default=0, help="which baseline ipynb to reference if enabled")
    ap_r.add_argument("--with-lit", action="store_true", help="enable literature search & synthesis (if supported)")
    ap_r.add_argument("--which", choices=["baseline", "patched", "both"], default="both",
                      help="which notebook(s) to execute (consumed in prompt pipeline if supported)")
    ap_r.add_argument("--prompt-file", type=str, default="prompts/pipeline_prompt.yaml",
                      help="path to a plain text/yaml prompt file")
    ap_r.add_argument("--prompt-text", type=str, default=None,
                      help="inline text prompt (overrides file if given)")
    # 是否读取 baseline 代码
    group_b = ap_r.add_mutually_exclusive_group()
    group_b.add_argument("--use-baseline", dest="use_baseline", action="store_true",
                         help="include baseline notebook as prior context for prompt pipeline")
    group_b.add_argument("--no-baseline", dest="use_baseline", action="store_false",
                         help="do NOT include baseline notebook (default)")
    ap_r.set_defaults(use_baseline=False)
    # 是否引用 stage1_analysis_dir 作为参考
    group_s1 = ap_r.add_mutually_exclusive_group()
    group_s1.add_argument("--use-stage1-ref", dest="use_stage1_ref", action="store_true",
                          help="include Stage-1 analysis notebooks summary as context")
    group_s1.add_argument("--no-stage1-ref", dest="use_stage1_ref", action="store_false",
                          help="do NOT include Stage-1 analysis context (default)")
    ap_r.set_defaults(use_stage1_ref=True)

    # 可选的细分阶段
    ap_pg = sub.add_parser("generate", help="Prompt pipeline: generate only")
    ap_pg.add_argument("--baseline-id", type=int, default=0)
    ap_pg.add_argument("--prompt-file", type=str, default="prompts/pipeline_prompt.yaml")
    ap_pg.add_argument("--prompt-text", type=str, default=None)
    group_gb = ap_pg.add_mutually_exclusive_group()
    group_gb.add_argument("--use-baseline", dest="use_baseline", action="store_true")
    group_gb.add_argument("--no-baseline", dest="use_baseline", action="store_false")
    ap_pg.set_defaults(use_baseline=False)
    group_gs1 = ap_pg.add_mutually_exclusive_group()
    group_gs1.add_argument("--use-stage1-ref", dest="use_stage1_ref", action="store_true")
    group_gs1.add_argument("--no-stage1-ref", dest="use_stage1_ref", action="store_false")
    ap_pg.set_defaults(use_stage1_ref=True)

    ap_pe = sub.add_parser("execute", help="Prompt pipeline: execute only")
    ap_pe.add_argument("--baseline-id", type=int, default=0)
    group_eb = ap_pe.add_mutually_exclusive_group()
    group_eb.add_argument("--use-baseline", dest="use_baseline", action="store_true")
    group_eb.add_argument("--no-baseline", dest="use_baseline", action="store_false")
    ap_pe.set_defaults(use_baseline=False)
    group_es1 = ap_pe.add_mutually_exclusive_group()
    group_es1.add_argument("--use-stage1-ref", dest="use_stage1_ref", action="store_true")
    group_es1.add_argument("--no-stage1-ref", dest="use_stage1_ref", action="store_false")
    ap_pe.set_defaults(use_stage1_ref=True)

    ap_pa = sub.add_parser("analyze", help="Prompt pipeline: analyze only")
    ap_pa.add_argument("--baseline-id", type=int, default=0)
    group_ab = ap_pa.add_mutually_exclusive_group()
    group_ab.add_argument("--use-baseline", dest="use_baseline", action="store_true")
    group_ab.add_argument("--no-baseline", dest="use_baseline", action="store_false")
    ap_pa.set_defaults(use_baseline=False)
    group_as1 = ap_pa.add_mutually_exclusive_group()
    group_as1.add_argument("--use-stage1-ref", dest="use_stage1_ref", action="store_true")
    group_as1.add_argument("--no-stage1-ref", dest="use_stage1_ref", action="store_false")
    ap_pa.set_defaults(use_stage1_ref=True)

    args = ap.parse_args()

    cfg_text = open(args.config, "r", encoding="utf-8").read()
    cfg = json.loads(cfg_text)
    env = dict(os.environ); env.update(cfg)
    cfg = _expand_vars(cfg, env)

    print("[INFO] Pipeline mode: prompt")

    if args.cmd == "run":
        ret = cmd_prompt_defined(
            cfg,
            prompt_path=args.prompt_file,
            subcmd="run",
            use_baseline=args.use_baseline,
            use_stage1_ref=args.use_stage1_ref,
            baseline_id=args.baseline_id,
            inline_prompt_text=args.prompt_text,
        )
        print("[INFO] Trial dir:", ret.get("trial_dir", ""))
        print("[INFO] === Pipeline finished successfully ===\n")
        return

    if args.cmd == "generate":
        ret = cmd_prompt_defined(
            cfg,
            prompt_path=args.prompt_file,
            subcmd="generate",
            use_baseline=args.use_baseline,
            use_stage1_ref=args.use_stage1_ref,
            baseline_id=args.baseline_id,
            inline_prompt_text=args.prompt_text,
        )
        print("[INFO] === Pipeline finished successfully ===\n")
        return

    if args.cmd == "execute":
        ret = cmd_prompt_defined(
            cfg,
            subcmd="execute",
            use_baseline=args.use_baseline,
            use_stage1_ref=args.use_stage1_ref,
            baseline_id=args.baseline_id,
        )
        print("[INFO] === Pipeline finished successfully ===\n")
        return

    if args.cmd == "analyze":
        ret = cmd_prompt_defined(
            cfg,
            subcmd="analyze",
            use_baseline=args.use_baseline,
            use_stage1_ref=args.use_stage1_ref,
            baseline_id=args.baseline_id,
        )
        print("[INFO] === Pipeline finished successfully ===\n")
        return


if __name__ == "__main__":
    main()
