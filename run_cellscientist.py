#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_cellscientist.py
#
# Pipeline runner + robust metrics scoreboard.
#
# Metrics Definitions:
# - Success Rate (Total): All successful runs, including those fixed by Auto-Fix.
# - Clean Rate (Zero-Shot): Successful runs WITHOUT any bug/recovery signals.
# - Bug Rate: Fraction of attempts that triggered any error/fix logic.
#
# [UPDATE] Now includes Explicit Path Tracking to ensure simultaneous runs 
# do not cross-read artifacts during report generation.

from __future__ import annotations

import datetime
import os
import subprocess
import time
from typing import Any, Dict, Optional

from runner_config import (
    get_nested,
    load_pipeline_config,
    materialize_merged_configs,
    pipeline_extra_env,
    validate_configs,
)
from runner_metrics import (
    mean_safe,
    parse_phase1_log,
    parse_phase2_log,
    parse_phase3_log,
    phase1_scores_from_artifacts,
    phase2_scores_from_artifacts,
    phase3_scores_from_artifacts,
    pick_best,
    planned_phase1_budget,
    print_execution_plan,
    print_final_scoreboard,
    rates,
)
from runner_report import generate_final_report
from runner_utils import (
    append_phase_header,
    atomic_write_json,
    ensure_project_cwd,
    extract_best_path_from_log, # [NEW]
    read_text,
    results_root_for_dataset,
    run_cmd_streamed,
    setup_logging,
)


# =============================================================================
# ⚙️ Phase Map
# =============================================================================


PHASE_MAP: Dict[str, Dict[str, Any]] = {
    "Phase 1": {
        "folder": "Design_Analysis",
        "script": "cellscientist_phase_1.py",
        "config": "design_analysis_config.json",
        "cmd_args": [],
    },
    "Phase 2": {
        "folder": "Generate_Execution",
        "script": "cellscientist_phase_2.py",
        "config": "generate_execution_config.json",
        "cmd_args": ["run", "--use-idea"],
    },
    "Phase 3": {
        "folder": "Review_Feedback",
        "script": "cellscientist_phase_3.py",
        "config": "review_feedback_config.json",
        "cmd_args": [],
    },
}

PHASE_LOG_NAME = {
    "Phase 1": "phase1.log",
    "Phase 2": "phase2.log",
    "Phase 3": "phase3.log",
}


def _maybe_console():
    try:
        from rich.console import Console

        return Console()
    except Exception:
        return None


def main():
    ensure_project_cwd()
    console = _maybe_console()

    # ---------------------------------------------------------------------
    # Optional pipeline_config.json
    # ---------------------------------------------------------------------
    pipe_cfg = load_pipeline_config()
    extra_env: Optional[Dict[str, str]] = None

    if isinstance(pipe_cfg, dict):
        try:
            materialize_merged_configs(PHASE_MAP, pipe_cfg)
            extra_env = pipeline_extra_env(pipe_cfg)

            if console:
                from rich.panel import Panel

                gpu_name = None
                if isinstance(pipe_cfg.get("common"), dict):
                    gpu_name = pipe_cfg["common"].get("gpu_model") or pipe_cfg["common"].get("gpu_name")
                console.print(
                    Panel(
                        f"[bold]Pipeline config detected[/]\n"
                        f"- dataset_name: [green]{pipe_cfg.get('dataset_name','(from phase configs)')}[/]\n"
                        f"- CUDA_VISIBLE_DEVICES: [cyan]{(extra_env or {}).get('CUDA_VISIBLE_DEVICES','(default)')}[/]\n"
                        f"- GPU model: [magenta]{gpu_name or '(n/a)'}[/]",
                        border_style="blue",
                    )
                )
            else:
                print("[INFO] Pipeline config detected:", pipe_cfg.get("dataset_name"))
        except Exception as e:
            if console:
                from rich.panel import Panel

                console.print(Panel(f"[bold yellow]WARN[/] Failed to apply pipeline_config.json: {e}", border_style="yellow"))
            else:
                print(f"[WARN] Failed to apply pipeline_config.json: {e}")

    # ---------------------------------------------------------------------
    # Validate configs + setup logging
    # ---------------------------------------------------------------------
    ds_name = validate_configs(PHASE_MAP)
    results_root = results_root_for_dataset(ds_name)
    os.makedirs(results_root, exist_ok=True)
    logs_dir, pipeline_log_path, log_fp = setup_logging(results_root)

    pipeline_start = time.time()
    print_execution_plan(PHASE_MAP, ds_name, console=console)

    print(f"🚀 Pipeline starting for [{ds_name}] in 2 seconds...")
    time.sleep(2)

    stage_timings: Dict[str, Dict[str, float]] = {}
    phase_logs: Dict[str, str] = {}
    
    # [NEW] Explicitly track paths to avoid globbing race conditions
    explicit_paths: Dict[str, str] = {"Phase 2": None, "Phase 3": None}

    try:
        # -----------------------------------------------------------------
        # Run phases
        # -----------------------------------------------------------------
        for name, info in PHASE_MAP.items():
            folder = info["folder"]
            script = info["script"]
            config = info["config"]
            extra_args = info.get("cmd_args") or []

            cmd = ["python", script, "--config", config] if name != "Phase 1" else ["python", script, config]
            if extra_args:
                cmd.extend(extra_args)

            phase_log_file = os.path.join(logs_dir, PHASE_LOG_NAME.get(name, f"{name}.log".replace(" ", "_").lower()))
            phase_logs[name] = phase_log_file

            phase_fp = None
            try:
                phase_fp = open(phase_log_file, "a", encoding="utf-8")
                append_phase_header(phase_fp, ds_name, name, cmd, folder)
            except Exception as e:
                phase_fp = None
                print(f"[WARN] Failed to open phase log file {phase_log_file}: {e}")

            if console:
                console.rule(f"[bold blue]Running {name}[/]")
                console.print(f"📂 Context: [underline]{folder}[/]")
                console.print(f"💻 Command: [dim]{' '.join(cmd)}[/]\n")
            else:
                print(f"\n=== Running {name} (Dir: {folder}) ===")

            start_ts = time.time()
            stage_timings[name] = {"start": start_ts}

            try:
                run_cmd_streamed(cmd, cwd=folder, phase_fp=phase_fp, extra_env=extra_env)
            except subprocess.CalledProcessError as e:
                if phase_fp:
                    try:
                        phase_fp.write(f"\n[ERROR] Phase failed. Exit Code={e.returncode}\n")
                        phase_fp.flush()
                    except Exception:
                        pass
                if console:
                    from rich.panel import Panel

                    console.print(Panel(f"[bold red]❌ Pipeline Failed at {name}[/]\nExit Code: {e.returncode}", border_style="red"))
                else:
                    print(f"❌ Pipeline Failed at {name} with exit code {e.returncode}")
                raise
            finally:
                if phase_fp:
                    try:
                        phase_fp.write(f"\n[INFO] Phase finished at {datetime.datetime.now().isoformat()}\n")
                        phase_fp.flush()
                        phase_fp.close()
                    except Exception:
                        pass

            end_ts = time.time()
            stage_timings[name]["end"] = end_ts
            print(f"✅ {name} Completed ({end_ts - start_ts:.1f}s)\n")
            
            # [NEW] Robust Path Capture
            if name in ["Phase 2", "Phase 3"]:
                base_search_dir = ""
                if name == "Phase 2":
                    base_search_dir = os.path.join(results_root, "generate_execution")
                elif name == "Phase 3":
                    base_search_dir = os.path.join(results_root, "review_feedback")
                
                # Pass start_ts for fallback mechanism
                found_path = extract_best_path_from_log(
                    phase_log_file, 
                    name, 
                    base_search_dir, 
                    stage_timings[name]["start"]
                )
                
                if found_path:
                    explicit_paths[name] = found_path
                    print(f"📌 [TRACKER] Locked {name} Output: {found_path}")
                else:
                    print(f"[WARN] Could not lock output path for {name}. Report generation might rely on fallback.")

        pipeline_end = time.time()

        # -----------------------------------------------------------------
        # Scoreboard
        # -----------------------------------------------------------------
        stage1_cfg = PHASE_MAP["Phase 1"].get("_loaded_cfg", {})
        stage2_cfg = PHASE_MAP["Phase 2"].get("_loaded_cfg", {})
        stage3_cfg = PHASE_MAP["Phase 3"].get("_loaded_cfg", {})

        optim_dir = get_nested(stage3_cfg, ["review", "direction"], "maximize").lower()

        base = results_root_for_dataset(ds_name)
        design_dir = os.path.join(base, "design_analysis")
        ge_dir = os.path.join(base, "generate_execution")
        rf_dir = os.path.join(base, "review_feedback")

        # Phase 1
        p1_budget = planned_phase1_budget(stage1_cfg)
        p1_metric = "heuristic_score"
        p1_log_text = read_text(phase_logs.get("Phase 1", ""))
        p1_q = parse_phase1_log(p1_log_text) if p1_log_text else {"attempted": 0, "succeeded": 0, "bug": 0, "clean_success": 0, "exec_time": 0.0}
        p1_avg, p1_best = phase1_scores_from_artifacts(design_dir)
        
        # [MODIFIED] Using new rate definitions
        p1_total_sr, p1_clean_sr, p1_bug_rate = rates(p1_q["attempted"], p1_q["clean_success"], p1_q["succeeded"], p1_q["bug"])
        
        p1 = {
            "budget": p1_budget,
            "attempted": p1_q["attempted"],
            "succeeded": p1_q["succeeded"],
            "clean_succeeded": p1_q["clean_success"],
            "bug_attempts": p1_q["bug"],
            "success_rate": p1_total_sr, # Now 1.0 if all runs finished (via fix)
            "clean_rate": p1_clean_sr,   # Original Clean Rate
            "bug_rate": p1_bug_rate,
            "avg_at_budget": p1_avg,
            "best_at_budget": p1_best,
            "best_metric": p1_metric,
            "time_sec": stage_timings.get("Phase 1", {}).get("end", 0.0) - stage_timings.get("Phase 1", {}).get("start", 0.0),
            "exec_time": p1_q.get("exec_time", 0.0), # [NEW]
        }

        # Phase 2
        p2_budget = int(get_nested(stage2_cfg, ["experiment", "max_iterations"], 0) or 0)
        p2_metric = str(get_nested(stage2_cfg, ["experiment", "primary_metric"], "PCC"))
        p2_t0 = stage_timings.get("Phase 2", {}).get("start", 0.0)
        p2_t1 = stage_timings.get("Phase 2", {}).get("end", time.time())
        p2_log_text = read_text(phase_logs.get("Phase 2", ""))
        p2_q = parse_phase2_log(p2_log_text, p2_metric) if p2_log_text else {"attempted": 0, "succeeded": 0, "bug": 0, "clean_success": 0, "scores": [], "exec_time": 0.0}

        # Robust Sync with Artifacts
        artifact_scores_p2 = phase2_scores_from_artifacts(ge_dir, p2_metric, p2_t0, p2_t1)
        if len(artifact_scores_p2) >= len(p2_q.get("scores", [])):
            p2_q["scores"] = artifact_scores_p2
        
        if len(p2_q["scores"]) > p2_q["succeeded"]:
            p2_q["succeeded"] = len(p2_q["scores"])
        
        if p2_q["succeeded"] > p2_q["attempted"]:
            p2_q["attempted"] = p2_q["succeeded"]

        if p2_q["succeeded"] > 0:
            expected_clean = max(0, p2_q["succeeded"] - p2_q["bug"])
            if p2_q["clean_success"] < expected_clean:
                p2_q["clean_success"] = expected_clean

        p2_avg = mean_safe([float(x) for x in p2_q.get("scores", []) if isinstance(x, (int, float))])
        p2_best = pick_best(p2_q.get("scores", []), optim_dir)
        
        # [MODIFIED] Rates
        p2_total_sr, p2_clean_sr, p2_bug_rate = rates(p2_q["attempted"], p2_q["clean_success"], p2_q["succeeded"], p2_q["bug"])
        
        p2 = {
            "budget": p2_budget,
            "attempted": p2_q["attempted"],
            "succeeded": p2_q["succeeded"],
            "clean_succeeded": p2_q["clean_success"],
            "bug_attempts": p2_q["bug"],
            "success_rate": p2_total_sr,
            "clean_rate": p2_clean_sr,
            "bug_rate": p2_bug_rate,
            "avg_at_budget": p2_avg,
            "best_at_budget": p2_best,
            "best_metric": p2_metric,
            "time_sec": float(p2_t1 - p2_t0),
            "exec_time": p2_q.get("exec_time", 0.0), # [NEW]
        }

        # Phase 3
        p3_budget = int(get_nested(stage3_cfg, ["review", "max_iterations"], 0) or 0)
        p3_metric = str(get_nested(stage3_cfg, ["review", "target_metric"], "PCC"))
        p3_t0 = stage_timings.get("Phase 3", {}).get("start", 0.0)
        p3_t1 = stage_timings.get("Phase 3", {}).get("end", time.time())
        p3_log_text = read_text(phase_logs.get("Phase 3", ""))
        p3_q = parse_phase3_log(p3_log_text, p3_metric) if p3_log_text else {"attempted": 0, "succeeded": 0, "bug": 0, "clean_success": 0, "scores": [], "exec_time": 0.0}

        # Robust Sync with Artifacts
        artifact_scores_p3 = phase3_scores_from_artifacts(rf_dir, p3_metric, p3_t0, p3_t1)
        if len(artifact_scores_p3) >= len(p3_q.get("scores", [])):
            p3_q["scores"] = artifact_scores_p3
        
        if len(p3_q["scores"]) > p3_q["succeeded"]:
            p3_q["succeeded"] = len(p3_q["scores"])
            
        if p3_q["succeeded"] > p3_q["attempted"]:
            p3_q["attempted"] = max(p3_q["succeeded"], 0)

        if p3_q["succeeded"] > 0:
            expected_clean = max(0, p3_q["succeeded"] - p3_q["bug"])
            if p3_q["clean_success"] < expected_clean:
                p3_q["clean_success"] = expected_clean

        p3_avg = mean_safe([float(x) for x in p3_q.get("scores", []) if isinstance(x, (int, float))])
        p3_best = pick_best(p3_q.get("scores", []), optim_dir)
        
        # [MODIFIED] Rates
        p3_total_sr, p3_clean_sr, p3_bug_rate = rates(p3_q["attempted"], p3_q["clean_success"], p3_q["succeeded"], p3_q["bug"])
        
        p3 = {
            "budget": p3_budget,
            "attempted": p3_q["attempted"],
            "succeeded": p3_q["succeeded"],
            "clean_succeeded": p3_q["clean_success"],
            "bug_attempts": p3_q["bug"],
            "success_rate": p3_total_sr,
            "clean_rate": p3_clean_sr,
            "bug_rate": p3_bug_rate,
            "avg_at_budget": p3_avg,
            "best_at_budget": p3_best,
            "best_metric": p3_metric,
            "time_sec": float(p3_t1 - p3_t0),
            "exec_time": p3_q.get("exec_time", 0.0), # [NEW]
        }

        # Total
        total_attempted = (p1["attempted"] or 0) + (p2["attempted"] or 0) + (p3["attempted"] or 0)
        total_succeeded = (p1["succeeded"] or 0) + (p2["succeeded"] or 0) + (p3["succeeded"] or 0)
        total_clean = (p1["clean_succeeded"] or 0) + (p2["clean_succeeded"] or 0) + (p3["clean_succeeded"] or 0)
        total_bug = (p1["bug_attempts"] or 0) + (p2["bug_attempts"] or 0) + (p3["bug_attempts"] or 0)
        
        # [MODIFIED] Rates
        total_sr, total_clean_sr, total_bug_rate = rates(total_attempted, total_clean, total_succeeded, total_bug)

        total_scores = []
        if p2_metric == p3_metric and p2.get("avg_at_budget") is not None:
            total_scores.extend([float(x) for x in (p2_q.get("scores") or []) if isinstance(x, (int, float))])
        total_scores.extend([float(x) for x in (p3_q.get("scores") or []) if isinstance(x, (int, float))])

        total_avg = mean_safe(total_scores)
        total_best = pick_best(total_scores, optim_dir)
        
        # [NEW] Total Exec Time
        total_exec_time = (p1.get("exec_time", 0.0) + p2.get("exec_time", 0.0) + p3.get("exec_time", 0.0))
        
        total_row = {
            "budget": (p1_budget or 0) + (p2_budget or 0) + (p3_budget or 0),
            "attempted": total_attempted,
            "succeeded": total_succeeded,
            "clean_succeeded": total_clean,
            "bug_attempts": total_bug,
            "success_rate": total_sr,
            "clean_rate": total_clean_sr,
            "bug_rate": total_bug_rate,
            "avg_at_budget": total_avg,
            "best_at_budget": total_best,
            "best_metric": p3_metric,
            "time_sec": float(pipeline_end - pipeline_start),
            "exec_time": total_exec_time, # [NEW]
        }

        summary = {
            "dataset": ds_name,
            "generated_at": datetime.datetime.now().isoformat(),
            "logs_dir": logs_dir,
            "pipeline_log_path": pipeline_log_path,
            "phase_logs": phase_logs,
            "stages": {"Phase 1": p1, "Phase 2": p2, "Phase 3": p3, "Total": total_row},
        }

        summary_path = os.path.join(results_root, "pipeline_summary.json")
        atomic_write_json(summary_path, summary)
        print_final_scoreboard(summary, console=console)

        # -----------------------------------------------------------------
        # Final report + best-code export
        # -----------------------------------------------------------------
        try:
            generate_final_report(
                ds_name=ds_name,
                results_root=results_root,
                pipeline_summary=summary,
                stage1_cfg=stage1_cfg,
                stage2_cfg=stage2_cfg,
                stage3_cfg=stage3_cfg,
                stage_timings=stage_timings,
                phase_logs=phase_logs,
                pipeline_log_path=pipeline_log_path,
                pipe_cfg=pipe_cfg,
                # [NEW] Pass explicitly locked paths
                explicit_p2_path=explicit_paths["Phase 2"],
                explicit_p3_path=explicit_paths["Phase 3"],
                direction=optim_dir,
                metric=p3_metric,
                # [NEW] Pass logs_dir to direct final outputs there
                output_base_dir=logs_dir,
            )
        except Exception as e:
            print(f"[WARN] Final report generation skipped/failed: {e}")

        if console:
            from rich.panel import Panel

            console.print(Panel("[bold green]🏆 CellScientist Workflow Completed Successfully![/]", border_style="green"))
            console.print(f"📌 Saved summary to: [underline]{summary_path}[/]")
        else:
            print("\n🏆 CellScientist Workflow Completed Successfully!")
            print(f"📌 Saved summary to: {summary_path}")

    finally:
        try:
            log_fp.flush()
            log_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()