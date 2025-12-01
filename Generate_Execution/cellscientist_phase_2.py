#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cellscientist_phase_2.py
import sys, os, json, argparse, glob, shutil, datetime
import numpy as np

# Force Line Buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from design_execution.config_loader import load_full_config
from design_execution.prompt_orchestrator import (
    phase_generate, phase_execute, phase_analyze, run_full_pipeline
)

def _setup_stage1_resources(cfg: dict, enable_idea: bool = False):
    s1_dir = cfg.get("paths", {}).get("stage1_analysis_dir")
    if not s1_dir:
        print("[SETUP][WARN] 'stage1_analysis_dir' missing in config.", flush=True)
        return

    h5_path = None
    cand_h5 = os.path.join(s1_dir, "REFERENCE_DATA.h5")
    if os.path.exists(cand_h5):
        h5_path = os.path.abspath(cand_h5)
    else:
        h5s = glob.glob(os.path.join(s1_dir, "*.h5"))
        if h5s: h5_path = os.path.abspath(h5s[0])
    
    if h5_path:
        os.environ["STAGE1_H5_PATH"] = h5_path
        print(f"[SETUP] Data Anchor: {h5_path}", flush=True)
    else:
        print(f"[SETUP][WARN] No HDF5 found in {s1_dir}", flush=True)

    if enable_idea:
        idea_path = os.path.join(os.path.dirname(h5_path or s1_dir), "idea.json")
        if os.path.exists(idea_path):
            os.environ["STAGE1_IDEA_PATH"] = idea_path
            print(f"[SETUP] Idea File: {idea_path}", flush=True)
        else:
            custom_idea = cfg.get("prompt_branch", {}).get("idea_file")
            if custom_idea and os.path.exists(custom_idea):
                os.environ["STAGE1_IDEA_PATH"] = os.path.abspath(custom_idea)
                print(f"[SETUP] Config Idea File: {os.environ['STAGE1_IDEA_PATH']}", flush=True)
            else:
                print("[SETUP][WARN] --use-idea ON but no idea.json found.", flush=True)
    else:
        if "STAGE1_IDEA_PATH" in os.environ: del os.environ["STAGE1_IDEA_PATH"]
        print("[SETUP] Idea Mode: OFF", flush=True)

def _inject_api_key(cfg: dict):
    key = cfg.get("llm", {}).get("api_key")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        print(f"[SETUP] API Key Injected: ...{key[-4:]}", flush=True)

def _check_success(metrics: dict, threshold: float, metric_key: str) -> tuple[bool, float]:
    if not metrics: return False, -999.0
    
    winner = metrics.get("winner")
    if not winner:
        models = [k for k in metrics.get("models", {}).keys() if k != "config"]
        if not models: return False, -999.0
        winner = models[0]
        
    m_data = metrics.get(winner, metrics.get("models", {}).get(winner, {}))
    
    val = None
    if "aggregate" in m_data:
        val = m_data["aggregate"].get(metric_key)
    if val is None and "per_fold" in m_data:
        vals = []
        for f in m_data["per_fold"].values():
            v = f.get(metric_key, f.get("metrics", {}).get(metric_key))
            if v is not None: vals.append(float(v))
        if vals: val = np.mean(vals)
        
    score = float(val) if val is not None else -999.0
    print(f"[CHECK] {winner} | {metric_key}: {score:.4f} (Target > {threshold})", flush=True)
    
    return score > threshold, score

def _archive_run(trial_dir: str):
    """
    [MODIFIED] Renames workspace to prompt_run_TIMESTAMP.
    """
    if not os.path.exists(trial_dir): return
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = os.path.dirname(trial_dir)
    
    # [FIX] Clean naming pattern
    new_name = f"prompt_run_{ts}"
    new_path = os.path.join(parent, new_name)
    
    try:
        os.rename(trial_dir, new_path)
        print(f"[ARCHIVE] Saved run to: {new_name}", flush=True)
    except OSError:
        shutil.move(trial_dir, new_path)
        print(f"[ARCHIVE] Moved run to: {new_name}", flush=True)

def run_loop(cfg: dict, prompt_file: str, use_idea: bool):
    exp_cfg = cfg.get("experiment", {})
    max_iters = int(exp_cfg.get("max_iterations", 1))
    threshold = float(exp_cfg.get("success_threshold", 0.0))
    pm = exp_cfg.get("primary_metric", "PCC")
    
    print(f"\n[LOOP] Max Iters: {max_iters} | Target: {pm} > {threshold}", flush=True)
    
    p_path = (prompt_file or 
              (cfg.get("prompt_branch") or {}).get("prompt_file") or 
              "prompts/pipeline_prompt.yaml")

    # Unique Workspace Name per Process
    start_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    workspace_name = f"workspace_{start_ts}_{pid}"
    
    # [FIX] State tracking for best run
    best_score = -9999.0
    best_snapshot_dir = None
    
    print(f"[LOOP] Temporary Workspace: {workspace_name}", flush=True)

    for i in range(1, max_iters + 1):
        print(f"\n{'='*40}\n🔄 ITERATION {i}/{max_iters}\n{'='*40}", flush=True)
        
        try:
            _setup_stage1_resources(cfg, use_idea)
            
            # Run in the unique workspace (NOTE: phase_generate clears this dir each time)
            res = run_full_pipeline(cfg, p_path, run_name=workspace_name)
            
            trial_dir = res.get("trial_dir")
            success, score = _check_success(res.get("metrics", {}), threshold, pm)
            
            # 1. Check Success (Instant Stop)
            if success:
                print(f"\n🎉 [SUCCESS] Criteria Met! Archiving and Stopping.", flush=True)
                _archive_run(trial_dir) 
                # Cleanup potential snapshot if exists
                if best_snapshot_dir and os.path.exists(best_snapshot_dir):
                    shutil.rmtree(best_snapshot_dir, ignore_errors=True)
                return
            
            # 2. Track Best Score
            # If valid run (score > -999) and better than previous best
            if score > -999.0 and score > best_score:
                print(f"📈 [IMPROVEMENT] New best score: {score:.4f} (Prev: {best_score:.4f}). Snapshotting...", flush=True)
                best_score = score
                
                # Create a safe snapshot because next iteration will wipe workspace
                best_snapshot_dir = trial_dir + "_best_snapshot"
                if os.path.exists(best_snapshot_dir):
                    shutil.rmtree(best_snapshot_dir)
                shutil.copytree(trial_dir, best_snapshot_dir)
            
            print(f"⚠️ [CONTINUE] Threshold not met.", flush=True)
            
        except Exception as e:
            print(f"❌ [ERROR] Iteration {i} crashed: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Loop Finished - No Success found.
    # Archive the BEST run we found, not necessarily the last one.
    print(f"\n{'='*40}\n🏁 LOOP FINISHED\n{'='*40}", flush=True)
    
    if best_snapshot_dir and os.path.exists(best_snapshot_dir):
        print(f"[LOOP] Archiving BEST run found (Score: {best_score:.4f}).", flush=True)
        _archive_run(best_snapshot_dir)
        
        # Cleanup the stale workspace if it still exists
        final_workspace = best_snapshot_dir.replace("_best_snapshot", "")
        if os.path.exists(final_workspace):
            shutil.rmtree(final_workspace, ignore_errors=True)
            
    else:
        print("[LOOP] ❌ No valid runs completed successfully to archive.", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    def _add_common(p):
        p.add_argument("--prompt-file", type=str)
        p.add_argument("--use-idea", action="store_true")

    cmd_run = sub.add_parser("run")
    _add_common(cmd_run)
    
    cmd_gen = sub.add_parser("generate")
    _add_common(cmd_gen)
    
    sub.add_parser("execute")
    sub.add_parser("analyze")
    
    args = parser.parse_args()
    cfg = load_full_config(args.config)
    _inject_api_key(cfg)
    use_idea = getattr(args, "use_idea", False)
    
    if args.cmd == "run":
        run_loop(cfg, args.prompt_file, use_idea)
        
    elif args.cmd == "generate":
        _setup_stage1_resources(cfg, use_idea)
        p_path = args.prompt_file or (cfg.get("prompt_branch", {})).get("prompt_file") or "prompts/pipeline_prompt.yaml"
        # Standard generate creates timestamped folder by default
        phase_generate(cfg, p_path)
        
    elif args.cmd == "execute":
        phase_execute(cfg)
        
    elif args.cmd == "analyze":
        phase_analyze(cfg)

if __name__ == "__main__":
    main()