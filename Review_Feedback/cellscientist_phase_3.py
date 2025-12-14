#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import re
import glob
import shutil
import nbformat
import numpy as np
from copy import deepcopy
from datetime import datetime

# =============================================================================
# [ARCH UPGRADE] Import Centralized Utilities
# =============================================================================
from config_loader import load_full_config
from llm_utils import chat_json, resolve_llm_config

# [UPDATED] Import decoupled execution and error handling functions
# Ensure executor_engine.py is updated to accept extra_env and mutable_indices!
from executor_engine import run_notebook_pure, attempt_fix_notebook, dump_error_log

try:
    from experiment_report import write_experiment_report
except ImportError:
    print("[WARN] 'experiment_report.py' not found. Report generation will be disabled.")
    write_experiment_report = None

# =============================================================================
# 1. Helper: Resource Resolver (H5 Path Logic Ported from Phase 2)
# =============================================================================

def _resolve_h5_path(cfg):
    """
    Resolves the absolute path of the STAGE 1 H5 Data.
    Mimics Phase 2's robust finding logic.
    """
    s1_dir = cfg.get("paths", {}).get("stage1_analysis_dir")
    if not s1_dir: 
        return None

    # 1. Try as absolute path or relative to cwd
    abs_s1_dir = os.path.abspath(s1_dir)
    
    if not os.path.exists(abs_s1_dir):
        print(f"[WARN] Data directory not found: {abs_s1_dir}")
        return None

    # 2. Look for REFERENCE_DATA.h5 first (Standard Convention)
    cand_h5 = os.path.join(abs_s1_dir, "REFERENCE_DATA.h5")
    if os.path.exists(cand_h5):
        print(f"[DATA] Found Stage 1 Data (Explicit): {cand_h5}")
        return cand_h5

    # 3. Fallback: Any .h5 file
    h5_files = glob.glob(os.path.join(abs_s1_dir, "*.h5"))
    if h5_files:
        target_h5 = h5_files[0]
        print(f"[DATA] Found Stage 1 Data (Auto-detected): {target_h5}")
        return target_h5
    
    print(f"[WARN] No .h5 files found in {abs_s1_dir}")
    return None

def _resolve_relative_resources(cfg):
    """(Legacy wrapper, kept for backward compatibility)"""
    path = _resolve_h5_path(cfg)
    if path:
        os.environ["STAGE1_H5_PATH"] = path

# =============================================================================
# 2. Logging & Analysis Logic
# =============================================================================

def write_review_log(workspace, iteration, suggestion, score, current_best, static_baseline, status):
    """
    Writes a log comparing Candidate vs Best vs Baseline.
    """
    log_path = os.path.join(workspace, "optimization_history.md")
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    critique = suggestion.get("critique", "No critique provided.")
    edits = suggestion.get("edits", [])
    
    icon = "✅" if status == "IMPROVED" else "❌" if status == "FAILED" else "⚠️"
    
    # Calculate Delta
    delta_str = "-"
    if static_baseline is not None and score != -999:
        diff = score - static_baseline
        delta_str = f"{diff:+.4f}"

    log_entry = f"""
## Iteration {iteration} {icon} (Time: {timestamp})
**Status**: {status}

| Metric Type | Score |
| :--- | :--- |
| **Candidate (This Run)** | **{score:.4f}** |
| Current Best (Evolution) | {current_best:.4f} |
| Original Baseline | {static_baseline if static_baseline is not None else 'N/A'} |
| **Delta (vs Baseline)** | **{delta_str}** |

### 💡 Rationale
> {critique}

### 🛠 Code Changes
"""
    if not edits:
        log_entry += "\n*No code edits applied.*\n"
    else:
        for edit in edits:
            idx = edit.get("cell_index")
            src = edit.get("source", "")
            log_entry += f"\n**Cell {idx}**:\n```python\n{src}\n```\n"

    log_entry += "\n---\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)
    
    print(f"[LOG] Review log updated: {log_path}")

# =============================================================================
# 3. Path & Resource Management
# =============================================================================

def find_best_phase2_trial(cfg, explicit_path=None):
    """
    Finds the source trial directory.
    """
    # 1. Check Explicit Path
    if explicit_path:
        if not os.path.isabs(explicit_path):
            abs_path = os.path.abspath(explicit_path)
        else:
            abs_path = explicit_path

        print(f"[INIT] Checking specified path: {abs_path}")
        
        nb_path = os.path.join(abs_path, "notebook_prompt_exec.ipynb")
        metrics_path = os.path.join(abs_path, "metrics.json")
        
        if os.path.exists(nb_path) and os.path.exists(metrics_path):
            print(f"[INIT] Locked to specified source: {abs_path}")
            return abs_path
        else:
            raise FileNotFoundError(f"Specified path exists but missing 'notebook_prompt_exec.ipynb' or 'metrics.json': {abs_path}")

    # 2. Auto-detect most recent
    gen_root = cfg["paths"]["generate_execution_root"]
    prompt_root = os.path.join(gen_root, "prompt")
    
    if not os.path.exists(prompt_root):
        raise FileNotFoundError(f"Generation root not found: {prompt_root}")

    runs = sorted(glob.glob(os.path.join(prompt_root, "prompt_run_*")), reverse=True)
    
    print(f"[INIT] Searching for Phase 2 results in: {prompt_root}")
    for run in runs:
        nb_path = os.path.join(run, "notebook_prompt_exec.ipynb")
        metrics_path = os.path.join(run, "metrics.json")
        if os.path.exists(nb_path) and os.path.exists(metrics_path):
            print(f"[INIT] Found valid Phase 2 source (Auto-detected): {run}")
            return run
            
    raise FileNotFoundError("No valid Phase 2 execution results found (Auto-detection failed).")

def setup_phase3_workspace(cfg, source_trial_path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = cfg["paths"]["review_feedback_root"]
    workspace = os.path.join(out_root, f"review_run_{ts}")
    os.makedirs(workspace, exist_ok=True)
    
    src_nb = os.path.join(source_trial_path, "notebook_prompt_exec.ipynb")
    dst_nb = os.path.join(workspace, "notebook_base.ipynb")
    shutil.copy(src_nb, dst_nb)
    
    src_met = os.path.join(source_trial_path, "metrics.json")
    shutil.copy(src_met, os.path.join(workspace, "metrics_base.json"))
    
    with open(os.path.join(workspace, "optimization_history.md"), "w") as f:
        f.write(f"# Optimization History\nSource Trial: {source_trial_path}\n\n")
    
    return workspace, dst_nb

# =============================================================================
# 4. Notebook Analysis
# =============================================================================

def identify_mutable_cells(nb, cfg):
    """
    Identifies mutable cells, explicitly supporting Strategy (Markdown) + Model (Code).
    """
    force_indices = cfg["review"].get("force_cells")
    if force_indices and isinstance(force_indices, list) and len(force_indices) > 0:
        valid_indices = [i for i in force_indices if 0 <= i < len(nb.cells)]
        return valid_indices

    protected_keywords = cfg["review"].get("protected_sections", [])
    
    mutable_indices = []
    
    for i, cell in enumerate(nb.cells):
        src = cell.source
        
        if "StratifiedKFold" in src or "train_test_split" in src:
            continue
            
        if cell.cell_type == "markdown":
            if any(k in src for k in ["Strategy", "Idea", "Hypothesis", "Research"]):
                mutable_indices.append(i)
            continue

        if cell.cell_type == "code":
            is_protected = any(k in src for k in protected_keywords)
            if is_protected: continue
            
            is_model = "import torch.nn" in src or ("class " in src and "(nn.Module)" in src)
            is_digest = "Digest" in src and "Strategy" in src
            
            if is_model or is_digest:
                mutable_indices.append(i)
            
    print(f"[ROUTER] Mutable Cell Indices (Auto): {mutable_indices}")
    return mutable_indices

# =============================================================================
# 5. Metric Extraction
# =============================================================================

def _extract_val_from_model_data(model_data, metric_name):
    """Helper to dig metric from model dict."""
    val = None
    if "aggregate" in model_data:
        val = model_data["aggregate"].get(metric_name)
    if val is None and "per_fold" in model_data:
            vals = [float(x.get(metric_name, 0)) for x in model_data["per_fold"].values() if metric_name in x]
            if vals: val = np.mean(vals)
    return float(val) if val is not None else None

def get_candidate_metric_value(metrics_path, metric_name="PCC"):
    if not os.path.exists(metrics_path): return -999.0
    try:
        with open(metrics_path, 'r') as f: data = json.load(f)
        models = data.get("models", {})
        if not models: return -999.0
        
        non_baseline_keys = [k for k in models.keys() if "baseline" not in k.lower() and "reference" not in k.lower()]
        target_key = non_baseline_keys[-1] if non_baseline_keys else (data.get("winner") or list(models.keys())[0])

        val = _extract_val_from_model_data(models[target_key], metric_name)
        return val if val is not None else -999.0
    except Exception as e:
        return -999.0

def get_baseline_metric_value(metrics_path, metric_name="PCC"):
    if not os.path.exists(metrics_path): return None
    try:
        with open(metrics_path, 'r') as f: data = json.load(f)
        models = data.get("models", {})
        baseline_keys = [k for k in models.keys() if "baseline" in k.lower() or "reference" in k.lower()]
        if baseline_keys:
            return _extract_val_from_model_data(models[baseline_keys[0]], metric_name)
        return None
    except:
        return None

# =============================================================================
# 6. LLM Logic (Enhanced for Context)
# =============================================================================

def _expand_vars(text, context):
    if not text: return ""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(context.get(m.group(1), m.group(0))), text)

def generate_optimization_suggestion(cfg, nb, mutable_indices, current_metrics, iteration, best_metric_val, workspace, history_summary):
    """
    Generates optimization suggestions.
    """
    
    # 1. Content Construction: Split into Mutable (Target) and Immutable (Context)
    mutable_content = ""
    immutable_context_content = "" 
    
    for idx, cell in enumerate(nb.cells):
        src = cell.source
        
        # Simple Labeling
        label = "CODE"
        if cell.cell_type == "markdown":
            if any(k in src for k in ["Strategy", "Idea", "Hypothesis", "Research Plan"]):
                label = "STRATEGY_DOC"
            else:
                label = "MARKDOWN_NOTE"
        elif cell.cell_type == "code":
            if "class " in src and "(nn.Module)" in src:
                label = "MODEL_DEF"
            elif "Strategy Digest" in src:
                label = "STRATEGY_DIGEST"
            elif idx < 5 and ("import " in src or "from " in src):
                label = "SETUP/IMPORTS"
            
        if idx in mutable_indices:
            # Code to be optimized
            mutable_content += f"\n# --- CELL INDEX {idx} (TARGET TO OPTIMIZE: {label}) ---\n{src}\n"
        else:
            # Read-Only Context (Critical for maintaining variable definitions)
            if cell.cell_type == "code":
                immutable_context_content += f"\n# --- CELL INDEX {idx} (READ-ONLY CONTEXT) ---\n{src}\n"

    # 2. History Context
    history_text = ""
    if history_summary:
        history_text = "\n".join([
            f"- Iter {h['iter']}: {h['critique']} (Result: {h['status']}, Score: {h['score']:.4f})" 
            for h in history_summary
        ])
    else:
        history_text = "None (Starting from Phase 2 baseline)."

    # 3. Load Prompt
    cwd_path = os.path.join(os.getcwd(), "prompts", "review_optimize.yaml")
    script_path = os.path.join(os.path.dirname(__file__), "prompts", "review_optimize.yaml")
    prompt_path = cwd_path if os.path.exists(cwd_path) else script_path
    
    sys_prompt = "You are a code optimizer. Return JSON with 'edits' key."
    user_prompt = f"Optimize this code:\n{mutable_content}"

    if os.path.exists(prompt_path):
        try:
            import yaml
            with open(prompt_path, 'r', encoding='utf-8') as f: p_data = yaml.safe_load(f)
            
            ctx = {
                "target_metric": cfg["review"]["target_metric"], 
                "current_best": str(best_metric_val),
                "iteration": str(iteration),
                "mutable_cells_content": mutable_content,
                "immutable_context_content": immutable_context_content,
                "metrics_json": json.dumps(current_metrics, indent=2),
                "history_summary": history_text
            }
            if "system" in p_data: sys_prompt = _expand_vars(p_data["system"], ctx)
            if "user_template" in p_data: user_prompt = _expand_vars(p_data["user_template"], ctx)
        except Exception as e:
            print(f"[WARN] Failed to load prompt yaml: {e}")

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"[REVIEW] Requesting optimization (Iter {iteration})...")
    
    try:
        return chat_json(
            messages, 
            cfg=cfg,
            temperature=cfg["llm"].get("temperature", 0.7)
        )
    except Exception as e:
        print(f"[REVIEW] LLM Interaction Failed: {e}")
        return None

# =============================================================================
# 7. Execution Loop Logic
# =============================================================================

def execute_and_recover(nb_path, workdir, cfg, mutable_indices=None, extra_env=None):
    """
    Manages the Execution -> Error Reporting -> Fix Loop lifecycle.
    [UPDATED] Accepts mutable_indices (locks) and extra_env (paths).
    """
    timeout = cfg["exec"]["timeout_seconds"]
    max_fixes = cfg["exec"]["max_fix_rounds"]
    
    current_nb_path = nb_path
    
    # 1. Initial Execution
    nb = nbformat.read(current_nb_path, as_version=4)
    print(f"[EXEC] Running Notebook: {current_nb_path}")
    
    # Run Pure Execution (With Environment Injection)
    executed_nb, errors = run_notebook_pure(
        nb, 
        workdir, 
        timeout, 
        cuda_device_id=None,
        extra_env=extra_env # <--- INJECTED HERE
    )
    
    # If no errors, save and return
    if not errors:
        out_path = current_nb_path.replace(".ipynb", "_exec.ipynb")
        nbformat.write(executed_nb, out_path)
        return out_path, True
    
    # 2. Fix Loop
    fix_round = 0
    current_nb_obj = executed_nb # This contains the output traces
    
    while errors and fix_round < max_fixes:
        fix_round += 1
        print(f"\n{'!'*40}")
        print(f"[EXEC] Errors Found (Round {fix_round}) - Initiating Recovery...")
        
        # A. Explicitly Dump Errors (Visible to User)
        dump_error_log(workdir, errors, round_idx=fix_round)
        
        # B. Attempt Fix (With Mutability Lock)
        fixed_nb, changed, method = attempt_fix_notebook(
            current_nb_obj, 
            errors, 
            cfg,
            mutable_indices=mutable_indices # <--- LOCKED HERE
        )
        
        if not changed:
            print(f"[EXEC] Could not generate valid fix ({method}). Stopping.")
            break
            
        print(f"[EXEC] Applying Fix ({method}) -> Rerunning...")
        
        # C. Rerun (With Environment Injection)
        current_nb_obj, errors = run_notebook_pure(
            fixed_nb, 
            workdir, 
            timeout, 
            cuda_device_id=None,
            extra_env=extra_env # <--- INJECTED HERE TOO
        )
        
        if not errors:
            print(f"[EXEC] Fixed successfully!")
            break

    # 3. Final Save
    final_out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    nbformat.write(current_nb_obj, final_out_path)
    
    if errors:
        print(f"[EXEC] Final Execution Failed. Remaining Errors: {len(errors)}")
        dump_error_log(workdir, errors, round_idx=999) # 999 indicates final failure
        return final_out_path, False
        
    return final_out_path, True

# =============================================================================
# 8. Main Optimization Loop
# =============================================================================

def optimize_loop(cfg, workspace_dir, base_nb_path):
    # [NEW] 1. Resolve H5 Path & Prepare Env
    h5_path = _resolve_h5_path(cfg)
    exec_env = {}
    if h5_path:
        exec_env["STAGE1_H5_PATH"] = h5_path
    else:
        print("[ERROR] STAGE1_H5_PATH could not be resolved! Notebook execution may fail.")

    target_metric = cfg["review"]["target_metric"]
    threshold = float(cfg["review"]["pass_threshold"])
    max_iters = int(cfg["review"]["max_iterations"])
    
    # Initialization
    current_metrics_path = os.path.join(workspace_dir, "metrics_base.json")
    best_score_so_far = get_candidate_metric_value(current_metrics_path, target_metric)
    
    # Baseline Logic
    static_baseline_score = get_baseline_metric_value(current_metrics_path, target_metric)
    if static_baseline_score is None:
        static_baseline_score = best_score_so_far
        
    best_nb_path = base_nb_path 
    history_summary = []

    baseline_display = f"{static_baseline_score:.4f}" if static_baseline_score is not None else "N/A"

    print(f"\n[LOOP] Starting Optimization.")
    print(f"       🎯 Target: {target_metric}")
    print(f"       🏁 Original Baseline: {baseline_display}")
    print(f"       🥇 Current Best:      {best_score_so_far:.4f}")

    for i in range(1, max_iters + 1):
        print(f"\n{'='*60}")
        print(f"🚀 OPTIMIZATION ROUND {i}/{max_iters}")
        print(f"{'='*60}")
        
        nb = nbformat.read(best_nb_path, as_version=4)
        mutable_indices = identify_mutable_cells(nb, cfg)
        if not mutable_indices:
            print("[ERROR] No mutable cells found.")
            break
        
        try:
            with open(current_metrics_path, 'r') as f: curr_metrics_obj = json.load(f)
        except: curr_metrics_obj = {}
            
        suggestion = generate_optimization_suggestion(
            cfg, nb, mutable_indices, curr_metrics_obj, i, best_score_so_far, workspace_dir, history_summary
        )
        
        if not suggestion or "edits" not in suggestion:
            print("[WARN] Invalid LLM response. Skipping.")
            continue
            
        critique = suggestion.get("critique", "")
        print(f"[CRITIQUE] {critique}")
        
        # Apply Edits
        nb_next = deepcopy(nb)
        applied_count = 0
        for edit in suggestion["edits"]:
            idx = int(edit["cell_index"])
            if idx in mutable_indices:
                nb_next.cells[idx].source = edit["source"]
                applied_count += 1
        
        if applied_count == 0:
            print("[WARN] No edits applied.")
            continue
            
        candidate_nb_path = os.path.join(workspace_dir, f"notebook_iter_{i}.ipynb")
        nbformat.write(nb_next, candidate_nb_path)
        
        print(f"[EXEC] Running Candidate {i}...")
        
        # Use execute_and_recover WITH ENV INJECTION and LOCKS
        executed_nb_path, success = execute_and_recover(
            nb_path=candidate_nb_path,
            workdir=workspace_dir,
            cfg=cfg,
            mutable_indices=mutable_indices, # Lock Immutable Cells
            extra_env=exec_env               # Inject Correct H5 Path & OUTPUT_DIR
        )
        
        try:
            iter_metrics_path = os.path.join(workspace_dir, f"metrics_iter_{i}.json")
            generated_metrics = os.path.join(workspace_dir, "metrics.json")
            if os.path.exists(generated_metrics):
                shutil.copy(generated_metrics, iter_metrics_path)
            
            candidate_score = get_candidate_metric_value(iter_metrics_path, target_metric)
            current_run_baseline = get_baseline_metric_value(iter_metrics_path, target_metric)
            comparison_baseline = current_run_baseline if current_run_baseline is not None else static_baseline_score
            
            # Determine Status
            status = "FAILED"
            if not success:
                 status = "CRASH"
            elif candidate_score > best_score_so_far:
                 status = "IMPROVED"
            
            # Summary
            print(f"-"*40)
            print(f"[RESULT] Iteration {i} Summary")
            print(f"  > Candidate Score: {candidate_score:.4f}")
            comp_base_disp = f"{comparison_baseline:.4f}" if comparison_baseline is not None else "N/A"
            print(f"  > Baseline Score:  {comp_base_disp}")
            print(f"  > Best Previous:   {best_score_so_far:.4f}")
            print(f"  > Verdict:         {status}")
            print(f"-"*40)
            
            write_review_log(workspace_dir, i, suggestion, candidate_score, best_score_so_far, comparison_baseline, status)
            
            history_summary.append({
                "iter": i,
                "critique": critique[:150] + "...", 
                "status": status,
                "score": candidate_score
            })

            if status == "IMPROVED":
                print(f"🎉 NEW BEST! Updating baseline for next iteration.")
                best_score_so_far = candidate_score
                best_nb_path = executed_nb_path 
                current_metrics_path = iter_metrics_path
                
                # Stopping Condition
                beats_threshold = (best_score_so_far >= threshold)
                beats_baseline = True
                if static_baseline_score is not None and static_baseline_score != -999.0:
                    beats_baseline = (best_score_so_far > static_baseline_score)
                
                if beats_threshold and beats_baseline:
                    print(f"\n[SUCCESS] Goal Reached! Score {best_score_so_far:.4f} >= Threshold ({threshold}) AND > Baseline ({static_baseline_score:.4f})")
                    break
                elif beats_threshold and not beats_baseline:
                    print(f"\n[CONTINUE] Threshold passed ({best_score_so_far:.4f} >= {threshold}), but NOT beating Baseline ({static_baseline_score:.4f}). Continuing...")
            else:
                print(f"📉 Improvement failed. Reverting to previous best.")
                
        except Exception as e:
            print(f"[ERROR] Logic Error after execution: {e}")
            history_summary.append({"iter": i, "critique": "Logic Crash", "status": "CRASH", "score": -999})

    # =============================================================================
    # 9. Finalize: Save Best Artifacts
    # =============================================================================
    print(f"\n🏁 Optimization Finished. Global Best Score: {best_score_so_far:.4f}")
    
    # 1. Save Best Notebook
    if best_nb_path and os.path.exists(best_nb_path):
        final_nb_path = os.path.join(workspace_dir, "notebook_best.ipynb")
        shutil.copy(best_nb_path, final_nb_path)
        print(f"✅ Saved BEST Notebook to: {final_nb_path}")
    else:
        print(f"⚠️ Could not locate best notebook source: {best_nb_path}")

    # 2. Save Best Metrics & Report
    if current_metrics_path and os.path.exists(current_metrics_path):
        final_met_path = os.path.join(workspace_dir, "metrics_best.json")
        shutil.copy(current_metrics_path, final_met_path)
        print(f"✅ Saved BEST Metrics to: {final_met_path}")
        
        # Regenerate report specifically for the winner
        if write_experiment_report:
            try:
                with open(final_met_path, 'r') as f: m_obj = json.load(f)
                write_experiment_report(workspace_dir, m_obj, cfg, primary_metric=target_metric)
                print(f"✅ Regenerated Experiment Report based on BEST metrics.")
            except Exception as e: 
                print(f"[WARN] Final Report gen failed: {e}")
    else:
        print(f"⚠️ Could not locate best metrics source: {current_metrics_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="review_feedback_config.json")
    parser.add_argument("--source_path", default=None, 
                        help="Explicit path to the previous run directory (overrides config)")
    args = parser.parse_args()
    
    if not os.path.exists(args.config): return
    cfg = load_full_config(args.config)

    try:
        explicit_source = args.source_path
        if not explicit_source:
            explicit_source = cfg.get("paths", {}).get("explicit_source_path")
            
        source_trial = find_best_phase2_trial(cfg, explicit_source)
        workspace, base_nb = setup_phase3_workspace(cfg, source_trial)
        optimize_loop(cfg, workspace, base_nb)
    except Exception as e:
        print(f"[FATAL] {e}")

if __name__ == "__main__":
    main()