#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import re
import glob
import shutil
import time
import nbformat
import numpy as np
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# =============================================================================
# Import User's Existing Utilities (Robustness Layer)
# =============================================================================
try:
    from llm_client import LLMClient, resolve_llm_from_cfg
    from nb_autofix import execute_with_autofix
except ImportError:
    raise ImportError("Missing required modules: 'llm_client.py' or 'nb_autofix.py'. Please ensure they are in the same directory.")

# [NEW] Import Report Generator
try:
    from experiment_report import write_experiment_report
except ImportError:
    print("[WARN] 'experiment_report.py' not found. Report generation will be disabled.")
    write_experiment_report = None

# =============================================================================
# 1. Logging & Analysis Logic (Review Details)
# =============================================================================

def write_review_log(workspace, iteration, suggestion, score, baseline, status):
    """
    Writes a human-readable log of what the LLM changed and the result.
    """
    log_path = os.path.join(workspace, "optimization_history.md")
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    critique = suggestion.get("critique", "No critique provided.")
    edits = suggestion.get("edits", [])
    
    log_entry = f"""
## Iteration {iteration} (Time: {timestamp})
**Status**: {status}
**Metric**: {score:.4f} (Baseline: {baseline:.4f})

### 1. Critique
> {critique}

### 2. Applied Edits
"""
    for edit in edits:
        idx = edit.get("cell_index")
        src = edit.get("source", "")
        # Indent code for display
        src_block = "\n".join(["    " + line for line in src.splitlines()])
        log_entry += f"\n- **Cell {idx}**:\n```python\n{src_block}\n```\n"

    log_entry += "\n---\n"
    
    # Append to file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)
    
    print(f"[LOG] Review details written to: {log_path}")


# =============================================================================
# 2. Path & Resource Management
# =============================================================================

def _expand_vars(obj, env):
    """Recursively expand ${VAR} in config."""
    if isinstance(obj, dict): return {k: _expand_vars(v, env) for k, v in obj.items()}
    if isinstance(obj, list): return [_expand_vars(v, env) for v in obj]
    if isinstance(obj, str): return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), obj)
    return obj

def find_best_phase2_trial(cfg):
    """
    Locates the most relevant trial from Phase 2 to optimize.
    Priority: Latest run that has metrics.json.
    """
    gen_root = cfg["paths"]["generate_execution_root"]
    prompt_root = os.path.join(gen_root, "prompt")
    
    # List all runs sorted by time (descending)
    runs = sorted(glob.glob(os.path.join(prompt_root, "prompt_run_*")), reverse=True)
    
    print(f"[INIT] Searching for Phase 2 results in: {prompt_root}")
    
    for run in runs:
        # We look for the EXECUTED notebook, not the prompt one
        nb_path = os.path.join(run, "notebook_prompt_exec.ipynb")
        metrics_path = os.path.join(run, "metrics.json")
        
        if os.path.exists(nb_path) and os.path.exists(metrics_path):
            print(f"[INIT] Found valid Phase 2 source: {run}")
            return run
            
    raise FileNotFoundError("No valid Phase 2 execution results found (looking for notebook_prompt_exec.ipynb and metrics.json)")

def setup_phase3_workspace(cfg, source_trial_path):
    """
    Creates the review_feedback workspace and copies the base notebook.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = cfg["paths"]["review_feedback_root"]
    workspace = os.path.join(out_root, f"review_run_{ts}")
    os.makedirs(workspace, exist_ok=True)
    
    # Copy base assets
    src_nb = os.path.join(source_trial_path, "notebook_prompt_exec.ipynb")
    dst_nb = os.path.join(workspace, "notebook_base.ipynb")
    shutil.copy(src_nb, dst_nb)
    
    # Copy metrics for reference
    src_met = os.path.join(source_trial_path, "metrics.json")
    shutil.copy(src_met, os.path.join(workspace, "metrics_base.json"))
    
    # Init Log File
    with open(os.path.join(workspace, "optimization_history.md"), "w") as f:
        f.write(f"# Optimization History\nSource Trial: {source_trial_path}\n\n")
    
    print(f"[INIT] Workspace initialized at: {workspace}")
    return workspace, dst_nb

# =============================================================================
# 3. Notebook Analysis & Cell Routing (Safety Layer)
# =============================================================================

def identify_mutable_cells(nb, cfg):
    """
    Identifies which cells are 'Modeling/Innovation' (Mutable) vs 'Data/Eval' (Immutable).
    Supports explicit override via 'force_cells'.
    """
    # [NEW] Manual Override Logic
    force_indices = cfg["review"].get("force_cells")
    if force_indices and isinstance(force_indices, list) and len(force_indices) > 0:
        print(f"[ROUTER] ⚠️ MANUAL OVERRIDE: Forcing focus on cells {force_indices}")
        # Validate indices
        valid_indices = [i for i in force_indices if 0 <= i < len(nb.cells)]
        if len(valid_indices) != len(force_indices):
            print(f"[WARN] Some forced indices were out of bounds. Valid: {valid_indices}")
        return valid_indices

    # Standard Heuristic Logic
    protected_keywords = cfg["review"].get("protected_sections", [])
    target_keywords = cfg["review"].get("target_sections", ["SECTION 3", "Model", "Innovation"])
    
    mutable_indices = []
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code": continue
        src = cell.source
        
        # [CRITICAL] Safety Check: Explicitly protect Baseline
        if "baseline" in src.lower() and "class" in src.lower():
            continue

        # Heuristic 1: Explicit Section Headers
        is_target = any(k in src for k in target_keywords)
        is_protected = any(k in src for k in protected_keywords)
        
        # Heuristic 2: Code Content Analysis
        # If it defines a class inheriting from nn.Module, it's likely a model
        has_torch_nn = "import torch.nn" in src or "class " in src and "(nn.Module)" in src
        
        # Logic: It is mutable if it looks like a model AND isn't explicitly protected
        if (is_target or has_torch_nn) and not is_protected:
            mutable_indices.append(i)
            
    print(f"[ROUTER] Mutable Cell Indices (Auto-Detected): {mutable_indices}")
    if not mutable_indices:
        print("[WARN] No mutable cells found! The LLM will have nothing to edit. Check keywords in config.")
        
    return mutable_indices

def get_metric_value(metrics_path, metric_name="PCC"):
    """Robustly extracts the primary metric."""
    if not os.path.exists(metrics_path):
        return -999.0
        
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        # Determine winner model
        winner = data.get("winner")
        if not winner:
            models = data.get("models", {})
            if models:
                winner = list(models.keys())[0]
            else:
                return -999.0
            
        # Get value
        model_data = data["models"][winner]
        val = None
        
        # Try aggregate
        if "aggregate" in model_data:
            val = model_data["aggregate"].get(metric_name)
        
        # Try per_fold mean
        if val is None and "per_fold" in model_data:
             vals = [float(x.get(metric_name, 0)) for x in model_data["per_fold"].values() if metric_name in x]
             if vals: val = np.mean(vals)
             
        return float(val) if val is not None else -999.0
    except Exception as e:
        print(f"[METRIC] Failed to read metric from {metrics_path}: {e}")
        return -999.0

# =============================================================================
# 4. LLM Optimization Logic
# =============================================================================

def generate_optimization_suggestion(cfg, llm_client, nb, mutable_indices, current_metrics, iteration, best_metric_val, workspace):
    """
    Calls LLM to review the mutable code and suggest improvements.
    """
    # Prepare Code Context
    mutable_content = ""
    for idx in mutable_indices:
        mutable_content += f"\n# --- CELL INDEX {idx} ---\n{nb.cells[idx].source}\n"
    
    # Load Prompt
    prompt_path = "prompts/review_optimize.yaml"
    if not os.path.exists(prompt_path):
        print(f"[WARN] Prompt file {prompt_path} not found. Using fallback.")
        sys_prompt = "You are a code optimizer. Return JSON with 'edits' key."
        user_prompt = f"Optimize this code to improve {cfg['review']['target_metric']}:\n{mutable_content}"
    else:
        import yaml
        with open(prompt_path, 'r') as f:
            p_data = yaml.safe_load(f)
            sys_prompt = _expand_vars(p_data["system"], {
                "target_metric": cfg["review"]["target_metric"], 
                "current_best": str(best_metric_val)
            })
            if "user_template" not in p_data:
                 print("[WARN] 'user_template' key missing in yaml. Using simple concat.")
                 user_prompt = f"ITERATION {iteration}\nMETRICS: {json.dumps(current_metrics)}\nCODE:\n{mutable_content}"
            else:
                user_prompt = _expand_vars(p_data["user_template"], {
                    "iteration": str(iteration),
                    "target_metric": cfg["review"]["target_metric"],
                    "current_best": str(best_metric_val),
                    "mutable_cells_content": mutable_content,
                    "metrics_json": json.dumps(current_metrics, indent=2)
                })

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"[REVIEW] Requesting optimization from LLM (Iter {iteration})...")
    
    # [DEBUG] Define a debug directory for raw responses
    debug_dir = os.path.join(workspace, "llm_debug")
    
    try:
        response_str = llm_client.chat(
            messages, 
            enforce_json=False, 
            temperature=cfg["llm"].get("temperature", 0.7),
            debug_dir=debug_dir # <--- Pass debug dir to client
        )
        
        cleaned = response_str.strip()
        
        # Regex to find JSON block
        json_match = re.search(r"```json(.*?)```", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1).strip()
        elif cleaned.startswith("```"): 
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"): 
            cleaned = cleaned[:-3].strip()
            
        # [IMPROVED] Brute force JSON extraction if regex fails
        # Find first '{' and last '}'
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx:end_idx+1]
        
        return json.loads(cleaned)
    except Exception as e:
        print(f"[REVIEW] LLM Interaction Failed: {e}")
        return None

# =============================================================================
# 5. Main Optimization Loop
# =============================================================================

def optimize_loop(cfg, workspace_dir, base_nb_path):
    """
    The core Review-Feedback Loop.
    """
    llm_params = resolve_llm_from_cfg(cfg)
    # [CRITICAL] Ensure timeout is sufficient
    print(f"[CONFIG] LLM Timeout set to: {llm_params['timeout']}s")
    client = LLMClient(**llm_params)
    
    target_metric = cfg["review"]["target_metric"]
    threshold = float(cfg["review"]["pass_threshold"])
    max_iters = int(cfg["review"]["max_iterations"])
    
    # Load Initial State
    current_metrics_path = os.path.join(workspace_dir, "metrics_base.json")
    best_score = get_metric_value(current_metrics_path, target_metric)
    best_nb_path = base_nb_path
    best_metrics_path = current_metrics_path 
    
    print(f"\n[LOOP] Starting Optimization. Baseline {target_metric}: {best_score}")

    for i in range(1, max_iters + 1):
        print(f"\n{'='*60}")
        print(f"🚀 OPTIMIZATION ROUND {i}/{max_iters}")
        print(f"{'='*60}")
        
        # 1. Load current best notebook
        nb = nbformat.read(best_nb_path, as_version=4)
        
        # 2. Identify Cells to optimize
        mutable_indices = identify_mutable_cells(nb, cfg)
        if not mutable_indices:
            print("[ERROR] No mutable modeling cells found. Cannot optimize.")
            break
            
        # 3. Get LLM Suggestion
        try:
            with open(current_metrics_path, 'r') as f: curr_metrics_obj = json.load(f)
        except:
            curr_metrics_obj = {}
            
        suggestion = generate_optimization_suggestion(
            cfg, client, nb, mutable_indices, curr_metrics_obj, i, best_score, workspace_dir
        )
        
        if not suggestion or "edits" not in suggestion:
            print("[WARN] No valid edits received. Skipping round.")
            write_review_log(workspace_dir, i, {"critique": "LLM Failed or Returned Invalid JSON"}, best_score, best_score, "SKIPPED")
            continue
            
        print(f"[CRITIQUE] {suggestion.get('critique', 'No critique provided.')}")
        
        # 4. Apply Edits (Safety Protected)
        nb_next = deepcopy(nb)
        applied_count = 0
        for edit in suggestion["edits"]:
            try:
                idx = int(edit["cell_index"])
                new_src = edit["source"]
                if idx in mutable_indices:
                    nb_next.cells[idx].source = new_src
                    applied_count += 1
                    print(f"[APPLY] Modified Cell {idx}")
                else:
                    print(f"[BLOCK] LLM tried to edit Immutable Cell {idx}. Ignored.")
            except Exception as e:
                print(f"[WARN] Failed to apply edit: {e}")
        
        if applied_count == 0:
            print("[WARN] No edits were applied. Skipping execution.")
            write_review_log(workspace_dir, i, suggestion, best_score, best_score, "NO_CHANGES")
            continue
            
        # 5. Save Candidate Notebook
        candidate_nb_path = os.path.join(workspace_dir, f"notebook_iter_{i}.ipynb")
        nbformat.write(nb_next, candidate_nb_path)
        
        # 6. Execute (Robust Run with AutoFix)
        print(f"[EXEC] Running Candidate {i} with AutoFix...")
        try:
            executed_nb_path = execute_with_autofix(
                ipynb_path=candidate_nb_path,
                workdir=workspace_dir,
                timeout_seconds=cfg["exec"]["timeout_seconds"],
                max_fix_rounds=cfg["exec"]["max_fix_rounds"],
                phase_cfg=cfg,
                verbose=True
            )
            
            # 7. Evaluate
            cand_metrics_path = os.path.join(workspace_dir, "metrics.json")
            iter_metrics_path = os.path.join(workspace_dir, f"metrics_iter_{i}.json")
            if os.path.exists(cand_metrics_path):
                shutil.copy(cand_metrics_path, iter_metrics_path)
            
            score = get_metric_value(iter_metrics_path, target_metric)
            print(f"[RESULT] Iteration {i} Score: {score}")
            
            # [LOGGING] Write detailed history
            status = "IMPROVED" if score > best_score else "FAILED"
            write_review_log(workspace_dir, i, suggestion, score, best_score, status)

            # 8. Compare & Update
            if score > best_score:
                print(f"🎉 NEW BEST! {target_metric}={score:.4f} > {best_score:.4f}")
                best_score = score
                best_nb_path = executed_nb_path 
                current_metrics_path = iter_metrics_path
                best_metrics_path = iter_metrics_path 
                
                # [NEW] Generate Report for this winner
                if write_experiment_report and os.path.exists(best_metrics_path):
                    try:
                        with open(best_metrics_path, 'r') as f: m_obj = json.load(f)
                        report_path = write_experiment_report(workspace_dir, m_obj, cfg, primary_metric=target_metric)
                        print(f"[REPORT] Updated experiment report: {report_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to generate interim report: {e}")

                if best_score >= threshold:
                    print(f"[SUCCESS] Threshold {threshold} reached. Stopping.")
                    break
            else:
                print(f"📉 Improvement failed ({score:.4f} <= {best_score:.4f}). Reverting to previous best.")
                
        except Exception as e:
            print(f"[ERROR] Execution failed for Iteration {i}: {e}")
            write_review_log(workspace_dir, i, suggestion, -999, best_score, f"CRASH: {str(e)}")

    print(f"\n🏁 Optimization Finished. Best {target_metric}: {best_score}")
    print(f"📂 Final Notebook: {best_nb_path}")
    print(f"📂 Review History: {os.path.join(workspace_dir, 'optimization_history.md')}")
    
    # [NEW] Final Report Generation
    if write_experiment_report and os.path.exists(best_metrics_path):
        print(f"\n[INFO] Generating Final Experiment Report...")
        try:
            with open(best_metrics_path, 'r') as f: m_obj = json.load(f)
            report_path = write_experiment_report(workspace_dir, m_obj, cfg, primary_metric=target_metric)
            print(f"📝 Final Report: {report_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate final report: {e}")

# =============================================================================
# Main Entry
# =============================================================================

def main():
    print("\n[INFO] === Phase 3: Review & Feedback Started ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="review_feedback_config.json")
    args = parser.parse_args()
    
    # 1. Load Config
    if not os.path.exists(args.config):
        print(f"[FATAL] Config file {args.config} not found.")
        return

    with open(args.config, "r", encoding='utf-8') as f:
        cfg = json.load(f)
    
    # Env Var Expansion
    cfg_env = dict(os.environ); cfg_env.update(cfg)
    cfg = _expand_vars(cfg, cfg_env)

    # 2. Find Phase 2 Result
    try:
        source_trial = find_best_phase2_trial(cfg)
    except Exception as e:
        print(f"[FATAL] {e}")
        return

    # 3. Setup Phase 3 Workspace
    workspace, base_nb = setup_phase3_workspace(cfg, source_trial)
    
    # 4. Run Loop
    optimize_loop(cfg, workspace, base_nb)

if __name__ == "__main__":
    main()