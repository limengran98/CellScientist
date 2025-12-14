import os
import sys
import shutil
import json
import yaml
import time
import subprocess
import glob
import csv
import argparse
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# =============================================================================
# 0. Setup Import Path
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from llm_utils import chat_json
    from config_loader import load_full_config
except ImportError as e:
    print(f"[ERROR] Import failed. Ensure llm_utils.py is in {parent_dir}")
    raise e

# =============================================================================
# 1. Helper: Metric Parser (Robust Regex Version)
# =============================================================================

class MetricParser:
    @staticmethod
    def parse(csv_content: str, target_metrics: object) -> float:
        """
        Parses CSV and extracts the first float value found in the target column.
        Handles formats like "0.7960 ± 0.0095", "0.85 (0.01)", etc.
        """
        try:
            lines = csv_content.strip().splitlines()
            if not lines: return -999.0
            
            reader = csv.DictReader(lines)
            rows = list(reader)
            if not rows: return -999.0
            
            last_row = rows[-1]
            headers = list(last_row.keys())
            
            # 1. Determine which column to look for
            candidates = []
            if isinstance(target_metrics, list):
                candidates = target_metrics
            else:
                candidates = [str(target_metrics)]
            
            found_col = None
            for col in candidates:
                if col in headers:
                    found_col = col
                    break
            
            if not found_col:
                print(f"[EVAL] Warning: None of {candidates} found in CSV headers: {headers}")
                return -999.0
            
            val_str = str(last_row[found_col])
            
            # Use regex to extract the first valid number
            match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
            
            if match:
                clean_val = float(match.group())
                print(f"       ✅ Extracted Metric: '{clean_val}' (from raw: '{val_str}') in col '{found_col}'")
                return clean_val
            else:
                print(f"[EVAL] Could not find a valid number in string: '{val_str}'")
                return -999.0

        except Exception as e:
            print(f"[EVAL] Parse Error: {e}")
            return -999.0

# =============================================================================
# 2. Helper: Checkpoint Manager & Snapshot (Unified to agent_result)
# =============================================================================

def save_run_context(agent_result_dir: str, config_path: str, prompt_path: str):
    """
    [UPDATED] Saves Config and Prompt to the unified agent_result directory.
    """
    print(f"[INIT] 📝 Archiving Config and Prompt to agent_result...")
    
    # Save Config
    dst_cfg = os.path.join(agent_result_dir, "agent_config.json")
    shutil.copy2(config_path, dst_cfg)
    
    # Save Prompt
    dst_prompt = os.path.join(agent_result_dir, "agent_prompt.yaml")
    shutil.copy2(prompt_path, dst_prompt)
    
    print(f"       Saved: {dst_cfg}")
    print(f"       Saved: {dst_prompt}")

def save_checkpoint(src_dir: str, backup_dir: str):
    """
    Saves the current code state to a backup directory (Best Checkpoint).
    [UPDATED] Excludes 'agent_result' to avoid recursive copying/bloating.
    """
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
        
    print(f"[CHECKPOINT] 💾 Saving Best State to: {backup_dir}")
    # Ignore agent specific folders to strictly backup THE CODE
    shutil.copytree(src_dir, backup_dir, dirs_exist_ok=True, 
                    ignore=shutil.ignore_patterns('agent_result', '*.git', '__pycache__', '*.pyc', 'wandb'))

def rollback_checkpoint(backup_dir: str, dest_dir: str):
    """
    Restores the code from the Best Checkpoint to the working directory.
    """
    print(f"[ROLLBACK] 🔙 Restoring content from Best Checkpoint...")
    for item in os.listdir(backup_dir):
        s = os.path.join(backup_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def snapshot_code(shadow_dir: str, agent_result_dir: str, target_files: List[str], iter_idx: int):
    """
    [UPDATED] Saves code snapshot to agent_result/iterations/iter_N/code_snapshot
    """
    snapshot_dir = os.path.join(agent_result_dir, "iterations", f"iter_{iter_idx}", "code_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    
    for rel_path in target_files:
        src_file = os.path.join(shadow_dir, rel_path)
        if os.path.exists(src_file):
            filename = os.path.basename(rel_path)
            dest_file = os.path.join(snapshot_dir, filename)
            shutil.copy2(src_file, dest_file)
    
    print(f"[SNAPSHOT] 📸 Saved code for Iteration {iter_idx} to: {snapshot_dir}")

def save_best_artifacts(shadow_dir: str, agent_result_dir: str, target_files: List[str]):
    """
    [UPDATED] Saves *_best.py to agent_result/best_code_artifacts/
    """
    dest_dir = os.path.join(agent_result_dir, "best_code_artifacts")
    os.makedirs(dest_dir, exist_ok=True)

    print(f"[ARTIFACTS] 💎 Creating '_best' copies in agent_result...")
    for rel_path in target_files:
        full_path = os.path.join(shadow_dir, rel_path)
        if os.path.exists(full_path):
            base_name = os.path.basename(full_path)
            name_part, ext_part = os.path.splitext(base_name)
            
            # e.g., model.py -> model_best.py
            best_filename = f"{name_part}_best{ext_part}"
            best_path = os.path.join(dest_dir, best_filename)
            
            shutil.copy2(full_path, best_path)
            print(f"       ✨ Generated: {best_path}")

def save_metrics_history(shadow_dir: str, agent_result_dir: str, iter_idx: int):
    """
    [UPDATED] Copies metrics CSV from iteration folder to agent_result/metrics_history/
    """
    # Note: Executor now saves results_backup to agent_result/iterations/iter_N/
    iter_dir = os.path.join(agent_result_dir, "iterations", f"iter_{iter_idx}")
    src_csv = os.path.join(iter_dir, "results_backup.csv")
    
    history_dir = os.path.join(agent_result_dir, "metrics_history")
    os.makedirs(history_dir, exist_ok=True)
    
    if os.path.exists(src_csv):
        dest_csv = os.path.join(history_dir, f"metrics_iter_{iter_idx}.csv")
        shutil.copy2(src_csv, dest_csv)
        print(f"[HISTORY] 📊 Saved metrics history: {dest_csv}")
    else:
        print(f"[HISTORY] ⚠️ No metrics file found for Iteration {iter_idx}, skipping history save.")

# =============================================================================
# 3. Executor Engine
# =============================================================================

class Executor:
    def __init__(self, work_dir: str, agent_result_dir: str, cfg: Dict):
        self.work_dir = work_dir
        self.agent_result_dir = agent_result_dir # Unified result dir
        self.cmd = cfg["execution"]["command"]
        self.log_file = cfg["execution"]["log_file"]
        self.metrics_pattern = cfg["execution"].get("metrics_probing_pattern")
        self.timeout = cfg["execution"].get("timeout_seconds", 3600)
        self.target_metric = cfg["evaluation"]["metric_column"]
        self.direction = cfg["evaluation"]["direction"]

    def _find_file_robustly(self, pattern: str) -> Optional[str]:
        target_filename = os.path.basename(pattern)
        candidates = []

        if pattern:
            search_path = os.path.join(self.work_dir, pattern)
            glob_found = glob.glob(search_path, recursive=True)
            candidates.extend(glob_found)

        for root, _, files in os.walk(self.work_dir):
            # [CRITICAL] Ignore agent_result folder to avoid finding old backups or history
            if "agent_result" in root: continue 
            
            if target_filename in files:
                candidates.append(os.path.join(root, target_filename))

        parent_dir = os.path.dirname(self.work_dir)
        potential_escape_dirs = [
            os.path.join(parent_dir, "results"),
            os.path.join(parent_dir, "output"),
            os.path.join(parent_dir, "data"),
        ]
        
        for escape_dir in potential_escape_dirs:
            if os.path.exists(escape_dir):
                for root, _, files in os.walk(escape_dir):
                    if target_filename in files:
                        candidates.append(os.path.join(root, target_filename))

        if not candidates:
            return None

        candidates = list(set(candidates))
        try:
            candidates.sort(key=lambda x: os.path.getmtime(x))
            best_file = candidates[-1]
            
            if not best_file.startswith(self.work_dir):
                print(f"       ⚠️ [ESCAPE DETECTED] File found OUTSIDE workspace: {best_file}")
                local_dest_dir = os.path.join(self.work_dir, "recovered_results")
                os.makedirs(local_dest_dir, exist_ok=True)
                local_file = os.path.join(local_dest_dir, target_filename)
                shutil.copy(best_file, local_file)
                return local_file
                
            return best_file
        except Exception as e:
            print(f"       ⚠️ Error sorting candidate files: {e}")
            return None

    def run(self, iteration_idx: int) -> Tuple[str, float]:
        print(f"[EXEC] Running Iteration {iteration_idx}...")
        print(f"       Command: {self.cmd}")
        
        # [UPDATED] Save logs inside agent_result/iterations/iter_N/
        iter_dir = os.path.join(self.agent_result_dir, "iterations", f"iter_{iteration_idx}")
        os.makedirs(iter_dir, exist_ok=True)
        log_path = os.path.join(iter_dir, self.log_file)
        
        start_time = time.time()
        exit_code = 0
        timed_out = False
        
        with open(log_path, "w", encoding="utf-8") as f_log:
            try:
                process = subprocess.Popen(
                    self.cmd, 
                    shell=True, 
                    cwd=self.work_dir, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1,
                    encoding='utf-8',
                    errors='replace'
                )

                while True:
                    if time.time() - start_time > self.timeout:
                        process.kill()
                        timed_out = True
                        msg = "\n\n[FATAL] Execution Timeout Triggered! Process killed.\n"
                        print(msg)
                        f_log.write(msg)
                        break

                    output_line = process.stdout.readline()
                    
                    if output_line == '' and process.poll() is not None:
                        break
                    
                    if output_line:
                        sys.stdout.write(output_line)
                        sys.stdout.flush()
                        f_log.write(output_line)

                if timed_out:
                    exit_code = -1
                else:
                    exit_code = process.poll()

            except Exception as e:
                err_msg = f"\n\n[FATAL] System Execution Exception: {e}\n"
                print(err_msg)
                f_log.write(err_msg)
                exit_code = -2

        duration = time.time() - start_time
        status = "SUCCESS" if exit_code == 0 else "FAILED"
        print(f"\n[EXEC] Finished. Status: {status} (Exit Code: {exit_code}) | Time: {duration:.2f}s")

        metric_val = -999.0 if self.direction == "maximize" else 999.0 
        csv_content = ""
        found_csv_path = None

        if self.metrics_pattern:
            found_csv_path = self._find_file_robustly(self.metrics_pattern)
            
            if found_csv_path:
                try:
                    with open(found_csv_path, 'r', encoding='utf-8', errors='replace') as f:
                        csv_content = f.read()
                    
                    # [UPDATED] Backup result to the agent_result iteration folder
                    shutil.copy(found_csv_path, os.path.join(iter_dir, "results_backup.csv"))
                    
                    parsed_val = MetricParser.parse(csv_content, self.target_metric)
                    if parsed_val != -999.0:
                        metric_val = parsed_val
                    else:
                        print(f"       ⚠️ Failed to extract {self.target_metric} from CSV.")
                        
                except Exception as e:
                    print(f"       ⚠️ Found CSV but failed to read/parse: {e}")
            else:
                print(f"       ⚠️ Metrics file NOT found.")

        feedback = f"=== EXECUTION REPORT (Iter {iteration_idx}) ===\n"
        feedback += f"Command: {self.cmd}\n"
        feedback += f"Exit Code: {exit_code} ({status})\n"
        
        if found_csv_path:
            feedback += f"\n\n--- METRICS FOUND ({os.path.basename(found_csv_path)}) ---\n"
            feedback += f"Target Metric ({self.target_metric}): {metric_val}\n"
            feedback += csv_content
            feedback += "\n------------------------------------------------\n"
        else:
            feedback += "\n[WARN] No metrics file found. Training may have crashed or path is wrong.\n"

        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                logs = f.read()
                tail_logs = logs[-3000:] 
                feedback += f"\n--- STDOUT/STDERR TAIL (Last 3000 chars) ---\n{tail_logs}\n"

        return feedback, metric_val

# =============================================================================
# 4. Environment & File Ops
# =============================================================================

def setup_shadow_workspace(cfg: Dict) -> str:
    source_root = cfg["target_project"]["root_dir"]
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"Target project root not found: {source_root}")

    project_name = os.path.basename(source_root.rstrip("/"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    agent_ws_root = os.path.join(current_dir, cfg["evolution"]["workspace_root"])
    shadow_dir = os.path.join(agent_ws_root, f"{project_name}_{timestamp}")
    
    print(f"[INIT] Cloning target project to shadow workspace...")
    try:
        shutil.copytree(source_root, shadow_dir, 
                       ignore=shutil.ignore_patterns(
                           '*.git', '__pycache__', '*.pyc', 'wandb', '.idea', 'logs', 'checkpoints'
                       ))
    except Exception as e:
        print(f"[ERROR] Clone failed: {e}")
        raise e
        
    return shadow_dir

def read_code_context(shadow_dir: str, relative_files: List[str]) -> str:
    context_str = ""
    for rel_path in relative_files:
        full_path = os.path.join(shadow_dir, rel_path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                code = f.read()
            context_str += f"\n#Params: === FILE: {rel_path} ===\n{code}\n"
        else:
            print(f"[WARN] Target file not found: {full_path}")
            context_str += f"\n#Params: === FILE: {rel_path} (NOT FOUND) ===\n"
    return context_str

def apply_modifications(shadow_dir: str, agent_result_dir: str, modifications: List[Dict]):
    """
    [UPDATED] Applies changes. Moves backups to agent_result/code_backups.
    """
    backup_root = os.path.join(agent_result_dir, "code_backups")
    os.makedirs(backup_root, exist_ok=True)

    for mod in modifications:
        rel_path = mod.get("file_path")
        new_code = mod.get("code")
        
        if not rel_path or not new_code:
            continue
            
        full_path = os.path.join(shadow_dir, rel_path)
        
        if os.path.exists(full_path):
            # [MODIFIED] Save .bak to agent_result instead of source tree
            ts = int(time.time())
            backup_filename = f"{os.path.basename(rel_path)}_{ts}.bak"
            backup_path = os.path.join(backup_root, backup_filename)
            shutil.copy(full_path, backup_path)
            
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
        
        print(f"[UPDATE] Rewrote: {rel_path}")

# =============================================================================
# 5. LLM Interaction (Refined for Reflection & Memory)
# =============================================================================

def generate_optimization(cfg: Dict, code_context: str, execution_feedback: str, 
                          experiment_history: List[Dict], # [NEW PARAM]
                          iter_idx: int, prompt_path: str) -> Dict:
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_data = yaml.safe_load(f)
        
    sys_prompt = prompt_data.get("system", "")
    user_tmpl = prompt_data.get("user_template", "")
    
    # [NEW LOGIC] Format History
    history_str = ""
    if not experiment_history:
        history_str = "No previous experiments. This is the first attempt."
    else:
        history_str = "--- PREVIOUS EXPERIMENTS HISTORY (READ CAREFULLY) ---\n"
        for exp in experiment_history:
            icon = "✅" if exp['improved'] else "❌"
            history_str += (f"Iteration {exp['iter']} {icon}:\n"
                            f"  - Strategy: {exp.get('idea', 'N/A')}\n"
                            f"  - Score: {exp['score']}\n"
                            f"  - Result: {'IMPROVED' if exp['improved'] else 'FAILED (Reverted)'}\n\n")

    user_content = user_tmpl.replace("${code_context}", code_context)
    user_content = user_content.replace("${execution_feedback}", execution_feedback)
    user_content = user_content.replace("${iteration_count}", str(iter_idx))
    user_content = user_content.replace("${experiment_history}", history_str) # [NEW]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]

    print(f"[AGENT] Brainstorming Iteration {iter_idx} (Model: {cfg['llm']['model']})...")
    
    try:
        response = chat_json(messages, cfg, temperature=cfg["llm"].get("temperature", 0.7))
        return response
    except Exception as e:
        print(f"[ERROR] LLM Request Failed: {e}")
        return {}

# =============================================================================
# 6. Main Evolution Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CodeEvo Agent")
    parser.add_argument("--config", type=str, default=os.path.join(current_dir, "config.json"), 
                        help="Path to the configuration file")
    parser.add_argument("--prompt", type=str, default=os.path.join(current_dir, "prompts", "code_optimizer.yaml"), 
                        help="Path to the prompt YAML file")
    args = parser.parse_args()

    config_path = args.config
    prompt_path = args.prompt

    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        return
    
    if not os.path.exists(prompt_path):
        print(f"[ERROR] Prompt not found: {prompt_path}")
        return
    
    cfg = load_full_config(config_path)

    try:
        # 1. Setup Environment
        shadow_dir = setup_shadow_workspace(cfg)
        
        # [NEW] Create Unified Agent Result Directory
        agent_result_dir = os.path.join(shadow_dir, "agent_result")
        os.makedirs(agent_result_dir, exist_ok=True)
        print(f"[INIT] 📁 Created Agent Result Directory: {agent_result_dir}")
        
        # Save Config & Prompt to agent_result
        save_run_context(agent_result_dir, config_path, prompt_path)
        
        best_checkpoint_dir = shadow_dir + "_BEST_CHECKPOINT"
        
        # Pass agent_result_dir to Executor
        executor = Executor(shadow_dir, agent_result_dir, cfg)
        
        target_files = cfg["target_project"]["target_files"]
        max_iters = cfg["evolution"]["max_iterations"]
        
        # [NEW CONFIG] Evolution Mode Control
        # If True: Runs all max_iterations, keeping the best. 
        # If False: Stops immediately after the first improvement.
        run_all_iterations = cfg["evolution"].get("run_all_iterations", True)
        
        direction = cfg["evaluation"]["direction"]
        
        best_iter_idx = 0
        experiment_history = [] # [NEW] Initialize Memory
        
        # 2. Iteration 0: Baseline Run
        print(f"\n{'='*60}")
        print(f"🏁 INITIALIZING BASELINE RUN (Iter 0)")
        print(f"{'='*60}")
        
        # Snapshot Code to agent_result
        snapshot_code(shadow_dir, agent_result_dir, target_files, 0)
        
        last_feedback, best_score = executor.run(0)
        
        # Save Baseline Metrics to agent_result
        save_metrics_history(shadow_dir, agent_result_dir, 0)
        
        # Save Initial Baseline as Best
        save_checkpoint(shadow_dir, best_checkpoint_dir)
        print(f"🏆 Initial Best Score ({executor.target_metric}): {best_score}")
        
        # 3. Optimization Loop
        for i in range(1, max_iters + 1):
            print(f"\n{'='*60}")
            print(f"🚀 EVOLUTION ITERATION {i}/{max_iters}")
            print(f"   Current Best: {best_score}")
            print(f"{'='*60}")
            
            # A. Read Code
            current_code = read_code_context(shadow_dir, target_files)
            
            # B. Contextual Feedback (Simplified, as Memory handles History)
            
            # C. LLM Generation [UPDATED CALL]
            opt_result = generate_optimization(cfg, current_code, last_feedback, experiment_history, i, prompt_path)
            
            if not opt_result or "modifications" not in opt_result:
                print("[WARN] Invalid LLM response. Retrying/Skipping.")
                if not opt_result: break 
                continue

            idea = opt_result.get("idea_summary", "No summary provided")
            print(f"\n💡 AGENT HYPOTHESIS: {idea}\n")
            
            # Save Thought to agent_result/iterations/iter_N/
            iter_log_dir = os.path.join(agent_result_dir, "iterations", f"iter_{i}")
            os.makedirs(iter_log_dir, exist_ok=True)
            with open(os.path.join(iter_log_dir, "agent_thought.json"), "w", encoding='utf-8') as f:
                json.dump(opt_result, f, indent=2, ensure_ascii=False)
            
            # D. Apply Code Changes (Backups go to agent_result)
            apply_modifications(shadow_dir, agent_result_dir, opt_result.get("modifications", []))
            
            # Snapshot Code (Modified)
            snapshot_code(shadow_dir, agent_result_dir, target_files, i)
            
            # E. Execute New Code
            current_feedback, current_score = executor.run(i)
            
            # Save Metrics History
            save_metrics_history(shadow_dir, agent_result_dir, i)
            
            # F. EVALUATION
            is_improved = False
            
            if current_score == -999.0 or current_score == 999.0:
                is_improved = False
            else:
                if direction == "maximize":
                    if current_score > best_score: is_improved = True
                else: 
                    if current_score < best_score: is_improved = True

            # [NEW] Record to History
            experiment_history.append({
                "iter": i,
                "idea": idea,
                "score": current_score,
                "improved": is_improved
            })

            if is_improved:
                print(f"\n✅ IMPROVEMENT DETECTED ({best_score} -> {current_score})")
                print(f"   Action: Keeping changes and updating checkpoint.")
                best_score = current_score
                best_iter_idx = i
                save_checkpoint(shadow_dir, best_checkpoint_dir)
                last_feedback = current_feedback
                
                # [NEW] CHECK CONFIG: Stop or Continue?
                if not run_all_iterations:
                    print(f"🛑 [CONFIG] run_all_iterations=False. Improvement found, stopping early.")
                    break
                else:
                    print(f"🔄 [CONFIG] run_all_iterations=True. Saved improvement, continuing evolution...")
                    
            else:
                print(f"\n❌ NO IMPROVEMENT ({best_score} vs {current_score})")
                print(f"   Action: Rolling back to previous best state.")
                rollback_checkpoint(best_checkpoint_dir, shadow_dir)
                last_feedback = (
                    f"*** SYSTEM NOTIFICATION ***\n"
                    f"Your last attempt FAILED to improve the metric.\n"
                    f"Attempted Score: {current_score} vs Best Score: {best_score}.\n"
                    f"Action Taken: The code has been ROLLED BACK to the previous best state.\n"
                    f"Instruction: Check the EXPERIMENT HISTORY. Do not repeat the same failure pattern.\n"
                    f"***************************\n\n"
                ) + current_feedback

        # --- [FINALIZATION PHASE] ---
        print(f"\n{'='*60}")
        print(f"🧹 FINALIZING WORKSPACE (Restoring Best State & Cleanup)")
        print(f"{'='*60}")
        
        # 1. Restore Best State
        rollback_checkpoint(best_checkpoint_dir, shadow_dir)
        
        # 2. Save Best Artifacts (Code) to agent_result
        save_best_artifacts(shadow_dir, agent_result_dir, target_files)
        
        # 3. Save & Print Best Metrics File to agent_result
        best_metrics_src = os.path.join(agent_result_dir, "metrics_history", f"metrics_iter_{best_iter_idx}.csv")
        best_metrics_dst = os.path.join(agent_result_dir, "metrics_best.csv")
        
        if os.path.exists(best_metrics_src):
            shutil.copy2(best_metrics_src, best_metrics_dst)
            print(f"\n🏆 BEST Iteration was: {best_iter_idx}")
            print(f"💾 Saved Final Best Results to: {best_metrics_dst}")
            print(f"\n--- 📄 FULL CONTENT OF metrics_best.csv ---")
            try:
                with open(best_metrics_dst, 'r', encoding='utf-8') as f:
                    print(f.read())
            except Exception as e:
                print(f"(Could not read file content: {e})")
            print(f"-------------------------------------------\n")
        else:
            print(f"⚠️ Could not locate best metrics file: {best_metrics_src}")

        # 4. Cleanup Checkpoint
        print(f"[CLEANUP] Deleting temporary checkpoint: {best_checkpoint_dir}")
        try:
            shutil.rmtree(best_checkpoint_dir)
        except Exception as e:
            print(f"[WARN] Failed to delete checkpoint folder: {e}")

        print(f"\n{'='*60}")
        print(f"✅ EVOLUTION COMPLETE")
        print(f"🏆 Final Best Score: {best_score} (Iter {best_iter_idx})")
        print(f"📂 Unified Result Folder: {agent_result_dir}")
        print(f"📂 Final Runnable Project: {shadow_dir}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"[FATAL] Main Loop Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()