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
# 1. Helper: Metric Parser
# =============================================================================

class MetricParser:
    @staticmethod
    def parse(csv_content: str, target_metric: str) -> float:
        """
        Parses the CSV content string to extract the target metric value.
        Assumes the CSV has headers. logic retrieves the value from the LAST row.
        """
        try:
            lines = csv_content.strip().splitlines()
            if not lines: return -999.0
            
            # Use csv module for robust parsing
            reader = csv.DictReader(lines)
            rows = list(reader)
            if not rows: return -999.0
            
            # Get the last row (assuming it represents the final result of the run)
            last_row = rows[-1]
            
            if target_metric not in last_row:
                print(f"[EVAL] Warning: Metric '{target_metric}' not found in CSV headers: {list(last_row.keys())}")
                return -999.0
            
            val_str = last_row[target_metric]
            return float(val_str)
        except Exception as e:
            print(f"[EVAL] Parse Error: {e}")
            return -999.0

# =============================================================================
# 2. Helper: Checkpoint Manager (Snapshot & Rollback)
# =============================================================================

def save_checkpoint(src_dir: str, backup_dir: str):
    """
    Saves the current code state to a backup directory (Best Checkpoint).
    """
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
        
    print(f"[CHECKPOINT] 💾 Saving Best State (Code + Results) to: {backup_dir}")
    
    # [修复点 1] 删除了 'results'，确保结果文件被备份
    # agent_iterations 依然忽略，避免递归备份日志
    shutil.copytree(src_dir, backup_dir, dirs_exist_ok=True, 
                    ignore=shutil.ignore_patterns('agent_iterations', '*.git', '__pycache__', '*.pyc', 'wandb'))

def rollback_checkpoint(backup_dir: str, dest_dir: str):
    """
    Restores the code from the Best Checkpoint to the working directory.
    """
    print(f"[ROLLBACK] 🔙 Restoring content from Best Checkpoint...")
    
    # We overwrite files in dest_dir with files from backup_dir
    for item in os.listdir(backup_dir):
        s = os.path.join(backup_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            # Recursively copy/overwrite directories
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            # Overwrite files
            shutil.copy2(s, d)

# =============================================================================
# 3. Executor Engine (Robust Search + Streaming)
# =============================================================================

class Executor:
    def __init__(self, work_dir: str, cfg: Dict):
        self.work_dir = work_dir
        self.cmd = cfg["execution"]["command"]
        self.log_file = cfg["execution"]["log_file"]
        self.metrics_pattern = cfg["execution"].get("metrics_probing_pattern")
        self.timeout = cfg["execution"].get("timeout_seconds", 3600)
        
        # Evaluation Settings
        self.target_metric = cfg["evaluation"]["metric_column"]
        self.direction = cfg["evaluation"]["direction"] # "maximize" or "minimize"

    def _find_file_robustly(self, pattern: str) -> Optional[str]:
        """
        Robustly finds the metrics file using 3 strategies:
        1. Configured Pattern.
        2. Workspace Scan (os.walk).
        3. Parent Directory Scan (Handle 'File Escape' where script writes to ../results).
        """
        target_filename = os.path.basename(pattern)
        candidates = []

        # --- Strategy A: Configured Glob Pattern ---
        if pattern:
            search_path = os.path.join(self.work_dir, pattern)
            glob_found = glob.glob(search_path, recursive=True)
            candidates.extend(glob_found)

        # --- Strategy B: Full Workspace Scan ---
        for root, _, files in os.walk(self.work_dir):
            if target_filename in files:
                candidates.append(os.path.join(root, target_filename))

        # --- Strategy C: Parent Directory Scan (Escape Detection) ---
        # The agent runs in .../CodeEvo/TranSiGen_Time/.
        # If script writes to "../results", it goes to .../CodeEvo/results.
        parent_dir = os.path.dirname(self.work_dir) # ../
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

        # --- Final Decision: Sort by Modification Time (Latest First) ---
        # Remove duplicates
        candidates = list(set(candidates))
        try:
            # Sort by time, newest at the end
            candidates.sort(key=lambda x: os.path.getmtime(x))
            best_file = candidates[-1]
            
            # If the file was found OUTSIDE the workspace, bring it back
            if not best_file.startswith(self.work_dir):
                print(f"       ⚠️ [ESCAPE DETECTED] File found OUTSIDE workspace: {best_file}")
                print(f"       🔄 Copying it back to workspace for safety...")
                
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
        """
        Executes command, STREAMS output to console AND file, then finds metrics.
        """
        print(f"[EXEC] Running Iteration {iteration_idx}...")
        print(f"       Command: {self.cmd}")
        
        iter_dir = os.path.join(self.work_dir, "agent_iterations", f"iter_{iteration_idx}")
        os.makedirs(iter_dir, exist_ok=True)
        log_path = os.path.join(iter_dir, self.log_file)
        
        start_time = time.time()
        exit_code = 0
        timed_out = False
        
        # 1. Execute Shell Command with Real-time Streaming
        with open(log_path, "w", encoding="utf-8") as f_log:
            try:
                # Merge stderr into stdout
                process = subprocess.Popen(
                    self.cmd, 
                    shell=True, 
                    cwd=self.work_dir, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1, # Line buffered
                    encoding='utf-8',
                    errors='replace'
                )

                # Real-time loop
                while True:
                    # Check for timeout
                    if time.time() - start_time > self.timeout:
                        process.kill()
                        timed_out = True
                        msg = "\n\n[FATAL] Execution Timeout Triggered! Process killed.\n"
                        print(msg)
                        f_log.write(msg)
                        break

                    # Read line from pipe
                    output_line = process.stdout.readline()
                    
                    # If empty string and process finished, break
                    if output_line == '' and process.poll() is not None:
                        break
                    
                    if output_line:
                        # Print to Console (Streaming)
                        sys.stdout.write(output_line)
                        sys.stdout.flush()
                        
                        # Write to Log File
                        f_log.write(output_line)

                # Get exit code
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

        # 2. Probe for Metrics File (Using Robust Search)
        metric_val = -999.0 if self.direction == "maximize" else 999.0 # Default bad value
        csv_content = ""
        found_csv_path = None

        if self.metrics_pattern:
            found_csv_path = self._find_file_robustly(self.metrics_pattern)
            
            if found_csv_path:
                mod_time = datetime.fromtimestamp(os.path.getmtime(found_csv_path)).strftime('%H:%M:%S')
                print(f"       🎯 Found Metrics File ({mod_time}): {found_csv_path}")

                try:
                    with open(found_csv_path, 'r', encoding='utf-8', errors='replace') as f:
                        csv_content = f.read()
                    
                    # Backup result
                    shutil.copy(found_csv_path, os.path.join(iter_dir, "results_backup.csv"))
                    
                    # Parse Metric
                    parsed_val = MetricParser.parse(csv_content, self.target_metric)
                    if parsed_val != -999.0:
                        metric_val = parsed_val
                        print(f"       📊 Extracted {self.target_metric}: {metric_val}")
                    else:
                        print(f"       ⚠️ Failed to extract {self.target_metric} from CSV.")
                        
                except Exception as e:
                    print(f"       ⚠️ Found CSV but failed to read/parse: {e}")
            else:
                print(f"       ⚠️ Metrics file NOT found anywhere in project.")
                print(f"          (Searched for: {os.path.basename(self.metrics_pattern)})")

        # 3. Construct Feedback Report for LLM
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

        # Add Log Tail (Last 3000 chars) for LLM Context
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
    print(f"       Source: {source_root}")
    print(f"       Shadow: {shadow_dir}")
    
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

def apply_modifications(shadow_dir: str, modifications: List[Dict]):
    for mod in modifications:
        rel_path = mod.get("file_path")
        new_code = mod.get("code")
        
        if not rel_path or not new_code:
            continue
            
        full_path = os.path.join(shadow_dir, rel_path)
        
        if os.path.exists(full_path):
            shutil.copy(full_path, full_path + ".bak")
            
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
        
        print(f"[UPDATE] Rewrote: {rel_path}")

# =============================================================================
# 5. LLM Interaction
# =============================================================================

def generate_optimization(cfg: Dict, code_context: str, execution_feedback: str, iter_idx: int, prompt_path: str) -> Dict:
    # prompt_path is now passed from main()
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_data = yaml.safe_load(f)
        
    sys_prompt = prompt_data.get("system", "")
    user_tmpl = prompt_data.get("user_template", "")
    
    user_content = user_tmpl.replace("${code_context}", code_context)
    user_content = user_content.replace("${execution_feedback}", execution_feedback)
    user_content = user_content.replace("${iteration_count}", str(iter_idx))

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
        
        # Define Checkpoint Directory (for rolling back)
        best_checkpoint_dir = shadow_dir + "_BEST_CHECKPOINT"
        
        executor = Executor(shadow_dir, cfg)
        target_files = cfg["target_project"]["target_files"]
        max_iters = cfg["evolution"]["max_iterations"]
        direction = cfg["evaluation"]["direction"] # "maximize" or "minimize"
        
        # 2. Iteration 0: Baseline Run
        print(f"\n{'='*60}")
        print(f"🏁 INITIALIZING BASELINE RUN (Iter 0)")
        print(f"{'='*60}")
        
        last_feedback, best_score = executor.run(0)
        
        # Save Initial Baseline as Best
        save_checkpoint(shadow_dir, best_checkpoint_dir)
        print(f"🏆 Initial Best Score ({executor.target_metric}): {best_score}")
        
        # 3. Optimization Loop
        for i in range(1, max_iters + 1):
            print(f"\n{'='*60}")
            print(f"🚀 EVOLUTION ITERATION {i}/{max_iters}")
            print(f"   Current Best: {best_score}")
            print(f"{'='*60}")
            
            # A. Read Code (From Current/Restored State)
            current_code = read_code_context(shadow_dir, target_files)
            
            # B. Inject Context into Feedback (Comparison for LLM)
            contextual_feedback = (
                f"*** HISTORY CONTEXT ***\n"
                f"Current Best Metric ({executor.target_metric}): {best_score}\n"
                f"Goal: {direction} this metric.\n"
                f"***********************\n\n"
            ) + last_feedback
            
            # C. LLM Analysis & Generation
            opt_result = generate_optimization(cfg, current_code, contextual_feedback, i, prompt_path)
            
            if not opt_result or "modifications" not in opt_result:
                print("[WARN] Invalid LLM response. Retrying/Skipping.")
                if not opt_result: break 
                continue

            idea = opt_result.get("idea_summary", "No summary provided")
            print(f"\n💡 AGENT HYPOTHESIS: {idea}\n")
            
            # Save Thought
            iter_log_dir = os.path.join(shadow_dir, "agent_iterations", f"iter_{i}")
            os.makedirs(iter_log_dir, exist_ok=True)
            with open(os.path.join(iter_log_dir, "agent_thought.json"), "w", encoding='utf-8') as f:
                json.dump(opt_result, f, indent=2, ensure_ascii=False)
            
            # D. Apply Code Changes
            apply_modifications(shadow_dir, opt_result.get("modifications", []))
            
            # E. Execute New Code
            current_feedback, current_score = executor.run(i)
            
            # F. EVALUATION & SELECTION
            is_improved = False
            
            # Handle default bad values (e.g. crash)
            if current_score == -999.0 or current_score == 999.0:
                is_improved = False
            else:
                if direction == "maximize":
                    if current_score > best_score: is_improved = True
                else: # minimize
                    if current_score < best_score: is_improved = True

            if is_improved:
                print(f"\n✅ IMPROVEMENT DETECTED ({best_score} -> {current_score})")
                print(f"   Action: Keeping changes and updating checkpoint.")
                best_score = current_score
                save_checkpoint(shadow_dir, best_checkpoint_dir)
                
                # Feedback for next round is just the standard report
                last_feedback = current_feedback
            else:
                print(f"\n❌ NO IMPROVEMENT ({best_score} vs {current_score})")
                print(f"   Action: Rolling back to previous best state.")
                rollback_checkpoint(best_checkpoint_dir, shadow_dir)
                
                # Feedback must explicitly tell LLM it failed
                last_feedback = (
                    f"*** SYSTEM NOTIFICATION ***\n"
                    f"Your last attempt FAILED to improve the metric.\n"
                    f"Attempted Score: {current_score} vs Best Score: {best_score}.\n"
                    f"Action Taken: The code has been ROLLED BACK to the previous best state.\n"
                    f"Instruction: Try a DIFFERENT approach. Do not repeat the same logic.\n"
                    f"***************************\n\n"
                ) + current_feedback

        # --- [修复点 2] 最终收尾工作 ---
        print(f"\n{'='*60}")
        print(f"🧹 FINALIZING WORKSPACE (Restoring Best State & Cleanup)")
        print(f"{'='*60}")
        
        # 1. 再次执行回滚，确保主文件夹里是 Best State
        rollback_checkpoint(best_checkpoint_dir, shadow_dir)
        
        # 2. 删除临时的 checkpoint 文件夹
        print(f"[CLEANUP] Deleting temporary checkpoint: {best_checkpoint_dir}")
        try:
            shutil.rmtree(best_checkpoint_dir)
        except Exception as e:
            print(f"[WARN] Failed to delete checkpoint folder: {e}")

        print(f"\n{'='*60}")
        print(f"✅ EVOLUTION COMPLETE")
        print(f"🏆 Final Best Score: {best_score}")
        print(f"📂 Final Project Location: {shadow_dir}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"[FATAL] Main Loop Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()