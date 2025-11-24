#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os, json, argparse, re, glob
import numpy as np
from design_execution.prompt_pipeline import run_prompt_pipeline as _prompt_run

# Try imports
try:
    from design_execution.prompt_pipeline import prompt_generate as _prompt_generate
    from design_execution.prompt_pipeline import prompt_execute as _prompt_execute
    from design_execution.prompt_pipeline import prompt_analyze as _prompt_analyze
except Exception:
    _prompt_generate = _prompt_execute = _prompt_analyze = None


def _expand_vars(obj, env):
    """Recursively expand ${VAR} in dictionary values."""
    if isinstance(obj, dict):
        return {k: _expand_vars(v, env) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_vars(v, env) for v in obj]
    elif isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), obj)
    return obj


def _resolve_stage1_resources(cfg: dict, enable_idea: bool = False):
    """
    Locates HDF5 data. 
    ONLY if enable_idea is True, locates idea.json and sets STAGE1_IDEA_PATH.
    """
    # 1. Get the directory
    s1_dir = cfg.get("paths", {}).get("stage1_analysis_dir")
    if not s1_dir:
        print("[SETUP][WARN] 'stage1_analysis_dir' not defined in config.")
        return

    # 2. Locate HDF5 (Always needed)
    h5_path = None
    cand_h5 = os.path.join(s1_dir, "REFERENCE_DATA.h5")
    if os.path.exists(cand_h5):
        h5_path = os.path.abspath(cand_h5)
    else:
        # Fallback: any .h5
        h5s = glob.glob(os.path.join(s1_dir, "*.h5"))
        if h5s:
            h5_path = os.path.abspath(h5s[0])
    
    if h5_path:
        os.environ["STAGE1_H5_PATH"] = h5_path
        print(f"[SETUP] Found Data Anchor: {h5_path}")
    else:
        print(f"[SETUP][WARN] No HDF5 file found in {s1_dir}")
        return

    # 3. Locate Idea JSON (Conditional)
    if enable_idea:
        # We look in the SAME directory as the found H5 file
        idea_path = os.path.join(os.path.dirname(h5_path), "idea.json")
        
        if os.path.exists(idea_path):
            os.environ["STAGE1_IDEA_PATH"] = idea_path
            print(f"[SETUP] Found Idea File:  {idea_path}")
        else:
            # Fallback to config path
            custom_idea = cfg.get("prompt_branch", {}).get("idea_file")
            if custom_idea and os.path.exists(custom_idea):
                 os.environ["STAGE1_IDEA_PATH"] = os.path.abspath(custom_idea)
                 print(f"[SETUP] Using Config Idea File: {os.environ['STAGE1_IDEA_PATH']}")
            else:
                 print(f"[SETUP][WARN] --use-idea is ON, but 'idea.json' NOT found alongside H5.")
    else:
        # Ensure var is cleared if not using idea
        if "STAGE1_IDEA_PATH" in os.environ:
            del os.environ["STAGE1_IDEA_PATH"]
        print("[SETUP] Idea-Driven Mode: OFF")


def check_success(metrics: dict, threshold: float, metric_key: str) -> tuple[bool, float]:
    """
    Check if metrics meet the target threshold.
    Returns: (is_success, current_score)
    """
    if not metrics:
        return False, -999.0
    
    # 1. Find winner or default to the first model
    winner_key = metrics.get("winner")
    if not winner_key:
        # If no winner in metrics structure, try to find the first key in models
        all_models = metrics.get("models", {})
        # Exclude non-model keys like config, trial_dir
        keys = [k for k in all_models.keys() if k not in ["config", "trial_dir"]]
        if not keys:
            return False, -999.0
        winner_key = keys[0]
        
    model_data = metrics.get(winner_key, metrics.get("models", {}).get(winner_key, {}))
    
    # 2. Get metric (Prioritize Aggregate, otherwise average over folds)
    val = None
    if "aggregate" in model_data and isinstance(model_data["aggregate"], dict):
        val = model_data["aggregate"].get(metric_key)
    
    if val is None and "per_fold" in model_data:
        # Robustness: If no aggregate field, try to calculate average manually
        folds = model_data["per_fold"]
        vals = []
        for f in folds.values():
            if isinstance(f, dict):
                v = f.get(metric_key, f.get("metrics", {}).get(metric_key))
                if v is not None: vals.append(float(v))
        if vals: val = np.mean(vals)
        
    current_score = float(val) if val is not None else -999.0
    print(f"[CHECK] Winner: {winner_key} | {metric_key}: {current_score:.4f} (Target > {threshold})")
    
    if current_score > threshold:
        return True, current_score
    return False, current_score


def cmd_run_loop(cfg: dict, prompt_path: str, use_idea: bool):
    """
    Core loop logic: Read max_iterations and threshold from config,
    exit immediately if the condition is met.
    """
    exp_cfg = cfg.get("experiment", {})
    # Default value protection
    max_iters = int(exp_cfg.get("max_iterations", 1))
    threshold = float(exp_cfg.get("success_threshold", 0.0))
    primary_metric = exp_cfg.get("primary_metric", "PCC")

    print(f"\n[LOOP] Starting Experiment Loop")
    print(f"      Max Iterations: {max_iters}")
    print(f"      Stop Condition: {primary_metric} > {threshold}")

    best_score = -float('inf')
    best_trial = None

    for i in range(1, max_iters + 1):
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {i}/{max_iters}")
        print(f"{'='*60}")

        try:
            # 1. Prepare resources (Idea / H5)
            _resolve_stage1_resources(cfg, enable_idea=use_idea)
            
            # 2. Determine Prompt path
            actual_prompt = (prompt_path or 
                             (cfg.get("prompt_branch") or {}).get("prompt_file") or 
                             "prompts/pipeline_prompt.yaml")

            # 3. Run Pipeline (Generate -> Execute -> Analyze)
            ret = _prompt_run(cfg, actual_prompt)
            
            # 4. Check results
            metrics = ret.get("metrics", {})
            trial_dir = ret.get("trial_dir")
            
            is_success, score = check_success(metrics, threshold, primary_metric)
            
            # Record best result
            if score > best_score:
                best_score = score
                best_trial = trial_dir

            # 5. Decide whether to exit
            if is_success:
                print(f"\n🎉 [SUCCESS] Iteration {i} PASSED!")
                print(f"      Metric: {primary_metric} = {score:.4f} (> {threshold})")
                print(f"      Trial:  {trial_dir}")
                print("🛑 Stopping loop early as success criteria met.")
                return # <--- Stop program immediately

            else:
                print(f"⚠️ [CONTINUE] Threshold not met ({score:.4f} <= {threshold}). Retrying...")

        except Exception as e:
            print(f"❌ [ERROR] Iteration {i} crashed: {e}")
            import traceback
            traceback.print_exc()

    # If loop finished without success
    print(f"\n🏁 [DONE] Loop finished without hitting threshold.")
    print(f"      Best {primary_metric}: {best_score:.4f}")
    print(f"      Best Trial: {best_trial}")


def main():
    print("\n[INFO] === Pipeline started ===")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Common helper
    def add_common(p):
        p.add_argument("--prompt-file", type=str, default=None)
        p.add_argument("--use-idea", action="store_true", help="Enable Idea-Driven Generation")

    # 1. RUN (Loop Mode)
    ap_r = sub.add_parser("run")
    add_common(ap_r)
    # Note: max_retries and threshold moved to config; add argument logic here if CLI override is needed later

    # 2. GENERATE (Single Step)
    ap_g = sub.add_parser("generate")
    add_common(ap_g)
    
    # 3. OTHER STEPS
    sub.add_parser("execute")
    sub.add_parser("analyze")

    args = ap.parse_args()
    
    # Load Config
    cfg_text = open(args.config, "r", encoding="utf-8").read()
    cfg = json.loads(cfg_text)
    env = dict(os.environ); env.update(cfg)
    cfg = _expand_vars(cfg, env)

    use_idea = getattr(args, "use_idea", False)

    if args.cmd == "run": 
        cmd_run_loop(cfg, prompt_path=args.prompt_file, use_idea=use_idea)

    elif args.cmd == "generate": 
        _resolve_stage1_resources(cfg, enable_idea=use_idea)
        p_path = args.prompt_file or (cfg.get("prompt_branch") or {}).get("prompt_file") or "prompts/pipeline_prompt.yaml"
        # Call original generate
        _prompt_generate(cfg, p_path)

    elif args.cmd == "execute": 
        _prompt_execute(cfg) 

    elif args.cmd == "analyze": 
        _prompt_analyze(cfg)

if __name__ == "__main__":
    main()