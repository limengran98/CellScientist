#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os, json, argparse, re, glob
from design_execution.prompt_pipeline import run_prompt_pipeline as _prompt_run

# Try imports
try:
    from design_execution.prompt_pipeline import prompt_generate as _prompt_generate
    from design_execution.prompt_pipeline import prompt_execute as _prompt_execute
    from design_execution.prompt_pipeline import prompt_analyze as _prompt_analyze
except Exception:
    _prompt_generate = _prompt_execute = _prompt_analyze = None


def _expand_vars(obj, env):
    if isinstance(obj, dict):
        return {k: _expand_vars(v, env) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_vars(v, env) for v in obj]
    elif isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), obj)
    return obj

# --- 核心修改：增加 enable_idea 控制 ---
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

# ---------------------------------

def cmd_prompt_defined(
    cfg: dict, 
    *, 
    prompt_path: str | None = None, 
    subcmd: str = "run", 
    use_idea: bool = False,  # <--- 接收参数
    **kwargs
):
    print(f"\n[INFO] === Starting prompt phase: {subcmd} ===")
    
    # 1. Load Resources (Controlled)
    _resolve_stage1_resources(cfg, enable_idea=use_idea)
    
    prompt_path = (prompt_path or (cfg.get("prompt_branch") or {}).get("prompt_file") or "prompts/pipeline_prompt.yaml")
    print(f"[INFO] Using prompt file: {prompt_path}")

    if subcmd == "run":
        ret = _prompt_run(cfg, prompt_path)
        print(f"[INFO] Trial directory: {ret['trial_dir']}")
        print("[INFO] === Prompt phase completed ===\n")
        return ret

    if subcmd == "generate": return _prompt_generate(cfg, prompt_path)
    elif subcmd == "execute": return _prompt_execute(cfg)
    elif subcmd == "analyze": return _prompt_analyze(cfg)
    else: raise ValueError(f"Unknown subcmd: {subcmd}")

def main():
    print("\n[INFO] === Pipeline started ===")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pipeline-mode", default="prompt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Common args helper
    def add_common(p):
        p.add_argument("--prompt-file", type=str, default="prompts/pipeline_prompt.yaml")
        p.add_argument("--use-idea", action="store_true", help="Enable Idea-Driven Generation (reads idea.json)") # <--- 新增

    ap_r = sub.add_parser("run")
    add_common(ap_r)
    # Legacy args (ignored but kept for compat)
    ap_r.add_argument("--baseline-id", type=int, default=0)
    ap_r.add_argument("--with-lit", action="store_true")
    ap_r.add_argument("--which", default="both")
    ap_r.add_argument("--prompt-text", type=str, default=None)
    ap_r.add_argument("--use-baseline", action="store_true")
    ap_r.add_argument("--use-stage1-ref", action="store_true")
    
    ap_g = sub.add_parser("generate")
    add_common(ap_g)
    
    sub.add_parser("execute")
    sub.add_parser("analyze")

    args = ap.parse_args()
    
    cfg_text = open(args.config, "r", encoding="utf-8").read()
    cfg = json.loads(cfg_text)
    env = dict(os.environ); env.update(cfg)
    cfg = _expand_vars(cfg, env)

    # Get use_idea flag safely
    use_idea = getattr(args, "use_idea", False)

    if args.cmd == "run": 
        cmd_prompt_defined(cfg, prompt_path=args.prompt_file, subcmd="run", use_idea=use_idea)
    elif args.cmd == "generate": 
        cmd_prompt_defined(cfg, prompt_path=args.prompt_file, subcmd="generate", use_idea=use_idea)
    elif args.cmd == "execute": 
        cmd_prompt_defined(cfg, subcmd="execute")
    elif args.cmd == "analyze": 
        cmd_prompt_defined(cfg, subcmd="analyze")

if __name__ == "__main__":
    main()