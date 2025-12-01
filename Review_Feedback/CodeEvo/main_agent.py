import os
import sys
import shutil
import json
import yaml
import glob
from datetime import datetime
from typing import List, Dict

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
# 1. Environment Logic (带时间戳)
# =============================================================================

def setup_shadow_workspace(cfg: Dict) -> str:
    """
    创建一个“影子”工作区。
    模式：带时间戳的独立副本。
    每次运行都会在 results/CodeEvo 下创建一个新的 TranSiGen_YYYYMMDD_HHMMSS 文件夹。
    """
    source_root = cfg["target_project"]["root_dir"]
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"Target project root not found: {source_root}")

    project_name = os.path.basename(source_root.rstrip("/"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 解析相对路径：../../results/CodeEvo
    agent_ws_root = os.path.join(current_dir, cfg["agent"]["workspace_root"])
    
    # 最终路径：.../results/CodeEvo/TranSiGen_20251201_183022
    shadow_dir = os.path.join(agent_ws_root, f"{project_name}_{timestamp}")
    
    print(f"[INIT] Cloning target project to shadow workspace...")
    print(f"       Source: {source_root}")
    print(f"       Shadow: {shadow_dir}")
    
    try:
        shutil.copytree(source_root, shadow_dir, 
                       ignore=shutil.ignore_patterns('*.git', '__pycache__', '*.pyc', 'results', 'logs', 'wandb'))
    except Exception as e:
        print(f"[ERROR] Clone failed: {e}")
        raise e
        
    return shadow_dir

def read_code_context(shadow_dir: str, relative_files: List[str]) -> str:
    context_str = ""
    print(f"[READ] Reading context...")
    for rel_path in relative_files:
        full_path = os.path.join(shadow_dir, rel_path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                code = f.read()
            context_str += f"\n#Params: === FILE: {rel_path} ===\n{code}\n"
        else:
            print(f"[WARN] File not found: {full_path}")
    return context_str

# =============================================================================
# 2. Agent Logic
# =============================================================================

def generate_optimization(cfg: Dict, code_context: str) -> Dict:
    prompt_path = os.path.join(current_dir, "prompts", "code_optimizer.yaml")
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_data = yaml.safe_load(f)
        
    sys_prompt = prompt_data.get("system", "")
    user_tmpl = prompt_data.get("user_template", "")
    user_content = user_tmpl.replace("${code_context}", code_context)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]

    print(f"[AGENT] Brainstorming improvements (Model: {cfg['llm']['model']})...")
    # 使用工具函数调用 LLM
    response = chat_json(messages, cfg, temperature=cfg["llm"].get("temperature", 0.7))
    return response

def apply_and_save(shadow_dir: str, optimization_result: Dict):
    """
    1. 保存修改建议文本 (Markdown + JSON)。
    2. 覆盖代码文件。
    """
    idea = optimization_result.get("idea_summary", "No summary provided")
    modifications = optimization_result.get("modifications", [])
    
    # --- 保存建议文本 (你之前没看见的就是这个) ---
    
    # 1. 保存为 JSON (数据存档)
    json_path = os.path.join(shadow_dir, "agent_idea.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(optimization_result, f, indent=2, ensure_ascii=False)
    
    # 2. 保存为 Markdown (给你看的，更直观)
    report_path = os.path.join(shadow_dir, "OPTIMIZATION_REPORT.md")
    report_content = f"""# Agent Optimization Report
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 💡 Core Idea (Scientific Hypothesis)
{idea}

## 🛠 Modified Files
"""
    for m in modifications:
        report_content += f"- `{m.get('file_path')}`\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"[SAVE] 📝 Optimization Report saved to: {report_path}")

    # --- 覆盖代码 ---
    if not modifications:
        print("[WARN] No modifications returned by Agent.")
        return

    for mod in modifications:
        rel_path = mod.get("file_path")
        new_code = mod.get("code")
        
        if not rel_path or not new_code:
            continue
            
        full_path = os.path.join(shadow_dir, rel_path)
        
        # 写入文件（直接覆盖 target_files）
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
        
        print(f"[UPDATE] Rewrote: {rel_path}")

    return report_path

# =============================================================================
# 3. Main
# =============================================================================

def main():
    config_path = os.path.join(current_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        return
    
    cfg = load_full_config(config_path)

    try:
        # 1. 建立影子环境 (带时间戳)
        shadow_dir = setup_shadow_workspace(cfg)

        # 2. 读取上下文
        target_files = cfg["target_project"]["target_files"]
        code_context = read_code_context(shadow_dir, target_files)

        # 3. 生成优化方案
        opt_result = generate_optimization(cfg, code_context)
        
        # 4. 应用修改并保存报告
        if opt_result and isinstance(opt_result, dict):
            report_path = apply_and_save(shadow_dir, opt_result)
            
            print(f"\n{'='*60}")
            print(f"✅ MISSION COMPLETE")
            print(f"{'='*60}")
            print(f"📂 New Project Location:  {shadow_dir}")
            print(f"📄 Optimization Report:   {report_path}")
            print(f"{'='*60}\n")
        else:
            print("[FAIL] LLM did not return valid JSON.")

    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()