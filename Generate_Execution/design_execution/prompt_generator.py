# design_execution/prompt_generator.py
import os, json, re
import nbformat
from typing import Dict, Any, Tuple
from pathlib import Path

# Import centralized LLM tools
from .llm_utils import chat_text

def _strip_code_fences(s: str) -> str:
    if not s: return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|python|markdown)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
    return s

def _load_ideas_if_available() -> str:
    idea_path = os.environ.get("STAGE1_IDEA_PATH")
    if not idea_path or not os.path.exists(idea_path): 
        return ""
    try:
        with open(idea_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ideas = data.get("ideas", [])
        if not ideas: return ""
        text = ["**Candidate Ideas:**"]
        for i, idea in enumerate(ideas):
            text.append(f"{i+1}. {idea.get('title')}: {idea.get('description')}")
        return "\n".join(text)
    except: 
        return ""

def _synthesize_strategy(cfg: Dict[str, Any], raw_ideas: str, debug_dir: str) -> str:
    strategy_path = os.path.join(debug_dir, "research_strategy.md")
    prompts_map = cfg.get("prompts", {})
    idea_prompt_data = prompts_map.get("idea", {})
    sys_content = idea_prompt_data.get("system", "Act as Lead Architect.")
    user_prompt = f"Here are the available ideas:\n{raw_ideas}\n\nWrite the Strategy Summary."
    
    print("\n[GEN] 🧠 Synthesizing Research Strategy...", flush=True)
    try:
        strategy = chat_text(
            [{"role": "system", "content": sys_content}, {"role": "user", "content": user_prompt}],
            llm_config=cfg.get("llm", {}),
            temperature=0.7, 
            max_tokens=10000,
            timeout=600
        )
        with open(strategy_path, "w", encoding="utf-8") as f: f.write(strategy)
        print(f"[GEN] ✅ Strategy saved to: {strategy_path}", flush=True)
        return strategy
    except Exception as e:
        print(f"[GEN][ERROR] Strategy Synthesis Failed: {e}", flush=True)
        return ""

def generate_notebook_content(
    cfg: Dict[str, Any], 
    spec_path: str, 
    debug_dir: str
) -> Tuple[nbformat.NotebookNode, str, str]:
    
    os.makedirs(debug_dir, exist_ok=True)
    import yaml
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    
    sys_txt = spec.get("system", "You are an expert.")
    spec_dump = yaml.safe_dump({k:v for k,v in spec.items() if k!='system'}, sort_keys=False)

    raw_ideas = _load_ideas_if_available()
    strategy_md = ""
    
    if raw_ideas:
        strategy_md = _synthesize_strategy(cfg, raw_ideas, debug_dir)
        if strategy_md:
            full_user_content = f"""
================================================================================
# PART 1: RESEARCH STRATEGY (MANDATORY IMPLEMENTATION)
================================================================================
**INSTRUCTION**: You are the Lead Engineer. Implement the model described below.

{strategy_md}

================================================================================
# PART 2: TECHNICAL SPECIFICATION
================================================================================
{spec_dump}
"""
            print("[GEN] 💻 Generating Code (Strategy-Driven)...", flush=True)
        else:
            full_user_content = f"# TECHNICAL SPECIFICATION\n{spec_dump}"
            print("[GEN] 💻 Generating Code (Fallback to Freestyle)...", flush=True)
    else:
        full_user_content = f"""
================================================================================
# TECHNICAL SPECIFICATION
================================================================================
**INSTRUCTION**: Design a high-performance model based on your expert judgement.
Follow these rules strictly:

{spec_dump}
"""
        print("[GEN] 💻 Generating Code (Freestyle/No-Idea)...", flush=True)

    messages = [{"role": "system", "content": sys_txt}, {"role": "user", "content": full_user_content}]
    pb = cfg.get("prompt_branch", {})
    
    raw_text = chat_text(
        messages, 
        llm_config=cfg.get("llm", {}),
        temperature=float(pb.get("temperature", 0.4)),
        max_tokens=int(pb.get("max_tokens", 40000)),
        timeout=1200
    )
    
    try:
        cleaned = _strip_code_fences(raw_text)
        nb_json = json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(f"Notebook Generation Failed (Invalid JSON): {e}")

    nb = nbformat.v4.new_notebook()
    cells_data = []
    
    # [MODIFIED] Robustly extract cells and hypergraph structure
    hypergraph_data = {}
    
    if isinstance(nb_json, dict):
        cells_data = nb_json.get("cells") or nb_json.get("notebook", {}).get("cells", [])
        hypergraph_data = nb_json.get("hypergraph", {})
    elif isinstance(nb_json, list):
        cells_data = nb_json

    # [MODIFIED] Inject Hypergraph Metadata into Notebook Level
    if hypergraph_data:
        nb.metadata["execution"] = {"hypergraph": hypergraph_data}

    for c in cells_data:
        cell = nbformat.v4.new_code_cell(c.get("code", ""))
        
        # [MODIFIED] Inject Cell-Level Metadata for Visualization
        # We store id, name, purpose into metadata.subtask
        subtask_meta = {
            "id": c.get("id"),
            "name": c.get("name"),
            "purpose": c.get("purpose")
        }
        cell.metadata["subtask"] = subtask_meta
        
        nb.cells.append(cell)
        
    if strategy_md:
        nb.cells.insert(0, nbformat.v4.new_markdown_cell(f"# 🧠 Research Strategy\n\n{strategy_md}"))
    
    return nb, full_user_content, strategy_md