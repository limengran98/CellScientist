# design_execution/prompt_generator.py
import os, json, re, ast
import nbformat
from typing import Dict, Any, Tuple
from pathlib import Path

# Import centralized LLM tools
from .llm_utils import chat_text

# [OPTIMIZED] Ultra-Robust JSON Extractor
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Surgical extraction of JSON object from LLM output.
    Handles: Markdown fences, Thinking process text, Raw JSON, AND Python Dicts.
    """
    if not text:
        raise ValueError("LLM returned empty response.")

    text = text.strip()
    
    # Candidate list to try parsing
    candidates = []

    # 1. Regex: ```json ... ``` (Most reliable)
    match_fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match_fenced:
        candidates.append(match_fenced.group(1))

    # 2. Regex: Outermost braces { ... } (Greedy match for raw JSON/Dict in text)
    match_braces = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_braces:
        candidates.append(match_braces.group(1))

    # 3. The raw text itself (fallback)
    candidates.append(text)

    for candidate in candidates:
        # Strategy A: Standard JSON Parse
        try:
            return json.loads(candidate)
        except:
            pass
        
        # Strategy B: Python Literal Eval (The "Nuclear Option" for Robustness)
        # LLMs often output Python dicts (single quotes, True/False) instead of strict JSON.
        # ast.literal_eval handles this safely.
        try:
            return ast.literal_eval(candidate)
        except:
            pass

    # 4. Debug Info if all fail
    preview = text[:500] + " ... " + text[-200:]
    raise ValueError(f"Could not parse JSON (tried json.loads and ast.literal_eval). Preview:\n{preview}")

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
        if not strategy:
            print("[GEN] ⚠️ Strategy synthesis returned empty string.", flush=True)
            return ""
            
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
    
    # 1. Strategy Branching (Idea vs Freestyle)
    if raw_ideas:
        # -- MODE: IDEA DRIVEN --
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
            # Fallback if synthesis failed
            strategy_md = "**Strategy**: Fallback to Freestyle Design (Synthesis Failed)."
            full_user_content = f"# TECHNICAL SPECIFICATION\n{spec_dump}"
            print("[GEN] 💻 Generating Code (Fallback to Freestyle)...", flush=True)
    else:
        # -- MODE: FREESTYLE (No Idea File) --
        # [MODIFIED] Create a default strategy text for Freestyle mode
        strategy_md = (
            "## 🧠 Research Strategy (Freestyle)\n\n"
            "**Mode**: Expert Autonomous Design\n\n"
            "This model architecture was designed autonomously by the AI Architect based on the "
            "provided Technical Specifications, without external idea constraints. "
            "The focus is on robust baseline performance and standard best practices."
        )
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
    
    # Call LLM
    raw_text = chat_text(
        messages, 
        llm_config=cfg.get("llm", {}),
        temperature=float(pb.get("temperature", 0.4)),
        max_tokens=int(pb.get("max_tokens", 40000)),
        timeout=1200
    )
    
    # [MODIFIED] Use robust extraction with ast fallback
    try:
        nb_json = extract_json_from_text(raw_text)
    except Exception as e:
        # Save raw output for debugging
        debug_path = os.path.join(debug_dir, "LLM_GEN_FAILURE.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(raw_text or "EMPTY RESPONSE")
        print(f"[GEN] ❌ Debug info saved to {debug_path}", flush=True)
        raise RuntimeError(f"Notebook Generation Failed (JSON Parse): {e}")

    nb = nbformat.v4.new_notebook()
    cells_data = []
    
    hypergraph_data = {}
    
    if isinstance(nb_json, dict):
        cells_data = nb_json.get("cells") or nb_json.get("notebook", {}).get("cells", [])
        hypergraph_data = nb_json.get("hypergraph", {})
    elif isinstance(nb_json, list):
        cells_data = nb_json

    if hypergraph_data:
        nb.metadata["execution"] = {"hypergraph": hypergraph_data}

    # 1. Add Code Cells
    for c in cells_data:
        cell = nbformat.v4.new_code_cell(c.get("code", ""))
        subtask_meta = {
            "id": c.get("id"),
            "name": c.get("name"),
            "purpose": c.get("purpose")
        }
        cell.metadata["subtask"] = subtask_meta
        nb.cells.append(cell)
        
    # 2. [FIXED] Insert Strategy as a pure Markdown cell at index 0
    if strategy_md:
        # Check if it already has a header, if not add one
        if not strategy_md.strip().startswith("#"):
            final_md = f"# 🧠 Research Strategy\n\n{strategy_md}"
        else:
            final_md = strategy_md
            
        md_cell = nbformat.v4.new_markdown_cell(final_md)
        nb.cells.insert(0, md_cell)
    
    return nb, full_user_content, strategy_md