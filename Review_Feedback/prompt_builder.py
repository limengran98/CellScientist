# -*- coding: utf-8 -*-
"""
Utilities for building the prompt-defined notebook.
FINAL VERSION:
1. Robust HTTP Fallback (No "None" errors).
2. Conditional Strategy Generation (Respects --use-idea flag via env var).
3. Strict Pathing for idea.json.
"""

from __future__ import annotations
import os, json, re, time
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import nbformat

try:
    import yaml
except Exception:
    yaml = None

# ---------- YAML & Env ----------

def read_yaml(path: str) -> Dict[str, Any]:
    if yaml is None: raise RuntimeError("PyYAML is required.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[PROMPT][WARN] Failed to read YAML {path}: {e}")
        return {}

def expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict): return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list): return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str): return os.path.expandvars(obj)
    return obj

# ---------- LLM Wrapper (Robust) ----------

def _strip_code_fences(s: str) -> str:
    if not s: return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|python|markdown)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
    return s

def _robust_http_fallback(messages, api_key, base_url, model, temperature, max_tokens, timeout) -> str:
    import requests
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens, "stream": False
    }
    
    print(f"\n[DEBUG] >>> SENDING TO API ({model})...")
    
    max_retries = 3
    for i in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content: return content
                print(f"[LLM] Warning: Empty content received. Retrying...")
            else:
                print(f"[LLM] HTTP Error {resp.status_code}: {resp.text}")
                
        except Exception as e:
            print(f"[LLM] Connection attempt {i+1} failed: {e}")
            time.sleep(2)
            
    raise RuntimeError("LLM API failed after retries.")

def chat_text(messages, *, cfg, timeout, debug_dir=None, temperature=0.2, max_tokens=20000, retries=2) -> str:
    provider = cfg.get("llm", {}).get("provider", "openai")
    prov_cfg = cfg.get("providers", {}).get(provider, {})
    base_url = cfg.get("llm", {}).get("base_url") or prov_cfg.get("base_url")
    api_key = cfg.get("llm", {}).get("api_key") or prov_cfg.get("api_key")
    model = cfg.get("llm", {}).get("model") or prov_cfg.get("model")

    return _robust_http_fallback(messages, api_key, base_url, model, temperature, max_tokens, timeout)

# ---------- Step 1: Idea & Strategy Logic ----------

def load_ideas_from_env() -> str:
    """Reads idea.json ONLY if STAGE1_IDEA_PATH is set (i.e., --use-idea was passed)."""
    idea_path = os.environ.get("STAGE1_IDEA_PATH")
    if not idea_path or not os.path.exists(idea_path): 
        return "" # Return empty to signal "No Idea Mode"

    try:
        with open(idea_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ideas = data.get("ideas", [])
        if not ideas: return ""

        text = ["**Candidate Ideas:**"]
        for i, idea in enumerate(ideas):
            text.append(f"{i+1}. {idea.get('title')}: {idea.get('description')}")
        return "\n".join(text)
    except: return ""

def synthesize_strategy(cfg: Dict[str, Any], raw_ideas: str, debug_dir: str) -> str:
    """Generates strategy document from raw ideas."""
    strategy_path = os.path.join(debug_dir, "research_strategy.md")
    
    # Locate prompt file
    current_file_dir = Path(__file__).resolve().parent
    root_dir = current_file_dir.parent
    prompt_file = root_dir / "prompts" / "idea.yml"
    
    if prompt_file.exists():
        yml_data = read_yaml(str(prompt_file))
        sys_content = yml_data.get("system", "Act as Lead Architect.")
    else:
        sys_content = "Act as Lead Architect."

    user_prompt = f"Here are the available ideas:\n{raw_ideas}\n\nWrite the Strategy Summary."
    
    print("\n[PROMPT] 🧠 Synthesizing Research Strategy...")
    try:
        # 1000 tokens limit as requested
        strategy = chat_text(
            [{"role": "system", "content": sys_content}, {"role": "user", "content": user_prompt}],
            cfg=cfg, timeout=600, debug_dir=debug_dir, temperature=0.7, max_tokens=10000 
        )
        with open(strategy_path, "w", encoding="utf-8") as f: f.write(strategy)
        print(f"[PROMPT] ✅ Strategy saved to: {strategy_path}")
        return strategy
    except Exception as e:
        print(f"[PROMPT][ERROR] Strategy Synthesis Failed: {e}")
        return ""

# ---------- Step 2: Notebook Generation ----------

def _wrap_cells_to_notebook(cells_spec: List[Dict[str, Any]]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    for c in cells_spec:
        nb.cells.append(nbformat.v4.new_code_cell(c.get("code", "")))
    return nb

def generate_notebook_from_prompt(cfg: Dict[str, Any], spec_path: str, debug_dir: str) -> Tuple[nbformat.NotebookNode, str]:
    os.makedirs(debug_dir, exist_ok=True)
    
    # 1. Attempt to Load Ideas
    raw_ideas = load_ideas_from_env()
    strategy_md = ""
    
    # 2. Load Technical Spec
    spec = expand_env_vars(read_yaml(spec_path))
    sys_txt = spec.get("system", "")
    spec_dump = yaml.safe_dump({k:v for k,v in spec.items() if k!='system'}, sort_keys=False)

    # 3. Branching Logic: Strategy vs Freestyle
    if raw_ideas:
        # -- MODE: IDEA DRIVEN --
        strategy_md = synthesize_strategy(cfg, raw_ideas, debug_dir)
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
            print("[PROMPT] 💻 Generating Code (Strategy-Driven)...")
        else:
            # Fallback if synthesis failed but ideas existed
            full_user_content = f"# TECHNICAL SPECIFICATION\n{spec_dump}"
            print("[PROMPT] 💻 Generating Code (Fallback to Freestyle)...")
    else:
        # -- MODE: FREESTYLE (No --use-idea or no file) --
        full_user_content = f"""
================================================================================
# TECHNICAL SPECIFICATION
================================================================================
**INSTRUCTION**: Design a high-performance model based on your expert judgement.
Follow these rules strictly:

{spec_dump}
"""
        print("[PROMPT] 💻 Generating Code (Freestyle/No-Idea)...")

    messages = [{"role": "system", "content": sys_txt}, {"role": "user", "content": full_user_content}]
    
    # 4. Generate Code
    pb = cfg.get("prompt_branch", {})
    try:
        raw_text = chat_text(
            messages, cfg=cfg, timeout=1200, debug_dir=debug_dir, 
            temperature=float(pb.get("temperature", 0.4)), 
            max_tokens=int(pb.get("max_tokens", 40000))
        )
        cleaned = _strip_code_fences(raw_text)
        nb_json = json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(f"Notebook Generation Failed: {e}")

    # 5. Build Notebook
    nb = nbformat.v4.new_notebook()
    cells = []
    if isinstance(nb_json, dict):
        if "cells" in nb_json: cells = nb_json["cells"]
        elif "notebook" in nb_json: cells = nb_json["notebook"].get("cells", [])
    elif isinstance(nb_json, list): # Handle edge case where LLM returns list
        cells = nb_json

    for c in cells:
        nb.cells.append(nbformat.v4.new_code_cell(c.get("code", "")))
        
    # Only inject Strategy Cell if we actually have one
    if strategy_md:
        nb.cells.insert(0, nbformat.v4.new_markdown_cell(f"# 🧠 Research Strategy\n\n{strategy_md}"))
    
    return nb, full_user_content