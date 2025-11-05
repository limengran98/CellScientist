#!/usr/bin/env python3
# review_orchestrator.py
# This module reads the completed hypergraph from Design_Analysis
# and applies an LLM-based "Expert Review" to each run.

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
import asyncio

# --- LLM Async Caller ---
async def chat_json(
    messages: List[Dict[str, str]], 
    *, 
    api_key: str, 
    base_url: Optional[str], 
    model: str, 
    temperature: float = 0.2, 
    max_tokens: int = 800
) -> Dict[str, Any]:
    """Async: Return JSON object from an OpenAI-compatible /chat/completions endpoint."""
    
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=120) as response:
                response.raise_for_status()
                data = await response.json()
        
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()

        if content:
            try:
                return json.loads(content)
            except Exception:
                pass # Fallback
        
        tools = msg.get("tool_calls") or []
        if tools and tools[0].get("function", {}).get("arguments"):
            try:
                return json.loads(tools[0]["function"]["arguments"])
            except Exception:
                pass
        
        raise ValueError(f"LLM response was not valid JSON: {content[:100]}...")

    except Exception as e:
        print(f"    > [Review] LLM call failed: {e}")
        return {"error": f"LLM call failed: {e}"}

# --- Review Logic ---

def _safe_read_text(p: str) -> str:
    """Safely read text content from a file path."""
    try:
        return Path(p).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

async def _llm_critique(
    notebook_content: str, 
    llm_cfg: Dict[str, Any],
    prompts: Dict[str, Any],
    threshold: float
) -> dict:
    """
    Perform a critique of a notebook using an LLM.
    """
    # [FIX] Apply the same smart logic for API key and URL
    api_key_val = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    base_url_val = llm_cfg.get("base_url_env")
    
    api_key = os.environ.get(api_key_val)
    if not api_key and (api_key_val.startswith("sk-") or api_key_val.startswith("fk-")):
        print("    > [Review] Using raw API key from config file.")
        api_key = api_key_val
        
    base_url = os.environ.get(base_url_val)
    if not base_url and base_url_val and (base_url_val.startswith("http://") or base_url_val.startswith("https://")):
        print("    > [Review] Using raw Base URL from config file.")
        base_url = base_url_val
    
    model = llm_cfg.get("model")
    
    sys_prompt = prompts.get("system_prompt")
    critique_template = prompts.get("critique_template")
    
    if not all([api_key, model, sys_prompt, critique_template]):
        print("⚠️ [Review] LLM critique misconfigured. Check config and prompts.")
        return {"error": "Review LLM misconfigured", "llm_used": False}

    max_content_len = 20000 # Truncate for safety
    content_to_review = notebook_content[:max_content_len]
    if len(notebook_content) > max_content_len:
        print(f"⚠️ [Review] Notebook content truncated to {max_content_len} chars for review.")

    user_prompt = critique_template.replace("{{CONTENT}}", content_to_review)
    user_prompt += f"\n\nNote: The pass/fail threshold is {threshold}."
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"    > [Review] Sending critique request to model: {model}")
    
    critique_json = await chat_json(
        messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=llm_cfg.get("temperature", 0.2),
        max_tokens=llm_cfg.get("max_tokens", 2048)
    )
    
    if "error" in critique_json:
        print(f"    ❌ [Review] LLM critique failed: {critique_json['error']}")
        critique_json["llm_used"] = False
        return critique_json

    print(f"    ✅ [Review] LLM critique received.")
    
    # Standardize score format
    scores = critique_json.get("scores", {})
    critique_json["scores"] = {
        "scientific": float(scores.get("scientific", 0.0)),
        "novelty": float(scores.get("novelty", 0.0)),
        "reproducibility": float(scores.get("reproducibility", 0.0)),
        "interpretability": float(scores.get("interpretability", 0.0)),
    }
    critique_json["overall"] = float(critique_json.get("overall", 0.0))
    critique_json["llm_used"] = True
    
    return critique_json

async def _review_run(run: Dict[str, Any], llm_cfg: Dict[str, Any], prompts: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Async wrapper to review a single run."""
    run_name = run.get("run_name", "UnknownRun")
    print(f"  🔬 [Review] Starting critique for: {run_name}")
    
    nb_path = run.get("paths", {}).get("notebook_executed_path")
    if not nb_path:
        return {"error": "Notebook path not found"}

    nb_content = _safe_read_text(nb_path)
    if not nb_content:
        return {"error": f"Notebook file empty or not found at {nb_path}"}
        
    critique = await _llm_critique(nb_content, llm_cfg, prompts, threshold)
    return critique

async def expert_review_hypergraph(config: Dict[str, Any]):
    """
    Main function to load the hypergraph, review all runs, and update the file.
    """
    llm_cfg = config.get("review_llm", {})
    prompts = config.get("prompts", {})
    threshold = config.get("review_threshold", 0.7)
    hypergraph_path_str = config.get("hypergraph_file_path")
    
    if not hypergraph_path_str:
        print("❌ [Review] 'hypergraph_file_path' not specified in review_config.json. Exiting.")
        return

    hp_path = Path(hypergraph_path_str)
    if not hp_path.exists():
        print(f"❌ [Review] Hypergraph file not found at: {hp_path}. Exiting.")
        print("Please run the Design_Analysis phase first.")
        return

    try:
        with open(hp_path, 'r', encoding='utf-8') as f:
            hypergraph = json.load(f)
    except Exception as e:
        print(f"❌ [Review] Failed to load hypergraph file: {e}")
        return
        
    runs = hypergraph.get("runs", [])
    if not runs:
        print("⚠️ [Review] No runs found in hypergraph file. Nothing to review.")
        return

    # --- Run reviews in parallel ---
    tasks = []
    for run in runs:
        tasks.append(_review_run(run, llm_cfg, prompts, threshold))
    
    critiques = await asyncio.gather(*tasks)
    
    # --- Aggregate results ---
    all_scores = []
    failed_runs = []
    for run, critique in zip(runs, critiques):
        run["expert_review"] = critique # Attach critique to the run
        if "error" not in critique:
            all_scores.append(critique.get("scores", {}))
            if critique.get("overall", 1.0) < threshold:
                failed_runs.append(critique)
    
    if not all_scores:
        print("❌ [Review] No valid scores were generated. Cannot aggregate.")
        return

    def avg(key):
        vals = [s.get(key, 0.0) for s in all_scores] or [0.0]
        return sum(vals) / len(vals)

    mean_scores = {
        "scientific": avg("scientific"),
        "novelty": avg("novelty"),
        "reproducibility": avg("reproducibility"),
        "interpretability": avg("interpretability")
    }
    overall_avg = sum(mean_scores.values()) / 4.0

    decision = "All Clear" if not failed_runs else "Revisions Needed"
    
    directives = []
    if failed_runs and config.get("generate_new_directives", False):
        print("ℹ️ [Review] Generating new directives based on failed runs...")
        # (Placeholder for a "meta-LLM" call to synthesize directives)
        # For now, just take directives from the first failed run
        directives = failed_runs[0].get("suggestions", ["No suggestions provided."])
        
    # Update the top-level hypergraph object
    hypergraph["expert_review_summary"] = {
        "review_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "overall_average_score": overall_avg,
        "mean_scores": mean_scores,
        "decision": decision,
        "failed_runs_count": len(failed_runs),
        "new_directives": directives
    }
    
    # Save the updated hypergraph file
    try:
        hp_path.write_text(json.dumps(hypergraph, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ [Review] Review complete. Hypergraph file updated: {hp_path}")
        print(f"  > Overall Average Score: {overall_avg:.3f}")
        print(f"  > Decision: {decision}")
    except Exception as e:
        print(f"❌ [Review] Failed to save updated hypergraph file: {e}")

# This allows run_review_feedback.py to import and call
if __name__ == "__main__":
    print("This module is not intended to be run directly. Run 'run_review_feedback.py'.")
    # As a fallback, try to run anyway
    try:
        THIS_DIR = Path(__file__).parent
        config_path = str(THIS_DIR / "review_config.json")
        prompts_dir = str(THIS_DIR / "prompts")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        with open(prompts_dir / "review.yml", 'r', encoding='utf-8') as f:
            import yaml
            prompts = yaml.safe_load(f)
            config["prompts"] = prompts
            
        asyncio.run(expert_review_hypergraph(config))
        
    except Exception as e:
        print(f"Error during standalone run: {e}")

