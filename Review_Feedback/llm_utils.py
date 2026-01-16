# llm_utils.py
import os
import json
import re
import time
import requests
import ast
import threading
from typing import Dict, Any, List, Optional, Union

# =============================================================================
# [NEW] Token Meter (Telemetry & Cost Tracking)
# =============================================================================

class TokenMeter:
    """
    Thread-safe singleton to track LLM usage (Tokens & Latency).
    Used to generate Cost-Accuracy Pareto Frontier data for manuscripts.
    """
    _lock = threading.Lock()
    _stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_latency_sec": 0.0,
        "api_calls": 0,
        "model_breakdown": {} 
    }

    @classmethod
    def record(cls, response: dict, latency: float, model: str):
        """Parse OpenAI-compatible usage format and record stats."""
        usage = response.get("usage", {})
        # Handle cases where usage might be None or empty
        p_tok = usage.get("prompt_tokens", 0) if usage else 0
        c_tok = usage.get("completion_tokens", 0) if usage else 0
        t_tok = usage.get("total_tokens", p_tok + c_tok) if usage else 0

        with cls._lock:
            cls._stats["prompt_tokens"] += p_tok
            cls._stats["completion_tokens"] += c_tok
            cls._stats["total_tokens"] += t_tok
            cls._stats["total_latency_sec"] += latency
            cls._stats["api_calls"] += 1
            
            # Breakdown by model
            if model not in cls._stats["model_breakdown"]:
                cls._stats["model_breakdown"][model] = {"prompt": 0, "completion": 0, "calls": 0}
            cls._stats["model_breakdown"][model]["prompt"] += p_tok
            cls._stats["model_breakdown"][model]["completion"] += c_tok
            cls._stats["model_breakdown"][model]["calls"] += 1

    @classmethod
    def get_and_reset(cls) -> dict:
        """Return current snapshot and reset counters (Call after each iteration)."""
        with cls._lock:
            snapshot = cls._stats.copy()
            # Deep copy breakdown to avoid reference issues
            snapshot["model_breakdown"] = json.loads(json.dumps(cls._stats["model_breakdown"]))
            
            # Reset
            cls._stats = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_latency_sec": 0.0,
                "api_calls": 0,
                "model_breakdown": {}
            }
        return snapshot

# =============================================================================
# 1. Config & Resolution
# =============================================================================

def resolve_llm_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Centralized config resolution.
    Respects the structure of review_feedback_config.json
    """
    llm = cfg.get("llm") or {}
    providers = cfg.get("providers") or {}
    
    # Provider resolution
    provider_name = llm.get("provider") or cfg.get("default_provider")
    if not provider_name and providers:
        provider_name = next(iter(providers.keys()))
    
    prof = providers.get(provider_name, {}) if provider_name else {}

    # Field resolution (Priority: LLM Config > Provider Config > Env Var > Default)
    model = llm.get("model") or prof.get("model") or "gpt-4"
    
    base_url = llm.get("base_url") or prof.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    api_key = (
        llm.get("api_key") 
        or prof.get("api_key") 
        or os.environ.get("OPENAI_API_KEY") 
        or os.environ.get(f"{(provider_name or 'OPENAI').upper()}_API_KEY")
    )

    # Respect user's timeout and params
    timeout = int(llm.get("timeout", prof.get("timeout", 300)))
    temperature = float(llm.get("temperature", prof.get("temperature", 0.7)))
    max_tokens = int(llm.get("max_tokens", prof.get("max_tokens", 40000)))

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "timeout": timeout,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

# =============================================================================
# 2. Robust Parsing (Enhanced)
# =============================================================================

def extract_json_from_text(text: str) -> Union[Dict, List]:
    """
    Surgical extraction of JSON from mixed LLM output.
    [UPGRADE]: Added ast.literal_eval fallback for single-quoted dicts.
    """
    if not text: 
        raise ValueError("Empty response from LLM")
    text = text.strip()

    # Pre-processing: Remove potential markdown/text wrappers
    text = re.sub(r'^[^{[]*', '', text) 
    text = re.sub(r'[^}\]]*$', '', text)

    # Strategy 1: Regex extraction of code blocks (```json ... ```)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try: 
            return json.loads(match.group(1))
        except: 
            pass

    # Strategy 2: Outer braces { ... } - Standard JSON
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        candidate = match.group(1)
        try: 
            return json.loads(candidate)
        except: 
            # Sub-strategy: Try ast.literal_eval (Handles Python dicts with single quotes)
            try: 
                return ast.literal_eval(candidate)
            except: 
                pass

    # Strategy 3: Direct Parse
    try: 
        return json.loads(text)
    except: 
        pass
    
    # Strategy 4: Fallback AST Parse (Last resort)
    try:
        return ast.literal_eval(text)
    except:
        pass

    # Debug Preview if all failed
    preview = text[:200] + " ... " + text[-200:]
    raise ValueError(f"Failed to parse JSON. Preview:\n{preview}")

# =============================================================================
# 3. Chat Functions
# =============================================================================

def _post_request(url, headers, payload, timeout, retries=2):
    """
    Internal HTTP helper.
    Note: Usage recording happens in chat_text because we need model context.
    """
    for i in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            # print(f"[LLM] HTTP {r.status_code}: {r.text[:100]}...")
            time.sleep(1)
        except Exception as e:
            print(f"[LLM] Conn Error (Attempt {i+1}): {e}")
            time.sleep(2)
    raise RuntimeError(f"LLM Request failed after {retries} retries.")

def chat_text(
    messages: List[Dict], 
    cfg: Dict[str, Any], 
    temperature: Optional[float] = None,
    debug_dir: Optional[str] = None,
    **kwargs # [FIX] Swallow 'meta' or other unexpected args
) -> str:
    """Standard Chat Completion returning String."""
    conf = resolve_llm_config(cfg)
    
    url = (conf["base_url"] or "").rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {conf['api_key']}", "Content-Type": "application/json"}
    
    payload = {
        "model": conf["model"],
        "messages": messages,
        "temperature": temperature if temperature is not None else conf["temperature"],
        "max_tokens": conf["max_tokens"],
        "stream": False
    }

    # [TELEM] Start Timer
    t0 = time.time()
    
    try:
        data = _post_request(url, headers, payload, conf["timeout"])
    except Exception as e:
        # Re-raise runtime errors from _post_request
        raise e
    
    # [TELEM] Stop Timer & Record
    duration = time.time() - t0
    TokenMeter.record(data, duration, conf["model"])
    
    # Save Debug Info
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, f"llm_resp_{int(time.time())}.json"), "w") as f:
            json.dump(data, f, indent=2)

    try:
        content = data["choices"][0]["message"]["content"]
        return content.strip()
    except:
        return ""

def chat_json(
    messages: List[Dict], 
    cfg: Dict[str, Any],
    temperature: float = 0.2,
    max_retries: int = 3,
    **kwargs # [FIX] Swallow 'meta' or other unexpected args
) -> Dict[str, Any]:
    """
    Chat Completion returning JSON with Auto-Retry Logic.
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # On retries, slightly increase temperature to get a different response
            curr_temp = temperature + (0.1 * attempt) if attempt > 0 else temperature
            
            # Call basic chat (which now handles TokenMeter)
            # Pass kwargs down to chat_text
            text = chat_text(messages, cfg, temperature=curr_temp, **kwargs)
            
            # Attempt extract
            return extract_json_from_text(text)
            
        except ValueError as e:
            last_error = e
            print(f"[LLM-JSON] Parse failed (Attempt {attempt+1}/{max_retries}). Retrying...")
            time.sleep(1)
            
    # If all retries fail, raise the last error
    print(f"[LLM-JSON] FATAL: Could not get valid JSON after {max_retries} attempts.")
    raise last_error