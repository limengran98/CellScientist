# design_execution/llm_utils.py
import os, json, re, time, requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# =============================================================================
# Config Resolution
# =============================================================================

def resolve_llm_config(llm_cfg: dict) -> dict:
    """
    Centralized logic to resolve API keys, Base URLs, and Models.
    Priority: Config File > Environment Variables > Defaults.
    """
    llm = llm_cfg or {}
    
    # 1. API Key
    api_key = llm.get("api_key")
    if not api_key:
        # Check standard env vars
        api_key = os.environ.get("OPENAI_API_KEY")

    # 2. Model
    model = llm.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o")

    # 3. Base URL
    # [CRITICAL FIX] Prioritize config 'base_url' (proxy) over hardcoded default.
    base_url = llm.get("base_url")
    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
    # Clean markdown links if present in URL (e.g. from loose copy-paste)
    if base_url and base_url.startswith("[") and base_url.endswith(")"):
        base_url = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", base_url)

    # Hyperparameters (Preserve config or reasonable defaults)
    max_tokens = int(llm.get("max_tokens") or 20000)
    temperature = float(llm.get("temperature", 0.5)) 
    timeout = int(llm.get("timeout") or 600)

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout
    }

# =============================================================================
# Chat Functions
# =============================================================================

def _post_request(url, headers, payload, timeout=600, retries=3):
    """Internal helper for robust HTTP posting with retries."""
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            
            # Log error but retry
            print(f"[LLM] HTTP {r.status_code} (Attempt {attempt+1}/{retries}): {r.text[:200]}", flush=True)
            time.sleep(1 + attempt)
        except Exception as e:
            # Catch network errors (like Errno 101)
            print(f"[LLM] Connection Error (Attempt {attempt+1}/{retries}): {e}", flush=True)
            time.sleep(1 + attempt)
    
    raise RuntimeError(f"LLM Request failed after {retries} retries. URL: {url}")

def chat_text(
    messages: List[Dict[str, str]], 
    llm_config: Dict[str, Any], 
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 600
) -> str:
    """
    Returns raw text content from LLM. Used for Idea Synthesis and Code Generation.
    """
    resolved = resolve_llm_config(llm_config)
    
    # Allow overrides
    temp = temperature if temperature is not None else resolved["temperature"]
    toks = max_tokens if max_tokens is not None else resolved["max_tokens"]
    
    url = (resolved["base_url"] or "").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {resolved['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": resolved["model"],
        "messages": messages,
        "temperature": temp,
        "max_tokens": toks,
        "stream": False
    }

    print(f"[LLM] Text Gen -> Model: {resolved['model']} | URL: {resolved['base_url']}", flush=True)
    
    data = _post_request(url, headers, payload, timeout=timeout)
    
    try:
        content = data["choices"][0]["message"]["content"]
        return content.strip()
    except KeyError:
        print(f"[LLM] Unexpected response structure: {data}", flush=True)
        return ""

def chat_json(
    messages: List[Dict[str, str]], 
    llm_config: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Returns parsed JSON object. Used for Auto-Fix and structured data.
    Includes robust parsing (Regex, Code Fences).
    """
    resolved = resolve_llm_config(llm_config)
    
    url = (resolved["base_url"] or "").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {resolved['api_key']}",
        "Content-Type": "application/json"
    }
    
    base_payload = {
        "model": resolved["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}
    }

    # Retry loop for JSON parsing
    for attempt in range(3):
        payload = dict(base_payload)
        # Nudge on retry
        if attempt > 0:
            payload["messages"] = messages + [
                {"role": "system", "content": "You MUST return valid JSON only. Check for missing braces."}
            ]

        try:
            data = _post_request(url, headers, payload, timeout=timeout)
            
            # Robust Extraction
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = (msg.get("content") or "").strip()
            
            # 1. Direct Load
            try: return json.loads(content)
            except: pass
            
            # 2. Regex Code Block
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
            if m:
                try: return json.loads(m.group(1))
                except: pass
            
            # 3. Regex Braces
            m2 = re.search(r"(\{.*\})", content, flags=re.S)
            if m2:
                try: return json.loads(m2.group(1))
                except: pass
                
        except Exception as e:
            print(f"[LLM] JSON Parse/Net Error (Attempt {attempt+1}): {e}", flush=True)

    print("[LLM] Failed to parse JSON after retries.", flush=True)
    return {}