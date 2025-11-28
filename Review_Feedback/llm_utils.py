# llm_utils.py
import os, json, re, time, requests
from typing import Dict, Any, List, Optional, Union

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

    # Respect user's 300s timeout for Thinking models
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
# 2. Robust Parsing
# =============================================================================

def extract_json_from_text(text: str) -> Union[Dict, List]:
    """
    Surgical extraction of JSON from mixed LLM output (Thinking, Markdown, etc).
    """
    if not text: raise ValueError("Empty response from LLM")
    text = text.strip()

    # Strategy 1: ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass

    # Strategy 2: Outer braces { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass

    # Strategy 3: Direct Parse
    try: return json.loads(text)
    except: pass

    # Debug Preview
    preview = text[:200] + " ... " + text[-200:]
    raise ValueError(f"Failed to parse JSON. Preview:\n{preview}")

# =============================================================================
# 3. Chat Functions
# =============================================================================

def _post_request(url, headers, payload, timeout, retries=2):
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
    debug_dir: Optional[str] = None
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

    data = _post_request(url, headers, payload, conf["timeout"])
    
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
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Chat Completion returning JSON (Robust Loop).
    Used for Auto-Fix and Review Optimization.
    """
    # Force JSON mode in prompt/payload if supported, but rely on extractor
    text = chat_text(messages, cfg, temperature=temperature)
    return extract_json_from_text(text)