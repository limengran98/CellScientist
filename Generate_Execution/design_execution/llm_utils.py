# design_execution/llm_utils.py
import os
import json
import re
import time
import requests
import threading
from typing import Dict, Any, List, Optional

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
# Config Resolution
# =============================================================================

def resolve_llm_config(llm_cfg: dict) -> dict:
    """
    Centralized logic to resolve API keys, Base URLs, and Models.
    Priority: Config File > Environment Variables > Defaults.
    """
    llm = llm_cfg or {}

    # 1. API Key
    api_key = llm.get("api_key") or os.environ.get("OPENAI_API_KEY")

    # 2. Model
    model = llm.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o")

    # 3. Base URL
    base_url = llm.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Clean markdown links if present in URL (e.g. from loose copy-paste)
    if base_url and base_url.startswith("[") and base_url.endswith(")"):
        base_url = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", base_url)

    # Hyperparameters (Preserve config or reasonable defaults)
    max_tokens = int(llm.get("max_tokens") or 20000)
    temperature = float(llm.get("temperature", 0.5))
    timeout = int(llm.get("timeout") or 600)

    # Debug print to verify Key loading (Masked)
    if api_key:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        # Uncomment the next line if you need to debug key loading repeatedly
        # print(f"[LLM] Resolved Config - Model: {model}, Key: {masked_key}", flush=True)
    else:
        print(f"[LLM] ⚠️ WARNING: No API Key found in config or environment!", flush=True)

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
    }

# =============================================================================
# HTTP Helper
# =============================================================================

def _post_request(url: str, headers: dict, payload: dict, timeout: int = 600, retries: int = 3) -> dict:
    """Internal helper for robust HTTP posting with retries."""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()

            # Log error but retry
            print(f"[LLM] HTTP {r.status_code} (Attempt {attempt+1}/{retries}): {r.text[:200]}", flush=True)
            time.sleep(1 + attempt)
        except Exception as e:
            last_err = e
            print(f"[LLM] Connection Error (Attempt {attempt+1}/{retries}): {e}", flush=True)
            time.sleep(1 + attempt)

    raise RuntimeError(f"LLM Request failed after {retries} retries. URL: {url}. Last error: {last_err}")

# =============================================================================
# Parsing Helpers
# =============================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)

def _strip_think(text: str) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()

def _extract_json_candidates(text: str) -> List[str]:
    """Return candidate JSON-ish strings in best-first order."""
    if not text:
        return []
    t = text.strip()

    cands: List[str] = []

    # 1) fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cands.append(m.group(1))

    # 2) first {...} to last } (greedy; then we will validate)
    m2 = re.search(r"(\{.*\})", t, flags=re.DOTALL)
    if m2:
        cands.append(m2.group(1))

    # 3) raw text
    cands.append(t)
    return cands

def _escape_control_chars_in_strings(s: str) -> str:
    """
    Make a best-effort pass to escape raw control chars that frequently break JSON,
    especially newlines inside string values.
    """
    if not s:
        return s

    out = []
    in_str = False
    esc = False

    for ch in s:
        o = ord(ch)

        if in_str:
            if esc:
                # keep escape as-is
                out.append(ch)
                esc = False
                continue

            if ch == "\\":  # start escape
                out.append(ch)
                esc = True
                continue

            if ch == '"':
                out.append(ch)
                in_str = False
                continue

            # Escape raw control chars inside strings
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if o < 32:
                # drop/escape any other control char
                out.append(f"\\u{o:04x}")
                continue

            out.append(ch)
            continue

        # not in string
        if ch == '"':
            out.append(ch)
            in_str = True
            esc = False
            continue

        # remove raw control characters outside strings (except whitespace)
        if o < 32 and ch not in ("\n", "\r", "\t"):
            continue

        out.append(ch)

    return "".join(out)

def _try_parse_json_like(candidate: str) -> Optional[Dict[str, Any]]:
    """Try json.loads then ast.literal_eval-like conversion without importing ast globally."""
    if not candidate:
        return None

    # 1) direct json
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) sanitize control chars and retry json
    try:
        fixed = _escape_control_chars_in_strings(candidate)
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) python literal fallback (common: single quotes / True False None)
    try:
        import ast
        py = candidate
        # if it's JSON booleans/null, make them pythonic for literal_eval
        py = re.sub(r"\btrue\b", "True", py, flags=re.IGNORECASE)
        py = re.sub(r"\bfalse\b", "False", py, flags=re.IGNORECASE)
        py = re.sub(r"\bnull\b", "None", py, flags=re.IGNORECASE)
        obj = ast.literal_eval(py)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None

def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from messy LLM output."""
    if not text:
        return {}

    cleaned = _strip_think(text)
    for cand in _extract_json_candidates(cleaned):
        obj = _try_parse_json_like(cand)
        if obj is not None:
            return obj
    return {}

# =============================================================================
# Chat Functions (Updated)
# =============================================================================

def chat_text(
    messages: List[Dict[str, str]],
    llm_config: Dict[str, Any],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    **kwargs  # [FIX] Swallow 'meta' or other unexpected args to prevent crash
) -> str:
    """
    Returns raw text content from LLM. Used for Idea Synthesis and Code Generation.
    [MODIFIED] Tracks Usage Stats via TokenMeter.
    """
    resolved = resolve_llm_config(llm_config)

    temp = temperature if temperature is not None else resolved["temperature"]
    toks = max_tokens if max_tokens is not None else resolved["max_tokens"]

    url = (resolved["base_url"] or "").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {resolved['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": resolved["model"],
        "messages": messages,
        "temperature": temp,
        "max_tokens": toks,
        "stream": False,
    }

    print(f"[LLM] Text Gen -> Model: {resolved['model']} | URL: {resolved['base_url']}", flush=True)

    # empty-content retry loop (does not duplicate HTTP retry logic)
    for attempt in range(3):
        t0 = time.time() # [METER] Start
        try:
            data = _post_request(url, headers, payload, timeout=timeout)
            duration = time.time() - t0 # [METER] Stop

            # [METER] Record Stats
            TokenMeter.record(data, duration, resolved["model"])

            content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
            content = (content or "").strip()
            if content:
                return content

            print(f"[LLM] ⚠️ Warning: Received empty content from API (Attempt {attempt+1}/3).", flush=True)
            time.sleep(0.8 + 0.8 * attempt)
        except Exception as e:
            # Catch network/request errors inside loop to allow retries if handled by caller or just fail gracefully
            # But here _post_request handles its own retries, so if it raises, it's fatal.
            # We re-raise or handle? _post_request raises RuntimeError after retries.
            raise e

    return ""

def chat_json(
    messages: List[Dict[str, str]],
    llm_config: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 600,
    **kwargs # [FIX] Swallow 'meta' or other unexpected args
) -> Dict[str, Any]:
    """
    Returns parsed JSON object. Used for Auto-Fix and structured data.
    [MODIFIED] Tracks Usage Stats via TokenMeter.
    """
    resolved = resolve_llm_config(llm_config)

    url = (resolved["base_url"] or "").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {resolved['api_key']}",
        "Content-Type": "application/json",
    }

    base_payload = {
        "model": resolved["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(3):
        payload = dict(base_payload)

        # Nudge on retry
        if attempt > 0:
            payload["messages"] = messages + [
                {"role": "system", "content": "Return ONLY a valid JSON object. No markdown, no prose, no <think>."}
            ]

        try:
            t0 = time.time() # [METER] Start
            data = _post_request(url, headers, payload, timeout=timeout)
            duration = time.time() - t0 # [METER] Stop

            # [METER] Record Stats
            TokenMeter.record(data, duration, resolved["model"])

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = (msg.get("content") or "").strip()

            if not content:
                print(f"[LLM] ⚠️ Empty JSON content (Attempt {attempt+1}/3).", flush=True)
                time.sleep(0.8 + 0.8 * attempt)
                continue

            obj = _parse_json_from_text(content)
            if obj:
                return obj

            print(f"[LLM] JSON Parse failed (Attempt {attempt+1}/3).", flush=True)
            time.sleep(0.8 + 0.8 * attempt)

        except Exception as e:
            print(f"[LLM] JSON Parse/Net Error (Attempt {attempt+1}/3): {e}", flush=True)
            time.sleep(0.8 + 0.8 * attempt)

    print("[LLM] Failed to parse JSON after retries.", flush=True)
    return {}