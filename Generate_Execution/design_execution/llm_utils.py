# design_execution/llm_utils.py
import os
import json
import re
import time
import requests
from typing import Dict, Any, List, Optional

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
# Chat Functions
# =============================================================================

def chat_text(
    messages: List[Dict[str, str]],
    llm_config: Dict[str, Any],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
) -> str:
    """
    Returns raw text content from LLM. Used for Idea Synthesis and Code Generation.

    [PATCH] Treat empty content as transient and retry a few times.
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
        data = _post_request(url, headers, payload, timeout=timeout)
        try:
            content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
            content = (content or "").strip()
            if content:
                return content

            print(f"[LLM] ⚠️ Warning: Received empty content from API (Attempt {attempt+1}/3).", flush=True)
            time.sleep(0.8 + 0.8 * attempt)
        except Exception as e:
            print(f"[LLM] Unexpected response structure (Attempt {attempt+1}/3): {e}", flush=True)
            time.sleep(0.8 + 0.8 * attempt)

    return ""

def chat_json(
    messages: List[Dict[str, str]],
    llm_config: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Returns parsed JSON object. Used for Auto-Fix and structured data.

    [PATCH] More robust extraction; also retries on empty/invalid JSON.
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
            data = _post_request(url, headers, payload, timeout=timeout)
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
