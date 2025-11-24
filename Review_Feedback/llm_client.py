import os, json, time, requests

class LLMUnavailable(Exception):
    pass

def resolve_llm_from_cfg(cfg: dict) -> dict:
    """
    Parse LLM connection parameters.
    """
    cfg = cfg or {}
    llm = (cfg.get("llm") or {})
    providers_map = (cfg.get("providers") or {}) or ((cfg.get("llm_providers") or {}).get("providers") or {})
    default_provider = llm.get("provider") or cfg.get("default_provider")
    if not default_provider and providers_map:
        default_provider = next(iter(providers_map.keys()))
    prof = providers_map.get(default_provider, {}) if default_provider else {}

    model = llm.get("model") or prof.get("model") or "gpt-4"
    base_url = llm.get("base_url") or prof.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://vip.yi-zhan.top/v1")
    env_key = os.getenv("OPENAI_API_KEY") or os.getenv(f"{(default_provider or 'openai').upper()}_API_KEY")
    api_key = llm.get("api_key") or env_key or prof.get("api_key") or "any_string_if_required"

    # [FIX] 默认超时提高到 300 秒，适配 Thinking 模型
    timeout = int(llm.get("timeout", prof.get("timeout", 300)))
    temperature = float(llm.get("temperature", prof.get("temperature", 0.7)))
    max_tokens = int(llm.get("max_tokens", prof.get("max_tokens", 40000)))

    return {
        "provider": default_provider or "openai_compat",
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "timeout": timeout,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


class LLMClient:
    def __init__(self, provider="openai_compat", model="gpt-5", base_url=None, api_key=None, **kwargs):
        self.provider = provider
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://vip.yi-zhan.top/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "any_string_if_required")
        self.timeout = kwargs.get("timeout", 300) # Default to 300s

    def chat(self, messages, temperature=0.7, max_tokens=40000, enforce_json=False, retries=2, debug_dir=None):
        print(f"[LLM] provider={self.provider} model={self.model} timeout={self.timeout}s")
        
        url = self.base_url.rstrip('/') + '/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': float(temperature),
            'max_tokens': int(max_tokens),
        }
        if enforce_json:
            payload['response_format'] = {'type': 'json_object'}

        last_err = None
        for i in range(int(retries) + 1):
            try:
                # print(f"[LLM] Sending Request (Attempt {i+1})...")
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                
                # [DEBUG] Save raw response if debug_dir is provided
                if debug_dir:
                    os.makedirs(debug_dir, exist_ok=True)
                    log_file = os.path.join(debug_dir, f"llm_response_try_{i+1}.txt")
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write(f"Status: {r.status_code}\n")
                        f.write(r.text)

                if r.status_code != 200:
                    raise LLMUnavailable(f'HTTP {r.status_code}: {r.text[:200]}')
                
                data = r.json()
                choice = (data.get('choices') or [{}])[0]
                message = choice.get('message') or {}
                content = message.get('content') or choice.get('text') or ''
                
                if not content and message.get('tool_calls'):
                     # Fallback for tool calls if any
                     content = json.dumps(message['tool_calls'])

                if content.strip():
                    return content.strip()

                time.sleep(1)
            except Exception as e:
                print(f"[LLM] Error attempt {i+1}: {e}")
                last_err = e
                time.sleep(2)

        print(f"[LLM] FAILED after retries. Last Error: {last_err}")
        raise LLMUnavailable(f'Empty or invalid response: {last_err}')