import os, json, time, requests

class LLMUnavailable(Exception):
    pass


def resolve_llm_from_cfg(cfg: dict) -> dict:
    """
    Parse LLM connection parameters from the merged generate_execution_config.json file.
    Priority order:
      1. Explicit cfg["llm"].* fields override provider mappings.
      2. Otherwise, read from cfg["providers"][<provider>] or cfg["llm_providers"]["providers"].
      3. The provider is determined by cfg["llm"]["provider"] or cfg["default_provider"] or
         cfg["llm_providers"]["default_provider"].

    Returns:
      {
        "provider": str,
        "model": str,
        "base_url": str,
        "api_key": str,
        "timeout": int,
        "temperature": float,
        "max_tokens": int
      }
    """
    cfg = cfg or {}
    llm = (cfg.get("llm") or {})
    providers_map = (cfg.get("providers") or {}) or ((cfg.get("llm_providers") or {}).get("providers") or {})
    default_provider = llm.get("provider") or cfg.get("default_provider") or ((cfg.get("llm_providers") or {}).get("default_provider"))
    if not default_provider and providers_map:
        default_provider = next(iter(providers_map.keys()))
    prof = providers_map.get(default_provider, {}) if default_provider else {}

    # derive fields with override
    model = llm.get("model") or prof.get("model") or ((prof.get("models") or [None])[0]) or "gpt-5"
    base_url = llm.get("base_url") or prof.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://vip.yi-zhan.top/v1")
    
    # Environment variable takes precedence over config file api_key
    env_key = os.getenv("OPENAI_API_KEY") or os.getenv(f"{(default_provider or 'openai').upper()}_API_KEY")
    api_key = llm.get("api_key") or env_key or prof.get("api_key") or "any_string_if_required"

    timeout = int(llm.get("timeout", prof.get("timeout", 60)))
    temperature = float(llm.get("temperature", prof.get("temperature", 0.2)))
    max_tokens = int(llm.get("max_tokens", prof.get("max_tokens", 20000)))

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
        self.timeout = kwargs.get("timeout", 60)

    def chat(self, messages, temperature=0.2, max_tokens=1500, enforce_json=False, retries=2, debug_dir=None):
        print(f"[LLM] provider={self.provider} model={self.model} base_url={self.base_url}")
        """Stable HTTP chat call compatible with many OpenAI-like gateways.
        Handles content in message.content, choice.text, and tool_calls[].function.arguments.
        Falls back from response_format=json_object when unsupported.
        """
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
        for i in range(int(retries)):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    raise LLMUnavailable(f'HTTP {r.status_code}: {r.text[:200]}')
                data = r.json()

                # ---- robust extraction across providers ----
                choice = (data.get('choices') or [{}])[0]
                message = choice.get('message') or {}
                content = message.get('content') or choice.get('text') or ''
                if isinstance(content, list):
                    content = ''.join(seg.get('text', '') if isinstance(seg, dict) else str(seg) for seg in content)
                content = (content or '').strip()
                if (not content) and message.get('tool_calls'):
                    try:
                        content = (message['tool_calls'][0]['function']['arguments'] or '').strip()
                    except Exception:
                        pass

                # If still empty and we requested JSON, retry inline without response_format
                if not content and 'response_format' in payload:
                    payload.pop('response_format', None)
                    r2 = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                    data = r2.json()
                    choice = (data.get('choices') or [{}])[0]
                    message = choice.get('message') or {}
                    content = message.get('content') or choice.get('text') or ''
                    if isinstance(content, list):
                        content = ''.join(seg.get('text', '') if isinstance(seg, dict) else str(seg) for seg in content)
                    content = (content or '').strip()
                    if (not content) and message.get('tool_calls'):
                        try:
                            content = (message['tool_calls'][0]['function']['arguments'] or '').strip()
                        except Exception:
                            pass

                # Dump raw response for diagnostics
                if debug_dir:
                    try:
                        os.makedirs(debug_dir, exist_ok=True)
                        with open(os.path.join(debug_dir, f'raw_response_llm_client_try{i+1}.json'), 'w', encoding='utf-8') as fdbg:
                            json.dump(data, fdbg, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                if content.strip():
                    return content.strip()

                time.sleep(1)
            except Exception as e:
                last_err = e
                time.sleep(1)

        raise LLMUnavailable(f'Empty or invalid response after {retries} attempts: {last_err}')
