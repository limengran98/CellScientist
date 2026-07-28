from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schemas import ContractError


@dataclass(frozen=True)
class LLMDecision:
    candidate_id: str
    address: Optional[str]
    prompt_text: str
    response_text: str
    usage: Dict[str, int]
    http_attempts: Tuple[Dict[str, Any], ...] = ()

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt_text.encode("utf-8")).hexdigest()

    @property
    def response_sha256(self) -> str:
        return hashlib.sha256(self.response_text.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ContractError("LLM output did not contain a JSON object")


class OpenAICompatiblePolicy:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.last_call_trace: List[Dict[str, Any]] = []
        self.last_response_text: Optional[str] = None
        self.last_usage: Optional[Dict[str, int]] = None
        self.last_validation_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def model(self) -> str:
        return str(self.config.get("model", "unknown"))

    def _chat(self, prompt: str) -> tuple[str, Dict[str, int]]:
        self.last_call_trace = []
        self.last_response_text = None
        self.last_usage = None
        self.last_validation_error = None
        key_name = str(self.config["api_key_env"])
        key = os.environ.get(key_name)
        if not key:
            raise ContractError(f"Missing credential environment variable: {key_name}")
        payload = {
            "model": str(self.config["model"]),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a constrained search controller. Return exactly one "
                        "JSON object and never invent a candidate ID."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.config.get("temperature", 0.0)),
            "max_tokens": int(self.config.get("max_tokens", 1200)),
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint_env = str(self.config.get("base_url_env", ""))
        base_url = os.environ.get(endpoint_env) if endpoint_env else None
        base_url = base_url or str(self.config.get("base_url", ""))
        if not base_url:
            raise ContractError(
                "Missing LLM endpoint; set the configured base_url_env variable"
            )
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        last_error: Optional[Exception] = None
        max_attempts = int(self.config.get("max_attempts", 3))
        if max_attempts < 1:
            raise ContractError("llm.max_attempts must be positive")
        for attempt in range(max_attempts):
            attempt_started = time.monotonic()
            attempt_record: Dict[str, Any] = {
                "retry_index": attempt,
                "attempt_number": attempt + 1,
                "http_success": False,
                "http_status": None,
                "latency_seconds": None,
                "error_type": None,
                "response_sha256": None,
                "provider_usage": None,
                "usage_reported": False,
            }
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=int(self.config.get("timeout_seconds", 180)),
                ) as response:
                    raw_body = response.read().decode("utf-8")
                    status = getattr(response, "status", None)
                    if status is None and hasattr(response, "getcode"):
                        status = response.getcode()
                attempt_record["http_success"] = True
                attempt_record["http_status"] = status
                attempt_record["response_body_sha256"] = hashlib.sha256(
                    raw_body.encode("utf-8")
                ).hexdigest()
                # Retain the raw successful body until the OpenAI-compatible
                # content field is parsed. This keeps malformed successful
                # responses auditable without conflating them with transport
                # failures.
                self.last_response_text = raw_body
                value = json.loads(raw_body)
                text = value["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ContractError("LLM returned empty content")
                raw_usage = value.get("usage", {})
                usage = {
                    "prompt_tokens": int(raw_usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
                    "total_tokens": int(raw_usage.get("total_tokens", 0)),
                }
                cleaned = text.strip()
                self.last_response_text = cleaned
                self.last_usage = usage
                attempt_record["response_sha256"] = hashlib.sha256(
                    cleaned.encode("utf-8")
                ).hexdigest()
                attempt_record["provider_usage"] = usage
                attempt_record["usage_reported"] = (
                    isinstance(raw_usage, Mapping)
                    and any(
                        key in raw_usage
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                    )
                )
                attempt_record["latency_seconds"] = time.monotonic() - attempt_started
                self.last_call_trace.append(attempt_record)
                return cleaned, usage
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
                ContractError,
            ) as exc:
                last_error = exc
                attempt_record["latency_seconds"] = time.monotonic() - attempt_started
                attempt_record["error_type"] = type(exc).__name__
                if isinstance(exc, urllib.error.HTTPError):
                    attempt_record["http_status"] = int(exc.code)
                if (
                    attempt_record["http_success"]
                    and self.last_response_text is not None
                    and attempt_record["response_sha256"] is None
                ):
                    attempt_record["response_sha256"] = hashlib.sha256(
                        self.last_response_text.encode("utf-8")
                    ).hexdigest()
                self.last_call_trace.append(attempt_record)
                if attempt + 1 < max_attempts:
                    time.sleep(1 + attempt)
        self.last_validation_error = (
            f"LLM request failed: {type(last_error).__name__}"
        )
        raise ContractError(f"LLM request failed: {type(last_error).__name__}")

    def choose(
        self,
        prompt: str,
        allowed_candidate_ids: Sequence[str],
        allowed_addresses: Sequence[str],
    ) -> LLMDecision:
        response, usage = self._chat(prompt)
        try:
            value = _extract_json_object(response)
            candidate_id = str(value.get("candidate_id", ""))
            if candidate_id not in set(allowed_candidate_ids):
                raise ContractError(
                    f"LLM selected an invalid candidate: {candidate_id}"
                )
            raw_address = value.get("address")
            address = None if raw_address is None else str(raw_address)
            if allowed_addresses and address not in set(allowed_addresses):
                raise ContractError(f"LLM selected an invalid address: {address}")
        except ContractError as exc:
            self.last_validation_error = str(exc)
            raise
        return LLMDecision(
            candidate_id=candidate_id,
            address=address,
            prompt_text=prompt,
            response_text=response,
            usage=usage,
            http_attempts=tuple(dict(row) for row in self.last_call_trace),
        )
