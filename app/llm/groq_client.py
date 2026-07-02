"""Groq (Llama) LLM client — runtime default.

Implements ``LLMClient`` using the Groq Python SDK. Chosen for lowest latency to
protect the 30s per-call timeout (CLAUDE.md §4). Reads model and credentials from
``app.config`` (``GROQ_API_KEY``, ``LLM_MODEL``).

Hardening for the evaluator:
- Hard per-request timeout (default 20s < the 30s eval limit).
- One retry on transient errors (timeouts / 5xx / rate limits).
- JSON mode via ``response_format={"type": "json_object"}`` for the router.
"""

from __future__ import annotations

import time

from app.config import get_settings
from app.llm.base import LLMClient, LLMError

# Errors worth one retry (transient). Import lazily-safe: the groq package
# exposes these, but we guard so import of this module never hard-fails.
try:  # pragma: no cover - import guard
    from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    _TRANSIENT = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
except Exception:  # pragma: no cover
    _TRANSIENT = ()


class GroqClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 20.0,
        max_retries: int = 1,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.llm_model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set")

        from groq import Groq

        # Disable the SDK's own retries; we control retry/backoff explicitly.
        self._client = Groq(api_key=self.api_key, timeout=timeout_s, max_retries=0)

    def complete(
        self,
        system: str,
        messages: list[dict],
        json_mode: bool = False,
    ) -> str:
        payload = [{"role": "system", "content": system}]
        payload.extend(
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        )

        kwargs: dict = {
            "model": self.model,
            "messages": payload,
            "temperature": 0,          # deterministic routing
            "max_tokens": 1024,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                if content is None:
                    raise LLMError("empty completion from Groq")
                return content
            except _TRANSIENT as exc:  # transient → retry once
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break
            except Exception as exc:   # non-transient → fail fast
                raise LLMError(f"Groq completion failed: {exc}") from exc

        raise LLMError(f"Groq completion failed after retries: {last_exc}") from last_exc
