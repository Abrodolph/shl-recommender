"""Gemini LLM client — optional swap.

Implements ``LLMClient`` for Google Gemini so the runtime provider can be
switched by env var without touching the rest of the app (CLAUDE.md §4). Not the
default; Groq is used at runtime for latency.

This is a working-shaped implementation guarded by import + key availability: if
``google-generativeai`` isn't installed or ``GEMINI_API_KEY`` is unset, it raises
a clear ``LLMError`` rather than failing at import time.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.base import LLMClient, LLMError


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 20.0,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.llm_model
        self.timeout_s = timeout_s
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover - optional dep
            raise LLMError(
                "google-generativeai is not installed; add it to use the Gemini provider"
            ) from exc
        genai.configure(api_key=self.api_key)
        self._genai = genai

    def complete(
        self,
        system: str,
        messages: list[dict],
        json_mode: bool = False,
    ) -> str:
        generation_config: dict = {"temperature": 0}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        try:
            model = self._genai.GenerativeModel(
                self.model,
                system_instruction=system,
                generation_config=generation_config,
            )
            # Gemini uses "model" for the assistant role.
            history = [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [m["content"]],
                }
                for m in messages
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            resp = model.generate_content(
                history, request_options={"timeout": self.timeout_s}
            )
            if not getattr(resp, "text", None):
                raise LLMError("empty completion from Gemini")
            return resp.text
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini completion failed: {exc}") from exc
