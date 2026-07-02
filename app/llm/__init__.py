"""LLM client package.

Exposes a provider-agnostic ``LLMClient`` interface (``base.py``) with concrete
implementations for Groq (default) and Gemini (optional swap). Swapping providers
should require changing only ``LLM_PROVIDER`` (see ``app.config``).

Use :func:`get_client` — a cached factory that reads the provider from settings.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.base import LLMClient, LLMError

_client: LLMClient | None = None


def build_client(provider: str | None = None) -> LLMClient:
    """Construct a fresh client for ``provider`` (defaults to settings)."""
    provider = (provider or get_settings().llm_provider or "groq").lower()
    if provider == "groq":
        from app.llm.groq_client import GroqClient

        return GroqClient()
    if provider == "gemini":
        from app.llm.gemini_client import GeminiClient

        return GeminiClient()
    raise LLMError(f"unknown LLM_PROVIDER: {provider!r}")


def get_client() -> LLMClient:
    """Process-wide client singleton, chosen by ``LLM_PROVIDER``."""
    global _client
    if _client is None:
        _client = build_client()
    return _client


__all__ = ["LLMClient", "LLMError", "build_client", "get_client"]
