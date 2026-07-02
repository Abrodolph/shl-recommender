"""LLMClient interface.

Defines the abstract contract every provider client implements. Keeping this
interface stable lets the runtime switch between Groq and Gemini by changing one
module + env var, as decided in CLAUDE.md §4.

The whole agent uses exactly ONE method — ``complete(system, messages,
json_mode)`` — because the design is one LLM call per turn. ``json_mode=True``
requests strict JSON output (used by the router); the returned value is always
the raw string content (JSON parsing happens in the caller).
"""

from __future__ import annotations

import abc


class LLMError(RuntimeError):
    """Raised when the provider call fails after retries/timeout. Callers must
    catch this and degrade to a safe fallback (never let it escape /chat)."""


class LLMClient(abc.ABC):
    """Provider-agnostic chat-completion interface."""

    @abc.abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict],
        json_mode: bool = False,
    ) -> str:
        """Run one completion and return the assistant message content.

        Args:
            system: System prompt (instructions / policy).
            messages: Conversation history as ``[{"role","content"}, ...]`` with
                roles ``user``/``assistant`` (the full replayed history).
            json_mode: If True, ask the provider for strict JSON output.

        Returns:
            The raw assistant text (JSON string when ``json_mode`` is True).

        Raises:
            LLMError: on transport/timeout/provider failure after retries.
        """
        raise NotImplementedError
