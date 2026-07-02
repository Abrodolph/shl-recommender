"""Pydantic v2 request/response models — the hard-eval guarantee.

Defines and enforces the exact API contract from CLAUDE.md §3:

- ``Message``       — {role, content} item in the conversation history.
- ``ChatRequest``   — {messages: [Message, ...]}.
- ``Recommendation`` — exactly {name, url, test_type}.
- ``ChatResponse``  — {reply, recommendations, end_of_conversation}.

Rules encoded here:
- ``recommendations`` is [] when gathering context or refusing; 1–10 items when
  committing to a shortlist.
- ``end_of_conversation`` is true only when the task is complete.
- Any deviation from this schema = zero score on that trace, so validation is
  strict and paired with a safe fallback in the agent/main layer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Evaluator hard limit: a committed shortlist is 1–10 items (CLAUDE.md §3).
MAX_RECOMMENDATIONS = 10


class Message(BaseModel):
    """One item in the replayed conversation history.

    Roles follow the OpenAI-style chat convention. ``system`` is accepted for
    robustness (the evaluator may or may not send one) but the agent treats only
    user/assistant turns as conversational content.
    """

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("content")
    @classmethod
    def _content_is_string(cls, v: str) -> str:
        # Allow empty content (some turns can be blank) but never None/non-str.
        return v


class ChatRequest(BaseModel):
    """Stateless request: the full conversation history on every call."""

    model_config = ConfigDict(extra="ignore")

    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    """A single recommended SHL assessment — exactly {name, url, test_type}.

    ``extra="forbid"`` guarantees no stray keys leak into the contract. Each
    field is required and must be non-empty; the URL must be a catalog URL, which
    is enforced structurally upstream (the LLM never emits URLs — see CLAUDE.md).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    test_type: str = Field(..., min_length=1)

    @field_validator("name", "url", "test_type")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @field_validator("url")
    @classmethod
    def _looks_like_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must be an absolute http(s) URL")
        return v


class ChatResponse(BaseModel):
    """The exact response contract returned by ``POST /chat``.

    - ``recommendations``: [] while clarifying/refusing, else 1–10 items.
    - ``end_of_conversation``: true only when the shortlist is delivered and
      there is nothing left to do.
    """

    model_config = ConfigDict(extra="forbid")

    reply: str = Field(..., min_length=1)
    recommendations: list[Recommendation] = Field(
        default_factory=list, max_length=MAX_RECOMMENDATIONS
    )
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def _within_limit(cls, v: list[Recommendation]) -> list[Recommendation]:
        # min_length is 0 (empty is valid); max is the evaluator's hard cap.
        if len(v) > MAX_RECOMMENDATIONS:
            raise ValueError(
                f"at most {MAX_RECOMMENDATIONS} recommendations allowed, got {len(v)}"
            )
        return v


# A response that is always schema-valid, used as the last-resort fallback so a
# malformed internal state can never escape as a contract violation.
SAFE_FALLBACK = ChatResponse(
    reply=(
        "Sorry, I hit a problem handling that. Could you restate what role or "
        "skills you're hiring for?"
    ),
    recommendations=[],
    end_of_conversation=False,
)
