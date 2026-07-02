"""FastAPI application entrypoint.

Responsibilities:
- Define the FastAPI app and wire up routes.
- ``GET /health`` — returns ``{"status": "ok"}`` with HTTP 200. Must be reachable
  as soon as the process is up, even during model/index warmup.
- ``POST /chat`` — stateless endpoint. Accepts the full conversation history and
  returns ``{reply, recommendations, end_of_conversation}``. Delegates all logic
  to ``app.agent`` and always returns a schema-valid response (Pydantic + a
  try/except fallback so a malformed response can never escape).

Hard constraints (see CLAUDE.md §3, §9):
- Response schema is exact and always valid.
- One LLM call per turn; stay well under the 30s timeout.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.agent import handle
from app.schemas import SAFE_FALLBACK, ChatRequest, ChatResponse

app = FastAPI(title="SHL Assessment Recommender", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Always 200 with ``{"status": "ok"}`` once the process is
    up — deliberately independent of catalog/model warmup (CLAUDE.md §3)."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Stateless chat turn.

    Delegates to ``app.agent.handle`` (one LLM call → route → retrieve →
    templated reply), always wrapped so any internal failure degrades to
    :data:`app.schemas.SAFE_FALLBACK` rather than escaping as an invalid
    contract.
    """
    try:
        return handle(request.messages)
    except Exception:
        return SAFE_FALLBACK
