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
