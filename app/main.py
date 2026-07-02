"""FastAPI application entrypoint.

Responsibilities:
- Define the FastAPI app and wire up routes.
- ``GET /health`` — returns ``{"status": "ok"}`` with HTTP 200 as soon as the
  process is up, independent of model/index warmup (which runs in the background).
- ``POST /chat`` — stateless. Accepts the full conversation history and returns
  ``{reply, recommendations, end_of_conversation}``. Delegates to ``app.agent`` and
  guards the call with a hard timeout so a slow LLM still yields a valid fallback
  well inside the evaluator's 30s limit.

Hard constraints (CLAUDE.md §3, §9):
- Response schema is exact and always valid (Pydantic + try/except + timeout).
- One LLM call per turn; stay well under the 30s timeout.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent import handle
from app.config import get_settings
from app.schemas import SAFE_FALLBACK, ChatRequest, ChatResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("shl.main")

_settings = get_settings()
# Leave headroom under the evaluator's 30s cap; return a fallback if we exceed it.
_TIMEOUT_S = max(5, _settings.request_timeout_s - 5)   # ~25s
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chat")

_warm = {"ready": False}


def _warmup() -> None:
    """Load the catalog, retriever, and embedding model so the first real /chat
    isn't cold. Runs in a background thread; failures are logged, not fatal."""
    try:
        t0 = time.perf_counter()
        from app.catalog import get_catalog
        from app.retrieval import get_retriever

        get_catalog()
        retriever = get_retriever()
        retriever._encode_query("warmup")   # force the sentence-transformer to load
        _warm["ready"] = True
        log.info("warmup complete in %d ms", int((time.perf_counter() - t0) * 1000))
    except Exception:
        log.exception("warmup failed (service still serves; first call may be slow)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off warmup without blocking startup, so /health answers immediately.
    threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield
    _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="SHL Assessment Recommender", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Always 200 with ``{"status": "ok"}`` once the process is
    up — deliberately independent of catalog/model warmup (CLAUDE.md §3)."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Stateless chat turn. Delegates to ``app.agent.handle`` under a hard timeout;
    any timeout/failure degrades to :data:`SAFE_FALLBACK` (still a valid contract)."""
    t0 = time.perf_counter()
    try:
        future = _executor.submit(handle, request.messages)
        response = future.result(timeout=_TIMEOUT_S)
    except FuturesTimeout:
        log.warning("chat timed out after %ss; returning fallback", _TIMEOUT_S)
        response = SAFE_FALLBACK
    except Exception:
        log.exception("chat failed; returning fallback")
        response = SAFE_FALLBACK
    log.info(
        "chat done latency_ms=%d recs=%d",
        int((time.perf_counter() - t0) * 1000),
        len(response.recommendations),
    )
    return response
