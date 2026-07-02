"""Agent orchestration: one LLM call per turn -> route -> act.

Responsibilities (CLAUDE.md §4):
- Take the full conversation history and make a SINGLE LLM call (via ``app.router``)
  that classifies intent and extracts normalized constraints.
- Dispatch on the intent:
    CLARIFY   -> LLM-generated question, recommendations=[]
    RECOMMEND -> hybrid retrieval -> top-k -> catalog lookup -> templated reply
    REFINE    -> re-derive constraints from full history -> re-retrieve
    COMPARE   -> grounded answer, recommendations=[]
    REFUSE    -> refusal reply, recommendations=[]
- Apply the catalog post-filter (canonical name/url/test_type by id) so a
  recommended URL can never be one that isn't in the catalog.
- Wrap everything so ANY failure degrades to a valid fallback ChatResponse.

Conversation policy defaults live here (deterministic refusal backstop, commit
before the 8-turn budget runs out, ≤10 recommendations).
"""

from __future__ import annotations

import logging
import time

from app import guardrails
from app.replies import compare_reply, shortlist_reply
from app.router import RouterResult, _fallback_query, route
from app.schemas import SAFE_FALLBACK, ChatResponse, Message, Recommendation

log = logging.getLogger("shl.agent")

# Retrieval depth. Recall@10 is the metric -> retrieve inclusively toward 10.
TOP_K = 10
# Evaluator hard cap: user+assistant combined <= 8 turns. Once the history is
# this long, don't ask another question that needs a reply — commit instead.
# (history length N means our reply is turn N+1; at N>=6 a further clarify +
# the user's answer would reach the cap, so we commit now.)
COMMIT_HISTORY_THRESHOLD = 6


def handle(
    messages: list,
    *,
    router_fn=None,
    retriever=None,
    catalog=None,
) -> ChatResponse:
    """Handle one turn. Always returns a schema-valid ChatResponse.

    Dependencies are injectable for testing; by default they resolve to the
    process singletons (router LLM, hybrid retriever, catalog).
    """
    t0 = time.perf_counter()
    intent = "ERROR"
    try:
        msgs = _normalize_messages(messages)
        last_user = _last_user(msgs)

        # Defensive: nothing usable in the payload -> ask, don't call the LLM.
        if not any(
            m.get("role") == "user" and (m.get("content") or "").strip() for m in msgs
        ):
            intent = "CLARIFY_EMPTY"
            return ChatResponse(
                reply=SAFE_FALLBACK.reply, recommendations=[], end_of_conversation=False
            )

        # Deterministic backstop: obvious off-topic / injection -> refuse without
        # spending the LLM call (a second net beside the router's REFUSE).
        if guardrails.should_refuse(last_user):
            intent = "REFUSE_GUARD"
            return _refuse(guardrails.refusal_reply())

        router_fn = router_fn or (lambda m: route(m))
        result: RouterResult = router_fn(msgs)
        intent = result.intent

        # Near the turn budget, don't ask another question — commit with what we
        # have (CLAUDE.md §3/§9: never exceed 8 turns).
        if result.intent == "CLARIFY" and len(msgs) >= COMMIT_HISTORY_THRESHOLD:
            result.intent = intent = "RECOMMEND_FORCED"

        if result.intent == "REFUSE":
            return _refuse(result.reply_text or guardrails.refusal_reply())

        if result.intent == "COMPARE":
            return _compare(result, catalog)

        if result.intent in ("RECOMMEND", "REFINE", "RECOMMEND_FORCED"):
            history_text = " ".join(
                m["content"] for m in msgs if m.get("role") == "user" and m.get("content")
            )
            # End the conversation only when the user signals they're done, or
            # when we're out of turns (forced commit). Matches the traces, which
            # keep end_of_conversation false on a fresh/edited shortlist and flip
            # it true only on the user's confirmation turn.
            final = result.intent == "RECOMMEND_FORCED" or _is_final(last_user)
            return _recommend(
                result, history_text, final=final, retriever=retriever, catalog=catalog
            )

        # CLARIFY (default): ask, no recommendations.
        return ChatResponse(
            reply=result.reply_text or SAFE_FALLBACK.reply,
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception:
        log.exception("agent.handle failed; returning safe fallback")
        return SAFE_FALLBACK
    finally:
        log.info(
            "turn intent=%s latency_ms=%d msgs=%s",
            intent,
            int((time.perf_counter() - t0) * 1000),
            len(messages) if hasattr(messages, "__len__") else "?",
        )


# --- intent handlers ----------------------------------------------------------
def _recommend(
    result: RouterResult, history_text, final, retriever, catalog
) -> ChatResponse:
    from app.assembly import assemble_ids, default_flags
    from app.catalog import get_catalog
    from app.retrieval import get_retriever

    catalog = catalog or get_catalog()
    retriever = retriever or get_retriever()

    # Combine the router's focused query with the full user history so
    # constraints are never dropped on a refine/confirm turn (statelessly
    # re-deriving the whole picture each turn). Without this, a final "locking it
    # in" turn retrieves only from that sparse message and the battery collapses.
    query = " ".join(
        p for p in (result.search_query or _fallback_query(result.constraints), history_text)
        if p
    ).strip()
    # Retrieve a generous candidate pool, then assemble a battery: role/skill
    # items + guaranteed personality/cognitive defaults (unless opted out). This
    # is the main Recall@10 lever (see eval/TUNING_LOG.md).
    retrieved = retriever.retrieve_ids(query, k=TOP_K * 2) if query else []
    add_p, add_c = default_flags(result.constraints, history_text)
    ids = assemble_ids(
        retrieved, k=TOP_K, add_personality=add_p, add_cognitive=add_c,
        valid_ids=set(catalog.ids),
    )
    items = catalog.recommendations_for(ids)[:TOP_K]   # canonical + de-duped

    if not items:
        # Nothing to recommend (e.g. empty query) -> ask instead of committing []
        return ChatResponse(
            reply=result.reply_text or SAFE_FALLBACK.reply,
            recommendations=[],
            end_of_conversation=False,
        )

    recs = [Recommendation(**it) for it in items]
    reply = shortlist_reply(result.constraints, items)
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=final)


def _compare(result: RouterResult, catalog) -> ChatResponse:
    # One LLM call per turn: reuse the router's grounded reply_text rather than
    # making a second call. Comparisons never introduce new picks (TRACES §c).
    items = [{"name": n} for n in result.named_assessments]
    reply = compare_reply(items, result.reply_text)
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _refuse(text: str) -> ChatResponse:
    return ChatResponse(
        reply=text or guardrails.refusal_reply(),
        recommendations=[],
        end_of_conversation=False,
    )


# --- helpers ------------------------------------------------------------------
def _normalize_messages(messages: list) -> list[dict]:
    """Coerce Pydantic Message / dicts into plain ``{role, content}`` dicts."""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, Message):
            out.append({"role": m.role, "content": m.content})
        elif isinstance(m, dict):
            out.append({"role": m.get("role"), "content": m.get("content", "")})
        else:
            out.append({"role": getattr(m, "role", None), "content": getattr(m, "content", "")})
    return out


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return ""


# Phrases that signal the user is satisfied and the task is complete. Drawn from
# the trace confirmations ("That's good.", "Locking it in.", "Final list...").
_FINAL_PHRASES = (
    "that's good", "thats good", "that works", "sounds good", "looks good",
    "perfect", "great, thank", "great thanks", "thank you", "thanks", "confirmed",
    "confirm", "final", "finalize", "finalise", "lock it in", "locking it in",
    "we're good", "were good", "all set", "good to go", "ship it", "no changes",
    "that's all", "thats all", "that is all", "nothing else", "that's it",
)
# If the user is also asking for a change, it's a refine, not a finalize.
_MODIFY_MARKERS = (
    "add", "remove", "drop", "swap", "replace", "change", "instead", " also ",
    "without", "include", "exclude", "but ", "shorter", "longer", "different",
)


def _is_final(text: str) -> bool:
    """True if the latest user message signals completion (and isn't also asking
    for an edit). Drives end_of_conversation for a committed shortlist."""
    t = f" {text.lower().strip()} "
    if any(m in t for m in _MODIFY_MARKERS):
        return False
    return any(p in t for p in _FINAL_PHRASES)
