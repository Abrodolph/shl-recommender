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

from app import guardrails
from app.replies import compare_reply, shortlist_reply
from app.router import RouterResult, _fallback_query, route
from app.schemas import SAFE_FALLBACK, ChatResponse, Message, Recommendation

# Retrieval depth. Recall@10 is the metric -> retrieve inclusively toward 10.
TOP_K = 10
# Force a commit rather than clarify again once the history is this long, so we
# never run past the evaluator's 8-turn cap while still gathering context.
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
    try:
        msgs = _normalize_messages(messages)
        last_user = _last_user(msgs)

        # Deterministic backstop: obvious off-topic / injection -> refuse without
        # spending the LLM call (a second net beside the router's REFUSE).
        if guardrails.should_refuse(last_user):
            return _refuse(guardrails.refusal_reply())

        router_fn = router_fn or (lambda m: route(m))
        result: RouterResult = router_fn(msgs)

        # Near the turn budget, don't ask another question — commit with what we
        # have (CLAUDE.md §3/§9: never exceed 8 turns).
        if result.intent == "CLARIFY" and len(msgs) >= COMMIT_HISTORY_THRESHOLD:
            result.intent = "RECOMMEND"

        if result.intent == "REFUSE":
            return _refuse(result.reply_text or guardrails.refusal_reply())

        if result.intent == "COMPARE":
            return _compare(result, catalog)

        if result.intent in ("RECOMMEND", "REFINE"):
            history_text = " ".join(
                m["content"] for m in msgs if m.get("role") == "user" and m.get("content")
            )
            return _recommend(
                result, history_text, retriever=retriever, catalog=catalog
            )

        # CLARIFY (default): ask, no recommendations.
        return ChatResponse(
            reply=result.reply_text or SAFE_FALLBACK.reply,
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception:
        return SAFE_FALLBACK


# --- intent handlers ----------------------------------------------------------
def _recommend(result: RouterResult, history_text, retriever, catalog) -> ChatResponse:
    from app.assembly import assemble_ids, default_flags
    from app.catalog import get_catalog
    from app.retrieval import get_retriever

    catalog = catalog or get_catalog()
    retriever = retriever or get_retriever()

    query = result.search_query or _fallback_query(result.constraints)
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
    # First commit (RECOMMEND) closes the task; a REFINE leaves room for more edits.
    end = result.intent == "RECOMMEND"
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end)


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
