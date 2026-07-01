"""Agent orchestration: one LLM call per turn → route → act.

Responsibilities (CLAUDE.md §4):
- Take the full conversation history and make a SINGLE LLM call (via ``app.router``)
  that classifies intent and extracts normalized constraints.
- Dispatch on the intent:
    CLARIFY   → LLM-generated question, recommendations=[]
    RECOMMEND → hybrid retrieval (app.retrieval) → top-k → templated reply
    REFINE    → re-derive constraints from full history → re-retrieve
    COMPARE   → grounded answer from catalog text, recommendations=[]
    REFUSE    → refusal reply, recommendations=[]
- Apply the catalog post-filter (drop any item whose id isn't in the catalog) and
  look up canonical URLs by id.
- Validate against the Pydantic schema; on any failure return a safe fallback that
  is still a valid ChatResponse.

Conversation policy defaults live here (turn-1 clarify on vague query, commit once
role + one discriminating attribute is known, cap clarifying turns, honor the
8-turn budget).
"""
