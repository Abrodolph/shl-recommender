"""Intent classification + constraint extraction — the one LLM call per turn.

Responsibilities (CLAUDE.md §4):
- Build the system/user prompt from the full conversation history.
- Make exactly ONE LLM call (through the ``app.llm`` interface) that returns a
  structured result: intent (CLARIFY / RECOMMEND / REFINE / COMPARE / REFUSE),
  the normalized retrieval query, extracted constraints (role, seniority, skills,
  test-type intent, named assessments for COMPARE), and optional reply text.
- Parse/validate the LLM output defensively; never let malformed model output
  break the turn.

The LLM never emits URLs — only assessment ids/names. URL lookup and post-filter
happen downstream in ``app.catalog``.
"""
