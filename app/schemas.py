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
