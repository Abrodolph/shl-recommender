"""Refusal, prompt-injection, and in-scope checks.

Responsibilities (CLAUDE.md §1, §9):
- Detect off-topic requests (general hiring advice, legal questions) and
  prompt-injection attempts, and drive a refusal with recommendations=[].
- Keep the agent strictly scoped to SHL assessment discussion.

Works alongside the router's REFUSE intent as a defensive backstop.
"""
