"""Behavior probes — scripted mini-conversations with binary assertions.

Responsibilities (CLAUDE.md §5, §6):
- Refuses off-topic requests and prompt injection.
- No recommendation on turn 1 for a vague query.
- Honors edits/refine (shortlist changes when constraints change).
- No hallucinated items (every recommended id/url is in the catalog).

Each probe returns pass/fail for the eval report.
"""
