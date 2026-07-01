"""Simulated-user replay harness (a graded deliverable — CLAUDE.md §6).

Responsibilities:
- For each trace in ``eval/traces/``, have an LLM play the user from the trace's
  persona/facts and run a real multi-turn conversation against ``POST /chat``.
- Collect the final shortlist per conversation and the full transcript.
- Feed results into ``metrics.py`` and ``probes.py`` and hand off to ``report.py``.

Mirrors SHL's automated evaluator so the approach doc can cite real before/after
numbers.
"""
