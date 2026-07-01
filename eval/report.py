"""Write the evaluation report (CLAUDE.md §6).

Aggregates per-trace pass/fail and metrics from ``replay.py``, ``metrics.py``, and
``probes.py`` into a human-readable report so the approach doc can state real
before/after numbers (e.g. "dense-only 0.62 → hybrid+RRF 0.81").
"""
