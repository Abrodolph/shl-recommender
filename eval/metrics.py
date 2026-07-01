"""Evaluation metrics (CLAUDE.md §5, §6).

Responsibilities:
- ``Recall@k`` (k=10) of returned shortlists vs the labeled shortlist per trace,
  and the Mean Recall@10 across traces.
- Groundedness: % of returned URLs present in the catalog (must be 100% by
  construction).
"""
