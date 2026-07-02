"""Evaluation metrics (CLAUDE.md §5, §6).

- ``recall_at_k``: fraction of a trace's labeled shortlist that appears in the
  retrieved/returned top-k ids.
- ``mean_recall_at_k``: mean of per-trace recall.
- ``groundedness``: fraction of returned ids/urls present in the catalog (must be
  1.0 by construction).

Ids are the unit of comparison (a returned item == an expected item iff their
catalog ids match). URL/name -> id resolution lives in ``eval.recall_eval``.
"""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int = 10) -> float:
    """|expected ∩ top-k(retrieved)| / |expected|. Empty expected -> 1.0."""
    if not expected_ids:
        return 1.0
    topk = set(retrieved_ids[:k])
    hit = len(expected_ids & topk)
    return hit / len(expected_ids)


def mean_recall_at_k(per_trace: list[tuple[list[str], set[str]]], k: int = 10) -> float:
    """Mean Recall@k over (retrieved_ids, expected_ids) pairs."""
    if not per_trace:
        return 0.0
    return sum(recall_at_k(r, e, k) for r, e in per_trace) / len(per_trace)


def groundedness(returned_ids: list[str], catalog_ids: set[str]) -> float:
    """Fraction of returned ids present in the catalog. Empty -> 1.0."""
    if not returned_ids:
        return 1.0
    grounded = sum(1 for i in returned_ids if i in catalog_ids)
    return grounded / len(returned_ids)
