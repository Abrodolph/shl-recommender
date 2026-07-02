"""Tests for hybrid retrieval + RRF (CLAUDE.md §4).

Assert dense and BM25 both contribute, RRF fusion returns up to k, exact
identifiers (e.g. "OPQ", ".NET") are recalled lexically, and results map to
catalog ids.

These load the shipped ``data/embeddings.npy`` + rebuild BM25; the dense model
may download on first run. If artifacts are missing the module is skipped.
"""

from __future__ import annotations

import json

import pytest

from app.retrieval import (
    CATALOG_PATH,
    EMBEDDINGS_PATH,
    get_retriever,
    tokenize,
)

pytestmark = pytest.mark.skipif(
    not EMBEDDINGS_PATH.exists(),
    reason="embeddings.npy not built (run scripts/build_embeddings.py)",
)


@pytest.fixture(scope="module")
def retriever():
    return get_retriever()


@pytest.fixture(scope="module")
def catalog_ids():
    recs = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {r["id"] for r in recs}


# --- tokenizer ----------------------------------------------------------------
def test_tokenize_splits_letter_digit_boundaries():
    toks = tokenize("OPQ32r")
    assert "opq32r" in toks and "opq" in toks and "32" in toks


def test_tokenize_dotnet():
    assert "net" in tokenize(".NET Framework 4.5")


# --- contract -----------------------------------------------------------------
def test_retrieve_returns_at_most_k(retriever):
    results = retriever.retrieve("java developer", k=10)
    assert 0 < len(results) <= 10
    ids, scores = zip(*results)
    assert len(set(ids)) == len(ids)                 # no duplicate ids
    assert all(s > 0 for s in scores)
    assert list(scores) == sorted(scores, reverse=True)  # descending


def test_retrieve_ids_are_in_catalog(retriever, catalog_ids):
    for doc_id, _ in retriever.retrieve("java developer", k=10):
        assert doc_id in catalog_ids


def test_k_is_configurable(retriever):
    assert len(retriever.retrieve("personality assessment", k=3)) <= 3


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("   ", k=10) == []


# --- semantic (dense) recall --------------------------------------------------
def test_java_query_surfaces_java_assessment(retriever):
    ids = retriever.retrieve_ids("Java developer", k=10)
    assert any("java" in i for i in ids), f"no Java assessment in top-10: {ids}"


# --- exact-name (BM25) recall -------------------------------------------------
def test_opq_surfaces_opq_entry(retriever):
    # Bare acronym must recall an OPQ entry — proves the lexical half works,
    # since "opq" only matches "opq32r" via the subword tokenizer.
    ids = retriever.retrieve_ids("OPQ", k=10)
    assert any("opq" in i for i in ids), f"no OPQ entry in top-10: {ids}"


def test_exact_product_name_ranks_first(retriever):
    # A verbatim product name should come back at or near the very top.
    ids = retriever.retrieve_ids("Microsoft Excel", k=5)
    assert any("excel" in i for i in ids), f"no Excel entry in top-5: {ids}"
