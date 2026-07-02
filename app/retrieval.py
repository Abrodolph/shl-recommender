"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion (K=10).

Responsibilities (CLAUDE.md §4):
- Dense retrieval over precomputed ``data/embeddings.npy`` using
  sentence-transformers ``all-MiniLM-L6-v2`` (catches semantic intent).
- Lexical retrieval with ``rank_bm25`` over catalog text (catches exact
  identifiers like "Java", "OPQ32r", ".NET").
- Fuse both ranked lists with Reciprocal Rank Fusion and return the top-k
  (default k=10). Metric is Recall@10 → retrieve inclusively, fill toward 10.
- Build the BM25 index in-memory at boot; embeddings are loaded from the shipped
  file so nothing large downloads at cold start.

Returns catalog ids/scores; URL resolution is handled by ``app.catalog``.

The per-assessment text and the BM25 tokenizer live here and are imported by
``scripts/build_embeddings.py`` so the dense and lexical indexes describe the
exact same documents in the exact same id order.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_IDS_PATH = DATA_DIR / "embeddings_ids.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RRF constant. Larger => flatter contribution from rank position (60 is the
# value from the original Cormack et al. RRF paper and the common default).
RRF_K = 60
# How many candidates to pull from each retriever before fusion. Generous so the
# fused top-k has real competition (Recall@10 rewards inclusive retrieval).
DEFAULT_CANDIDATE_N = 50


# --- shared document text + tokenizer (imported by build_embeddings) ----------
def assessment_text(record: dict) -> str:
    """Human-readable text used for BOTH dense embedding and BM25.

    Combines name + human-readable test-type labels (``keys``) + description.
    The name is placed first (and its exact tokens are recoverable by the
    tokenizer) so lexical search can hit exact identifiers.
    """
    parts: list[str] = []
    name = (record.get("name") or "").strip()
    if name:
        parts.append(name)
    # ``keys`` are the human-readable test-type labels, e.g. "Knowledge & Skills".
    labels = record.get("keys") or []
    if labels:
        parts.append(", ".join(labels))
    desc = (record.get("description") or "").strip()
    if desc:
        parts.append(desc)
    # job_levels add cheap seniority signal (entry / manager / …) when present.
    levels = record.get("job_levels") or []
    if levels:
        parts.append(", ".join(levels))
    return "\n".join(parts)


def tokenize(text: str) -> list[str]:
    """Lexical tokenizer for BM25.

    Lowercases and extracts alphanumeric runs, then ALSO splits letter/digit
    boundaries so exact-name queries hit subwords: "OPQ32r" → opq32r, opq, 32, r;
    ".NET" → net; "Java 8" → java, 8. This is what lets a bare "OPQ" recall
    "OPQ32r" lexically.
    """
    tokens: list[str] = []
    for run in re.findall(r"[a-z0-9]+", text.lower()):
        tokens.append(run)
        parts = re.findall(r"[a-z]+|[0-9]+", run)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


# --- retriever ----------------------------------------------------------------
@dataclass
class Retriever:
    ids: list[str]
    embeddings: np.ndarray          # (N, D), L2-normalized rows
    bm25: object                    # rank_bm25.BM25Okapi
    _model: object = None           # lazily loaded SentenceTransformer

    def _encode_query(self, query: str) -> np.ndarray | None:
        """Encode a query into a normalized vector, or None if the dense model
        can't be loaded (retrieval then falls back to BM25-only)."""
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(EMBEDDING_MODEL)
            vec = self._model.encode([query], normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)[0]
        except Exception:
            return None

    def _dense_ranking(self, query: str, n: int) -> list[str]:
        qv = self._encode_query(query)
        if qv is None or self.embeddings.size == 0:
            return []
        sims = self.embeddings @ qv                     # cosine (rows normalized)
        top = np.argsort(-sims)[:n]
        return [self.ids[i] for i in top]

    def _bm25_ranking(self, query: str, n: int) -> list[str]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = np.argsort(-np.asarray(scores))[:n]
        # Drop zero-score hits so pure-lexical misses don't pad the pool.
        return [self.ids[i] for i in order if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        k: int = 10,
        n: int | None = None,
        rrf_k: int = RRF_K,
    ) -> list[tuple[str, float]]:
        """Hybrid retrieve. Runs dense + BM25, fuses with RRF, returns up to ``k``
        ``(id, rrf_score)`` pairs, highest first. ``n`` is the per-retriever
        candidate depth (defaults to :data:`DEFAULT_CANDIDATE_N`); ``rrf_k`` is the
        RRF constant (larger => flatter rank contribution)."""
        if not query or not query.strip():
            return []
        n = n or DEFAULT_CANDIDATE_N
        dense = self._dense_ranking(query, n)
        lexical = self._bm25_ranking(query, n)

        scores: dict[str, float] = {}
        for ranking in (dense, lexical):
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return fused[:k]

    def retrieve_ids(self, query: str, k: int = 10, n: int | None = None) -> list[str]:
        return [doc_id for doc_id, _ in self.retrieve(query, k=k, n=n)]


def _build_bm25(texts: list[str]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(t) for t in texts])


def load_retriever(
    catalog_path: Path = CATALOG_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    embeddings_ids_path: Path = EMBEDDINGS_IDS_PATH,
) -> Retriever:
    """Load embeddings + id order, and (re)build the BM25 index over the same
    documents in the same order. Raises if artifacts are missing/misaligned."""
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in catalog}

    embeddings = np.load(embeddings_path).astype(np.float32)
    ids = json.loads(embeddings_ids_path.read_text(encoding="utf-8"))
    if len(ids) != embeddings.shape[0]:
        raise ValueError(
            f"embeddings/id mismatch: {embeddings.shape[0]} vectors, {len(ids)} ids"
        )

    # Rebuild BM25 texts from the catalog in the SAME id order as the embeddings.
    texts = [assessment_text(by_id[i]) for i in ids]
    bm25 = _build_bm25(texts)
    return Retriever(ids=ids, embeddings=embeddings, bm25=bm25)


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Process-wide retriever singleton (built once at first use / boot)."""
    global _retriever
    if _retriever is None:
        _retriever = load_retriever()
    return _retriever


def retrieve(query: str, k: int = 10, n: int | None = None) -> list[tuple[str, float]]:
    """Module-level convenience wrapper over the singleton retriever."""
    return get_retriever().retrieve(query, k=k, n=n)
