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

Returns catalog ids/records; URL resolution is handled by ``app.catalog``.
"""
