"""Build-time: encode ``data/catalog.json`` into ``data/embeddings.npy``.

Responsibilities (CLAUDE.md §4, §7):
- Load the catalog, build a text representation per assessment, and encode it with
  sentence-transformers ``all-MiniLM-L6-v2``.
- Save the dense embedding matrix to ``data/embeddings.npy`` (shipped in the repo)
  so nothing large downloads at cold start.

Keep the row order aligned with the catalog so ids map to embedding rows.
Build-time only — NOT a runtime dependency of the API.
"""
