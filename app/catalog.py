"""Catalog loading, id→url lookup, and the hallucination post-filter.

Responsibilities (CLAUDE.md §4, §9):
- Load ``data/catalog.json`` (the single source of truth) at boot.
- Provide id/name → canonical URL and record lookups.
- Post-filter: given proposed items from the agent, drop any whose id is not in
  the catalog, so a recommended URL can never be one that isn't in the catalog.
- Expose helpers for retrieval (catalog text corpus) and for COMPARE (full text of
  a few named items).

Golden rule enforced here: never emit a URL not in ``catalog.json``.
"""
