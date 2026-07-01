"""Tests for the API contract / Pydantic schema (CLAUDE.md §3, §9).

Assert that ChatResponse is always valid: recommendations is [] or 1–10 items,
each item has exactly {name, url, test_type}, and malformed inputs fall back to a
valid response.
"""
