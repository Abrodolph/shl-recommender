"""Tests for hybrid retrieval + RRF (CLAUDE.md §4).

Assert dense and BM25 both contribute, RRF fusion returns up to k=10, exact
identifiers (e.g. "Java", "OPQ32r") are recalled, and results map to catalog ids.
"""
