"""Gemini LLM client — optional swap.

Implements ``LLMClient`` from ``base.py`` for Google Gemini, provided so the runtime
provider can be switched by env var without touching the rest of the app
(CLAUDE.md §4). Not the default; Groq is used at runtime for latency.
"""
