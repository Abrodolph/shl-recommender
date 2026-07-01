"""Groq (Llama) LLM client — runtime default.

Implements ``LLMClient`` from ``base.py`` using the Groq Python SDK. Chosen for
lowest latency to protect the 30s per-call timeout (CLAUDE.md §4). Reads model and
credentials from ``app.config`` (``GROQ_API_KEY``, ``LLM_MODEL``).
"""
