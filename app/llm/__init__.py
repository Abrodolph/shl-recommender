"""LLM client package.

Exposes a provider-agnostic ``LLMClient`` interface (``base.py``) with concrete
implementations for Groq (default) and Gemini (optional swap). Swapping providers
should require changing only this package plus env vars (see ``app.config``).
"""
