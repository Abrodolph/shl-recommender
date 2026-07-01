"""Env-driven configuration (12-factor).

Loads settings from environment variables (see ``.env.example``):
- ``GROQ_API_KEY``  — LLM provider credential.
- ``LLM_PROVIDER``  — e.g. "groq" (Gemini swappable).
- ``LLM_MODEL``     — e.g. "llama-3.3-70b-versatile".

Also centralizes paths (``data/catalog.json``, ``data/embeddings.npy``) and
retrieval/agent tunables (top-k, embedding model name, turn cap).
"""
