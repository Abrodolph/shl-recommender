"""Env-driven configuration (12-factor).

Loads settings from environment variables (see ``.env.example``):
- ``GROQ_API_KEY``  — LLM provider credential.
- ``LLM_PROVIDER``  — e.g. "groq" (Gemini swappable).
- ``LLM_MODEL``     — e.g. "llama-3.3-70b-versatile".

Also centralizes paths (``data/catalog.json``, ``data/embeddings.npy``) and
retrieval/agent tunables (top-k, embedding model name, turn cap).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env if python-dotenv is available; never a hard dependency at runtime.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    """Immutable, env-driven settings snapshot.

    Read once at import time via :func:`get_settings`. Nothing here should fail
    to construct even when secrets are absent (so ``/health`` can answer during
    warmup and tests run without a key).
    """

    # LLM provider config.
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "groq"))
    llm_model: str = field(
        default_factory=lambda: _get("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    groq_api_key: str = field(default_factory=lambda: _get("GROQ_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: _get("GEMINI_API_KEY"))

    # Data artifacts.
    catalog_path: Path = field(default_factory=lambda: DATA_DIR / "catalog.json")
    embeddings_path: Path = field(default_factory=lambda: DATA_DIR / "embeddings.npy")

    # Retrieval / agent tunables.
    embedding_model: str = field(
        default_factory=lambda: _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    top_k: int = field(default_factory=lambda: int(_get("TOP_K", "10")))
    max_turns: int = field(default_factory=lambda: int(_get("MAX_TURNS", "8")))
    request_timeout_s: int = field(
        default_factory=lambda: int(_get("REQUEST_TIMEOUT_S", "30"))
    )

    @property
    def has_llm_key(self) -> bool:
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return bool(self.groq_api_key)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
