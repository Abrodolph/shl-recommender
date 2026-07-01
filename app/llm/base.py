"""LLMClient interface.

Defines the abstract contract every provider client implements (e.g. a single
``complete(messages, ...) -> str`` / structured call). Keeping this interface stable
lets the runtime switch between Groq and Gemini by changing one module + env var,
as decided in CLAUDE.md §4.
"""
