"""Refusal, prompt-injection, and in-scope checks.

Responsibilities (CLAUDE.md §1, §9):
- Detect off-topic requests (general hiring advice, legal questions) and
  prompt-injection attempts.
- Provide a canned refusal reply so REFUSE responses never depend on the LLM.

This is a lightweight DETERMINISTIC net that runs alongside the router's REFUSE
intent. The router does the nuanced classification; these keyword/pattern checks
are a cheap backstop that catches the obvious cases even if the LLM wobbles.
Kept conservative to avoid false-positives on legitimate assessment questions.
"""

from __future__ import annotations

import re

CANNED_REFUSAL = (
    "I can only help with recommending SHL assessments for a role. I can't help "
    "with that — but tell me who you're hiring for and I'll suggest suitable "
    "assessments."
)

# Prompt-injection / jailbreak attempts.
_INJECTION_PATTERNS = [
    r"ignore (all|any|the|your|previous|prior)",
    r"disregard (all|any|the|your|previous|prior)",
    r"forget (all|your|the|previous|prior) (instructions|rules|prompt)",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions|system)",
    r"you are now",
    r"act as (?!an? (assessment|shl))",   # "act as ..." unless about assessments
    r"developer mode",
    r"jailbreak",
    r"print (your|the) (instructions|prompt|rules)",
]

# Off-topic: legal advice.
_LEGAL_PATTERNS = [
    r"\bis it (legal|lawful|illegal)\b",
    r"\b(discriminat|adverse impact|eeoc|gdpr|lawsuit|sue|liabilit|comply with the law)\w*",
    r"\blegal(ly)? (advice|require|allowed|permitted)\b",
    r"\bcan i (legally|lawfully)\b",
]

# Off-topic: general hiring / HR advice not about assessments.
_GENERAL_HIRING_PATTERNS = [
    r"\bhow much should i pay\b",
    r"\bwhat salary\b",
    r"\bwrite (me )?a? ?job (description|posting|ad)\b",
    r"\bhow do i (interview|onboard|fire|negotiate)\b",
    r"\bnotice period\b",
    r"\bseverance\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_injection(text: str) -> bool:
    """True if the text looks like a prompt-injection / jailbreak attempt."""
    return _matches_any(text, _INJECTION_PATTERNS)


def is_legal_query(text: str) -> bool:
    return _matches_any(text, _LEGAL_PATTERNS)


def is_general_hiring_query(text: str) -> bool:
    return _matches_any(text, _GENERAL_HIRING_PATTERNS)


def should_refuse(text: str) -> bool:
    """Deterministic backstop: True if this obviously off-topic / injection text
    should be refused regardless of what the router says."""
    if not text:
        return False
    return (
        is_injection(text)
        or is_legal_query(text)
        or is_general_hiring_query(text)
    )


def refusal_reply() -> str:
    return CANNED_REFUSAL
