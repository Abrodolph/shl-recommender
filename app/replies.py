"""Templated shortlist replies and COMPARE formatting.

Responsibilities (CLAUDE.md §4, "Reply text: hybrid"):
- Generate the schema-critical shortlist reply from a deterministic template
  ("Here are N assessments for a {seniority} {role}...") so the response never
  depends on LLM formatting.
- Format COMPARE answers from catalog evidence (the fluent prose comes from the
  LLM; we just frame/ground it).

``constraints`` is the normalized dict the router emits:
``{role, seniority, skills[], test_type_prefs[], other_signals[]}`` (all optional).
Accepts a plain dict or any object exposing those attributes.
"""

from __future__ import annotations


def _get(constraints, key: str):
    if constraints is None:
        return None
    if isinstance(constraints, dict):
        return constraints.get(key)
    return getattr(constraints, key, None)


def _describe_role(constraints) -> str:
    """Build a human phrase like "mid-level Java developer" from constraints."""
    seniority = (_get(constraints, "seniority") or "").strip()
    role = (_get(constraints, "role") or "").strip()
    skills = _get(constraints, "skills") or []

    # Avoid doubling when seniority is already part of the role phrase
    # (e.g. seniority "graduate" + role "graduate trainee").
    if seniority and role and seniority.lower() in role.lower():
        seniority = ""
    core = " ".join(p for p in (seniority, role) if p).strip()
    if not core:
        # No role given: fall back to skills, else a generic phrase.
        if skills:
            core = f"{', '.join(skills[:3])} role"
        else:
            core = "this role"
    return core


def shortlist_reply(constraints, items: list) -> str:
    """Deterministic reply text for a committed shortlist.

    ``items`` is the list of recommendation dicts ({name,url,test_type}). The
    count is stated explicitly; a compact skill/focus clause is appended when
    available. Never raises — always returns a non-empty string."""
    n = len(items)
    role_phrase = _describe_role(constraints)
    skills = _get(constraints, "skills") or []

    noun = "assessment" if n == 1 else "assessments"
    reply = f"Here are {n} {noun} for a {role_phrase}"

    focus = [s for s in skills if s][:4]
    if focus:
        reply += f", covering {_join_and(focus)}"
    reply += "."

    reply += (
        " These span the skills, ability, and personality signals that fit the role"
        " — tell me if you'd like to adjust the mix or drop anything."
    )
    return reply


def compare_reply(items: list, llm_text: str) -> str:
    """Frame a grounded COMPARE answer.

    The LLM supplies the fluent comparison (``llm_text``), grounded on catalog
    text upstream. If the LLM text is empty, fall back to naming the items being
    compared so we still return something schema-valid and on-topic."""
    text = (llm_text or "").strip()
    if text:
        return text
    names = [_item_name(i) for i in items if _item_name(i)]
    if len(names) >= 2:
        return (
            f"Both {_join_and(names)} are in your shortlist. They differ in what they "
            "measure and how long they take — let me know which dimension matters most "
            "and I'll break down the trade-off."
        )
    if names:
        return f"{names[0]} is in your shortlist. What would you like to compare it against?"
    return "Which assessments would you like me to compare?"


# --- helpers ------------------------------------------------------------------
def _item_name(item) -> str:
    if isinstance(item, dict):
        return item.get("name") or ""
    return getattr(item, "name", "") or ""


def _join_and(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"
