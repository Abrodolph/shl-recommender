"""Shortlist assembly: turn retrieved ids into a battery (CLAUDE.md §4 §5).

Retrieval is recall-oriented but the labeled shortlists are *batteries* — a
role-specific skills spine PLUS two recurring defaults the user rarely names:
a personality measure (OPQ32r, test_type P) and a cognitive/ability measure
(SHL Verify Interactive G+, test_type A). The traces bear this out: OPQ32r is in
7/10 final shortlists and Verify G+ recurs (see eval/TRACES_ANALYSIS.md).

So after retrieval we assemble a shortlist that keeps the top retrieved items and
*guarantees* the personality/cognitive defaults are present for hiring-style
requests — unless the user opted out. This is the single biggest Recall@10 lever
(measured +0.19 mean recall; see eval/TUNING_LOG.md).

Ordering: role/skill items first, then the defaults appended — matching how the
traces present batteries (skills, then Verify, then OPQ).
"""

from __future__ import annotations

# Catalog ids of the two recurring battery defaults.
PERSONALITY_DEFAULT = "occupational-personality-questionnaire-opq32r"  # test_type P
COGNITIVE_DEFAULT = "shl-verify-interactive-g"                          # test_type A


def assemble_ids(
    retrieved_ids: list[str],
    k: int = 10,
    add_personality: bool = True,
    add_cognitive: bool = True,
    exclude: set[str] | None = None,
    valid_ids: set[str] | None = None,
) -> list[str]:
    """Compose the final shortlist ids from retrieved ids + guaranteed defaults.

    Reserves slots for the enabled defaults so they can't be crowded out, then
    fills the rest with the top retrieved items (skills first). Never exceeds
    ``k`` and never duplicates. If ``valid_ids`` is given, a default is only
    injected (and only reserves a slot) when it exists in that set — so we never
    waste a slot on a default the catalog would drop."""
    exclude = exclude or set()
    defaults: list[str] = []
    if add_personality and PERSONALITY_DEFAULT not in exclude:
        defaults.append(PERSONALITY_DEFAULT)
    if add_cognitive and COGNITIVE_DEFAULT not in exclude:
        defaults.append(COGNITIVE_DEFAULT)
    if valid_ids is not None:
        defaults = [d for d in defaults if d in valid_ids]

    # Reserve slots for defaults; fill the remainder with retrieved (skills-first).
    reserve = min(len(defaults), k)
    room = max(k - reserve, 0)
    skills: list[str] = []
    for d in retrieved_ids:
        if d in defaults or d in exclude or d in skills:
            continue
        skills.append(d)
        if len(skills) >= room:
            break

    out = skills + defaults
    return out[:k]


# --- policy: derive default toggles from the user's constraints ---------------
import re

_REMOVE_VERBS = "drop|remove|without|skip|exclude|replace|no"


def _opted_out(text: str, *topic_markers: str) -> bool:
    """True if the user asked to remove/skip the given default topic, e.g.
    "drop the OPQ", "no personality test", "remove Verify G+".

    The removal verb must be *near* the topic (within a few words), so a positive
    mention elsewhere ("keep Verify G+") plus an unrelated "drop the OPQ" doesn't
    wrongly opt out of the cognitive default."""
    topics = "|".join(re.escape(m) for m in topic_markers)
    pattern = rf"\b(?:{_REMOVE_VERBS})\b[\w\s'+]{{0,20}}?\b(?:{topics})"
    return re.search(pattern, text) is not None


def default_flags(constraints: dict | None, history_text: str = "") -> tuple[bool, bool]:
    """Decide whether to include the personality / cognitive defaults.

    Defaults are ON for hiring-style requests, but turned OFF when the user
    explicitly opts out (in constraints or anywhere in the conversation), or when
    they've asked for a purely knowledge/skills screen with an explicit narrow
    test-type preference."""
    text = (history_text or "").lower()
    add_personality = not _opted_out(text, "personality", "opq", "behavioural", "behavioral")
    add_cognitive = not _opted_out(text, "cognitive", "aptitude", "verify g", "ability test")

    prefs = [p.lower() for p in (constraints or {}).get("test_type_prefs", [])]
    if prefs:
        wants_personality = any("person" in p or "behav" in p for p in prefs)
        wants_cognitive = any(
            "cognit" in p or "abilit" in p or "aptitud" in p or "reason" in p
            for p in prefs
        )
        wants_knowledge_only = all(
            "know" in p or "skill" in p or "simul" in p or "technical" in p
            for p in prefs
        )
        # If the user named test types, honor personality/cognitive only when
        # requested OR when they didn't restrict to a knowledge-only screen.
        if wants_knowledge_only and not wants_personality:
            add_personality = False
        if wants_knowledge_only and not wants_cognitive:
            add_cognitive = False
    return add_personality, add_cognitive
