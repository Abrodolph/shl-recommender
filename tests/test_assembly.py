"""Tests for app.assembly: test_type-aware battery assembly + default policy.

Covers the Recall@10 lever (guaranteed personality/cognitive defaults) and the
opt-out policy that keeps the "honors edits" behavior intact (a REFINE that drops
OPQ must actually drop it).
"""

from __future__ import annotations

from app.assembly import (
    COGNITIVE_DEFAULT,
    PERSONALITY_DEFAULT,
    assemble_ids,
    default_flags,
)

VALID = {PERSONALITY_DEFAULT, COGNITIVE_DEFAULT} | {f"skill-{i}" for i in range(20)}


def test_defaults_injected_and_capped():
    ids = assemble_ids(["skill-0", "skill-1"], k=10, valid_ids=VALID)
    assert PERSONALITY_DEFAULT in ids and COGNITIVE_DEFAULT in ids
    assert ids[:2] == ["skill-0", "skill-1"]     # skills first, defaults appended
    assert len(ids) == 4


def test_defaults_reserved_when_pool_is_full():
    pool = [f"skill-{i}" for i in range(20)]
    ids = assemble_ids(pool, k=10, valid_ids=VALID)
    assert len(ids) == 10
    assert PERSONALITY_DEFAULT in ids and COGNITIVE_DEFAULT in ids
    assert ids.count(PERSONALITY_DEFAULT) == 1   # no duplicates


def test_no_duplicate_when_default_already_retrieved():
    pool = [PERSONALITY_DEFAULT, "skill-0"]
    ids = assemble_ids(pool, k=10, valid_ids=VALID)
    assert ids.count(PERSONALITY_DEFAULT) == 1


def test_valid_ids_prevents_wasted_slot():
    # Defaults not in the catalog are neither injected nor reserve a slot.
    pool = [f"skill-{i}" for i in range(20)]
    ids = assemble_ids(pool, k=10, valid_ids={f"skill-{i}" for i in range(20)})
    assert len(ids) == 10
    assert PERSONALITY_DEFAULT not in ids


def test_disable_flags():
    ids = assemble_ids(
        ["skill-0"], k=10, add_personality=False, add_cognitive=False, valid_ids=VALID
    )
    assert ids == ["skill-0"]


# --- default_flags policy -----------------------------------------------------
def test_flags_default_on_for_plain_hiring():
    add_p, add_c = default_flags({}, "Hiring a senior Java developer")
    assert add_p and add_c


def test_flags_personality_opt_out_phrases():
    for text in [
        "Drop the OPQ32r",
        "remove the personality test",
        "no personality please",
        "screen for Excel, skip personality",
    ]:
        add_p, _ = default_flags({}, text.lower())
        assert add_p is False, text


def test_flags_cognitive_opt_out():
    add_p, add_c = default_flags({}, "drop verify g+, keep the rest")
    assert add_c is False


def test_flags_knowledge_only_prefs_disable_defaults():
    add_p, add_c = default_flags(
        {"test_type_prefs": ["knowledge", "skills"]}, "excel and word only"
    )
    assert add_p is False and add_c is False


def test_flags_explicit_personality_pref_keeps_it():
    add_p, _ = default_flags(
        {"test_type_prefs": ["knowledge", "personality"]}, "some text"
    )
    assert add_p is True
