"""Tests for app.replies: templated shortlist reply + COMPARE framing."""

from __future__ import annotations

from app.replies import compare_reply, shortlist_reply

ITEMS = [
    {"name": "Core Java (Advanced Level) (New)", "url": "u1", "test_type": "K"},
    {"name": "OPQ32r", "url": "u2", "test_type": "P"},
]


def test_shortlist_reply_states_count_and_role():
    reply = shortlist_reply(
        {"role": "Java developer", "seniority": "mid-level", "skills": ["Java", "SQL"]},
        ITEMS,
    )
    assert "2 assessments" in reply
    assert "mid-level Java developer" in reply
    assert "Java" in reply and "SQL" in reply


def test_shortlist_reply_singular():
    reply = shortlist_reply({"role": "analyst"}, ITEMS[:1])
    assert "1 assessment" in reply
    assert "assessments" not in reply.split("These")[0]


def test_shortlist_reply_handles_missing_constraints():
    reply = shortlist_reply({}, ITEMS)
    assert "2 assessments" in reply
    assert reply.strip()


def test_shortlist_reply_falls_back_to_skills_when_no_role():
    reply = shortlist_reply({"skills": ["Python"]}, ITEMS)
    assert reply.strip()
    assert "Python" in reply


def test_shortlist_reply_accepts_none_constraints():
    reply = shortlist_reply(None, ITEMS)
    assert "2 assessments" in reply


def test_shortlist_reply_dedupes_seniority_in_role():
    # seniority "graduate" already part of role "graduate trainee" -> no doubling.
    reply = shortlist_reply({"role": "graduate trainee", "seniority": "graduate"}, ITEMS)
    assert "graduate graduate" not in reply
    assert "graduate trainee" in reply


def test_compare_reply_uses_llm_text():
    assert compare_reply(ITEMS, "OPQ measures personality; Java measures skill.") == (
        "OPQ measures personality; Java measures skill."
    )


def test_compare_reply_fallback_names_items():
    reply = compare_reply(ITEMS, "")
    assert "Core Java (Advanced Level) (New)" in reply and "OPQ32r" in reply


def test_compare_reply_fallback_single_item():
    reply = compare_reply(ITEMS[:1], "   ")
    assert "OPQ32r" not in reply
    assert "Core Java" in reply


def test_compare_reply_fallback_no_items():
    assert compare_reply([], None).strip()
