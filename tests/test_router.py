"""Tests for app.router: intent + constraint extraction with a MOCKED LLM.

We inject a fake LLMClient that returns canned JSON, so these tests are
deterministic and never hit the network. Covers each intent plus malformed JSON
and non-JSON output (both must degrade to a safe CLARIFY).
"""

from __future__ import annotations

import json

from app.llm.base import LLMClient
from app.router import RouterResult, parse_router_output, route


class FakeLLM(LLMClient):
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    def complete(self, system, messages, json_mode=False) -> str:
        self.calls += 1
        return self._response


def _route(payload) -> RouterResult:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return route([{"role": "user", "content": "hi"}], client=FakeLLM(body))


# --- one intent per case ------------------------------------------------------
def test_recommend_intent_parsed():
    r = _route(
        {
            "intent": "RECOMMEND",
            "constraints": {
                "role": "Java developer",
                "seniority": "mid",
                "skills": ["Java", "SQL"],
                "test_type_prefs": ["knowledge"],
                "other_signals": [],
            },
            "search_query": "mid Java SQL developer knowledge",
            "named_assessments": [],
            "reply_text": "",
        }
    )
    assert r.intent == "RECOMMEND"
    assert r.is_recommend
    assert r.constraints["role"] == "Java developer"
    assert r.constraints["skills"] == ["Java", "SQL"]
    assert r.search_query == "mid Java SQL developer knowledge"


def test_clarify_intent_keeps_reply():
    r = _route(
        {
            "intent": "CLARIFY",
            "constraints": {},
            "search_query": "",
            "reply_text": "What seniority are you hiring for?",
        }
    )
    assert r.intent == "CLARIFY"
    assert r.reply_text == "What seniority are you hiring for?"
    assert r.search_query == ""


def test_refine_intent_is_recommend_like():
    r = _route({"intent": "REFINE", "constraints": {"role": "dev", "skills": ["Go"]}})
    assert r.intent == "REFINE"
    assert r.is_recommend
    # No search_query given -> synthesized from constraints.
    assert "Go" in r.search_query and "dev" in r.search_query


def test_compare_intent_captures_named_assessments():
    r = _route(
        {
            "intent": "COMPARE",
            "constraints": {},
            "named_assessments": ["OPQ32r", "Verify G+"],
            "reply_text": "They measure different things.",
        }
    )
    assert r.intent == "COMPARE"
    assert r.named_assessments == ["OPQ32r", "Verify G+"]
    assert r.reply_text


def test_refuse_intent_has_reply():
    r = _route({"intent": "REFUSE", "reply_text": "I can't help with that."})
    assert r.intent == "REFUSE"
    assert r.reply_text == "I can't help with that."


# --- defensive parsing --------------------------------------------------------
def test_malformed_json_defaults_to_clarify():
    r = _route("{not valid json")
    assert r.intent == "CLARIFY"
    assert r.reply_text  # safe non-empty question


def test_non_object_json_defaults_to_clarify():
    r = _route("[1, 2, 3]")
    assert r.intent == "CLARIFY"


def test_unknown_intent_defaults_to_clarify():
    r = _route({"intent": "BANANA", "reply_text": "x"})
    assert r.intent == "CLARIFY"


def test_clarify_without_reply_gets_safe_default():
    r = _route({"intent": "CLARIFY", "reply_text": ""})
    assert r.intent == "CLARIFY"
    assert r.reply_text.strip()


def test_refuse_without_reply_gets_safe_default():
    r = _route({"intent": "REFUSE"})
    assert r.intent == "REFUSE"
    assert r.reply_text.strip()


def test_constraints_normalized_from_messy_types():
    # skills given as a bare string, junk types coerced.
    r = parse_router_output(
        json.dumps(
            {
                "intent": "RECOMMEND",
                "constraints": {
                    "role": "  Analyst  ",
                    "seniority": "",
                    "skills": "Excel",
                    "test_type_prefs": ["", "cognitive"],
                    "other_signals": None,
                },
                "search_query": "analyst excel",
            }
        )
    )
    assert r.constraints["role"] == "Analyst"
    assert r.constraints["seniority"] is None
    assert r.constraints["skills"] == ["Excel"]
    assert r.constraints["test_type_prefs"] == ["cognitive"]
    assert r.constraints["other_signals"] == []


def test_llm_exception_defaults_to_clarify():
    class BoomLLM(LLMClient):
        def complete(self, system, messages, json_mode=False):
            raise RuntimeError("boom")

    r = route([{"role": "user", "content": "hi"}], client=BoomLLM())
    assert r.intent == "CLARIFY"
    assert r.reply_text.strip()


def test_single_llm_call_per_route():
    fake = FakeLLM(json.dumps({"intent": "RECOMMEND", "constraints": {}}))
    route([{"role": "user", "content": "hi"}], client=fake)
    assert fake.calls == 1
