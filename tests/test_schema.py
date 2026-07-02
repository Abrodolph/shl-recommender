"""Tests for the API contract / Pydantic schema (CLAUDE.md §3, §9).

Assert that ChatResponse is always valid: recommendations is [] or 1–10 items,
each item has exactly {name, url, test_type}, and malformed inputs are rejected
(so the agent layer can fall back to a valid response).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import (
    MAX_RECOMMENDATIONS,
    ChatRequest,
    ChatResponse,
    Message,
    Recommendation,
)

VALID_URL = "https://www.shl.com/products/product-catalog/view/spring-new/"


def _rec(i: int = 0) -> Recommendation:
    return Recommendation(name=f"Test {i}", url=VALID_URL, test_type="K")


# --- Message / ChatRequest ----------------------------------------------------
def test_message_valid_roles():
    for role in ("user", "assistant", "system"):
        assert Message(role=role, content="hi").role == role


def test_message_bad_role_rejected():
    with pytest.raises(ValidationError):
        Message(role="tool", content="hi")


def test_chat_request_requires_at_least_one_message():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_chat_request_parses_history():
    req = ChatRequest(
        messages=[
            {"role": "user", "content": "Hiring a Java dev"},
            {"role": "assistant", "content": "What seniority?"},
            {"role": "user", "content": "Mid-level"},
        ]
    )
    assert len(req.messages) == 3
    assert req.messages[0].role == "user"


def test_chat_request_ignores_extra_message_keys():
    req = ChatRequest(messages=[{"role": "user", "content": "hi", "name": "x"}])
    assert req.messages[0].content == "hi"


# --- Recommendation -----------------------------------------------------------
def test_recommendation_valid():
    rec = Recommendation(name="Spring (New)", url=VALID_URL, test_type="K")
    assert rec.test_type == "K"


def test_recommendation_multi_letter_test_type_ok():
    # Traces render multi-key items as comma-joined codes, e.g. "K,S".
    rec = Recommendation(name="Microsoft Excel 365", url=VALID_URL, test_type="K,S")
    assert rec.test_type == "K,S"


@pytest.mark.parametrize("field", ["name", "url", "test_type"])
def test_recommendation_missing_field_rejected(field):
    kwargs = {"name": "X", "url": VALID_URL, "test_type": "K"}
    del kwargs[field]
    with pytest.raises(ValidationError):
        Recommendation(**kwargs)


@pytest.mark.parametrize("field", ["name", "url", "test_type"])
def test_recommendation_blank_field_rejected(field):
    kwargs = {"name": "X", "url": VALID_URL, "test_type": "K"}
    kwargs[field] = "   "
    with pytest.raises(ValidationError):
        Recommendation(**kwargs)


def test_recommendation_non_http_url_rejected():
    with pytest.raises(ValidationError):
        Recommendation(name="X", url="www.shl.com/foo", test_type="K")


def test_recommendation_forbids_extra_keys():
    with pytest.raises(ValidationError):
        Recommendation(
            name="X", url=VALID_URL, test_type="K", description="not allowed"
        )


def test_recommendation_trims_whitespace():
    rec = Recommendation(name="  Spring  ", url=VALID_URL, test_type=" K ")
    assert rec.name == "Spring"
    assert rec.test_type == "K"


# --- ChatResponse -------------------------------------------------------------
def test_response_empty_recommendations_valid():
    resp = ChatResponse(reply="What seniority?", recommendations=[])
    assert resp.recommendations == []
    assert resp.end_of_conversation is False


def test_response_default_recommendations_is_empty():
    resp = ChatResponse(reply="hi")
    assert resp.recommendations == []


def test_response_ten_recommendations_valid():
    resp = ChatResponse(
        reply="Here you go",
        recommendations=[_rec(i) for i in range(MAX_RECOMMENDATIONS)],
        end_of_conversation=True,
    )
    assert len(resp.recommendations) == MAX_RECOMMENDATIONS


def test_response_too_many_recommendations_rejected():
    with pytest.raises(ValidationError):
        ChatResponse(
            reply="Here you go",
            recommendations=[_rec(i) for i in range(MAX_RECOMMENDATIONS + 1)],
        )


def test_response_blank_reply_rejected():
    with pytest.raises(ValidationError):
        ChatResponse(reply="", recommendations=[])


def test_response_forbids_extra_keys():
    with pytest.raises(ValidationError):
        ChatResponse(reply="hi", recommendations=[], foo="bar")


def test_response_bad_recommendation_item_rejected():
    with pytest.raises(ValidationError):
        ChatResponse(reply="hi", recommendations=[{"name": "X"}])


# --- Endpoints ----------------------------------------------------------------
def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_endpoint_returns_valid_contract(monkeypatch):
    # Stub the agent so this exercises the endpoint plumbing / contract shape
    # deterministically (no LLM/network). Agent behavior is covered separately.
    def fake_handle(messages):
        return ChatResponse(
            reply="What seniority are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    monkeypatch.setattr("app.main.handle", fake_handle)
    client = TestClient(app)
    r = client.post(
        "/chat", json={"messages": [{"role": "user", "content": "Hiring a Java dev"}]}
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert body["recommendations"] == []
    assert isinstance(body["reply"], str) and body["reply"]
    assert body["end_of_conversation"] is False
    # Each returned item (when present) must carry exactly the 3 contract keys.
    for item in body["recommendations"]:
        assert set(item.keys()) == {"name", "url", "test_type"}


def test_chat_endpoint_rejects_empty_messages():
    client = TestClient(app)
    r = client.post("/chat", json={"messages": []})
    assert r.status_code == 422


def test_chat_endpoint_rejects_bad_role():
    client = TestClient(app)
    r = client.post("/chat", json={"messages": [{"role": "tool", "content": "x"}]})
    assert r.status_code == 422
