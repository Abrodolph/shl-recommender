"""Endpoint robustness tests (CLAUDE.md §3, §9).

Health is always 200; /chat degrades to a valid fallback on timeout or error and
never violates the contract. The agent is stubbed so these are fast and offline.
"""

from __future__ import annotations

import time

import app.main as main
from app.schemas import ChatResponse
from fastapi.testclient import TestClient

BODY = {"messages": [{"role": "user", "content": "Hiring a Java dev"}]}


def test_health_always_ok():
    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_chat_ok_path(monkeypatch):
    monkeypatch.setattr(
        main, "handle",
        lambda msgs: ChatResponse(reply="ok", recommendations=[], end_of_conversation=False),
    )
    r = TestClient(main.app).post("/chat", json=BODY)
    assert r.status_code == 200
    assert set(r.json()) == {"reply", "recommendations", "end_of_conversation"}


def test_chat_timeout_returns_fallback(monkeypatch):
    def slow(msgs):
        time.sleep(2)
        return ChatResponse(reply="late", recommendations=[])

    monkeypatch.setattr(main, "handle", slow)
    monkeypatch.setattr(main, "_TIMEOUT_S", 0.2)   # force the guard to trip fast
    r = TestClient(main.app).post("/chat", json=BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["recommendations"] == []           # SAFE_FALLBACK
    assert body["reply"].strip()


def test_chat_exception_returns_fallback(monkeypatch):
    def boom(msgs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(main, "handle", boom)
    r = TestClient(main.app).post("/chat", json=BODY)
    assert r.status_code == 200
    assert r.json()["recommendations"] == []


def test_chat_rejects_malformed_body():
    client = TestClient(main.app)
    assert client.post("/chat", json={"messages": []}).status_code == 422
    assert client.post("/chat", json={"nope": 1}).status_code == 422


def test_warmup_callable_is_safe():
    # Should never raise even if artifacts are odd; sets ready flag on success.
    main._warm["ready"] = False
    main._warmup()
    # Either it warmed (ready True) or it logged and moved on; never raises.
    assert main._warm["ready"] in (True, False)
