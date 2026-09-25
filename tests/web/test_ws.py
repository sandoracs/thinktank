"""WebSocket tests: lossless replay, human participation, hand raise, tokens
(DESIGN.md §14.2, §17, and the M2 acceptance criteria)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

HUMAN_SESSION = {
    "title": "WS human",
    "topic": "AI co-authorship?",
    "participants": [{"agent": "ai_optimist"}, {"human": "Sándor"}],
    "stop": {"max_rounds": 1, "max_cost_usd": 5.0},
}


def _recv_until(
    ws: WebSocketTestSession, predicate: Callable[[dict[str, Any]], bool], limit: int = 60
) -> dict[str, Any] | None:
    """Receive messages (JSON) until ``predicate(msg)`` matches."""
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        if predicate(msg):
            return msg
    return None


def _wait_ended(client: TestClient, sid: str, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    status = "unknown"
    while time.time() < deadline:
        status = client.get(f"/api/sessions/{sid}").json()["status"]
        if status == "ended":
            return status
        time.sleep(0.1)
    return status


def test_replay_after_reload_is_complete(client: TestClient) -> None:
    """M2 criterion: after a page reload, the transcript is intact (after_seq)."""
    body = {
        "title": "Replay",
        "topic": "T",
        "participants": [
            {"agent": "ai_moderator"},
            {"agent": "ai_optimist"},
            {"agent": "skeptic_methodologist"},
        ],
        "moderator": "ai_moderator",
        "stop": {"max_rounds": 1, "max_cost_usd": 5.0},
    }
    session_id = client.post("/api/sessions", json=body).json()["id"]
    client.post(f"/api/sessions/{session_id}/start")
    deadline = time.time() + 30
    while time.time() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == "ended":
            break
        time.sleep(0.1)
    assert client.get(f"/api/sessions/{session_id}").json()["status"] == "ended"

    expected = client.get(f"/api/sessions/{session_id}/events").json()
    expected_seqs = [e["seq"] for e in expected]

    # A fresh observer connects after the fact: the full stream must replay.
    with client.websocket_connect(f"/ws/sessions/{session_id}?after_seq=0") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello"
        assert hello["last_seq"] == expected_seqs[-1]

        seen: list[int] = []
        messages: list[dict[str, Any]] = []
        ended = False
        while not ended:
            msg = json.loads(ws.receive_text())
            if msg["type"] != "event":
                continue
            seen.append(msg["seq"])
            if msg["event"]["type"] == "MessagePosted":
                messages.append(msg)
            if msg["event"]["type"] == "SessionEnded":
                ended = True

        assert seen == expected_seqs, "replayed seqs must match the stored stream exactly"
        speakers = [m["event"]["payload"]["message"]["speaker_id"] for m in messages]
        for expected_speaker in ("ai_moderator", "ai_optimist", "skeptic_methodologist"):
            assert expected_speaker in speakers
        # Every message event carries a rendered transcript fragment (server-side rendering).
        assert all(m["fragments"]["transcript"] for m in messages)


def test_replay_from_after_seq_only_sends_tail(client: TestClient) -> None:
    body = {
        "title": "Tail",
        "topic": "T",
        "participants": [
            {"agent": "ai_moderator"},
            {"agent": "ai_optimist"},
        ],
        "moderator": "ai_moderator",
        "stop": {"max_rounds": 1, "max_cost_usd": 5.0},
    }
    session_id = client.post("/api/sessions", json=body).json()["id"]
    client.post(f"/api/sessions/{session_id}/start")
    deadline = time.time() + 30
    while time.time() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == "ended":
            break
        time.sleep(0.1)

    all_events = client.get(f"/api/sessions/{session_id}/events").json()
    cut = 3
    tail_expected = [e["seq"] for e in all_events if e["seq"] > cut]

    with client.websocket_connect(f"/ws/sessions/{session_id}?after_seq={cut}") as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
        seen: list[int] = []
        while True:
            msg = json.loads(ws.receive_text())
            if msg["type"] != "event":
                continue
            seen.append(msg["seq"])
            if msg["event"]["type"] == "SessionEnded":
                break
        assert seen == tail_expected


def test_human_speaks_over_websocket(client: TestClient) -> None:
    """M2 criterion: a human takes the floor and the message lands in the stream."""
    creation = client.post("/api/sessions", json=HUMAN_SESSION).json()
    sid = creation["id"]
    token = creation["humans"][0]["token"]
    pid = creation["humans"][0]["participant"]
    assert pid == "Sándor"

    client.post(f"/api/sessions/{sid}/start")
    time.sleep(1.0)  # let the agent speak so the human's turn is pending

    ws_url = f"/ws/sessions/{sid}?participant={pid}&token={token}&after_seq=0"
    with client.websocket_connect(ws_url) as ws:
        turn = _recv_until(ws, lambda m: m["type"] == "your_turn")
        assert turn is not None, "the pending human turn must be offered on (re)connect"
        ws.send_text(json.dumps({"type": "say", "content": "Disclosure must be mandatory."}))

        posted = _recv_until(
            ws,
            lambda m: m["type"] == "event"
            and m["event"]["type"] == "MessagePosted"
            and m["event"]["payload"]["message"]["speaker_id"] == "Sándor",
        )
        assert posted is not None
        assert posted["event"]["payload"]["message"]["content"] == "Disclosure must be mandatory."
        assert posted["fragments"]["transcript"]  # rendered row present

    _wait_ended(client, sid)
    state = client.get(f"/api/sessions/{sid}").json()
    speakers = [
        e["payload"]["message"]["speaker_id"]
        for e in client.get(f"/api/sessions/{sid}/events").json()
        if e["type"] == "MessagePosted"
    ]
    assert "Sándor" in speakers
    assert state["status"] == "ended"


def test_bad_token_is_rejected(client: TestClient) -> None:
    creation = client.post("/api/sessions", json=HUMAN_SESSION).json()
    sid = creation["id"]
    pid = creation["humans"][0]["participant"]


    with client.websocket_connect(f"/ws/sessions/{sid}?participant={pid}&token=wrong-token") as ws:
        # The server closes with 4401; starlette surfaces it on the next receive.
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_text()
        assert excinfo.value.code == 4401


def test_raise_hand_is_event_sourced(client: TestClient) -> None:
    creation = client.post("/api/sessions", json=HUMAN_SESSION).json()
    sid = creation["id"]
    token = creation["humans"][0]["token"]
    pid = creation["humans"][0]["participant"]

    client.post(f"/api/sessions/{sid}/start")
    time.sleep(1.0)  # human turn pending

    ws_url = f"/ws/sessions/{sid}?participant={pid}&token={token}&after_seq=0"
    with client.websocket_connect(ws_url) as ws:
        _recv_until(ws, lambda m: m["type"] == "your_turn")
        ws.send_text(json.dumps({"type": "raise_hand"}))
        raised = _recv_until(
            ws,
            lambda m: bool(
                m["type"] == "event"
                and m["event"]["type"] == "HandRaised"
                and m["event"]["payload"]["participant_id"] == pid
            ),
        )
        assert raised is not None
        assert "kéz" in raised["fragments"]["sidebar"]
        # answer so the session can finish
        ws.send_text(json.dumps({"type": "say", "content": "Done."}))
        _recv_until(
            ws,
            lambda m: m["type"] == "event" and m["event"]["type"] == "MessagePosted",
        )

    events = client.get(f"/api/sessions/{sid}/events").json()
    assert any(e["type"] == "HandRaised" for e in events)
    client.post(f"/api/sessions/{sid}/stop")
    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/api/sessions/{sid}").json()["status"] == "ended":
            break
        time.sleep(0.1)
