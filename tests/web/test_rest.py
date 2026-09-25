"""REST API tests: agents, plugins, session lifecycle, events (DESIGN.md §14.1)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import BaseModel

from roundtable.llm.client import ChatMessage, Purpose
from roundtable.llm.fake import FakeLLM
from roundtable.web.app import create_app

TWO_AGENT_SESSION = {
    "title": "Web test",
    "topic": "Is LLM co-authorship acceptable?",
    "participants": [
        {"agent": "ai_moderator"},
        {"agent": "ai_optimist"},
        {"agent": "skeptic_methodologist"},
    ],
    "moderator": "ai_moderator",
    "stop": {"max_rounds": 1, "max_cost_usd": 5.0},
}

def _wait_status(client: TestClient, session_id: str, wanted: set[str], timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    status = "unknown"
    while time.time() < deadline:
        status = client.get(f"/api/sessions/{session_id}").json()["status"]
        if status in wanted:
            return status
        time.sleep(0.1)
    return status


def test_agent_library_seeded(client: TestClient) -> None:
    agents = client.get("/api/agents").json()
    ids = {a["id"] for a in agents}
    assert {"ai_moderator", "ai_optimist", "skeptic_methodologist", "pragmatic_editor"} <= ids
    by_id = {a["id"]: a for a in agents}
    assert by_id["ai_optimist"]["persona"]["name"] == "Tóth Lilla"
    assert by_id["ai_moderator"]["drift_mode"] == "locked"


def _new_agent_body(agent_id: str = "cautious_statistician") -> dict[str, object]:
    return {
        "id": agent_id,
        "model": "anthropic/claude-3-5-sonnet",
        "temperature": 0.5,
        "persona": {
            "name": "Dr. Stat",
            "role": "cautious statistician",
            "expertise": ["statistics"],
            "values": ["rigor"],
            "boundaries": ["no medical advice"],
        },
        "drift": {"mode": "bounded"},
        "consistency_check": True,
        "consistency_threshold": 4,
    }


def test_agent_crud_api(client: TestClient) -> None:
    # create
    created = client.post("/api/agents", json=_new_agent_body())
    assert created.status_code == 201
    assert created.json()["id"] == "cautious_statistician"
    assert created.json()["drift_mode"] == "bounded"

    # listed
    ids = {a["id"] for a in client.get("/api/agents").json()}
    assert "cautious_statistician" in ids

    # update
    body = _new_agent_body()
    body["temperature"] = 0.9
    body["persona"]["role"] = "extra-cautious statistician"
    updated = client.put("/api/agents/cautious_statistician", json=body)
    assert updated.status_code == 200
    listing = {a["id"]: a for a in client.get("/api/agents").json()}
    assert listing["cautious_statistician"]["temperature"] == 0.9
    assert listing["cautious_statistician"]["persona"]["role"] == "extra-cautious statistician"

    # delete
    deleted = client.delete("/api/agents/cautious_statistician")
    assert deleted.status_code == 204
    ids = {a["id"] for a in client.get("/api/agents").json()}
    assert "cautious_statistician" not in ids


def test_agent_create_conflict_and_errors(client: TestClient) -> None:
    assert client.post("/api/agents", json=_new_agent_body()).status_code == 201
    # duplicate id -> 409
    assert client.post("/api/agents", json=_new_agent_body()).status_code == 409
    # invalid payload -> 400
    bad = client.post("/api/agents", json={"id": "x", "persona": {}})
    assert bad.status_code == 400
    # update unknown -> 404
    body = _new_agent_body("ghost")
    assert client.put("/api/agents/ghost", json=body).status_code == 404
    # delete unknown -> 404
    assert client.delete("/api/agents/ghost").status_code == 404


def test_agent_form_creates_template(client: TestClient) -> None:
    form = {
        "id": "form_agent",
        "model": "anthropic/claude-3-5-sonnet",
        "temperature": "0.6",
        "persona_name": "Form Person",
        "persona_role": "form-built",
        "expertise": "a, b",
        "values": "care",
        "boundaries": "no harm",
        "drift_mode": "free",
        "consistency_check": "on",
        "consistency_threshold": "3",
    }
    response = client.post("/agents", data=form, follow_redirects=False)
    assert response.status_code == 303
    listing = {a["id"]: a for a in client.get("/api/agents").json()}
    assert "form_agent" in listing
    assert listing["form_agent"]["drift_mode"] == "free"
    assert listing["form_agent"]["persona"]["name"] == "Form Person"
    assert listing["form_agent"]["persona"]["expertise"] == ["a", "b"]


def test_agent_form_page_renders(client: TestClient) -> None:
    page = client.get("/agents/new")
    assert page.status_code == 200
    assert "Új agent" in page.text
    assert "drift_mode" in page.text


def test_plugins_lists_builtin_strategies(client: TestClient) -> None:
    plugins = client.get("/api/plugins").json()
    names = {s["name"] for s in plugins["turn_strategies"]}
    assert {"round_robin", "hand_raise"} <= names


def test_create_start_end_session(client: TestClient) -> None:
    created = client.post("/api/sessions", json=TWO_AGENT_SESSION)
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["humans"] == []  # no human in this config

    session_id = body["id"]
    started = client.post(f"/api/sessions/{session_id}/start")
    assert started.status_code == 200
    assert started.json() == {"status": "running"}

    assert _wait_status(client, session_id, {"ended"}) == "ended"
    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["ended_reason"] in ("max_rounds", "cost_limit", "manual", "all_disabled")
    assert state["message_count"] >= 3  # moderator open + two speakers

    # The dashboard lists the finished session.
    dashboard = client.get("/").text
    assert "Web test" in dashboard

    # The REST list matches.
    listing = client.get("/api/sessions").json()
    assert any(s["id"] == session_id and s["status"] == "ended" for s in listing)


def test_unknown_agent_rejected(client: TestClient) -> None:
    body = {
        "title": "Bad",
        "topic": "T",
        "participants": [{"agent": "does_not_exist"}, {"agent": "ai_optimist"}],
        "stop": {"max_rounds": 1},
    }
    response = client.post("/api/sessions", json=body)
    assert response.status_code == 400
    assert "does_not_exist" in response.json()["detail"]


def test_starting_unknown_session_404s(client: TestClient) -> None:
    assert client.get("/api/sessions/not-a-uuid").status_code == 404
    assert client.post("/api/sessions/not-a-uuid/start").status_code == 404


def test_events_after_seq_and_type_filter(client: TestClient) -> None:
    session_id = client.post("/api/sessions", json=TWO_AGENT_SESSION).json()["id"]
    client.post(f"/api/sessions/{session_id}/start")
    assert _wait_status(client, session_id, {"ended"}) == "ended"

    all_events = client.get(f"/api/sessions/{session_id}/events").json()
    assert len(all_events) >= 6
    assert [e["seq"] for e in all_events] == list(range(1, len(all_events) + 1))

    tail = client.get(f"/api/sessions/{session_id}/events", params={"after_seq": 3}).json()
    assert tail[0]["seq"] == 4
    assert len(tail) == len(all_events) - 3

    messages = client.get(
        f"/api/sessions/{session_id}/events", params={"types": "MessagePosted"}
    ).json()
    assert messages
    assert all(e["type"] == "MessagePosted" for e in messages)

    bad = client.get(f"/api/sessions/{session_id}/events", params={"types": "BogusType"})
    assert bad.status_code == 400


def test_pause_resume_stop_with_pending_human(client: TestClient) -> None:
    body = {
        "title": "Controls",
        "topic": "T",
        "participants": [{"agent": "ai_optimist"}, {"human": "Sándor"}],
        "stop": {"max_rounds": 5, "max_cost_usd": 5.0},
    }
    session_id = client.post("/api/sessions", json=body).json()["id"]
    client.post(f"/api/sessions/{session_id}/start")

    # Let the agent speak so the human's turn is pending.
    assert _wait_status(client, session_id, {"running", "ended"}, timeout=10) in {"running", "ended"}
    time.sleep(1.0)
    state = client.get(f"/api/sessions/{session_id}").json()
    if state["status"] != "running":
        return  # session already finished; nothing to control

    assert client.post(f"/api/sessions/{session_id}/pause").json() == {"status": "paused"}
    assert client.post(f"/api/sessions/{session_id}/pause").status_code == 409
    assert client.post(f"/api/sessions/{session_id}/resume").json() == {"status": "running"}

    assert client.post(f"/api/sessions/{session_id}/stop").json() == {"status": "stopping"}
    assert _wait_status(client, session_id, {"ended"}, timeout=10) == "ended"
    events = [e["type"] for e in client.get(f"/api/sessions/{session_id}/events").json()]
    assert "SessionPaused" in events
    assert "SessionResumed" in events


def test_builder_form_creates_and_redirects(client: TestClient) -> None:
    form = {
        "title": "Form build",
        "topic": "AI és tudomány",
        "language": "hu",
        "questions": "Jó ez?\nÉs ez?",
        "agents": ["ai_moderator", "ai_optimist"],
        "humans": "Sándor",
        "moderator": "ai_moderator",
        "strategy": "round_robin",
        "max_rounds": "2",
        "max_cost_usd": "2.0",
    }
    response = client.post("/sessions", data=form, follow_redirects=False)
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/sessions/")
    assert "participant=S%C3%A1ndor" in location or "participant=Sándor" in location
    assert "token=" in location

    session_id = location.split("/sessions/")[1].split("?")[0]
    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["status"] in ("running", "paused", "ended")
    assert state["config"]["title"] == "Form build"
    names = [p["human"] or p["agent"] for p in state["config"]["participants"]]
    assert "Sándor" in names
    # stop the session so the app can shut down cleanly
    client.post(f"/api/sessions/{session_id}/stop")
    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == "ended":
            break
        time.sleep(0.1)


def test_builder_form_applies_strategy_params(client: TestClient) -> None:
    """M5: the schema-driven strategy params reach the session config."""
    form = {
        "title": "Bidding form",
        "topic": "T",
        "language": "hu",
        "questions": "Jó ez?",
        "agents": ["ai_moderator", "ai_optimist"],
        "strategy": "bidding",
        "strategy_params": '{"bid_model": "fake", "bid_temperature": 0.1}',
        "max_rounds": "1",
        "max_cost_usd": "2.0",
    }
    response = client.post("/sessions", data=form, follow_redirects=False)
    assert response.status_code == 303
    session_id = response.headers["location"].split("/sessions/")[1].split("?")[0]
    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["config"]["strategy"]["name"] == "bidding"
    assert state["config"]["strategy"]["params"]["bid_model"] == "fake"
    assert state["config"]["strategy"]["params"]["bid_temperature"] == 0.1
    client.post(f"/api/sessions/{session_id}/stop")


def test_plugins_endpoint_exposes_schema(client: TestClient) -> None:
    """M5: /api/plugins returns each strategy's Params schema for form generation."""
    data = client.get("/api/plugins").json()
    by_name = {s["name"]: s for s in data["turn_strategies"]}
    assert set(by_name) >= {"round_robin", "hand_raise", "bidding"}
    schema = by_name["bidding"]["params_schema"]
    assert "bid_model" in schema["properties"]
    assert "bid_temperature" in schema["properties"]


def test_approval_flow_over_api(tmp_path: Path) -> None:
    """M6: APPROVED mode surfaces a pending approval; the API approves it."""
    import sqlite3

    proposal = {
        "stance_updates": [
            {
                "question_id": "q1",
                "new_position": "disclose-always",
                "new_confidence": 0.9,
                "influenced_by": ["ai_optimist"],
                "reason": "the evidence was strong",
            }
        ],
        "attitude_updates": [],
        "mood": "persuaded",
    }

    def responder(
        model: str,
        messages: list[ChatMessage],
        purpose: Purpose,
        response_model: type[BaseModel] | None,
    ) -> str:
        if purpose == "reflection":
            return json.dumps(proposal)
        return "a considered position"

    db = tmp_path / "approval.db"
    app = create_app(
        llm=FakeLLM(responder=responder),
        database_url=f"sqlite+aiosqlite:///{db}",
        human_timeout_s=30,
    )
    with TestClient(app) as c:
        # Register an APPROVED-mode agent in the library.
        agent_cfg = {
            "id": "approved_agent",
            "model": "fake-model",
            "persona": {"name": "Approved", "role": "measured participant"},
            "drift": {"mode": "approved"},
        }
        con = sqlite3.connect(db)
        con.execute(
            "INSERT INTO agent_templates (id, config, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (agent_cfg["id"], json.dumps(agent_cfg), "2026-01-01 00:00:00", "2026-01-01 00:00:00"),
        )
        con.commit()
        con.close()

        body = {
            "title": "approval",
            "topic": "AI co-authorship",
            "questions": [{"id": "q1", "text": "Must AI usage be disclosed?"}],
            "participants": [{"agent": "approved_agent"}],
            "stop": {"max_rounds": 1, "max_cost_usd": 5.0},
        }
        session_id = c.post("/api/sessions", json=body).json()["id"]
        c.post(f"/api/sessions/{session_id}/start")
        assert _wait_status(c, session_id, {"ended"}, timeout=10) == "ended"

        # A pending approval is visible; the persona is unchanged.
        approvals = c.get(f"/api/sessions/{session_id}/approvals").json()
        assert len(approvals) == 1
        assert approvals[0]["agent_id"] == "approved_agent"
        assert approvals[0]["status"] == "pending"

        # The live table renders the approval panel with the pending request.
        live_html = c.get(f"/sessions/{session_id}").text
        assert 'id="approvals-panel"' in live_html
        assert "Jóváhagyás" in live_html

        # Approve it.
        r = c.post(
            f"/api/sessions/{session_id}/approvals/approved_agent", json={"decision": "approve"}
        )
        assert r.status_code == 200
        approvals = c.get(f"/api/sessions/{session_id}/approvals").json()
        assert approvals[0]["status"] == "approved"

        # The decision and the applied persona change are in the event stream.
        events = c.get(f"/api/sessions/{session_id}/events").json()
        types = {e["type"] for e in events}
        assert "ApprovalDecided" in types
        assert "PersonaUpdated" in types
        state = c.get(f"/api/sessions/{session_id}").json()
        assert state["persona_states"]["approved_agent"]["stances"]["q1"]["position"] == "disclose-always"

        # A second decision on the same (now closed) approval is a conflict.
        r = c.post(
            f"/api/sessions/{session_id}/approvals/approved_agent", json={"decision": "reject"}
        )
        assert r.status_code == 409


def test_replay_page_renders(client: TestClient) -> None:
    """M6: a finished session can be opened in the step-through replay view."""
    session_id = client.post("/api/sessions", json=TWO_AGENT_SESSION).json()["id"]
    client.post(f"/api/sessions/{session_id}/start")
    assert _wait_status(client, session_id, {"ended"}, timeout=15) == "ended"

    page = client.get(f"/sessions/{session_id}/replay")
    assert page.status_code == 200
    assert "Visszajátszás" in page.text
    assert 'id="rp-events"' in page.text
    assert 'id="rp-play"' in page.text

    # The event stream it steps through is the source of truth, in seq order.
    events = client.get(f"/api/sessions/{session_id}/events").json()
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)
    assert any(e["type"] == "SessionEnded" for e in events)


def test_agent_inspector_timeline(tmp_path: Path) -> None:
    """M4: the inspector exposes core, current state, and the drift timeline."""
    proposal = {
        "stance_updates": [
            {
                "question_id": "q1",
                "new_position": "disclose-always",
                "new_confidence": 0.9,
                "influenced_by": ["skeptic_methodologist"],
                "reason": "the replication point was solid",
            }
        ],
        "attitude_updates": [
            {"participant_id": "skeptic_methodologist", "new_value": 0.6, "reason": "earned respect"}
        ],
        "mood": "persuaded",
    }

    def responder(
        model: str,
        messages: list[ChatMessage],
        purpose: Purpose,
        response_model: type[BaseModel] | None,
    ) -> str:
        if purpose == "reflection":
            return json.dumps(proposal)
        return "a considered position"

    app = create_app(
        llm=FakeLLM(responder=responder),
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'inspector.db'}",
        human_timeout_s=30,
    )
    with TestClient(app) as c:
        body = {
            "title": "inspector",
            "topic": "AI co-authorship",
            "questions": [{"id": "q1", "text": "Must AI usage be disclosed?"}],
            "participants": [{"agent": "ai_optimist"}, {"agent": "skeptic_methodologist"}],
            "stop": {"max_rounds": 1, "max_cost_usd": 5.0},
        }
        sid = c.post("/api/sessions", json=body).json()["id"]
        c.post(f"/api/sessions/{sid}/start")
        deadline = time.time() + 30
        while time.time() < deadline:
            if c.get(f"/api/sessions/{sid}").json()["status"] == "ended":
                break
            time.sleep(0.2)

        api = c.get(f"/api/sessions/{sid}/agents/ai_optimist")
        assert api.status_code == 200, api.text
        data = api.json()
        assert data["persona"]["name"] == "Tóth Lilla"
        assert data["drift_mode"] == "free"
        assert data["current"]["stances"]["q1"]["position"] == "disclose-always"
        assert data["current"]["attitudes"]["skeptic_methodologist"] == 0.6
        assert data["current"]["mood"] == "persuaded"
        assert [row["version"] for row in data["timeline"]] == [1]
        event_types = {e["type"] for e in data["events"]}
        assert {"ReflectionProposed", "PersonaUpdated"} <= event_types

        page = c.get(f"/sessions/{sid}/agents/ai_optimist")
        assert page.status_code == 200
        for needle in ("Tóth Lilla", "Drift-idővonal", "disclose-always", "persuaded"):
            assert needle in page.text, needle

        # Unknown agent / session 404.
        assert c.get(f"/api/sessions/{sid}/agents/does_not_exist").status_code == 404
        assert c.get("/api/sessions/not-a-uuid/agents/ai_optimist").status_code == 404
