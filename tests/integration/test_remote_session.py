"""End-to-end session with a remote participant (M7).

Confirms a :class:`RemoteAgent` — a participant whose brain lives in an
external HTTP service — joins the table over the fixed protocol, speaks its
turn, and the debate still ends cleanly alongside LLM agents. A failing remote
is recorded as an error and never wedges the session.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from roundtable.core.manager import SessionManager
from roundtable.core.state import project
from roundtable.domain.events import EventType
from roundtable.domain.models import (
    AgentConfig,
    DebateQuestion,
    ParticipantRef,
    PersonaCore,
    RemoteConfig,
    SessionConfig,
    StopConditions,
)
from roundtable.llm.client import ChatMessage, Purpose
from roundtable.llm.fake import FakeLLM
from roundtable.storage.db import init_db, make_engine, make_session_factory
from roundtable.storage.repositories import EventStore


def _responder(model: str, messages: list[ChatMessage], purpose: Purpose, response_model: object) -> str:
    if purpose == "speech":
        return "an LLM-backed position"
    return "ok"


def _ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/speak"):
        return httpx.Response(200, json={"content": "an external service's position"})
    return httpx.Response(200, json={})


def _fail_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/speak"):
        return httpx.Response(500, json={"error": "down"})
    return httpx.Response(200, json={})


async def _build(tmp_path: Path, handler: object) -> tuple[SessionManager, EventStore, httpx.AsyncClient, object]:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'remote.db'}")
    await init_db(engine)
    store = EventStore(make_session_factory(engine))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    manager = SessionManager(store=store, llm=FakeLLM(responder=_responder), remote_client=client)
    return manager, store, client, engine


def _session() -> SessionConfig:
    return SessionConfig(
        title="remote debate",
        topic="Should AI co-author papers?",
        questions=[DebateQuestion(id="q1", text="Should AI co-author papers?")],
        participants=[ParticipantRef(agent="a"), ParticipantRef(agent="b"), ParticipantRef(remote="remote1")],
        stop=StopConditions(max_rounds=1, max_cost_usd=5.0),
    )


@pytest.mark.asyncio
async def test_remote_agent_speaks_in_a_real_session(tmp_path: Path) -> None:
    manager, store, client, db = await _build(tmp_path, _ok_handler)
    try:
        agents = {
            "a": AgentConfig(id="a", model="fake", persona=PersonaCore(name="A", role="moderator")),
            "b": AgentConfig(id="b", model="fake", persona=PersonaCore(name="B", role="skeptic")),
        }
        remotes = {"remote1": RemoteConfig(id="remote1", url="http://remote.test/base")}
        session = await manager.create_session(_session(), agents, remotes)
        await manager.start(session)
        sid = session.session_id

        events = await store.get_events(sid)
        speakers = [e.payload["message"]["speaker_id"] for e in events if e.type is EventType.MESSAGE_POSTED]
        # Round-robin: every active seat, including the remote one, speaks once.
        assert set(speakers) == {"a", "b", "remote1"}
        assert len(speakers) == 3

        final = project(sid, events)
        assert final.status == "ended"
        assert final.message_count == 3

        # The remote's speech carried its content and the remote meta marker.
        remote_msgs = [
            e.payload["message"]
            for e in events
            if e.type is EventType.MESSAGE_POSTED and e.payload["message"]["speaker_id"] == "remote1"
        ]
        assert remote_msgs[0]["content"] == "an external service's position"
        assert remote_msgs[0]["meta"]["remote_url"] == "http://remote.test/base"

        # Remote agent never reflected (the engine skips non-AI participants).
        reflected = [
            e for e in events if e.type is EventType.REFLECTION_PROPOSED and e.payload.get("agent_id") == "remote1"
        ]
        assert reflected == []
    finally:
        await client.aclose()
        await db.dispose()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_remote_speech_error_is_recorded_and_session_still_ends(tmp_path: Path) -> None:
    manager, store, client, db = await _build(tmp_path, _fail_handler)
    try:
        agents = {"a": AgentConfig(id="a", model="fake", persona=PersonaCore(name="A", role="moderator"))}
        remotes = {"remote1": RemoteConfig(id="remote1", url="http://remote.test/base")}
        cfg = _session()
        cfg.participants = [ParticipantRef(agent="a"), ParticipantRef(remote="remote1")]
        session = await manager.create_session(cfg, agents, remotes)
        await manager.start(session)

        events = await store.get_events(session.session_id)
        # A remote error surfaces as an ERROR event, not a crash.
        assert any(e.type is EventType.ERROR for e in events)
        assert any(e.type is EventType.TURN_SKIPPED for e in events)
        final = project(session.session_id, events)
        assert final.status == "ended"
    finally:
        await client.aclose()
        await db.dispose()  # type: ignore[union-attr]
