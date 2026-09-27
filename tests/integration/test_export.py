"""Export tests (DESIGN.md §14.1, M6 acceptance: JSONL export ready for analysis)."""

from __future__ import annotations

import csv
import io
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from thinktank.core.manager import SessionManager
from thinktank.domain.models import (
    AgentConfig,
    DebateQuestion,
    ParticipantRef,
    PersonaCore,
    SessionConfig,
    StopConditions,
)
from thinktank.llm.client import ChatMessage, Purpose
from thinktank.llm.fake import FakeLLM
from thinktank.storage.db import init_db, make_engine, make_session_factory
from thinktank.storage.repositories import EventStore


def _responder(
    model: str, messages: list[ChatMessage], purpose: Purpose, response_model: object
) -> str:
    if purpose == "speech":
        return "a considered position"
    return "ok"


def _agents() -> dict[str, AgentConfig]:
    return {
        pid: AgentConfig(id=pid, model="fake", persona=PersonaCore(name=pid, role="r"))
        for pid in ("a", "b")
    }


def _session() -> SessionConfig:
    return SessionConfig(
        title="export",
        topic="Should AI co-author papers?",
        questions=[DebateQuestion(id="q1", text="Should AI co-author papers?")],
        participants=[ParticipantRef(agent=pid) for pid in ("a", "b")],
        stop=StopConditions(max_rounds=1, max_cost_usd=5.0),
    )


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[tuple[SessionManager, EventStore, uuid.UUID]]:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'export.db'}")
    await init_db(engine)
    store = EventStore(make_session_factory(engine))
    manager = SessionManager(store=store, llm=FakeLLM(responder=_responder))
    session = await manager.create_session(_session(), _agents())
    await manager.start(session)
    try:
        yield manager, store, session.state.session_id  # type: ignore[return-value]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_jsonl_export_is_analysis_ready(harness: tuple[SessionManager, EventStore, uuid.UUID]) -> None:
    _manager, store, session_id = harness
    text = await store.export(session_id, format="jsonl")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    records = [json.loads(ln) for ln in lines]
    # Every record carries the stable analysis fields.
    assert all(r["session_id"] == str(session_id) for r in records)
    assert [r["seq"] for r in records] == sorted(r["seq"] for r in records)

    by_type = {r["type"]: r for r in records}
    assert by_type["SessionStarted"]["type"] == "SessionStarted"

    # Message payloads are inlined: the transcript is directly readable.
    posted = [r for r in records if r["type"] == "MessagePosted"]
    assert len(posted) >= 2
    assert all(isinstance(r["content"], str) and r["content"] for r in posted)
    assert all(r["speaker"] in {"a", "b"} for r in posted)


@pytest.mark.asyncio
async def test_csv_export_is_a_transcript(harness: tuple[SessionManager, EventStore, uuid.UUID]) -> None:
    _manager, store, session_id = harness
    text = await store.export(session_id, format="csv")

    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["seq", "created_at", "type", "speaker", "content"]
    spoken = [r for r in rows[1:] if r[2] == "MessagePosted"]
    assert len(spoken) >= 2
    assert all(r[4].strip() for r in spoken)  # every spoken turn has content


@pytest.mark.asyncio
async def test_export_rejects_unknown_format(harness: tuple[SessionManager, EventStore, uuid.UUID]) -> None:
    _manager, store, session_id = harness
    with pytest.raises(ValueError, match="unsupported export format"):
        await store.export(session_id, format="xml")
