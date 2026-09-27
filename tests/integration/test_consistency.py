"""Persona-consistency check (M6).

The opt-in judge scores the candidate speech against the frozen persona core.
Below the threshold the speech is regenerated once with the judge's feedback;
either way a ``ConsistencyViolation`` event is recorded. Off by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thinktank.core.manager import SessionManager
from thinktank.domain.events import ConsistencyViolationPayload, EventType, MessagePostedPayload
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

PERSONA = PersonaCore(
    name="A",
    role="safety advocate",
    values=["safety first"],
    boundaries=["never dismiss safety concerns"],
)
CANDIDATE = "we can ignore the safety concerns to ship faster"
REWRITTEN = "rewritten to respect the persona core"


def _responder(judge_score: int):
    def responder(model: str, messages: list[ChatMessage], purpose: Purpose, response_model: object) -> str:
        if purpose == "judge":
            return json.dumps(
                {
                    "score": judge_score,
                    "justification": "stays faithful" if judge_score >= 4 else "dismisses safety concerns",
                }
            )
        if purpose == "speech":
            last_user = messages[-1].content if messages else ""
            if "inconsistent" in last_user:  # regeneration prompt carries the judge feedback
                return REWRITTEN
            return CANDIDATE
        return "ok"

    return responder


def _broken_responder(model: str, messages: list[ChatMessage], purpose: Purpose, response_model: object) -> str:
    if purpose == "judge":
        return "not-json"  # unparseable -> treated as judge failure
    return CANDIDATE


async def _build(tmp_path: Path, responder: object) -> tuple[SessionManager, EventStore, FakeLLM, object]:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'cons.db'}")
    await init_db(engine)
    store = EventStore(make_session_factory(engine))
    llm = FakeLLM(responder=responder)  # type: ignore[arg-type]
    manager = SessionManager(store=store, llm=llm)
    return manager, store, llm, engine


def _agent(**overrides: object) -> AgentConfig:
    base: dict[str, object] = {
        "id": "a",
        "model": "fake",
        "persona": PERSONA,
        "consistency_check": True,
        "consistency_threshold": 3,
    }
    base.update(overrides)
    return AgentConfig(**base)  # type: ignore[arg-type]


def _session() -> SessionConfig:
    return SessionConfig(
        title="consistency",
        topic="Should we ship without review?",
        questions=[DebateQuestion(id="q1", text="Should we ship without review?")],
        participants=[ParticipantRef(agent="a")],
        stop=StopConditions(max_rounds=1, max_cost_usd=5.0),
    )


@pytest.mark.asyncio
async def test_low_score_triggers_single_regeneration(tmp_path: Path) -> None:
    manager, store, llm, db = await _build(tmp_path, _responder(2))
    try:
        session = await manager.create_session(_session(), {"a": _agent()})
        await manager.start(session)
        events = await store.get_events(session.session_id)

        purposes = [c["purpose"] for c in llm.calls]
        # One candidate, one judge, one regeneration.
        assert purposes.count("speech") == 2
        assert purposes.count("judge") == 1

        violations = [e for e in events if e.type is EventType.CONSISTENCY_VIOLATION]
        assert len(violations) == 1
        payload = violations[0].payload_as(ConsistencyViolationPayload)
        assert payload.agent_id == "a"
        assert payload.score == 2
        assert payload.regenerated is True
        assert payload.justification

        message = next(e for e in events if e.type is EventType.MESSAGE_POSTED).payload_as(MessagePostedPayload).message
        assert message.content == REWRITTEN
    finally:
        await db.dispose()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_high_score_keeps_candidate(tmp_path: Path) -> None:
    manager, store, llm, db = await _build(tmp_path, _responder(5))
    try:
        session = await manager.create_session(_session(), {"a": _agent()})
        await manager.start(session)
        events = await store.get_events(session.session_id)

        purposes = [c["purpose"] for c in llm.calls]
        assert purposes.count("speech") == 1
        assert purposes.count("judge") == 1

        violations = [e for e in events if e.type is EventType.CONSISTENCY_VIOLATION]
        assert len(violations) == 1
        payload = violations[0].payload_as(ConsistencyViolationPayload)
        assert payload.regenerated is False
        assert payload.score == 5

        message = next(e for e in events if e.type is EventType.MESSAGE_POSTED).payload_as(MessagePostedPayload).message
        assert message.content == CANDIDATE
    finally:
        await db.dispose()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_disabled_by_default(tmp_path: Path) -> None:
    manager, store, llm, db = await _build(tmp_path, _responder(1))
    try:
        session = await manager.create_session(_session(), {"a": _agent(consistency_check=False)})
        await manager.start(session)
        events = await store.get_events(session.session_id)

        assert [c["purpose"] for c in llm.calls].count("judge") == 0
        assert [e for e in events if e.type is EventType.CONSISTENCY_VIOLATION] == []
        message = next(e for e in events if e.type is EventType.MESSAGE_POSTED).payload_as(MessagePostedPayload).message
        assert message.content == CANDIDATE
    finally:
        await db.dispose()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_judge_failure_keeps_candidate(tmp_path: Path) -> None:
    manager, store, llm, db = await _build(tmp_path, _broken_responder)
    try:
        session = await manager.create_session(_session(), {"a": _agent()})
        await manager.start(session)
        events = await store.get_events(session.session_id)

        assert [c["purpose"] for c in llm.calls].count("speech") == 1
        assert [e for e in events if e.type is EventType.CONSISTENCY_VIOLATION] == []
        message = next(e for e in events if e.type is EventType.MESSAGE_POSTED).payload_as(MessagePostedPayload).message
        assert message.content == CANDIDATE
    finally:
        await db.dispose()  # type: ignore[union-attr]
