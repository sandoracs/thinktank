"""End-to-end session with the bidding strategy (M5).

Confirms a strategy registered purely through the plugin entry point drives a
real session: the engine asks the strategy for the next speaker, the bidding
calls go through the LLM gateway, and messages are posted and the round ends.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from roundtable.core.manager import SessionManager
from roundtable.core.state import project
from roundtable.domain.events import EventType
from roundtable.domain.models import (
    AgentConfig,
    DebateQuestion,
    ParticipantRef,
    PersonaCore,
    SessionConfig,
    StopConditions,
    StrategyRef,
)
from roundtable.llm.client import ChatMessage, Purpose
from roundtable.llm.fake import FakeLLM
from roundtable.storage.db import init_db, make_engine, make_session_factory
from roundtable.storage.repositories import EventStore

BIDS = {"a": 0.4, "b": 0.8, "c": 0.6}


def _responder(model: str, messages: list[ChatMessage], purpose: Purpose, response_model: object) -> str:
    if purpose == "bid":
        user = messages[-1].content if messages else ""
        for pid, value in BIDS.items():
            if f"Your participant id: {pid}" in user:
                return json.dumps({"bid": value})
        return json.dumps({"bid": 0.0})
    if purpose == "speech":
        return "a considered position"
    return "ok"


def _agents() -> dict[str, AgentConfig]:
    return {
        pid: AgentConfig(id=pid, model="fake", persona=PersonaCore(name=pid, role="r"))
        for pid in ("a", "b", "c")
    }


def _session() -> SessionConfig:
    return SessionConfig(
        title="bidding",
        topic="Should AI co-author papers?",
        questions=[DebateQuestion(id="q1", text="Should AI co-author papers?")],
        participants=[ParticipantRef(agent=pid) for pid in ("a", "b", "c")],
        strategy=StrategyRef(name="bidding", params={"bid_model": "fake"}),
        stop=StopConditions(max_rounds=1, max_cost_usd=5.0),
    )


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[tuple[SessionManager, EventStore]]:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'bidding.db'}")
    await init_db(engine)
    store = EventStore(make_session_factory(engine))
    manager = SessionManager(store=store, llm=FakeLLM(responder=_responder))
    try:
        yield manager, store
    finally:
        await engine.dispose()


async def test_bidding_session_runs_and_orders_by_bid(harness: tuple[SessionManager, EventStore]) -> None:
    manager, store = harness
    session = await manager.create_session(_session(), _agents())
    await manager.start(session)
    sid = session.session_id

    events = await store.get_events(sid)
    speakers = [
        e.payload.get("speaker_id")
        for e in events
        if e.type is EventType.TURN_ASSIGNED
    ]
    # b (0.8) > c (0.6) > a (0.4); one round, every active speaks once.
    assert speakers == ["b", "c", "a"]

    posted = [e for e in events if e.type is EventType.MESSAGE_POSTED]
    assert len(posted) == 3

    final = project(sid, events)
    assert final.status == "ended"
    assert final.message_count == 3
