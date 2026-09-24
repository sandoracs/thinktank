"""End-to-end engine run with FakeLLM (M1 core deliverable)."""

from __future__ import annotations

import pytest

from roundtable.core.context import ContextBuilder
from roundtable.core.manager import SessionManager
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
from roundtable.llm.fake import FakeLLM
from roundtable.storage.db import init_db, make_engine, make_session_factory
from roundtable.storage.repositories import EventStore


def _build(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    store = EventStore(make_session_factory(engine))
    config = SessionConfig(
        title="AI co-authorship",
        topic="Should AI co-author papers?",
        questions=[DebateQuestion(id="q1", text="Should AI co-author papers?")],
        participants=[
            ParticipantRef(agent="moderator"),
            ParticipantRef(agent="agent"),
            ParticipantRef(human="human"),
        ],
        moderator="moderator",
        strategy=StrategyRef(name="round_robin"),
        stop=StopConditions(max_rounds=2, max_cost_usd=1.0),
    )
    agents = {
        "moderator": AgentConfig(
            id="moderator", model="", persona=PersonaCore(name="Moderator", role="moderator")
        ),
        "agent": AgentConfig(id="agent", model="", persona=PersonaCore(name="Agent", role="skeptic")),
    }
    mgr = SessionManager(
        store=store,
        llm=FakeLLM(responses=[f"turn {i}" for i in range(30)]),
        context_builder=ContextBuilder(),
        default_model="fake-model",
        human_timeout_s=0.02,
    )
    return engine, store, mgr, config, agents


@pytest.mark.asyncio
async def test_full_run_produces_messages_and_events(tmp_path) -> None:
    engine, store, mgr, config, agents = _build(tmp_path)
    await init_db(engine)
    sess = await mgr.create_session(config, agents)
    await mgr.start(sess)

    events = await store.get_events(sess.session_id)
    assert events[0].type is EventType.SESSION_CREATED
    assert events[-1].type is EventType.SESSION_ENDED
    types = {e.type for e in events}
    assert EventType.ROUND_STARTED in types
    assert EventType.TURN_ASSIGNED in types
    assert EventType.TURN_SKIPPED in types  # the human seat times out
    assert sum(1 for e in events if e.type is EventType.MESSAGE_POSTED) >= 3  # open + agent + close
    await engine.dispose()
