"""End-to-end engine run with FakeLLM (M1 core deliverable)."""

from __future__ import annotations

from pathlib import Path

import pytest

from thinktank.core.context import ContextBuilder
from thinktank.core.manager import SessionManager
from thinktank.domain.events import EventType
from thinktank.domain.models import (
    AgentConfig,
    DebateQuestion,
    ParticipantRef,
    PersonaCore,
    SessionConfig,
    StopConditions,
    StrategyRef,
)
from thinktank.llm.fake import FakeLLM
from thinktank.storage.db import init_db, make_engine, make_session_factory
from thinktank.storage.repositories import EventStore


def _build(tmp_path: Path):
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
async def test_full_run_produces_messages_and_events(tmp_path: Path) -> None:
    engine, store, mgr, config, agents = _build(tmp_path)
    await init_db(engine)
    sess = await mgr.create_session(config, agents)
    await mgr.start(sess)

    events = await store.get_events(sess.session_id)
    assert events[0].type is EventType.SESSION_CREATED
    assert events[-1].type is EventType.SESSION_ENDED
    types = {e.type for e in events}
    assert EventType.ROUND_STARTED in types
    # Regression: the run loop must honor max_rounds (it used to break
    # after round 1 because a 2-tuple stop check is always truthy).
    assert sum(1 for e in events if e.type is EventType.ROUND_STARTED) == 2
    assert EventType.TURN_ASSIGNED in types
    assert EventType.TURN_SKIPPED in types  # the human seat times out
    assert sum(1 for e in events if e.type is EventType.MESSAGE_POSTED) >= 3  # open + agent + close
    await engine.dispose()
