"""Replay-from-seq test (event sourcing, lossless replay)."""

from __future__ import annotations

from pathlib import Path

import pytest

from thinktank.core.manager import SessionManager
from thinktank.domain.events import EventType
from thinktank.domain.models import (
    AgentConfig,
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
        title="T",
        topic="Q",
        participants=[ParticipantRef(agent="mod"), ParticipantRef(agent="ag")],
        moderator="mod",
        strategy=StrategyRef(name="round_robin"),
        stop=StopConditions(max_rounds=1, max_cost_usd=1.0),
    )
    agents = {
        "mod": AgentConfig(id="mod", model="", persona=PersonaCore(name="M", role="moderator")),
        "ag": AgentConfig(id="ag", model="", persona=PersonaCore(name="A", role="skeptic")),
    }
    mgr = SessionManager(
        store=store,
        llm=FakeLLM(responses=[f"r{i}" for i in range(30)]),
        default_model="fake-model",
    )
    return engine, store, mgr, config, agents


@pytest.mark.asyncio
async def test_replay_from_seq(tmp_path: Path) -> None:
    engine, store, mgr, config, agents = _build(tmp_path)
    await init_db(engine)
    sess = await mgr.create_session(config, agents)
    await mgr.start(sess)

    all_events = await store.get_events(sess.session_id)
    assert len(all_events) >= 5
    assert all_events[0].type is EventType.SESSION_CREATED
    assert all_events[-1].type is EventType.SESSION_ENDED

    # after_seq is exclusive: strictly the events after seq 3.
    replay = await store.get_events(sess.session_id, after_seq=3)
    assert replay[0].seq == 4
    assert len(replay) == len(all_events) - 3
    await engine.dispose()
