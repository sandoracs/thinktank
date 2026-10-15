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


@pytest.mark.asyncio
async def test_stop_mid_round_still_closes_the_round(tmp_path: Path) -> None:
    from thinktank.domain.models import StopConditions

    engine, store, mgr, config, agents = _build(tmp_path)
    # Stop after the moderator's opening + one speech: the cap lands mid-round.
    config = config.model_copy(update={"stop": StopConditions(max_rounds=3, max_messages=2, max_cost_usd=None)})
    await init_db(engine)
    sess = await mgr.create_session(config, agents)
    await mgr.start(sess)

    events = await store.get_events(sess.session_id)
    started = [e for e in events if e.type is EventType.ROUND_STARTED]
    ended = [e for e in events if e.type is EventType.ROUND_ENDED]
    assert len(started) == len(ended)
    assert ended[-1].payload.get("partial") is True
    await engine.dispose()


@pytest.mark.asyncio
async def test_ai_moderator_only_opens_and_closes(tmp_path: Path) -> None:
    engine, store, mgr, config, agents = _build(tmp_path)
    config = config.model_copy(update={"moderator_round_summary": False})
    await init_db(engine)
    sess = await mgr.create_session(config, agents)
    await mgr.start(sess)

    events = await store.get_events(sess.session_id)
    assigned = [e.payload.get("speaker_id") for e in events if e.type is EventType.TURN_ASSIGNED]
    assert "moderator" not in assigned  # never scheduled in a round
    posted = [e.payload["message"] for e in events if e.type is EventType.MESSAGE_POSTED]
    mod_msgs = [m for m in posted if m["speaker_id"] == "moderator"]
    assert len(mod_msgs) == 2  # opening + closing
    assert all(m["kind"] == "moderator" for m in mod_msgs)
    assert posted[0]["speaker_id"] == "moderator"
    assert posted[-1]["speaker_id"] == "moderator"
    reflected = [e.payload.get("agent_id") for e in events if e.type is EventType.REFLECTION_PROPOSED]
    assert "moderator" not in reflected
    await engine.dispose()


@pytest.mark.asyncio
async def test_human_moderator_is_prompted_for_open_and_close(tmp_path: Path) -> None:
    engine, store, mgr, config, agents = _build(tmp_path)
    # The human seat moderates; the AI "moderator" template becomes a debater.
    config = config.model_copy(update={"moderator": "human", "moderator_round_summary": False})
    await init_db(engine)
    sess = await mgr.create_session(config, agents)
    await mgr.start(sess)

    events = await store.get_events(sess.session_id)
    human_turns = [
        e for e in events if e.type is EventType.TURN_ASSIGNED and e.payload.get("speaker_id") == "human"
    ]
    assert [e.payload.get("strategy") for e in human_turns] == ["moderator", "moderator"]
    skipped = [e for e in events if e.type is EventType.TURN_SKIPPED and e.payload.get("speaker_id") == "human"]
    assert len(skipped) == 2  # timed out in the test, both open and close
    await engine.dispose()


@pytest.mark.asyncio
async def test_moderator_sums_up_between_rounds_by_default(tmp_path: Path) -> None:
    engine, store, mgr, config, agents = _build(tmp_path)  # max_rounds=2
    assert config.moderator_round_summary is True
    await init_db(engine)
    sess = await mgr.create_session(config, agents)
    await mgr.start(sess)

    events = await store.get_events(sess.session_id)
    seq = [
        (e.type, e.payload["message"]["speaker_id"] if e.type is EventType.MESSAGE_POSTED else None)
        for e in events
        if e.type in (EventType.MESSAGE_POSTED, EventType.ROUND_ENDED, EventType.ROUND_STARTED)
    ]
    posted = [e.payload["message"] for e in events if e.type is EventType.MESSAGE_POSTED]
    # open + one between-round summary (after round 1, not after the last) + close
    assert sum(1 for m in posted if m["speaker_id"] == "moderator") == 3
    # The summary sits between RoundEnded(1) and RoundStarted(2).
    first_end = next(i for i, (t, _) in enumerate(seq) if t is EventType.ROUND_ENDED)
    assert seq[first_end + 1] == (EventType.MESSAGE_POSTED, "moderator")
    assert seq[first_end + 2][0] is EventType.ROUND_STARTED
    speech = [c for c in mgr._llm.calls if c["purpose"] == "speech"]  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    assert any("Round 1 is over" in c["messages"][-1].content for c in speech)
    await engine.dispose()
