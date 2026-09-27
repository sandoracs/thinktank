"""Persona reflection / drift-policy integration tests (DESIGN.md §12, M4 acceptance).

M4 acceptance:
- FREE mode: a stance change is applied, recorded in ``persona_versions``, visible
  in the replayed ``SessionState``, and justified (the proposal is logged).
- LOCKED mode: the state does not change; with ``shadow_reflection`` the proposal
  is still logged (``ReflectionProposed(shadow=True)``); without it, no reflection
  runs at all.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from thinktank.core.manager import SessionManager
from thinktank.core.state import project
from thinktank.domain.events import (
    Event,
    EventType,
    PersonaUpdatedPayload,
    ReflectionProposedPayload,
)
from thinktank.domain.models import (
    AgentConfig,
    DebateQuestion,
    DriftConfig,
    DriftMode,
    ParticipantRef,
    PersonaCore,
    SessionConfig,
    StopConditions,
)
from thinktank.llm.client import ChatMessage
from thinktank.llm.fake import FakeLLM
from thinktank.storage.db import init_db, make_engine, make_session_factory
from thinktank.storage.repositories import EventStore

AGENT_ID = "skeptic"
Q1 = "q1"

PROPOSAL = {
    "stance_updates": [
        {
            "question_id": Q1,
            "new_position": "qualified-yes",
            "new_confidence": 0.8,
            "influenced_by": ["opponent"],
            "reason": "the evidence on disclosure persuaded me partially",
        }
    ],
    "attitude_updates": [],
    "mood": "open",
}

Harness = tuple[SessionManager, EventStore, dict[str, DriftConfig]]


def _reflection_responder(
    model: str, messages: list[ChatMessage], purpose: str, response_model: object
) -> str:
    if purpose == "speech":
        return "a considered position on the question"
    if purpose == "reflection":
        return json.dumps(PROPOSAL)
    return "ok"


def _agents(drift: DriftConfig) -> dict[str, AgentConfig]:
    return {
        AGENT_ID: AgentConfig(
            id=AGENT_ID,
            model="fake-model",
            persona=PersonaCore(name="Skeptic", role="critical analyst"),
            drift=drift,
        )
    }


def _session() -> SessionConfig:
    return SessionConfig(
        title="drift",
        topic="Should AI co-author papers?",
        questions=[DebateQuestion(id=Q1, text="Should AI co-author papers?")],
        participants=[ParticipantRef(agent=AGENT_ID)],
        stop=StopConditions(max_rounds=1, max_cost_usd=10.0),
    )


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    """Build the DB, yield the store + drift presets, dispose the engine on teardown."""
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'drift.db'}")
    await init_db(engine)
    store = EventStore(make_session_factory(engine))
    manager = SessionManager(store=store, llm=FakeLLM(responder=_reflection_responder))
    drifts: dict[str, DriftConfig] = {
        "free": DriftConfig(mode=DriftMode.FREE),
        "locked_shadow": DriftConfig(mode=DriftMode.LOCKED, shadow_reflection=True),
        "locked": DriftConfig(mode=DriftMode.LOCKED),
    }
    try:
        yield manager, store, drifts
    finally:
        await engine.dispose()


def _by_type(events: list[Event], etype: EventType) -> list[Event]:
    return [e for e in events if e.type is etype]


async def _start(harness: Harness, mode: str) -> tuple[EventStore, uuid.UUID]:
    manager, store, drifts = harness
    session = await manager.create_session(_session(), _agents(drifts[mode]))
    await manager.start(session)
    return store, session.session_id


@pytest.mark.asyncio
async def test_free_mode_applies_and_justifies(harness: Harness) -> None:
    store, sid = await _start(harness, "free")
    events = await store.get_events(sid)

    # Applied: a PersonaUpdated event carrying the new stance.
    updated = _by_type(events, EventType.PERSONA_UPDATED)
    assert len(updated) == 1
    payload = updated[0].payload_as(PersonaUpdatedPayload)
    assert payload.state.stances[Q1].position == "qualified-yes"
    assert payload.state.stances[Q1].confidence == 0.8

    # Justified: the proposal (influenced_by + reason) is logged, not shadow.
    proposed = _by_type(events, EventType.REFLECTION_PROPOSED)
    assert len(proposed) == 1
    prop = proposed[0].payload_as(ReflectionProposedPayload)
    assert prop.shadow is False
    stance_update = prop.proposal["stance_updates"][0]
    assert stance_update["influenced_by"] == ["opponent"]
    assert stance_update["reason"]

    # Recorded in the persona_versions projection (version 1).
    rows = await store.persona_history(AGENT_ID, sid)
    assert [row.version for row in rows] == [1]
    assert rows[0].state["stances"][Q1]["position"] == "qualified-yes"

    # Visible in the replayed SessionState.
    final = project(sid, events)
    assert final.persona_states[AGENT_ID].stances[Q1].position == "qualified-yes"
    assert final.persona_states[AGENT_ID].mood == "open"


@pytest.mark.asyncio
async def test_locked_shadow_logs_but_does_not_change(harness: Harness) -> None:
    store, sid = await _start(harness, "locked_shadow")
    events = await store.get_events(sid)

    # State does not change: no PersonaUpdated, one rejection.
    assert _by_type(events, EventType.PERSONA_UPDATED) == []
    assert len(_by_type(events, EventType.PERSONA_UPDATE_REJECTED)) == 1

    # Shadow proposal is logged.
    proposed = _by_type(events, EventType.REFLECTION_PROPOSED)
    assert len(proposed) == 1
    assert proposed[0].payload_as(ReflectionProposedPayload).shadow is True

    # No persona_versions row; the replayed state has no persona override.
    assert await store.persona_history(AGENT_ID, sid) == []
    assert AGENT_ID not in project(sid, events).persona_states


@pytest.mark.asyncio
async def test_locked_without_shadow_never_reflects(harness: Harness) -> None:
    store, sid = await _start(harness, "locked")
    events = await store.get_events(sid)

    # The design does not even run reflection for plain LOCKED (DESIGN.md §12.2).
    assert _by_type(events, EventType.REFLECTION_PROPOSED) == []
    assert _by_type(events, EventType.PERSONA_UPDATE_REJECTED) == []
    assert _by_type(events, EventType.PERSONA_UPDATED) == []
    assert await store.persona_history(AGENT_ID, sid) == []


BOUNDED_PROPOSAL = {
    "stance_updates": [
        {
            "question_id": "q1",
            "new_position": "disclose-always",
            "new_confidence": 0.9,
            "influenced_by": ["optimist"],
            "reason": "first",
        },
        {
            "question_id": "q2",
            "new_position": "yes-with-care",
            "new_confidence": 0.8,
            "influenced_by": ["optimist"],
            "reason": "second — should be skipped by the per-round limit",
        },
    ],
    "attitude_updates": [],
    "mood": "persuaded",
}


@pytest.mark.asyncio
async def test_bounded_mode_clamps_and_limits(tmp_path: Path) -> None:
    """M6: BOUNDED mode applies within the per-round limit and records a Clamped event."""

    def responder(model: str, messages: list[ChatMessage], purpose: str, response_model: object) -> str:
        if purpose == "speech":
            return "a considered position"
        if purpose == "reflection":
            return json.dumps(BOUNDED_PROPOSAL)
        return "ok"

    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'bounded.db'}")
    await init_db(engine)
    store = EventStore(make_session_factory(engine))
    manager = SessionManager(store=store, llm=FakeLLM(responder=responder))
    agent = {
        AGENT_ID: AgentConfig(
            id=AGENT_ID,
            model="fake-model",
            persona=PersonaCore(name="Skeptic", role="critical analyst"),
            # Only one stance change per round; the second proposed change is skipped.
            drift=DriftConfig(mode=DriftMode.BOUNDED, max_stance_changes_per_round=1),
        )
    }
    config = SessionConfig(
        title="bounded",
        topic="Should AI co-author papers?",
        questions=[
            DebateQuestion(id="q1", text="Should AI co-author papers?"),
            DebateQuestion(id="q2", text="Must AI usage be disclosed?"),
        ],
        participants=[ParticipantRef(agent=AGENT_ID)],
        stop=StopConditions(max_rounds=1, max_cost_usd=10.0),
    )
    try:
        session = await manager.create_session(config, agent)
        await manager.start(session)
        sid = session.state.session_id
        assert sid is not None

        events = await store.get_events(sid)
        assert _by_type(events, EventType.REFLECTION_PROPOSED)
        # A clamp (limit applied) is recorded, not a plain update.
        clamped = _by_type(events, EventType.PERSONA_UPDATE_CLAMPED)
        assert clamped, "expected a PersonaUpdateClamped event"
        state = project(sid, events)
        # Only the first stance change (within the limit) landed.
        assert AGENT_ID in state.persona_states
        assert Q1 in state.persona_states[AGENT_ID].stances
        assert "q2" not in state.persona_states[AGENT_ID].stances
        # The clamp note explains what was skipped.
        notes = clamped[0].payload.get("notes") or []
        assert any("q2" in n for n in notes)
    finally:
        await engine.dispose()
