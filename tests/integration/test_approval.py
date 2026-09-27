"""Approval workflow integration tests (M6 acceptance).

APPROVED drift mode: a reflection proposal becomes a pending approval instead of
being applied. Until a human approves or rejects it the old persona state holds;
the decision applies/rejects the change and records ``ApprovalDecided``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from thinktank.core.manager import SessionManager
from thinktank.core.state import SessionState
from thinktank.domain.events import Event, EventType
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
            "new_position": "disclose-always",
            "new_confidence": 0.9,
            "influenced_by": ["optimist"],
            "reason": "the replication evidence was strong",
        }
    ],
    "attitude_updates": [],
    "mood": "persuaded",
}


def _responder(model: str, messages: list[ChatMessage], purpose: str, response_model: object) -> str:
    if purpose == "speech":
        return "a considered position"
    if purpose == "reflection":
        return json.dumps(PROPOSAL)
    return "ok"


def _agent() -> dict[str, AgentConfig]:
    return {
        AGENT_ID: AgentConfig(
            id=AGENT_ID,
            model="fake-model",
            persona=PersonaCore(name="Skeptic", role="critical analyst"),
            drift=DriftConfig(mode=DriftMode.APPROVED),
        )
    }


def _session() -> SessionConfig:
    return SessionConfig(
        title="approved",
        topic="Should AI co-author papers?",
        questions=[DebateQuestion(id=Q1, text="Should AI co-author papers?")],
        participants=[ParticipantRef(agent=AGENT_ID)],
        stop=StopConditions(max_rounds=1, max_cost_usd=10.0),
    )


Harness = tuple[SessionManager, EventStore, uuid.UUID]


@pytest.fixture
async def approved(tmp_path: Path) -> AsyncIterator[Harness]:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'approved.db'}")
    await init_db(engine)
    store = EventStore(make_session_factory(engine))
    manager = SessionManager(store=store, llm=FakeLLM(responder=_responder))
    session = await manager.create_session(_session(), _agent())
    await manager.start(session)
    try:
        sid = session.state.session_id
        assert sid is not None
        yield manager, store, sid
    finally:
        await engine.dispose()


def _by_type(events: list[Event], etype: EventType) -> list[Event]:
    return [e for e in events if e.type is etype]


@pytest.mark.asyncio
async def test_approved_mode_requests_and_applies_on_approve(approved: Harness) -> None:
    manager, store, sid = approved
    # A pending approval was created and NO persona change was applied yet.
    approvals = await manager.list_approvals(sid)
    assert len(approvals) == 1
    assert approvals[0]["status"] == "pending"

    events = await store.get_events(sid)
    assert _by_type(events, EventType.APPROVAL_REQUESTED)
    assert not _by_type(events, EventType.PERSONA_UPDATED)
    state: SessionState = await manager.get_state(sid)
    assert AGENT_ID not in state.persona_states

    # Approve: the change is applied and the decision is recorded.
    assert await manager.decide_approval(sid, AGENT_ID, "approve") is True
    events = await store.get_events(sid)
    assert _by_type(events, EventType.PERSONA_UPDATED)
    assert _by_type(events, EventType.APPROVAL_DECIDED)[0].payload.get("decision") == "approve"
    state = await manager.get_state(sid)
    assert state.persona_states[AGENT_ID].stances[Q1].position == "disclose-always"
    assert state.persona_states[AGENT_ID].mood == "persuaded"

    # The approval row is closed; a second decision is a no-op (maps to 409 upstream).
    approvals = await manager.list_approvals(sid)
    assert approvals[0]["status"] == "approved"
    assert await manager.decide_approval(sid, AGENT_ID, "approve") is False


@pytest.mark.asyncio
async def test_approved_mode_reject_keeps_old_state(approved: Harness) -> None:
    manager, store, sid = approved
    assert await manager.decide_approval(sid, AGENT_ID, "reject") is True
    events = await store.get_events(sid)
    assert _by_type(events, EventType.PERSONA_UPDATE_REJECTED)
    assert _by_type(events, EventType.APPROVAL_DECIDED)[0].payload.get("decision") == "reject"
    # The old state holds: no persona version was written.
    state = await manager.get_state(sid)
    assert AGENT_ID not in state.persona_states
    assert await store.persona_history(AGENT_ID, sid) == []
    approvals = await manager.list_approvals(sid)
    assert approvals[0]["status"] == "rejected"
