"""State-projection regression tests (DESIGN.md §6 event sourcing).

The dashboard/inspector re-derive ``SessionState`` by folding the event
stream; a payload schema mismatch here 500s every view that touches the
session (caught live: a real session that disabled a flaky participant
broke the dashboard).
"""

from __future__ import annotations

import uuid

from thinktank.core.state import project
from thinktank.domain.events import Event, EventType
from thinktank.domain.models import ParticipantRef, SessionConfig


def _base_session() -> SessionConfig:
    return SessionConfig(
        title="t",
        topic="T",
        participants=[ParticipantRef(agent="a1"), ParticipantRef(agent="a2")],
        moderator="a1",
    )


def test_disabled_participant_with_reason_projects() -> None:
    """The engine emits PARTICIPANT_DISABLED with ``reason``; projection must accept it."""
    sid = uuid.uuid4()
    events = [
        Event(
            seq=0,
            type=EventType.SESSION_CREATED,
            payload={"config": _base_session().model_dump(mode="json")},
        ),
        Event(seq=1, type=EventType.SESSION_STARTED),
        Event(
            seq=2,
            type=EventType.PARTICIPANT_DISABLED,
            payload={"participant_id": "a2", "reason": "repeated_errors"},
        ),
    ]
    state = project(sid, events)
    assert state.session_id == sid
    assert state.disabled == ["a2"]


def test_hand_events_still_project() -> None:
    """Guard the sibling hand-raise/lower path that shares ParticipantRefPayload."""
    sid = uuid.uuid4()
    events = [
        Event(seq=0, type=EventType.SESSION_CREATED, payload={"config": _base_session().model_dump(mode="json")}),
        Event(seq=1, type=EventType.SESSION_STARTED),
        Event(seq=2, type=EventType.HAND_RAISED, payload={"participant_id": "a2"}),
        Event(seq=3, type=EventType.HAND_LOWERED, payload={"participant_id": "a2"}),
    ]
    state = project(sid, events)
    assert state.hands_raised == []
