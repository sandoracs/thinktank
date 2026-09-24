"""Session state and its projection from events (DESIGN.md §3, §7, §9).

``SessionState`` is a *derived, in-memory* view — never persisted as the source
of truth. It is rebuilt by :func:`project`, the exact same code path used for
live sessions, replays, and crash-resume (DESIGN.md §9.2). That identity is what
makes the replay test meaningful (DESIGN.md §17).

The strategy keeps its own transient state (who has spoken this round) via
``on_event``; this projection holds the durable facts the engine and UI need.
"""


import uuid
from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from roundtable.domain.events import (
    Event,
    EventType,
    MessagePostedPayload,
    ParticipantRefPayload,
    RoundPayload,
    SessionCreatedPayload,
    SessionEndedPayload,
    TurnAssignedPayload,
)
from roundtable.domain.models import Message, SessionConfig

# Durable session lifecycle values (DESIGN.md §7).
STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_ENDED = "ended"
STATUS_INTERRUPTED = "interrupted"


class SessionState(BaseModel):
    """Mutable derived view of a session, rebuilt from the event stream."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: uuid.UUID | None = None
    config: SessionConfig | None = None
    status: str = STATUS_CREATED

    current_round: int = 0
    round_complete: bool = False
    last_seq: int = 0

    message_count: int = 0
    messages: list[Message] = Field(default_factory=list)

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    hands_raised: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)
    current_speaker: str | None = None
    ended_reason: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def is_active(self, participant_id: str) -> bool:
        return participant_id not in set(self.disabled)


def apply_event(state: SessionState, event: Event) -> SessionState:
    """Apply a single event to the state (in place) and return it."""
    t = event.type
    state.last_seq = max(state.last_seq, event.seq)

    if t is EventType.SESSION_CREATED:
        payload = event.payload_as(SessionCreatedPayload)
        state.config = payload.config
        state.status = STATUS_CREATED
    elif t is EventType.SESSION_STARTED:
        state.status = STATUS_RUNNING
        state.started_at = event.created_at
    elif t is EventType.SESSION_PAUSED:
        state.status = STATUS_PAUSED
    elif t is EventType.SESSION_RESUMED:
        state.status = STATUS_RUNNING
    elif t is EventType.SESSION_ENDED:
        payload = event.payload_as(SessionEndedPayload)
        state.status = STATUS_ENDED
        state.ended_reason = payload.reason
        state.ended_at = event.created_at
    elif t is EventType.ROUND_STARTED:
        payload = event.payload_as(RoundPayload)
        state.current_round = payload.round
        state.round_complete = False
    elif t is EventType.ROUND_ENDED:
        state.round_complete = False
    elif t is EventType.TURN_ASSIGNED:
        payload = event.payload_as(TurnAssignedPayload)
        state.current_speaker = payload.speaker_id
    elif t is EventType.MESSAGE_POSTED:
        payload = event.payload_as(MessagePostedPayload)
        state.messages.append(payload.message)
        if payload.message.kind in ("speech", "moderator"):
            state.message_count += 1
    elif t is EventType.HAND_RAISED:
        payload = event.payload_as(ParticipantRefPayload)
        if payload.participant_id not in state.hands_raised:
            state.hands_raised.append(payload.participant_id)
    elif t is EventType.HAND_LOWERED:
        payload = event.payload_as(ParticipantRefPayload)
        if payload.participant_id in state.hands_raised:
            state.hands_raised.remove(payload.participant_id)
    elif t is EventType.LLM_CALL_COMPLETED:
        state.total_cost_usd += event.payload.get("cost_usd", 0.0)
        state.total_input_tokens += int(event.payload.get("input_tokens", 0))
        state.total_output_tokens += int(event.payload.get("output_tokens", 0))
    elif t is EventType.PARTICIPANT_DISABLED:
        payload = event.payload_as(ParticipantRefPayload)
        if payload.participant_id not in state.disabled:
            state.disabled.append(payload.participant_id)

    return state


def project(
    session_id: uuid.UUID | None,
    events: Iterable[Event],
    initial: SessionState | None = None,
) -> SessionState:
    """Fold events (in ``seq`` order) into a fresh or existing ``SessionState``."""
    state = initial if initial is not None else SessionState()
    if session_id is not None:
        state.session_id = session_id
    ordered = sorted(events, key=lambda e: e.seq)
    for event in ordered:
        apply_event(state, event)
    return state
