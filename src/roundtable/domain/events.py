"""Event types and payload schemas (event sourcing, DESIGN.md §6).

The ``events`` table is the single source of truth. Every domain fact is
expressed as an event; ``messages``, ``persona_versions`` etc. are projections
rebuilt by :func:`roundtable.core.state.project`. The event vocabulary here is a
closed set so the projection, the UI and the replay tool can all switch on
:class:`EventType` with confidence.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from roundtable.domain.models import Message, SessionConfig

T = TypeVar("T", bound=BaseModel)


class EventType(StrEnum):
    SESSION_CREATED = "SessionCreated"
    SESSION_STARTED = "SessionStarted"
    SESSION_PAUSED = "SessionPaused"
    SESSION_RESUMED = "SessionResumed"
    SESSION_ENDED = "SessionEnded"
    ROUND_STARTED = "RoundStarted"
    ROUND_ENDED = "RoundEnded"
    TURN_ASSIGNED = "TurnAssigned"
    MESSAGE_POSTED = "MessagePosted"
    TURN_SKIPPED = "TurnSkipped"
    HAND_RAISED = "HandRaised"
    HAND_LOWERED = "HandLowered"
    MODERATOR_INTERVENED = "ModeratorIntervened"
    LLM_CALL_COMPLETED = "LLMCallCompleted"
    MEMORY_WRITTEN = "MemoryWritten"
    REFLECTION_PROPOSED = "ReflectionProposed"
    PERSONA_UPDATED = "PersonaUpdated"
    PERSONA_UPDATE_REJECTED = "PersonaUpdateRejected"
    PERSONA_UPDATE_CLAMPED = "PersonaUpdateClamped"
    APPROVAL_REQUESTED = "ApprovalRequested"
    APPROVAL_DECIDED = "ApprovalDecided"
    CONSISTENCY_VIOLATION = "ConsistencyViolation"
    PARTICIPANT_DISABLED = "ParticipantDisabled"
    ERROR = "Error"


# --- payload models ---------------------------------------------------------
class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionCreatedPayload(_Payload):
    config: SessionConfig


class _NoPayload(_Payload):
    pass


class SessionEndedPayload(_Payload):
    reason: str  # max_rounds | cost_limit | moderator_closed | manual | error


class RoundPayload(_Payload):
    round: int


class TurnAssignedPayload(_Payload):
    speaker_id: str
    strategy: str
    reason: str | None = None


class MessagePostedPayload(_Payload):
    message: Message


class TurnSkippedPayload(_Payload):
    speaker_id: str
    reason: str  # human_timeout | passed | error


class ParticipantRefPayload(_Payload):
    participant_id: str


class ModeratorIntervenedPayload(_Payload):
    content: str
    speaker_id: str | None = None


class LLMCallCompletedPayload(_Payload):
    model: str
    purpose: str  # speech | summary | reflection | judge
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    agent_id: str | None = None


class ParticipantDisabledPayload(_Payload):
    participant_id: str
    reason: str


class ErrorPayload(_Payload):
    component: str
    message: str


PAYLOAD_MODELS: dict[EventType, type[_Payload]] = {
    EventType.SESSION_CREATED: SessionCreatedPayload,
    EventType.SESSION_STARTED: _NoPayload,
    EventType.SESSION_PAUSED: _NoPayload,
    EventType.SESSION_RESUMED: _NoPayload,
    EventType.SESSION_ENDED: SessionEndedPayload,
    EventType.ROUND_STARTED: RoundPayload,
    EventType.ROUND_ENDED: RoundPayload,
    EventType.TURN_ASSIGNED: TurnAssignedPayload,
    EventType.MESSAGE_POSTED: MessagePostedPayload,
    EventType.TURN_SKIPPED: TurnSkippedPayload,
    EventType.HAND_RAISED: ParticipantRefPayload,
    EventType.HAND_LOWERED: ParticipantRefPayload,
    EventType.MODERATOR_INTERVENED: ModeratorIntervenedPayload,
    EventType.LLM_CALL_COMPLETED: LLMCallCompletedPayload,
    EventType.PARTICIPANT_DISABLED: ParticipantDisabledPayload,
    EventType.ERROR: ErrorPayload,
}

# Events with a payload not yet given a dedicated schema in M0/M1 (M3-M6); they
# carry a free-form JSON payload until their milestone lands.
OPEN_PAYLOAD: tuple[EventType, ...] = (
    EventType.MEMORY_WRITTEN,
    EventType.REFLECTION_PROPOSED,
    EventType.PERSONA_UPDATED,
    EventType.PERSONA_UPDATE_REJECTED,
    EventType.PERSONA_UPDATE_CLAMPED,
    EventType.APPROVAL_REQUESTED,
    EventType.APPROVAL_DECIDED,
    EventType.CONSISTENCY_VIOLATION,
)


class Event(BaseModel):
    """A single immutable fact. ``seq`` is unique and increasing per session."""

    model_config = ConfigDict(frozen=True)

    seq: int
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def payload_model(self) -> _Payload:
        """Validate the raw payload against its schema and return it."""
        if self.type in PAYLOAD_MODELS:
            return PAYLOAD_MODELS[self.type](**self.payload)
        return _Payload(**self.payload)

    def payload_as(self, cls: type[T]) -> T:
        """Validate the payload against a specific schema and return it typed."""
        return cls(**self.payload)

    def get(self, field: str) -> Any:
        """Convenience accessor into the payload."""
        return self.payload.get(field)


def _jsonable(value: Any) -> Any:
    """Recursively convert Pydantic models to JSON-safe dicts for the payload."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def make_event(type: EventType, **payload: Any) -> Event:
    """Build an event with a ``seq`` of 0 (the store assigns the real seq).

    Pydantic models in ``payload`` are validated against the event's schema and
    then JSON-serialised so the stored payload is always plain data.
    """
    if type in PAYLOAD_MODELS:
        PAYLOAD_MODELS[type](**payload)  # fail fast on a bad payload
    return Event(seq=0, type=type, payload={k: _jsonable(v) for k, v in payload.items()})


def new_message_id() -> uuid.UUID:
    return uuid.uuid4()
