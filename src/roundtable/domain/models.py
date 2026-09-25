"""Core domain models.

These are the Pydantic value objects shared across the engine, storage, API and
UI. They follow DESIGN.md §5. The split between *immutable core* (persona
``core``) and *mutable state* (persona ``state``) is load-bearing: the drift
policies may only ever touch ``PersonaState`` fields, never the frozen core
(DESIGN.md §12.1).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------
class PersonaCore(BaseModel):
    """Frozen identity: never changes during (or across) a session."""

    model_config = ConfigDict(frozen=True)

    name: str
    role: str
    expertise: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    communication_style: str = ""
    temperament: str = ""
    background: str = ""
    boundaries: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Human-readable rendering for the system prompt (DESIGN.md §10.1)."""
        lines = [
            f"{self.name} — {self.role}",
        ]
        if self.expertise:
            lines.append("Expertise: " + ", ".join(self.expertise))
        if self.values:
            lines.append("Values: " + ", ".join(self.values))
        if self.communication_style:
            lines.append("Communication style: " + self.communication_style)
        if self.temperament:
            lines.append("Temperament: " + self.temperament)
        if self.background:
            lines.append("Background: " + self.background)
        if self.boundaries:
            lines.append("Hard boundaries: " + "; ".join(self.boundaries))
        return "\n".join(lines)


class Stance(BaseModel):
    """A position on one debate question, with a confidence in [0, 1]."""

    position: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PersonaState(BaseModel):
    """Mutable persona state; changes are governed by the drift policy."""

    stances: dict[str, Stance] = Field(default_factory=dict)
    attitudes: dict[str, float] = Field(default_factory=dict)  # participant id -> [-1, 1]
    mood: str = "neutral"

    def clone(self) -> PersonaState:
        return self.model_copy(deep=True)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
class DriftMode(StrEnum):
    LOCKED = "locked"
    BOUNDED = "bounded"
    APPROVED = "approved"
    FREE = "free"


class DriftConfig(BaseModel):
    mode: DriftMode = DriftMode.LOCKED
    max_confidence_delta: float = Field(default=0.2, ge=0.0)
    max_attitude_delta: float = Field(default=0.3, ge=0.0)
    max_stance_changes_per_round: int = Field(default=1, ge=0)
    shadow_reflection: bool = False


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class Layer(StrEnum):
    """The three memory layers (DESIGN.md §11)."""

    WORKING = "working"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"


class MemoryConfig(BaseModel):
    working_window: int = Field(default=12, ge=1)
    summarize_every: int = Field(default=8, ge=1)
    retrieval_k: int = Field(default=5, ge=1)
    long_term: bool = True


# ---------------------------------------------------------------------------
# Session / agent configuration
# ---------------------------------------------------------------------------
class ParticipantRef(BaseModel):
    """A seat at the table: exactly one of an agent template, a human, or remote."""

    agent: str | None = None
    human: str | None = None
    remote: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ParticipantRef:
        set_fields = [f for f in (self.agent, self.human, self.remote) if f is not None]
        if len(set_fields) != 1:
            msg = "ParticipantRef must set exactly one of agent/human/remote"
            raise ValueError(msg)
        return self

    @property
    def participant_id(self) -> str:
        """Stable identifier used throughout the engine and UI."""
        for value in (self.agent, self.human, self.remote):
            if value is not None:
                return value
        return ""  # unreachable: _exactly_one guarantees one is set

    @property
    def participant_kind(self) -> Literal["ai", "human", "remote"]:
        if self.agent is not None:
            return "ai"
        if self.human is not None:
            return "human"
        return "remote"


class AgentConfig(BaseModel):
    """Configuration for one AI participant; ``id`` is its long-lived identity."""

    id: str
    type: str = "llm"
    model: str
    summary_model: str = Field(default="", description="Optional cheaper model for side calls (summaries).")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=600, ge=1)
    persona: PersonaCore
    initial_state: PersonaState = Field(default_factory=PersonaState)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    consistency_check: bool = False
    carry_over_state: bool = False

    def with_defaults(self, default_model: str) -> AgentConfig:
        if self.model:
            return self
        return self.model_copy(update={"model": default_model})


class DebateQuestion(BaseModel):
    """A debate question the discussion is measured against (DESIGN.md §5.3)."""

    id: str
    text: str


class StrategyRef(BaseModel):
    name: str = "round_robin"
    params: dict[str, Any] = Field(default_factory=dict)


class StopConditions(BaseModel):
    max_rounds: int | None = Field(default=6, ge=1)
    max_messages: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=2.0, ge=0.0)
    max_duration_s: int | None = Field(default=None, ge=1)


class SessionConfig(BaseModel):
    """Full configuration snapshot captured into the first event (DESIGN.md §6)."""

    title: str
    topic: str
    questions: list[DebateQuestion] = Field(default_factory=list)
    participants: list[ParticipantRef] = Field(default_factory=list)
    moderator: str | None = None
    strategy: StrategyRef = Field(default_factory=StrategyRef)
    stop: StopConditions = Field(default_factory=StopConditions)
    reflection_every_rounds: int = Field(default=1, ge=1)
    language: str = "hu"

    @model_validator(mode="after")
    def _unique_participants(self) -> SessionConfig:
        ids = [p.participant_id for p in self.participants]
        if len(ids) != len(set(ids)):
            msg = f"Duplicate participants in session config: {ids}"
            raise ValueError(msg)
        if self.moderator is not None and self.moderator not in ids:
            msg = f"Moderator {self.moderator!r} is not among the participants"
            raise ValueError(msg)
        return self

    def agent_ids(self) -> list[str]:
        return [p.participant_id for p in self.participants if p.participant_kind == "ai"]

    def participant_ids(self) -> list[str]:
        return [p.participant_id for p in self.participants]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------
class Message(BaseModel):
    id: UUID
    session_id: UUID
    seq: int
    speaker_id: str
    kind: Literal["speech", "moderator", "system"]
    content: str
    reply_to: UUID | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Memory search
# ---------------------------------------------------------------------------
class MemoryHit(BaseModel):
    """One retrieval result (vector or FTS), fused by the backend."""

    id: int
    layer: Layer
    content: str
    score: float
    session_id: UUID | None = None
    source_seq: int | None = None
    created_at: datetime | None = None
