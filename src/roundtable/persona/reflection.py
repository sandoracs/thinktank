"""Reflection output schema (DESIGN.md §12.2).

The reflection prompt may only ever propose changes to :class:`PersonaState`
fields — the frozen :class:`PersonaCore` is not part of this schema, so a
drift policy can, by type, never touch the core (DESIGN.md §12.1).

The prompt explicitly permits a "no change" answer (empty ``stance_updates`` /
``attitude_updates``) to suppress artificial convergence (DESIGN.md §12.2).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from roundtable.domain.models import PersonaState, Stance


class StanceUpdate(BaseModel):
    question_id: str
    new_position: str
    new_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    influenced_by: list[str] = Field(default_factory=list)
    reason: str = ""


class AttitudeUpdate(BaseModel):
    participant_id: str
    new_value: float = Field(default=0.0, ge=-1.0, le=1.0)
    reason: str = ""


class ReflectionResult(BaseModel):
    stance_updates: list[StanceUpdate] = Field(default_factory=list)
    attitude_updates: list[AttitudeUpdate] = Field(default_factory=list)
    mood: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.stance_updates and not self.attitude_updates and self.mood is None


def apply_updates(current: PersonaState, proposal: ReflectionResult) -> PersonaState:
    """Apply a reflection proposal to a copy of ``current`` (pure)."""
    new = current.clone()
    for update in proposal.stance_updates:
        new.stances[update.question_id] = Stance(
            position=update.new_position,
            confidence=update.new_confidence,
        )
    for update in proposal.attitude_updates:
        new.attitudes[update.participant_id] = update.new_value
    if proposal.mood is not None:
        new.mood = proposal.mood
    return new
