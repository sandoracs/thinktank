"""Participant interface (DESIGN.md §3 principle 2, §8.1).

One interface for every seat at the table — AI, human, or remote. The engine
does not know *who* is speaking (DESIGN.md §9); it only calls :meth:`speak`,
:meth:`observe`, and :meth:`on_session_end`. ``speak`` returning ``None`` is
the universal "no turn" signal (a human timeout, a pass, a disabled agent),
which the engine records as ``TurnSkipped``.
"""

from __future__ import annotations

import uuid
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from roundtable.core.state import SessionState
from roundtable.domain.models import Message
from roundtable.persona.reflection import ReflectionResult


class TurnContext(BaseModel):
    """Context handed to a participant for one turn (DESIGN.md §8.1)."""

    session_id: uuid.UUID
    round: int
    turn_instruction: str | None = None  # e.g. a moderator's specific request


@runtime_checkable
class Participant(Protocol):
    id: str
    kind: Literal["ai", "human", "remote"]
    display_name: str

    async def speak(self, ctx: TurnContext) -> Message | None:
        """Produce the participant's message, or ``None`` to skip/pass/timeout."""
        ...

    async def observe(self, msg: Message) -> None:
        """Every posted message is delivered here (working-memory updates)."""
        ...

    async def on_session_end(self) -> None:
        """Session teardown hook (e.g. distil long-term lessons)."""
        ...

    async def reflect(self, state: SessionState) -> ReflectionResult | None:
        """Round-end self-reflection: propose persona-state changes, or ``None``."""
        ...
