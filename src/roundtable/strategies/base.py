"""Turn strategy interface (DESIGN.md §8.2).

A strategy decides *who speaks next* and defines what a "round" is. Strategies
are plugins: they are stateless across sessions but keep per-session state
reconstructed from the event stream via :meth:`on_event`, so replay is exact.

``Params`` exposes each strategy's configuration as a Pydantic model; the API
returns the ``model_json_schema()`` and the UI generates a form from it
(DESIGN.md §13) — a new strategy therefore shows up in the UI with no code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from roundtable.core.state import SessionState
from roundtable.domain.events import Event
from roundtable.llm.client import LLMClient


class EmptyParams(BaseModel):
    """Marker for strategies with no configuration."""


class TurnStrategy(ABC):
    name: ClassVar[str] = ""
    Params: ClassVar[type[BaseModel]] = EmptyParams

    def __init__(
        self,
        params: BaseModel | dict[str, Any] | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        if isinstance(params, dict):
            data = params
        elif params is not None:
            data = params.model_dump()
        else:
            data = {}
        self.params = self.Params(**data)
        # Optional LLM gateway for strategies that need one (e.g. bidding).
        self._llm = llm

    @abstractmethod
    async def next_speaker(self, state: SessionState) -> str | None:
        """Return the participant id to speak next, or ``None`` to end."""
        ...

    async def on_event(self, event: Event) -> None:
        """Optional: maintain internal state from the event stream (for replay)."""
        return
