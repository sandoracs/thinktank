"""Hand-raise priority wrapper (DESIGN.md §8.2).

A wrapper, not a standalone strategy: it delegates to an inner strategy but
always gives the floor to a raised hand first (FIFO). This keeps hand-raising
strategy-independent — it can be layered on top of round-robin, bidding, etc.
The engine lowers the chosen speaker's hand after they talk.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from roundtable.core.state import SessionState
from roundtable.domain.events import Event
from roundtable.domain.models import StrategyRef
from roundtable.strategies.base import TurnStrategy


class HandRaisePriority(TurnStrategy):
    name = "hand_raise"

    class Params(BaseModel):
        inner: StrategyRef = Field(default_factory=lambda: StrategyRef(name="round_robin"))

    def __init__(
        self,
        params: BaseModel | dict[str, Any] | None = None,
        inner: TurnStrategy | None = None,
    ) -> None:
        super().__init__(params)
        if inner is not None:
            self._inner: TurnStrategy = inner
        else:
            from roundtable.plugins.registry import build_strategy

            self._inner = build_strategy(StrategyRef(**self.params.model_dump().get("inner", {})))

    async def next_speaker(self, state: SessionState) -> str | None:
        raised = list(state.hands_raised)
        if raised:
            return raised[0]
        return await self._inner.next_speaker(state)

    async def on_event(self, event: Event) -> None:
        """Hand state lives in the projection; forward everything to the inner strategy."""
        await self._inner.on_event(event)
