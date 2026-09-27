"""Hand-raise priority wrapper (DESIGN.md §8.2).

A wrapper, not a standalone strategy: it delegates to an inner strategy but
always gives the floor to a raised hand first (FIFO). This keeps hand-raising
strategy-independent — it can be layered on top of round-robin, bidding, etc.
The engine lowers the chosen speaker's hand after they talk.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from thinktank.core.state import SessionState
from thinktank.domain.events import Event
from thinktank.domain.models import StrategyRef
from thinktank.llm.client import LLMClient
from thinktank.strategies.base import TurnStrategy


class HandRaisePriority(TurnStrategy):
    name = "hand_raise"

    class Params(BaseModel):
        inner: StrategyRef = Field(default_factory=lambda: StrategyRef(name="round_robin"))

    def __init__(
        self,
        params: BaseModel | dict[str, Any] | None = None,
        inner: TurnStrategy | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        super().__init__(params, llm=llm)
        if inner is not None:
            self._inner: TurnStrategy = inner
        else:
            from thinktank.plugins.registry import build_strategy

            inner_ref = StrategyRef(**self.params.model_dump().get("inner", {}))
            self._inner = build_strategy(inner_ref, llm=llm)

    async def next_speaker(self, state: SessionState) -> str | None:
        disabled = set(state.disabled)
        raised = [pid for pid in state.hands_raised if pid not in disabled]
        if raised:
            return raised[0]
        return await self._inner.next_speaker(state)

    async def on_event(self, event: Event) -> None:
        """Hand state lives in the projection; forward everything to the inner strategy."""
        await self._inner.on_event(event)
