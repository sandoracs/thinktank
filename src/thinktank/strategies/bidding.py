"""Bidding turn strategy (DESIGN.md §8.2).

A second built-in strategy, registered through the same entry-point mechanism
as every other plugin. Instead of a fixed order, each *unspeoken* active
participant submits a cheap bid in ``[0, 1]`` for how strongly it wants the
floor right now; the highest bid speaks. A round is complete once every active
participant has spoken once (the order within the round is what bidding
decides).

Ties are broken deterministically so a session and its replay agree: prefer a
participant who has not spoken yet, then the one who spoke longest ago, then
the configured session order.

The bid is a structured-output LLM call (``purpose="bid"``). The strategy
therefore requires an :class:`LLMClient` (injected via the registry); a
``bid_model`` is configurable so a cheap model can be used. Note: the bid call
is not surfaced as an ``LLMCallCompleted`` event (the engine does not yet exist
when the strategy is constructed) — keep ``bid_model`` cheap.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from thinktank.core.state import SessionState
from thinktank.domain.events import Event, EventType, SessionCreatedPayload
from thinktank.llm.client import ChatMessage, LLMClient
from thinktank.strategies.base import TurnStrategy


class BidResult(BaseModel):
    """A single participant's bid for the floor, in ``[0, 1]``."""

    bid: float = Field(default=0.0, ge=0.0, le=1.0)


class Bidding(TurnStrategy):
    name = "bidding"

    class Params(BaseModel):
        bid_model: str = ""
        bid_temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    def __init__(
        self,
        params: BaseModel | dict[str, object] | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        super().__init__(params, llm=llm)
        self._order: list[str] = []
        self._spoken: set[str] = set()
        self._last_seq: dict[str, int] = {}

    @property
    def _p(self) -> Bidding.Params:
        """The strategy's typed parameters (DESIGN.md §13 Params schema)."""
        if isinstance(self.params, Bidding.Params):
            return self.params
        return Bidding.Params.model_validate(self.params.model_dump())

    async def on_event(self, event: Event) -> None:
        if event.type is EventType.SESSION_CREATED:
            cfg = event.payload_as(SessionCreatedPayload)
            self._order = list(cfg.config.participant_ids())
        elif event.type is EventType.ROUND_STARTED:
            self._spoken = set()
        elif event.type is EventType.TURN_ASSIGNED:
            pid = event.payload.get("speaker_id")
            if pid is not None:
                self._spoken.add(pid)
                self._last_seq[pid] = event.seq

    # -- selection ---------------------------------------------------------
    async def next_speaker(self, state: SessionState) -> str | None:
        if not self._order and state.config is not None:
            self._order = list(state.config.participant_ids())
        disabled = set(state.disabled)
        active = [pid for pid in self._order if pid not in disabled]
        candidates = [pid for pid in active if pid not in self._spoken]
        if not candidates:
            state.round_complete = True
            return None

        bids = await self._collect_bids(state, candidates)
        # Highest bid wins; deterministic tie-break (see module docstring).
        best = max(
            candidates,
            key=lambda pid: (
                bids.get(pid, 0.0),
                0 if pid in self._spoken else 1,  # not-yet-spoken first on ties
                -self._last_seq.get(pid, -1),      # spoke longest ago first
                -self._order.index(pid),           # session order first
            ),
        )
        self._spoken.add(best)
        if len(self._spoken) >= len(active):
            state.round_complete = True
        return best

    async def _collect_bids(self, state: SessionState, candidates: list[str]) -> dict[str, float]:
        if self._llm is None:
            # No LLM available: fall back to a neutral, deterministic order so
            # the strategy still works (e.g. in tests or offline tooling).
            return dict.fromkeys(candidates, 0.0)
        results = await asyncio.gather(
            *(self._bid(state, pid) for pid in candidates), return_exceptions=True
        )
        bids: dict[str, float] = {}
        for pid, res in zip(candidates, results, strict=True):
            if isinstance(res, float):
                bids[pid] = res
            else:
                bids[pid] = 0.0  # bid call failed; treat as no interest
        return bids

    async def _bid(self, state: SessionState, pid: str) -> float:
        assert self._llm is not None
        topic = state.config.topic if state.config else ""
        questions = (
            "\n".join(f"- [{q.id}] {q.text}" for q in state.config.questions) if state.config else ""
        )
        transcript = "\n".join(f"{m.speaker_id}: {m.content}" for m in state.messages[-8:])
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are deciding whether to take the floor in a multi-party debate. "
                    "Respond with a single number in [0, 1] for how strongly you want to "
                    "speak right now (1 = urgently, 0 = not at all)."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Topic: {topic}\n"
                    f"Questions:\n{questions}\n"
                    f"Recent discussion:\n{transcript}\n"
                    f"Your participant id: {pid}"
                ),
            ),
        ]
        result = await self._llm.complete(
            model=self._p.bid_model or "bid",
            messages=messages,
            purpose="bid",
            temperature=self._p.bid_temperature,
            max_tokens=16,
            response_model=BidResult,
        )
        if result.parsed is not None:
            return float(getattr(result.parsed, "bid", 0.0))
        return 0.0
