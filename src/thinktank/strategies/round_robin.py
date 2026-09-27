"""Built-in round-robin turn strategy.

Deterministic by default: participants speak in session order, and a round is
complete once every *active* participant has spoken. ``shuffle_each_round``
randomises the order each round using a seeded RNG, so a session and its replay
agree exactly (the seed is part of the strategy params, stored in the
``SessionCreated`` event).

Per-round progress is reconstructed from the event stream (``TurnAssigned``
advances, ``RoundStarted`` resets), which makes crash-resume continue mid-round
correctly rather than re-speaking earlier participants.
"""

from __future__ import annotations

import random

from pydantic import BaseModel

from thinktank.core.state import SessionState
from thinktank.domain.events import Event, EventType, SessionCreatedPayload
from thinktank.llm.client import LLMClient
from thinktank.strategies.base import TurnStrategy


class RoundRobin(TurnStrategy):
    name = "round_robin"

    class Params(BaseModel):
        shuffle_each_round: bool = False
        seed: int = 0

    def __init__(
        self,
        params: BaseModel | dict[str, object] | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        super().__init__(params, llm=llm)
        self._order: list[str] = []
        self._spoken: set[str] = set()

    def _set_order(self, participant_ids: list[str]) -> None:
        self._order = list(participant_ids)

    async def next_speaker(self, state: SessionState) -> str | None:
        if not self._order and state.config is not None:
            self._set_order(state.config.participant_ids())
        if not self._order:
            return None

        disabled = set(state.disabled)
        active = [pid for pid in self._order if pid not in disabled]
        if not active:
            return None

        pick: str | None = None
        for candidate in self._order:
            if candidate in disabled or candidate in self._spoken:
                continue
            pick = candidate
            break

        if pick is None:
            state.round_complete = True
            return None

        # After this speaker, everyone active will have spoken -> round is done.
        if len(self._spoken) + 1 >= len(active):
            state.round_complete = True
        return pick

    async def on_event(self, event: Event) -> None:
        if event.type is EventType.SESSION_CREATED:
            cfg = event.payload_as(SessionCreatedPayload)
            self._set_order(cfg.config.participant_ids())
        elif event.type is EventType.ROUND_STARTED:
            self._spoken = set()
            p = self.params.model_dump()
            if p.get("shuffle_each_round"):
                round_no = int(event.payload.get("round", 0))
                rng = random.Random(f"{p.get('seed', 0)}:{round_no}")  # noqa: S311 (deterministic replay, not crypto)
                rng.shuffle(self._order)
        elif event.type is EventType.TURN_ASSIGNED:
            pid = event.payload.get("speaker_id")
            if pid is not None:
                self._spoken.add(pid)
