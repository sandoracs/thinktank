"""RoundRobin strategy unit tests."""

from __future__ import annotations

import pytest

from thinktank.core.state import SessionState
from thinktank.domain.events import EventType, make_event
from thinktank.domain.models import ParticipantRef, SessionConfig
from thinktank.strategies.round_robin import RoundRobin


def _config(ids: list[str]) -> SessionConfig:
    return SessionConfig(
        title="t",
        topic="t",
        participants=[ParticipantRef(agent=p) for p in ids],
        moderator=ids[0] if ids else None,
    )


def _state(cfg: SessionConfig, disabled: list[str] | None = None) -> SessionState:
    state = SessionState(config=cfg)
    if disabled:
        state.disabled = list(disabled)
    return state


def _created(ids: list[str]):
    return make_event(EventType.SESSION_CREATED, config=_config(ids))


async def _drive(strat: RoundRobin, state: SessionState, count: int) -> list[str | None]:
    order: list[str | None] = []
    for _ in range(count):
        s = await strat.next_speaker(state)
        order.append(s)
        if s is not None:
            await strat.on_event(make_event(EventType.TURN_ASSIGNED, speaker_id=s, strategy="round_robin"))
    return order


@pytest.mark.asyncio
async def test_round_robin_order() -> None:
    state = _state(_config(["a", "b", "c"]))
    strat = RoundRobin()
    await strat.on_event(_created(["a", "b", "c"]))
    assert await _drive(strat, state, 3) == ["a", "b", "c"]
    # All active have spoken, so the round is complete.
    assert await strat.next_speaker(state) is None


@pytest.mark.asyncio
async def test_round_robin_skips_disabled() -> None:
    state = _state(_config(["a", "b", "c"]), disabled=["b"])
    strat = RoundRobin()
    await strat.on_event(_created(["a", "b", "c"]))
    assert await _drive(strat, state, 2) == ["a", "c"]


@pytest.mark.asyncio
async def test_all_disabled_returns_none() -> None:
    state = _state(_config(["a", "b"]), disabled=["a", "b"])
    strat = RoundRobin()
    await strat.on_event(_created(["a", "b"]))
    assert await strat.next_speaker(state) is None
