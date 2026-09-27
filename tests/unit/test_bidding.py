"""Bidding strategy unit tests (DESIGN.md §8.2, M5).

Bidding picks the highest bidder for each slot; a round ends once every active
participant has spoken. Bids come from a (cheap, structured-output) LLM call,
so a scripted :class:`FakeLLM` supplies deterministic per-participant bids.
"""

from __future__ import annotations

import json

import pytest

from thinktank.core.state import SessionState
from thinktank.domain.models import ParticipantRef, SessionConfig
from thinktank.llm.client import ChatMessage, Purpose
from thinktank.llm.fake import FakeLLM
from thinktank.strategies.bidding import Bidding

BIDS = {"a": 0.5, "b": 0.9, "c": 0.7}


def _bid_responder(
    model: str, messages: list[ChatMessage], purpose: Purpose, response_model: object
) -> str:
    if purpose != "bid":
        return "n/a"
    user = messages[-1].content if messages else ""
    for pid, value in BIDS.items():
        if f"Your participant id: {pid}" in user:
            return json.dumps({"bid": value})
    return json.dumps({"bid": 0.0})


def _config(ids: list[str]) -> SessionConfig:
    return SessionConfig(
        title="t",
        topic="t",
        participants=[ParticipantRef(agent=p) for p in ids],
    )


def _state(ids: list[str], disabled: list[str] | None = None) -> SessionState:
    state = SessionState(config=_config(ids))
    if disabled:
        state.disabled = list(disabled)
    return state


def _strategy() -> Bidding:
    return Bidding(params={"bid_model": "fake"}, llm=FakeLLM(responder=_bid_responder))


async def _round(strat: Bidding, state: SessionState) -> list[str]:
    order: list[str] = []
    while not state.round_complete:
        speaker = await strat.next_speaker(state)
        if speaker is None:
            break
        order.append(speaker)
    return order


@pytest.mark.asyncio
async def test_bidding_highest_bid_spokes_first() -> None:
    state = _state(["a", "b", "c"])
    strat = _strategy()
    order = await _round(strat, state)
    # b (0.9) > c (0.7) > a (0.5); every active participant speaks once.
    assert order == ["b", "c", "a"]
    assert state.round_complete is True


@pytest.mark.asyncio
async def test_bidding_round_completes_when_all_spoken() -> None:
    state = _state(["a", "b"])
    strat = _strategy()
    assert await _round(strat, state) == ["b", "a"]
    # Round is done: the next call returns None.
    assert await strat.next_speaker(state) is None


@pytest.mark.asyncio
async def test_bidding_skips_disabled() -> None:
    state = _state(["a", "b", "c"], disabled=["b"])
    strat = _strategy()
    # b is disabled so only a and c speak; c (0.7) before a (0.5).
    assert await _round(strat, state) == ["c", "a"]


@pytest.mark.asyncio
async def test_bidding_without_llm_falls_back_to_config_order() -> None:
    state = _state(["a", "b", "c"])
    strat = Bidding(params={"bid_model": "fake"})  # no llm injected
    # All bids are neutral (0.0); the deterministic tie-break is config order.
    assert await _round(strat, state) == ["a", "b", "c"]
