"""Event schema tests."""

from __future__ import annotations

from thinktank.domain.events import OPEN_PAYLOAD, PAYLOAD_MODELS, EventType, RoundPayload


def test_every_event_type_is_covered() -> None:
    for t in EventType:
        assert t in PAYLOAD_MODELS or t in OPEN_PAYLOAD, f"no payload model for {t}"


def test_payload_round_trip() -> None:
    p = RoundPayload(round=3)
    p2 = RoundPayload.model_validate(p.model_dump())
    assert p2.round == 3
