"""Transcript rendering: agent-colored lines must stay readable."""

from __future__ import annotations

import uuid

from thinktank.core.state import SessionState
from thinktank.domain.events import EventType, make_event
from thinktank.domain.models import Message
from thinktank.web.render import readable_text, transcript_fragment


def _posted(speaker: str):
    message = Message(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        seq=1,
        speaker_id=speaker,
        kind="speech",
        content="A message for the debate.",
    )
    return make_event(EventType.MESSAGE_POSTED, message=message)


def test_agent_message_is_tinted_with_its_color() -> None:
    html = transcript_fragment(
        _posted("ai_a"),
        SessionState(),
        names={"ai_a": "Agent A"},
        colors={"ai_a": "#e8b62c"},
    )
    assert html is not None
    assert "background: #e8b62c" in html
    # The text color is picked for contrast on that background.
    assert f"color: {readable_text('#e8b62c')}" in html


def test_message_without_color_is_untinted() -> None:
    html = transcript_fragment(
        _posted("sando"),
        SessionState(),
        names={"sando": "Sándor"},
        colors={"ai_a": "#e8b62c"},
    )
    assert html is not None
    assert "background:" not in html


def test_readable_text_picks_contrasting_color() -> None:
    assert readable_text("#ffffff") == "#101418"
    assert readable_text("#000000") == "#f5f7fa"
    assert readable_text("#e8b62c") == "#101418"  # yellow -> dark text
    assert readable_text("#429dd7") == "#101418"  # mid blue: dark beats white
    assert readable_text("#7a1f1f") == "#f5f7fa"  # dark red -> white text
    assert readable_text("not-a-color") == "#101418"

