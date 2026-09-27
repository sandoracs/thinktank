"""Server-rendered HTML fragments for the live table (DESIGN.md §15).

Every event the WebSocket layer forwards also carries small pre-rendered
fragments: a ``transcript`` row (when the event is user-visible) and a fresh
``sidebar`` snapshot. Rendering here (not in the browser) keeps the client to
a thin "append this HTML" job and makes replay and live views pixel-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from thinktank.core.state import SessionState
from thinktank.domain.events import (
    Event,
    EventType,
    MessagePostedPayload,
    TurnAssignedPayload,
)

_TEMPLATES = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(("html",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def participant_names(state: SessionState) -> dict[str, str]:
    """Map participant id -> display name from the session config."""
    out: dict[str, str] = {}
    if state.config is None:
        return out
    for ref in state.config.participants:
        out[ref.participant_id] = ref.participant_id
    return out


def transcript_fragment(
    event: Event,
    state: SessionState,
    names: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
) -> str | None:
    """Render the transcript row for ``event``, or ``None`` if not a row.

    ``colors`` maps participant id -> hex color; an agent's message line is
    tinted with its color and the text is chosen to stay readable on it.
    """
    resolved = names if names is not None else participant_names(state)
    palette = colors or {}
    ctx: dict[str, Any] = {"event": event, "state": state, "names": resolved}

    if event.type is EventType.MESSAGE_POSTED:
        message = event.payload_as(MessagePostedPayload).message
        ctx["message"] = message
        ctx["speaker"] = ctx["names"].get(message.speaker_id, message.speaker_id)
        bg = palette.get(message.speaker_id)
        if bg:
            ctx["speaker_bg"] = bg
            ctx["speaker_fg"] = readable_text(bg)
        return _env().get_template("partials/transcript_message.html").render(**ctx)
    if event.type in (EventType.ROUND_STARTED, EventType.ROUND_ENDED):
        return _env().get_template("partials/transcript_round.html").render(**ctx)
    if event.type is EventType.TURN_ASSIGNED:
        speaker_id = event.payload_as(TurnAssignedPayload).speaker_id
        return _note(f"→ {ctx['names'].get(speaker_id, speaker_id)} speaks")
    if event.type is EventType.TURN_SKIPPED:
        speaker = str(event.payload.get("speaker_id"))
        return _note(f"{ctx['names'].get(speaker, speaker)} skipped ({event.payload.get('reason')})")
    if event.type is EventType.SESSION_ENDED:
        return _note(f"Session ended: {event.payload.get('reason')}")
    if event.type is EventType.ERROR:
        return _note(f"Error [{event.payload.get('component')}]: {event.payload.get('message')}")
    return None


def _note(note: str) -> str:
    return _env().get_template("partials/transcript_note.html").render(note=note)


def sidebar_fragment(
    state: SessionState,
    names: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
) -> str:
    """Render the full sidebar (status, meters, participant list)."""
    resolved = names if names is not None else participant_names(state)
    kinds: dict[str, str] = {}
    if state.config is not None:
        for ref in state.config.participants:
            kinds[ref.participant_id] = ref.participant_kind
    return _env().get_template("partials/sidebar.html").render(
        state=state, names=resolved, kinds=kinds, colors=colors or {}
    )


def readable_text(bg: str) -> str:
    """A near-black or near-white text color for ``bg`` (hex).

    Whichever of the two wins on WCAG contrast ratio against the background is
    returned, so the text stays readable on any assigned agent color.
    """
    if len(bg) != 7 or bg[0] != "#":
        return "#101418"
    try:
        r, g, b = (int(bg[i : i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return "#101418"

    def _chan(c: int) -> float:
        x = c / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _chan(r) + 0.7152 * _chan(g) + 0.0722 * _chan(b)
    dark_lum, light_lum = 0.0056, 0.956  # #101418, #f5f7fa

    def _contrast(a: float, c: float) -> float:
        hi, lo = (a, c) if a >= c else (c, a)
        return (hi + 0.05) / (lo + 0.05)

    return "#101418" if _contrast(lum, dark_lum) >= _contrast(lum, light_lum) else "#f5f7fa"
