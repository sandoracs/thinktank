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

from roundtable.core.state import SessionState
from roundtable.domain.events import (
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
) -> str | None:
    """Render the transcript row for ``event``, or ``None`` if not a row."""
    resolved = names if names is not None else participant_names(state)
    ctx: dict[str, Any] = {"event": event, "state": state, "names": resolved}

    if event.type is EventType.MESSAGE_POSTED:
        message = event.payload_as(MessagePostedPayload).message
        ctx["message"] = message
        ctx["speaker"] = ctx["names"].get(message.speaker_id, message.speaker_id)
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


def sidebar_fragment(state: SessionState, names: dict[str, str] | None = None) -> str:
    """Render the full sidebar (status, meters, participant list)."""
    resolved = names if names is not None else participant_names(state)
    kinds: dict[str, str] = {}
    if state.config is not None:
        for ref in state.config.participants:
            kinds[ref.participant_id] = ref.participant_kind
    return _env().get_template("partials/sidebar.html").render(state=state, names=resolved, kinds=kinds)
