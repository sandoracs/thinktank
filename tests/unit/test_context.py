"""ContextBuilder unit tests."""

from __future__ import annotations

import uuid

from thinktank.core.context import ContextBuilder, ParticipantInfo
from thinktank.core.state import SessionState
from thinktank.domain.models import AgentConfig, Message, PersonaCore, SessionConfig


def _session_id() -> uuid.UUID:
    return uuid.uuid4()


def test_builds_system_and_working_memory() -> None:
    sid = _session_id()
    config = SessionConfig(title="t", topic="Co-authorship", participants=[])
    agent = AgentConfig(id="me", model="", persona=PersonaCore(name="Skeptic", role="skeptic"))
    state = SessionState(
        session_id=sid,
        config=config,
        messages=[
            Message(id=uuid.uuid4(), session_id=sid, seq=1, speaker_id="other",
                    kind="speech", content="I agree with the plan."),
            Message(id=uuid.uuid4(), session_id=sid, seq=2, speaker_id="me",
                    kind="speech", content="I disagree, the evidence is weak."),
        ],
    )
    participants = [
        ParticipantInfo(id="me", name="Skeptic"),
        ParticipantInfo(id="other", name="Optimist"),
    ]
    messages = ContextBuilder().build(
        config=config, agent=agent, state=state, participants=participants,
        turn_instruction="State your position.",
    )

    assert messages[0].role == "system"
    assert "Skeptic" in messages[0].content  # persona rendered
    assert messages[-1].role == "user"  # final turn instruction
    roles = {m.role for m in messages[1:-1]}
    assert "assistant" in roles  # own message
    assert "user" in roles  # other participant's message, name-prefixed
    assert any(m.content.startswith("[Optimist]") for m in messages)
