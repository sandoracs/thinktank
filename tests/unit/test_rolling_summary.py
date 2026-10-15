"""Rolling in-session summary: messages leaving the working window are kept."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel

from thinktank.core.context import ParticipantInfo
from thinktank.core.state import SessionState
from thinktank.domain.events import Event, EventType
from thinktank.domain.models import AgentConfig, MemoryConfig, Message, PersonaCore, SessionConfig
from thinktank.llm.client import ChatMessage, Purpose
from thinktank.llm.fake import FakeLLM
from thinktank.participants.ai_agent import AIAgent
from thinktank.participants.base import TurnContext


def _state(sid: uuid.UUID, config: SessionConfig, n: int) -> SessionState:
    msgs = [
        Message(
            id=uuid.uuid4(),
            session_id=sid,
            seq=i + 1,
            speaker_id="other" if i % 2 else "me",
            kind="speech",
            content=f"point number {i}",
        )
        for i in range(n)
    ]
    return SessionState(session_id=sid, config=config, messages=msgs)


def _agent(state: SessionState, config: SessionConfig, llm: FakeLLM, *, rolling: bool = True) -> AIAgent:
    async def emit(etype: EventType, **payload: Any) -> Event:  # pragma: no cover - trivial
        return None  # type: ignore[return-value]

    cfg = AgentConfig(
        id="me",
        model="m",
        persona=PersonaCore(name="Skeptic", role="skeptic"),
        memory=MemoryConfig(working_window=4, rolling_summary=rolling),
    )
    return AIAgent(
        config=cfg,
        session_config=config,
        state=state,
        participants=[ParticipantInfo(id="me", name="Skeptic"), ParticipantInfo(id="other", name="Optimist")],
        llm=llm,
        emit=emit,
    )


def _responder(model: str, messages: list[ChatMessage], purpose: Purpose, rm: type[BaseModel] | None) -> str:
    return "SUMMARY-OF-EARLY-PART" if purpose == "summary" else "my reply"


async def test_evicted_messages_are_summarised_into_the_system_prompt() -> None:
    sid = uuid.uuid4()
    config = SessionConfig(title="t", topic="topic", participants=[])
    state = _state(sid, config, 20)
    llm = FakeLLM(responder=_responder)
    agent = _agent(state, config, llm)

    await agent.speak(TurnContext(session_id=sid, round=1))

    summary_calls = [c for c in llm.calls if c["purpose"] == "summary"]
    assert len(summary_calls) == 1
    # The summariser saw the evicted messages with display names, not ids.
    summarised = summary_calls[0]["messages"][-1].content
    assert "Optimist: point number 1" in summarised
    assert "point number 16" not in summarised  # still in the working window
    speech = next(c for c in llm.calls if c["purpose"] == "speech")
    assert "SUMMARY-OF-EARLY-PART" in speech["messages"][0].content
    working = speech["messages"][1:-1]
    assert "point number 0" not in "".join(m.content for m in working)
    assert "point number 19" in "".join(m.content for m in working)


async def test_summary_is_batched_not_per_turn() -> None:
    sid = uuid.uuid4()
    config = SessionConfig(title="t", topic="topic", participants=[])
    state = _state(sid, config, 20)
    llm = FakeLLM(responder=_responder)
    agent = _agent(state, config, llm)
    await agent.speak(TurnContext(session_id=sid, round=1))
    # One more message: the evicted one waits in the working context (batch=2).
    state.messages.append(state.messages[-1].model_copy(update={"id": uuid.uuid4(), "seq": 21}))
    await agent.speak(TurnContext(session_id=sid, round=1))
    assert sum(1 for c in llm.calls if c["purpose"] == "summary") == 1


async def test_rolling_summary_can_be_disabled() -> None:
    sid = uuid.uuid4()
    config = SessionConfig(title="t", topic="topic", participants=[])
    state = _state(sid, config, 20)
    llm = FakeLLM(responder=_responder)
    agent = _agent(state, config, llm, rolling=False)
    await agent.speak(TurnContext(session_id=sid, round=1))
    assert not [c for c in llm.calls if c["purpose"] == "summary"]


def test_session_end_transcript_uses_display_names() -> None:
    sid = uuid.uuid4()
    config = SessionConfig(title="t", topic="topic", participants=[])
    state = _state(sid, config, 2)
    agent = _agent(state, config, FakeLLM(responder=_responder))
    transcript = agent._transcript()  # pyright: ignore[reportPrivateUsage]
    assert "Optimist: point number 1" in transcript
    assert "other:" not in transcript
