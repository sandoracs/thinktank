"""Cross-session memory integration test (M3 acceptance).

M3 "done when": *in a second session, the agent references the previous
session's lesson* — the second session's agent must have the previous
session's distilled lesson in its context.

Both sessions run against :class:`FakeLLM` (deterministic, offline). The
responder returns purpose-appropriate content and records every prompt so
the test can assert that session 2's speech prompt contains session 1's
long-term lesson.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from thinktank.core.manager import SessionManager
from thinktank.domain.models import (
    AgentConfig,
    DebateQuestion,
    Layer,
    ParticipantRef,
    PersonaCore,
    SessionConfig,
    StopConditions,
)
from thinktank.llm.client import ChatMessage
from thinktank.llm.fake import FakeLLM
from thinktank.memory.distill import EPISODIC_SYSTEM
from thinktank.memory.embeddings import FakeEmbeddingProvider
from thinktank.memory.sqlite import SQLiteMemoryBackend
from thinktank.storage.db import init_db, init_memory_tables, make_engine, make_session_factory
from thinktank.storage.repositories import EventStore

AGENT_ID = "coauthor_agent"
LESSON_MARKER = "EVIDENCE_BASE_WEAK"

Harness = tuple[SessionManager, EventStore, FakeLLM, SQLiteMemoryBackend]

def _responder(
    model: str, messages: list[ChatMessage], purpose: str, response_model: object
) -> str:
    system = messages[0].content if messages else ""
    if purpose == "speech":
        return f"speech turn on the {LESSON_MARKER} point"
    if purpose == "summary" and EPISODIC_SYSTEM in system:
        # Episodic summary of session 1: mention a durable insight.
        return f"Session summary: the {LESSON_MARKER} was the crux of the debate."
    if purpose == "summary":
        # Long-term distillation: the durable lesson to carry forward.
        return f"Lesson: {LESSON_MARKER} — check the evidence base before conceding."
    return "ok"


def _agent() -> dict[str, AgentConfig]:
    return {
        AGENT_ID: AgentConfig(
            id=AGENT_ID,
            model="fake-model",
            persona=PersonaCore(
                name="Peter Stone",
                role="science editor",
                values=["strict sourcing"],
                boundaries=["does not give medical advice"],
            ),
        )
    }


def _session(title: str) -> SessionConfig:
    return SessionConfig(
        title=title,
        topic="AI co-authorship: disclosure and evidence standards in scientific publishing.",
        questions=[
            DebateQuestion(id="q1", text="May an LLM be listed as a co-author?"),
            DebateQuestion(id="q2", text="Must AI usage be disclosed in detail?"),
        ],
        participants=[ParticipantRef(agent=AGENT_ID)],
        moderator=AGENT_ID,
        stop=StopConditions(max_rounds=2, max_cost_usd=10.0),
    )


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncGenerator[Harness, None]:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'mem.db'}", load_vec=True)
    await init_db(engine)
    embedder = FakeEmbeddingProvider(dim=16)
    await init_memory_tables(engine, 16, embedder.name)
    factory = make_session_factory(engine)
    store = EventStore(factory)
    llm = FakeLLM(responder=_responder)
    manager = SessionManager(
        store=store,
        llm=llm,
        memory=SQLiteMemoryBackend(engine, embedder),
    )
    yield manager, store, llm, SQLiteMemoryBackend(engine, embedder)
    await engine.dispose()


@pytest.mark.asyncio
async def test_second_session_recalls_previous_lesson(harness: Harness) -> None:
    manager, store, llm, backend = harness

    # --- session 1: runs, ends, distils memory -----------------------------
    session1 = await manager.create_session(_session("Session 1"), _agent())
    await manager.start(session1)
    assert session1.state.status == "ended"
    sid1 = session1.state.session_id
    assert sid1 is not None
    long_hits = await backend.search(
        AGENT_ID,
        "evidence base disclosure",
        5,
        {Layer.LONG_TERM},
    )
    assert long_hits, "session 1 must have distilled a long-term lesson"
    assert LESSON_MARKER in long_hits[0].content

    # Also an episodic summary bound to session 1.
    episodic_hits = await backend.search(
        AGENT_ID,
        "summary crux debate",
        5,
        {Layer.EPISODIC},
        session_id=sid1,
    )
    assert episodic_hits
    assert episodic_hits[0].session_id == sid1

    # --- session 2: the agent's speech context must contain the lesson -----
    llm.calls.clear()
    session2 = await manager.create_session(_session("Session 2"), _agent())
    await manager.start(session2)
    assert session2.state.status == "ended"

    speech_calls = [c for c in llm.calls if c["purpose"] == "speech"]
    assert speech_calls, "session 2 must have produced at least one speech"
    first_prompt = "\n".join(m.content for m in speech_calls[0]["messages"])
    assert LESSON_MARKER in first_prompt, (
        "session 2's first speech prompt must carry session 1's long-term lesson"
    )

    # --- the events record the memory writes (auditability) -----------------
    events1 = await store.get_events(sid1)
    memory_events = [e for e in events1 if e.type.value == "MemoryWritten"]
    assert len(memory_events) == 2  # one episodic + one long-term
    assert {e.payload.get("layer") for e in memory_events} == {"episodic", "long_term"}
