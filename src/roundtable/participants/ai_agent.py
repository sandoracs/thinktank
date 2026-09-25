"""LLM-backed AI participant (DESIGN.md §8.1).

``speak`` runs the pipeline the design prescribes:
ContextBuilder -> LLMClient -> (consistency check, off by default) -> Message,
and emits an ``LLMCallCompleted`` event so the engine's cost accounting and the
live UI see every model call. Working memory is projected by the engine from
the event stream, so ``observe`` is a no-op.

At session end the agent distils its memory (DESIGN.md §11): an episodic
summary of this session, then up to a few durable long-term lessons, both
stored through the injected :class:`MemoryBackend` and logged as
``MemoryWritten`` events.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel

from roundtable.core.bus import EmitFn
from roundtable.core.context import ContextBuilder, ParticipantInfo
from roundtable.core.state import SessionState
from roundtable.domain.events import EventType, new_message_id
from roundtable.domain.models import (
    AgentConfig,
    DriftConfig,
    Layer,
    MemoryHit,
    Message,
    PersonaState,
    SessionConfig,
)
from roundtable.llm.client import ChatMessage, LLMClient, LLMResult, Purpose
from roundtable.memory.base import MemoryBackend
from roundtable.memory.distill import episodic_messages, has_lesson, long_term_messages
from roundtable.participants.base import TurnContext
from roundtable.persona.prompts import reflection_messages
from roundtable.persona.reflection import ReflectionResult


class AIAgent:
    """An agent that speaks using a configured model and persona."""

    kind: Literal["ai", "human", "remote"] = "ai"

    def __init__(
        self,
        *,
        config: AgentConfig,
        session_config: SessionConfig,
        state: SessionState,
        participants: list[ParticipantInfo],
        llm: LLMClient,
        context_builder: ContextBuilder | None = None,
        memory: MemoryBackend | None = None,
        emit: EmitFn,
    ) -> None:
        self.id = config.id
        self.display_name = config.persona.name
        self._config = config
        self._session = session_config
        self._state = state
        self._participants = participants
        self._llm = llm
        self._context_builder = context_builder or ContextBuilder()
        self._memory = memory
        self._emit = emit
        self._session_id: uuid.UUID | None = None

    async def speak(self, ctx: TurnContext) -> Message:
        self._session_id = ctx.session_id
        memories = await self._retrieve_memories(ctx)
        messages = self._context_builder.build(
            config=self._session,
            agent=self._config,
            state=self._state,
            participants=self._participants,
            memories=memories,
            turn_instruction=ctx.turn_instruction,
        )
        result = await self._llm.complete(
            model=self._config.model,
            messages=messages,
            purpose="speech",
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        await self._emit(
            EventType.LLM_CALL_COMPLETED,
            model=result.model,
            purpose="speech",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            duration_ms=result.duration_ms,
            agent_id=self.id,
        )
        # Consistency check (DESIGN.md §12.1) is opt-in and lands in M6.
        return Message(
            id=new_message_id(),
            session_id=ctx.session_id,
            seq=0,  # stamped with the event seq during projection
            speaker_id=self.id,
            kind="speech",
            content=result.content,
            meta={
                "model": result.model,
                "tokens_in": result.input_tokens,
                "tokens_out": result.output_tokens,
                "cost_usd": result.cost_usd,
            },
        )

    @property
    def drift(self) -> DriftConfig:
        """The agent's drift configuration (used by the engine's reflection loop)."""
        return self._config.drift

    @property
    def initial_state(self) -> PersonaState:
        """The agent's starting persona state (fallback when no update is applied)."""
        return self._config.initial_state

    async def reflect(self, state: SessionState) -> ReflectionResult | None:
        """Round-end self-reflection: propose persona-state changes (DESIGN.md §12.2)."""
        persona_state = state.persona_states.get(self.id) or self._config.initial_state
        messages, schema = reflection_messages(
            agent_name=self.display_name,
            config=self._session,
            state=state,
            persona_state=persona_state,
        )
        result = await self._complete(
            purpose="reflection",
            messages=messages,
            response_model=schema,
        )
        if result is None or result.parsed is None:
            return None
        proposal = result.parsed
        if not isinstance(proposal, ReflectionResult) or proposal.is_empty:
            return None
        return proposal

    async def observe(self, msg: Message) -> None:
        # Working memory is projected by the engine; nothing to store in M1.
        return None

    async def on_session_end(self) -> None:
        """Distil episodic + long-term memory (DESIGN.md §11)."""
        if self._memory is None or self._session_id is None:
            return
        transcript = self._transcript()
        if not transcript.strip():
            return

        summary = await self._complete(
            purpose="summary",
            messages=episodic_messages(
                agent_name=self.display_name,
                topic=self._session.topic,
                transcript=transcript,
            ),
        )
        if not summary or not summary.content.strip():
            return
        episodic_id = await self._memory.add(
            self.id,
            Layer.EPISODIC,
            summary.content.strip(),
            self._session_id,
            None,
            {"kind": "session_summary"},
        )
        await self._emit(
            EventType.MEMORY_WRITTEN,
            agent_id=self.id,
            layer=Layer.EPISODIC.value,
            memory_id=episodic_id,
            session_id=str(self._session_id),
            source_seq=None,
        )

        if not self._config.memory.long_term:
            return
        lessons = await self._complete(
            purpose="summary",
            messages=long_term_messages(
                agent_name=self.display_name,
                topic=self._session.topic,
                episodic_summary=summary.content,
            ),
        )
        if lessons is None or not has_lesson(lessons.content):
            return
        long_id = await self._memory.add(
            self.id,
            Layer.LONG_TERM,
            lessons.content.strip(),
            None,
            None,
            {"kind": "lesson", "from_session": str(self._session_id)},
        )
        await self._emit(
            EventType.MEMORY_WRITTEN,
            agent_id=self.id,
            layer=Layer.LONG_TERM.value,
            memory_id=long_id,
            session_id=None,
            source_seq=None,
        )

    async def _complete(
        self,
        *,
        purpose: Purpose,
        messages: list[ChatMessage],
        response_model: type[BaseModel] | None = None,
    ) -> LLMResult | None:
        """One side-call with the (optionally cheaper) summary model + event log."""
        model = self._config.summary_model or self._config.model
        try:
            result = await self._llm.complete(
                model=model,
                messages=messages,
                purpose=purpose,
                temperature=0.3,
                max_tokens=400,
                response_model=response_model,
            )
        except Exception:
            return None
        await self._emit(
            EventType.LLM_CALL_COMPLETED,
            model=result.model,
            purpose=purpose,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            duration_ms=result.duration_ms,
            agent_id=self.id,
        )
        return result

    def _transcript(self) -> str:
        lines: list[str] = []
        for msg in self._state.messages:
            name = msg.speaker_id
            lines.append(f"{name}: {msg.content}")
        return "\n".join(lines)

    async def _retrieve_memories(self, ctx: TurnContext) -> list[MemoryHit]:
        if self._memory is None:
            return []
        query = self._session.topic
        recent = self._state.messages[-2:]
        if recent:
            query = f"{query} {' '.join(m.content for m in recent)}"
        layers = {Layer.EPISODIC, Layer.LONG_TERM}
        return await self._memory.search(
            self.id,
            query,
            self._config.memory.retrieval_k,
            layers,
            session_id=ctx.session_id,
        )
