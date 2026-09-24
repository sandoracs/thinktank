"""LLM-backed AI participant (DESIGN.md §8.1).

``speak`` runs the pipeline the design prescribes:
ContextBuilder -> LLMClient -> (consistency check, off by default) -> Message,
and emits an ``LLMCallCompleted`` event so the engine's cost accounting and the
live UI see every model call. The engine owns working memory (it is projected
from the event stream), so ``observe`` is a no-op in M1; episodic summarisation
lands in M3.
"""

from __future__ import annotations

from typing import Literal

from roundtable.core.bus import EmitFn
from roundtable.core.context import ContextBuilder, ParticipantInfo
from roundtable.core.state import SessionState
from roundtable.domain.events import EventType, new_message_id
from roundtable.domain.models import (
    AgentConfig,
    Layer,
    MemoryHit,
    Message,
    SessionConfig,
)
from roundtable.llm.client import LLMClient
from roundtable.memory.base import MemoryBackend
from roundtable.participants.base import TurnContext


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

    async def speak(self, ctx: TurnContext) -> Message:
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

    async def observe(self, msg: Message) -> None:
        # Working memory is projected by the engine; nothing to store in M1.
        return None

    async def on_session_end(self) -> None:
        # Long-term distillation (DESIGN.md §11) lands in M3.
        return None

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
