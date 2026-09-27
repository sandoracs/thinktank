"""LLM-backed AI participant.

``speak`` runs the pipeline the design prescribes:
ContextBuilder -> LLMClient -> (consistency check, off by default) -> Message,
and emits an ``LLMCallCompleted`` event so the engine's cost accounting and the
live UI see every model call. Working memory is projected by the engine from
the event stream, so ``observe`` is a no-op.

At session end the agent distils its memory: an episodic
summary of this session, then up to a few durable long-term lessons, both
stored through the injected :class:`MemoryBackend` and logged as
``MemoryWritten`` events.
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from pydantic import BaseModel

from thinktank.core.bus import EmitFn
from thinktank.core.context import ContextBuilder, ParticipantInfo
from thinktank.core.state import SessionState
from thinktank.domain.events import EventType, new_message_id
from thinktank.domain.models import (
    AgentConfig,
    DriftConfig,
    Layer,
    MemoryHit,
    Message,
    PersonaState,
    SessionConfig,
)
from thinktank.llm.client import ChatMessage, LLMClient, LLMResult, Purpose
from thinktank.memory.base import MemoryBackend
from thinktank.memory.distill import episodic_messages, has_lesson, long_term_messages
from thinktank.participants.base import TurnContext
from thinktank.persona.consistency import ConsistencyJudgment, consistency_messages
from thinktank.persona.prompts import reflection_messages
from thinktank.persona.reflection import ReflectionResult

logger = logging.getLogger(__name__)


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
        content = result.content
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        cost_usd = result.cost_usd

        if self._config.consistency_check:
            content, extra_in, extra_out, extra_cost = await self._consistency_check(result.content)
            input_tokens += extra_in
            output_tokens += extra_out
            cost_usd += extra_cost

        return Message(
            id=new_message_id(),
            session_id=ctx.session_id,
            seq=0,  # stamped with the event seq during projection
            speaker_id=self.id,
            kind="speech",
            content=content,
            meta={
                "model": result.model,
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "cost_usd": cost_usd,
            },
        )

    async def _consistency_check(self, candidate: str) -> tuple[str, int, int, float]:
        """Optional persona-consistency pass (M6).

        The judge scores the candidate against the frozen core (1-5). Below the
        threshold the speech is regenerated once with the judge's feedback. A
        ``ConsistencyViolation`` event is emitted either way. Returns the final
        content plus the extra tokens/cost incurred by the pass.
        """
        threshold = self._config.consistency_threshold
        judge = await self._complete(
            purpose="judge",
            messages=consistency_messages(core=self._config.persona, candidate=candidate),
            response_model=ConsistencyJudgment,
        )
        if judge is None or judge.parsed is None:
            # Judge unavailable (model error / unparseable) — keep the candidate.
            return candidate, 0, 0, 0.0
        judgment = judge.parsed
        assert isinstance(judgment, ConsistencyJudgment)

        extra_in = extra_out = 0
        extra_cost = 0.0
        content = candidate
        regenerated = False
        if judgment.score < threshold:
            regen = await self._llm.complete(
                model=self._config.model,
                messages=consistency_messages(
                    core=self._config.persona,
                    candidate=candidate,
                    feedback=judgment.justification,
                ),
                purpose="speech",
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            await self._emit(
                EventType.LLM_CALL_COMPLETED,
                model=regen.model,
                purpose="speech",
                input_tokens=regen.input_tokens,
                output_tokens=regen.output_tokens,
                cost_usd=regen.cost_usd,
                duration_ms=regen.duration_ms,
                agent_id=self.id,
            )
            content = regen.content
            extra_in = regen.input_tokens
            extra_out = regen.output_tokens
            extra_cost = regen.cost_usd
            regenerated = True

        await self._emit(
            EventType.CONSISTENCY_VIOLATION,
            agent_id=self.id,
            score=judgment.score,
            justification=judgment.justification,
            regenerated=regenerated,
        )
        return content, extra_in, extra_out, extra_cost

    @property
    def drift(self) -> DriftConfig:
        """The agent's drift configuration (used by the engine's reflection loop)."""
        return self._config.drift

    @property
    def initial_state(self) -> PersonaState:
        """The agent's starting persona state (fallback when no update is applied)."""
        return self._config.initial_state

    async def reflect(self, state: SessionState) -> ReflectionResult | None:
        """Round-end self-reflection: propose persona-state changes."""
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
        """Distil episodic + long-term memory."""
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
            logger.exception("%s LLM call failed for %s", purpose, self.id)
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
