"""Context assembly for AI agents.

Builds the exact message sequence the design prescribes:

1. **System** — persona core (re-rendered every call), behavioural rules,
   participant list, topic, debate questions, current persona state.
2. **Memories** — retrieved episodic / long-term items (trimmable).
3. **Own summary** — episodic summary of the earlier section (trimmable).
4. **Working memory** — last N messages; others as ``user`` with a ``[Name]:``
   prefix, self as ``assistant``; consecutive same-role messages merged.
5. **Turn instruction** — final ``user`` message.

A per-section token budget is enforced with the trimming order
``memories -> working (oldest first) -> summary``; the core and current state
are never trimmed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from thinktank.core.language import language_name
from thinktank.core.state import SessionState
from thinktank.domain.models import AgentConfig, MemoryHit, Message, PersonaState, SessionConfig
from thinktank.llm.client import ChatMessage

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


def estimate_tokens(text: str) -> int:
    """Cheap offline token estimate; no downloads needed.

    ~4 chars/token holds for English, but Hungarian and other morphologically
    rich languages tokenise denser, so 3 chars/token is used to stay on the
    safe side of the real budget.
    """
    return max(1, len(text) // 3)


TokenCounter = Callable[[str], int]


@dataclass
class ParticipantInfo:
    """One row in the participant list: id, display name, short description."""

    id: str
    name: str
    description: str = ""


@dataclass
class ContextBudget:
    total: int = 16000
    system: int = 3000
    memories: int = 1500
    summary: int = 1500

    @classmethod
    def from_settings(cls, settings: object) -> ContextBudget:
        """Build from a ``Settings`` object (missing fields keep the defaults)."""
        default = cls()
        return cls(
            total=int(getattr(settings, "context_total_tokens", default.total)),
            system=int(getattr(settings, "context_system_tokens", default.system)),
            memories=int(getattr(settings, "context_memory_tokens", default.memories)),
            summary=int(getattr(settings, "context_summary_tokens", default.summary)),
        )


class ContextBuilder:
    """Compose and budget the message list for one agent turn."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.count = token_counter or estimate_tokens
        self._env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            autoescape=False,  # noqa: S701 -- LLM prompt text, not HTML
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # -- public ------------------------------------------------------------
    def build(
        self,
        *,
        config: SessionConfig,
        agent: AgentConfig,
        state: SessionState,
        participants: list[ParticipantInfo],
        memories: list[MemoryHit] | None = None,
        summary: str | None = None,
        turn_instruction: str | None = None,
        working_messages: list[Message] | None = None,
    ) -> list[ChatMessage]:
        """Assemble the turn context.

        ``working_messages`` overrides the default working-memory slice (the last
        ``memory.working_window`` messages); the agent passes it when a rolling
        summary already covers everything older than that list.
        """
        base_system = self._base_system(config, agent, state, participants)
        base_cost = self.count(base_system)

        memories_text = self._memories_text(memories or [])
        summary_text = summary or ""

        working = self._working_memory(state, agent, participants, working_messages)

        # Budget: ensure base fits, then size the trimmable sections.
        self._check_system_budget(agent.id, base_cost)
        memories_text = self._trim_to_budget(memories_text, self.budget.memories)
        summary_text = self._trim_to_budget(summary_text, self.budget.summary)

        system = self._assemble_system(base_system, memories_text, summary_text)

        remainder = self.budget.total - self.count(system)
        working = self._fit_working(working, remainder)

        messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]
        messages.extend(working)
        messages.append(self._turn_instruction(turn_instruction))
        return messages

    # -- internals ---------------------------------------------------------
    def _check_system_budget(self, agent_id: str, cost: int) -> None:
        # The core/state is never trimmed. Going over the soft cap used to raise,
        # which the engine counted as a failed turn and, three turns later,
        # disabled the agent for the rest of the session. Warn instead: the
        # working memory shrinks to make room (see _fit_working).
        if cost > self.budget.system:
            logger.warning(
                "System context for %s is ~%d tokens, over the %d soft cap; "
                "working memory will be trimmed to fit. Shorten the persona or raise "
                "THINKTANK_CONTEXT_SYSTEM_TOKENS.",
                agent_id,
                cost,
                self.budget.system,
            )

    def _base_system(
        self,
        config: SessionConfig,
        agent: AgentConfig,
        state: SessionState,
        participants: list[ParticipantInfo],
    ) -> str:
        template = self._env.get_template("system_prompt.j2")
        participants_text = "\n".join(
            f"- {p.name}" + (f": {p.description}" if p.description else "") for p in participants
        ) or "(you are the only participant)"
        return template.render(
            persona_name=agent.persona.name,
            language=language_name(config.language),
            persona=agent.persona.render(),
            questions=[q.model_dump() for q in config.questions],
            topic=config.topic,
            participants=participants_text,
            state_block=self._state_block(self._agent_state(state, agent)),
        )

    def _agent_state(self, state: SessionState, agent: AgentConfig) -> PersonaState:
        """The agent's live persona state (M4: per-agent updates from reflections)."""
        if agent.id in state.persona_states:
            return state.persona_states[agent.id]
        return agent.initial_state

    def _state_block(self, st: PersonaState) -> str:
        lines: list[str] = []
        if st.stances:
            lines.append("Stances:")
            for qid, stance in st.stances.items():
                lines.append(f"  - {qid}: {stance.position} (confidence {stance.confidence:.2f})")
        if st.attitudes:
            lines.append("Attitudes toward others:")
            for pid, value in st.attitudes.items():
                lines.append(f"  - {pid}: {value:+.2f}")
        if st.mood:
            lines.append(f"Mood: {st.mood}")
        return "\n".join(lines) or "(none recorded yet)"

    def _memories_text(self, hits: list[MemoryHit]) -> str:
        if not hits:
            return ""
        header = "Relevant memories:"
        body = "\n".join(f"- {h.content}" for h in hits)
        return f"{header}\n{body}"

    def _working_memory(
        self,
        state: SessionState,
        agent: AgentConfig,
        participants: list[ParticipantInfo],
        working_messages: list[Message] | None = None,
    ) -> list[ChatMessage]:
        names = {p.id: p.name for p in participants}
        window = (
            working_messages
            if working_messages is not None
            else state.messages[-agent.memory.working_window :]
        )
        raw: list[ChatMessage] = []
        for msg in window:
            if msg.speaker_id == agent.id:
                raw.append(ChatMessage(role="assistant", content=msg.content))
            else:
                name = names.get(msg.speaker_id, msg.speaker_id)
                raw.append(ChatMessage(role="user", content=f"[{name}]: {msg.content}"))
        return _merge_consecutive(raw)

    def _assemble_system(self, base: str, memories: str, summary: str) -> str:
        parts = [base.strip()]
        if memories.strip():
            parts.append(memories.strip())
        if summary.strip():
            parts.append(f"Your summary of the discussion so far:\n{summary.strip()}")
        return "\n\n".join(parts)

    def _fit_working(self, working: list[ChatMessage], remainder: int) -> list[ChatMessage]:
        if not working:
            return working
        # A positive remainder always keeps at least the most recent message for
        # continuity; a remainder that is already <= 0 (system+memories+summary
        # alone blew the budget) has no such allowance left to protect.
        budget = max(remainder, 0)
        min_keep = 1 if remainder > 0 else 0
        cost = sum(self.count(m.content) for m in working)
        while cost > budget and len(working) > min_keep:
            dropped = working.pop(0)  # oldest first
            cost -= self.count(dropped.content)
        return working

    def _trim_to_budget(self, text: str, budget: int) -> str:
        if not text or budget <= 0:
            return ""
        if self.count(text) <= budget:
            return text
        # Truncate on a word boundary to fit the section budget.
        words = text.split()
        kept: list[str] = []
        for word in words:
            if self.count(" ".join([*kept, word])) > budget and kept:
                break
            kept.append(word)
        return " ".join(kept) + "…" if len(kept) < len(words) else text

    def _turn_instruction(self, instruction: str | None) -> ChatMessage:
        template = self._env.get_template("turn_instruction.j2")
        return ChatMessage(role="user", content=template.render(instruction=instruction).strip())


def _merge_consecutive(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Merge runs of consecutive same-role messages into a single message."""
    if not messages:
        return []
    out: list[ChatMessage] = [messages[0]]
    for message in messages[1:]:
        prev = out[-1]
        if prev.role == message.role and prev.role in ("user", "assistant"):
            out[-1] = ChatMessage(role=prev.role, content=f"{prev.content}\n{message.content}")
        else:
            out.append(message)
    return out
