"""Context assembly for AI agents (DESIGN.md §10).

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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from thinktank.core.state import SessionState
from thinktank.domain.models import AgentConfig, MemoryHit, PersonaState, SessionConfig
from thinktank.llm.client import ChatMessage

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


def estimate_tokens(text: str) -> int:
    """Cheap offline token estimate (~4 chars/token); no downloads needed."""
    return max(1, len(text) // 4)


TokenCounter = Callable[[str], int]


@dataclass
class ParticipantInfo:
    """One row in the participant list: id, display name, short description."""

    id: str
    name: str
    description: str = ""


@dataclass
class ContextBudget:
    total: int = 6000
    system: int = 1500
    memories: int = 1000
    summary: int = 600


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
    ) -> list[ChatMessage]:
        base_system = self._base_system(config, agent, state, participants)
        base_cost = self.count(base_system)

        memories_text = self._memories_text(memories or [])
        summary_text = summary or ""

        working = self._working_memory(state, agent, participants)

        # Budget: ensure base fits, then size the trimmable sections.
        self._assert_fits(base_system, base_cost)
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
    def _assert_fits(self, text: str, cost: int) -> None:
        if cost > self.budget.system:
            # The core/state is non-trimmable, so a budget this small is a config
            # error; surface it rather than silently truncating identity.
            msg = (
                f"System context ({cost} tokens) exceeds the non-trimmable "
                f"system budget ({self.budget.system}). Reduce the persona or raise the budget."
            )
            raise ValueError(msg)

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
    ) -> list[ChatMessage]:
        names = {p.id: p.name for p in participants}
        window = state.messages[-agent.memory.working_window :]
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
