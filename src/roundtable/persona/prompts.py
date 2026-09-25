"""Reflection prompt (DESIGN.md §12.2).

The model may only propose changes to :class:`PersonaState` (stances,
attitudes, mood) — never to the frozen core. The prompt explicitly permits
the empty answer ("nothing changed") so that a round without pressure does
not manufacture artificial convergence.
"""

from __future__ import annotations

from roundtable.core.state import SessionState
from roundtable.domain.models import PersonaState, SessionConfig
from roundtable.llm.client import ChatMessage
from roundtable.persona.reflection import ReflectionResult

REFLECTION_SYSTEM = (
    "You are performing a private round-end reflection for an AI debate participant. "
    "Review the discussion so far and decide whether your own state has genuinely "
    "changed. You may update: (1) your stance on each debate question, with a "
    "confidence in [0, 1] and a short reason; (2) your attitude toward a specific "
    "participant, in [-1, 1], with a short reason; (3) your mood, a short phrase. "
    "Only propose what the discussion actually caused — if nothing changed, return "
    "an empty object. Do not restate your core values or boundaries; they are frozen."
)


def reflection_messages(
    *,
    agent_name: str,
    config: SessionConfig,
    state: SessionState,
    persona_state: PersonaState,
) -> tuple[list[ChatMessage], type[ReflectionResult]]:
    """Build the reflection prompt; the answer is parsed as :class:`ReflectionResult`."""
    questions = "\n".join(f"- [{q.id}] {q.text}" for q in config.questions) or "(none)"
    transcript = "\n".join(f"{m.speaker_id}: {m.content}" for m in state.messages[-24:])
    current = (
        f"Stances: {persona_state.stances or '(none)'}\n"
        f"Attitudes: {persona_state.attitudes or '(none)'}\n"
        f"Mood: {persona_state.mood}"
    )
    user = (
        f"Participant: {agent_name}\n"
        f"Topic: {config.topic}\n"
        f"Debate questions:\n{questions}\n"
        f"Your current state:\n{current}\n"
        f"Recent discussion:\n{transcript}"
    )
    return (
        [
            ChatMessage(role="system", content=REFLECTION_SYSTEM),
            ChatMessage(role="user", content=user),
        ],
        ReflectionResult,
    )
