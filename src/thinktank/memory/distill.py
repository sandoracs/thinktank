"""Memory distillation prompts.

Two side calls at session end, both with ``purpose="summary"``:

1. **Episodic summary** — the agent's own view of what happened in *this*
   session (positions taken, key exchanges, what changed for it). Stored
   per session (``session_id`` set).
2. **Long-term distillation** — from that summary, the durable lessons the
   agent should carry into *future* sessions. Stored cross-session
   (``session_id`` NULL). An empty answer is a valid outcome ("nothing worth
   keeping") and stores nothing.
"""

from __future__ import annotations

from thinktank.llm.client import ChatMessage

EPISODIC_SYSTEM = (
    "You are maintaining the personal episodic memory of an AI debate participant. "
    "Write a concise first-person summary of the session for that participant: "
    "its positions on the debate questions, the strongest exchanges it had, "
    "what it conceded or pushed back on, and how its view developed. "
    "Do not invent content that is not in the transcript. Output plain text only."
)

LONG_TERM_SYSTEM = (
    "You are distilling long-term memory for an AI debate participant. "
    "Given its episodic summary of a session, extract at most three durable, "
    "reusable lessons or insights it should apply in future debates on related "
    "topics (e.g. about the topic itself, or about how arguments in this domain "
    "tend to play out). Be specific, avoid generic advice, and do not invent "
    "claims the summary does not support. If there is nothing durable worth "
    "keeping, output exactly: NO_LESSON"
)


def episodic_messages(
    *,
    agent_name: str,
    topic: str,
    transcript: str,
) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=EPISODIC_SYSTEM),
        ChatMessage(
            role="user",
            content=(
                f"Participant: {agent_name}\n"
                f"Session topic: {topic}\n"
                f"Transcript:\n{transcript}"
            ),
        ),
    ]


def long_term_messages(
    *,
    agent_name: str,
    topic: str,
    episodic_summary: str,
) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=LONG_TERM_SYSTEM),
        ChatMessage(
            role="user",
            content=(
                f"Participant: {agent_name}\n"
                f"Session topic: {topic}\n"
                f"Episodic summary:\n{episodic_summary}"
            ),
        ),
    ]


def has_lesson(text: str) -> bool:
    """True unless the distiller signalled that nothing is worth keeping."""
    stripped = text.strip()
    return bool(stripped) and stripped.upper() != "NO_LESSON"
