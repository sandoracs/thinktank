"""Human participant (DESIGN.md §8.1).

``speak`` blocks on an :class:`asyncio.Future`; the WebSocket layer (M2) resolves
it when the person types (or taps a button). After ``human_timeout_s`` the
future is abandoned and ``speak`` returns ``None``, which the engine records as
``TurnSkipped(human_timeout)``.

``observe`` is a no-op: the human's client receives messages from the event
bus, not through the participant (DESIGN.md §8.1).
"""

from __future__ import annotations

import asyncio
from typing import Literal

from roundtable.domain.events import new_message_id
from roundtable.domain.models import Message
from roundtable.participants.base import TurnContext


class HumanParticipant:
    kind: Literal["ai", "human", "remote"] = "human"

    def __init__(
        self,
        name: str,
        *,
        human_timeout_s: float = 60.0,
        is_moderator: bool = False,
    ) -> None:
        self.id = name
        self.display_name = name
        self._timeout_s = human_timeout_s
        self._is_moderator = is_moderator
        self._pending: asyncio.Future[str] | None = None
        self._hand_raised = False

    @property
    def hand_raised(self) -> bool:
        return self._hand_raised

    async def speak(self, ctx: TurnContext) -> Message | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending = future
        timed_out = False
        try:
            content = await asyncio.wait_for(future, timeout=self._timeout_s)
        except TimeoutError:
            timed_out = True
            content = ""
        finally:
            self._pending = None

        if timed_out:
            return None

        # Speaking lowers a raised hand (the engine also does this from the
        # event stream; the participant lowers it for its own bookkeeping).
        self._hand_raised = False
        return Message(
            id=new_message_id(),
            session_id=ctx.session_id,
            seq=0,
            speaker_id=self.id,
            kind="moderator" if self._is_moderator else "speech",
            content=content,
        )

    # -- driven by the WebSocket layer -------------------------------------
    def submit(self, content: str) -> None:
        """Resolve the in-flight turn with a typed message (no-op if none)."""
        if self._pending is not None and not self._pending.done():
            self._pending.set_result(content)

    def raise_hand(self) -> None:
        self._hand_raised = True

    def lower_hand(self) -> None:
        self._hand_raised = False

    async def observe(self, msg: Message) -> None:
        return None

    async def on_session_end(self) -> None:
        return None

