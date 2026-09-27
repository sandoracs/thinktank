"""In-process event bus (DESIGN.md §3).

Two responsibilities, kept deliberately separate:
- **Durability:** every :meth:`emit` appends through the event store, which
  assigns the per-session ``seq`` under a transaction and persists the row.
  This is the single writer, so sequence numbers are total-ordered per session.
- **Liveness:** subscribers (the WebSocket layer in M2) are notified in-process
  with the stored event, so a connected client sees each fact exactly once in
  order.

The bus is the seam that later forwards to WebSockets; it never talks to the
network itself.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from thinktank.domain.events import Event

logger = logging.getLogger(__name__)
# Append signature: persist ``event`` for ``session_id`` and return it with a real seq.
AppendFn = Callable[[uuid.UUID, Event], Awaitable[Event]]
Subscriber = Callable[[Event], Awaitable[None]]
# Participant-facing emit: a session-bound closure ``(etype, **payload) -> Event``
# that persists, applies to live state, and feeds the strategy (see engine).
EmitFn = Callable[..., Awaitable[Event]]


def _is_coroutine(obj: object) -> bool:
    return asyncio.iscoroutine(obj)


@dataclass
class EventBus:
    """Append-and-notify event bus, one instance per session (or process-wide)."""

    append: AppendFn
    _subscribers: dict[uuid.UUID, list[Subscriber]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def emit(self, session_id: uuid.UUID, event: Event) -> Event:
        """Persist ``event`` (assigning ``seq``), then notify subscribers."""
        stored = await self._append_with_seq(session_id, event)
        await self._notify(session_id, stored)
        return stored

    async def _append_with_seq(self, session_id: uuid.UUID, event: Event) -> Event:
        async with self._lock:
            return await self.append(session_id, event)

    def subscribe(self, session_id: uuid.UUID, subscriber: Subscriber) -> Callable[[], None]:
        """Register ``subscriber``; returns an unsubscribe callable."""
        self._subscribers.setdefault(session_id, []).append(subscriber)

        def _unsubscribe() -> None:
            subs = self._subscribers.get(session_id)
            if subs and subscriber in subs:
                subs.remove(subscriber)

        return _unsubscribe

    def subscriber_count(self, session_id: uuid.UUID) -> int:
        return len(self._subscribers.get(session_id, ()))

    async def _notify(self, session_id: uuid.UUID, event: Event) -> None:
        for subscriber in list(self._subscribers.get(session_id, ())):
            try:
                result = subscriber(event)
                if _is_coroutine(result):
                    await result
            except Exception:
                # A misbehaving subscriber must never break the event stream.
                logger.exception("event subscriber failed for seq=%s", event.seq)
