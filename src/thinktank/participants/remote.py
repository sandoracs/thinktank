"""Remote participant: an external agent that joins the table over HTTP.

DESIGN.md §8.1 / M7. A remote agent is any service (AG2, LangGraph, a custom
bot, ...) that speaks the same protocol the design fixes:

    POST {url}/speak        body: context JSON   -> {"content": "..."}
    POST {url}/observe      body: message JSON   -> {}
    POST {url}/session_end  body: {"session_id": "..."} -> {}

``speak`` maps a non-2xx response or a missing ``content`` to an exception, so
the engine's standard error path (``Error`` + ``TurnSkipped``) applies and a
flaky remote never wedges the debate. ``observe`` / ``session_end`` are
best-effort: a failure is logged, never raised.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from typing import Literal

import httpx

from thinktank.core.state import SessionState
from thinktank.domain.events import new_message_id
from thinktank.domain.models import DriftConfig, DriftMode, Message, PersonaState, RemoteConfig
from thinktank.participants.base import TurnContext
from thinktank.persona.reflection import ReflectionResult

logger = logging.getLogger(__name__)


class RemoteAgent:
    """A participant whose brain lives in an external HTTP service."""

    kind: Literal["ai", "human", "remote"] = "remote"

    def __init__(
        self,
        config: RemoteConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.id = config.id
        self.display_name = config.display_name or config.id
        self._config = config
        self._injected_client = client
        self._session_id: uuid.UUID | None = None

    # -- Participant protocol ------------------------------------------------
    async def speak(self, ctx: TurnContext) -> Message | None:
        self._session_id = ctx.session_id
        payload = {
            "session_id": str(ctx.session_id),
            "round": ctx.round,
            "turn_instruction": ctx.turn_instruction,
        }
        response = await self._post("speak", payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("content") if isinstance(data, dict) else None
        if not content:
            msg = f"Remote agent {self.id} returned no content"
            raise ValueError(msg)
        return Message(
            id=new_message_id(),
            session_id=ctx.session_id,
            seq=0,  # stamped with the event seq during projection
            speaker_id=self.id,
            kind="speech",
            content=str(content),
            meta={"remote_url": self._config.url},
        )

    async def observe(self, msg: Message) -> None:
        self._session_id = msg.session_id
        try:
            await self._post("observe", {"session_id": str(msg.session_id), "message": msg.model_dump(mode="json")})
        except Exception:
            logger.warning("remote observe failed for %s", self.id, exc_info=True)

    async def on_session_end(self) -> None:
        if self._session_id is None:
            return
        try:
            await self._post("session_end", {"session_id": str(self._session_id)})
        except Exception:
            logger.warning("remote session_end failed for %s", self.id, exc_info=True)

    async def reflect(self, state: SessionState) -> ReflectionResult | None:
        # Remote agents own their own state; they do not propose local drift.
        return None

    # -- persona surface (so the engine's uniform interface holds) -----------
    @property
    def drift(self) -> DriftConfig:
        return DriftConfig(mode=DriftMode.LOCKED)

    @property
    def initial_state(self) -> PersonaState:
        return PersonaState()

    # -- transport -----------------------------------------------------------
    async def _post(self, path: str, payload: Mapping[str, object]) -> httpx.Response:
        url = self._config.url.rstrip("/") + "/" + path
        if self._injected_client is not None:
            return await self._injected_client.post(
                url, json=payload, timeout=self._config.timeout_s, headers=self._config.headers
            )
        async with httpx.AsyncClient(timeout=self._config.timeout_s, headers=self._config.headers) as client:
            return await client.post(url, json=payload)
