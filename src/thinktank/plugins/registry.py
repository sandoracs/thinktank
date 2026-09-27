"""Plugin registry.

Built-ins and external plugins are loaded through the *same* mechanism:
``importlib.metadata`` entry points grouped by extension surface. A broken
plugin is logged and skipped so the hub still starts.

The registry is the single place that maps a strategy name to its class and
instantiates it from a :class:`StrategyRef`, which is what lets a new strategy
appear in the UI and be usable with zero core changes.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from thinktank.domain.models import StrategyRef
from thinktank.llm.client import LLMClient
from thinktank.strategies.base import TurnStrategy

logger = logging.getLogger(__name__)

def load_strategies() -> dict[str, type[TurnStrategy]]:
    """Return ``{name: class}`` for every loadable turn strategy."""
    out: dict[str, type[TurnStrategy]] = {}
    try:
        eps = entry_points(group="thinktank.turn_strategies")
    except Exception as exc:  # pragma: no cover - importlib quirks
        logger.warning("entry-point lookup failed: %s", exc)
        return out
    for ep in eps:
        try:
            cls = ep.load()
        except Exception as exc:
            logger.warning("turn strategy plugin %s failed to load: %s", ep.name, exc)
            continue
        if not (isinstance(cls, type) and issubclass(cls, TurnStrategy) and cls.name):
            logger.warning("ignoring invalid turn strategy plugin %s", ep.name)
            continue
        out[cls.name] = cls
    return out


def build_strategy(ref: StrategyRef, llm: LLMClient | None = None) -> TurnStrategy:
    """Instantiate the strategy named by ``ref`` (validates against the registry).

    ``llm`` is offered to strategies that want an LLM gateway (e.g. bidding).
    Strategies without an ``llm`` parameter are constructed without it, so
    external plugins that predate the argument keep working.
    """
    strategies = load_strategies()
    cls = strategies.get(ref.name)
    if cls is None:
        available = ", ".join(sorted(strategies)) or "none"
        msg = f"Unknown turn strategy {ref.name!r}; available: {available}"
        raise ValueError(msg)
    if llm is not None:
        try:
            return cls(params=ref.params, llm=llm)
        except TypeError:
            pass  # strategy does not accept an llm argument
    return cls(params=ref.params)
