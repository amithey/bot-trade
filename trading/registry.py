"""
EngineRegistry — one live bot per account, owned by the process.

A ``LiveTradingEngine`` runs on a background daemon thread and keeps trading
whether or not anyone is looking at it.  Keeping the only reference to it in
``st.session_state`` was therefore a bug waiting to happen: a browser refresh
starts a fresh session, the old reference is dropped, and the thread carries
on placing trades with nothing able to stop it — while the refreshed page
happily starts a *second* engine on the same portfolio.

The registry moves ownership to the process.  Engines are keyed by account id,
so a refresh reattaches to the bot that is already running instead of
orphaning it, and a user cannot accidentally run two.

Capacity
--------
When the registry is full it **refuses to start a new bot** rather than
evicting an existing one: evicting would mean silently stopping a stranger's
trading, and a queue-jumping user is a far better outcome than someone's
stop-loss no longer being watched.

The ceiling is ``BOTTRADE_MAX_LIVE_ENGINES`` (default 100), a process-level
guard rather than a plan limit — per-plan symbol caps live in
:mod:`saas.plans`.

The default is measured, not guessed. ``python -m tools.loadtest_engines``
on the development machine gave a sustained ~7 cycles/second of CPU-bound
work regardless of how many bots ran, and 55 KB of memory per idle engine.
At a 30-second interval that is roughly 210 bots before CPU saturates, and
memory does not become interesting until far beyond that; 100 leaves
comfortable headroom on both.

What that measurement does *not* cover, and what will actually bite first:
thread scheduling under real network load, Yahoo Finance rate-limiting a
host that fetches for a hundred symbols, and Anthropic rate limits in the
AI modes. Re-run the tool on the real host before raising this.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_ENGINES = 100


class RegistryFullError(RuntimeError):
    """Raised when no engine slot is free and none can be reclaimed."""


def max_engines() -> int:
    try:
        value = int(os.environ.get("BOTTRADE_MAX_LIVE_ENGINES", "")
                    or _DEFAULT_MAX_ENGINES)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_ENGINES
    return max(1, value)


class EngineRegistry:
    """Process-wide account → engine map."""

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}
        self._lock = threading.RLock()

    # -- lookup ------------------------------------------------------------
    def get(self, account_id: str):
        with self._lock:
            return self._engines.get(account_id)

    def accounts(self) -> list[str]:
        with self._lock:
            return list(self._engines)

    def running_count(self) -> int:
        """Engines whose loop is actually alive."""
        with self._lock:
            engines = list(self._engines.values())
        return sum(1 for e in engines if _is_running(e))

    def size(self) -> int:
        with self._lock:
            return len(self._engines)

    # -- lifecycle ---------------------------------------------------------
    def reap(self) -> int:
        """Forget engines that are no longer running. Returns how many.

        A stopped engine holds a slot but no thread, so reaping before a
        capacity check means an idle account never blocks an active one.
        """
        with self._lock:
            dead = [acct for acct, eng in self._engines.items()
                    if not _is_running(eng)]
            for acct in dead:
                self._engines.pop(acct, None)
        if dead:
            logger.debug("[registry] reaped %d idle engine(s)", len(dead))
        return len(dead)

    def get_or_create(self, account_id: str, factory: Callable[[], object]):
        """Return this account's engine, building it once.

        Capacity is only enforced for accounts that do not already have an
        engine — an existing user reattaching after a refresh is never
        refused, however full the process is.
        """
        with self._lock:
            existing = self._engines.get(account_id)
            if existing is not None:
                return existing

        self.reap()

        with self._lock:
            existing = self._engines.get(account_id)
            if existing is not None:
                return existing
            if len(self._engines) >= max_engines():
                raise RegistryFullError(
                    f"This deployment is at its limit of {max_engines()} "
                    f"concurrent live bots. Try again shortly, or raise "
                    f"BOTTRADE_MAX_LIVE_ENGINES if the host can take it."
                )
            engine = factory()
            self._engines[account_id] = engine
            logger.info("[registry] engine created for %s (%d/%d in use)",
                        account_id, len(self._engines), max_engines())
            return engine

    def stop(self, account_id: str) -> bool:
        """Stop and forget one account's engine."""
        with self._lock:
            engine = self._engines.pop(account_id, None)
        if engine is None:
            return False
        _stop(engine)
        logger.info("[registry] engine stopped for %s", account_id)
        return True

    def stop_all(self) -> int:
        with self._lock:
            engines = list(self._engines.items())
            self._engines.clear()
        for _, engine in engines:
            _stop(engine)
        return len(engines)

    def snapshot(self) -> dict:
        """Operator-facing view of what this process is running."""
        with self._lock:
            items = list(self._engines.items())
        return {
            "max":     max_engines(),
            "held":    len(items),
            "running": sum(1 for _, e in items if _is_running(e)),
            "accounts": [
                {"account": acct, "running": _is_running(eng),
                 "ticker": _attr(eng, "_ticker"),
                 "mode": _attr(eng, "_strategy_mode")}
                for acct, eng in items
            ],
        }


# --------------------------------------------------------------------------- #
# Tolerant accessors — the registry must never crash a page render
# --------------------------------------------------------------------------- #
def _is_running(engine) -> bool:
    try:
        return bool(engine.is_running())
    except Exception:                                          # noqa: BLE001
        return False


def _stop(engine) -> None:
    try:
        engine.stop()
    except Exception:                                          # noqa: BLE001
        logger.warning("[registry] engine did not stop cleanly", exc_info=True)


def _attr(engine, name: str, default: str = "") -> str:
    try:
        return str(getattr(engine, name, default) or default)
    except Exception:                                          # noqa: BLE001
        return default


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_REGISTRY: Optional[EngineRegistry] = None
_REGISTRY_GUARD = threading.Lock()


def get_registry() -> EngineRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_GUARD:
            if _REGISTRY is None:
                _REGISTRY = EngineRegistry()
    return _REGISTRY
