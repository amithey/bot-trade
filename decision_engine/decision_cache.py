"""
SharedDecisionCache — one Claude call per market state, not per user.

Why this exists
---------------
An AI verdict on ``BTC-USD`` for the 14:30 five-minute bar is *identical*
for every user looking at that bar.  Running the decision loop per-user
makes the API bill scale with headcount; running it per **market state**
makes it scale with the number of symbols under watch — a number the
operator controls.

Two mechanisms:

* **Memoisation** — a decision is stored under a key describing the market
  state that produced it (ticker + interval + bar timestamp + strategy
  mode + the few user-side inputs that genuinely change the answer).
  Repeat lookups inside the bar's TTL return the stored verdict.
* **Single-flight** — when N threads miss the same key simultaneously, one
  computes and the rest block on that result.  A cold start with 200
  concurrent users costs one API call, not 200.

The cache is process-wide and thread-safe.  It stores plain objects, so it
works for ``TradingDecision``, boardroom rulings, or anything else.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------- #
# Key construction
# --------------------------------------------------------------------------- #
def position_bucket(in_position: bool, pnl_pct: Optional[float]) -> str:
    """Coarsen a user's position state so near-identical users share a call.

    A flat user and another flat user always share.  Two users long the same
    ticker share whenever their unrealised P&L rounds into the same 0.5%
    bucket — close enough that the risk analyst's read does not change.
    """
    if not in_position:
        return "flat"
    if pnl_pct is None:
        return "long"
    return f"long{round(float(pnl_pct) * 2) / 2:+.1f}"


def pnl_bucket(daily_pnl_pct: Optional[float]) -> str:
    """Round daily P&L to whole percent — the granularity the CRO reacts to."""
    if daily_pnl_pct is None:
        return "na"
    return f"{round(float(daily_pnl_pct)):+d}"


def make_key(**parts: Any) -> str:
    """Build a stable cache key from keyword parts.

    Values are JSON-serialised with sorted keys so ordering never matters,
    then hashed — keys stay short regardless of how many parts are supplied.
    """
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
@dataclass
class _Entry:
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)
    hits: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    coalesced: int = 0     # threads that waited on someone else's in-flight call
    errors: int = 0
    evictions: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses + self.coalesced

    @property
    def calls_saved(self) -> int:
        """Upstream calls avoided — every lookup that wasn't a miss."""
        return self.hits + self.coalesced

    @property
    def hit_rate(self) -> float:
        return (self.calls_saved / self.lookups) if self.lookups else 0.0

    def to_dict(self) -> dict:
        return {
            "lookups":     self.lookups,
            "hits":        self.hits,
            "misses":      self.misses,
            "coalesced":   self.coalesced,
            "errors":      self.errors,
            "evictions":   self.evictions,
            "calls_saved": self.calls_saved,
            "hit_rate":    round(self.hit_rate, 4),
        }


class SharedDecisionCache:
    """TTL cache with per-key single-flight.

    Parameters
    ----------
    default_ttl:
        Seconds a computed value stays fresh.  Should sit a little under the
        bar interval so a new bar always recomputes.
    max_entries:
        Soft cap; the oldest entries are dropped once exceeded.
    """

    def __init__(self, default_ttl: float = 240.0, max_entries: int = 2048) -> None:
        self._default_ttl = float(default_ttl)
        self._max_entries = int(max_entries)
        self._store: dict[str, _Entry] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.stats = CacheStats()

    # -- internals ---------------------------------------------------------
    def _key_lock(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _peek(self, key: str) -> Optional[_Entry]:
        with self._guard:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.time():
                self._store.pop(key, None)
                self.stats.evictions += 1
                return None
            entry.hits += 1
            return entry

    def _prune_locked(self) -> None:
        """Drop expired entries, then oldest-first until under the cap.

        Caller must hold ``self._guard``.
        """
        now = time.time()
        for k in [k for k, e in self._store.items() if e.expires_at <= now]:
            self._store.pop(k, None)
            self._locks.pop(k, None)
            self.stats.evictions += 1
        overflow = len(self._store) - self._max_entries
        if overflow > 0:
            oldest = sorted(self._store.items(), key=lambda kv: kv[1].created_at)
            for k, _ in oldest[:overflow]:
                self._store.pop(k, None)
                self._locks.pop(k, None)
                self.stats.evictions += 1

    # -- public API --------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """Return a fresh value, or ``None``.  Does not count as a lookup."""
        entry = self._peek(key)
        return entry.value if entry is not None else None

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Any],
        ttl: Optional[float] = None,
    ) -> tuple[Any, bool]:
        """Return ``(value, was_cached)``, computing at most once per key.

        ``compute`` runs while this thread holds the key's lock; concurrent
        callers for the same key block and then read the stored result.
        Exceptions propagate to the calling thread and nothing is cached, so
        a transient API failure does not poison the entry.
        """
        entry = self._peek(key)
        if entry is not None:
            self.stats.hits += 1
            return entry.value, True

        lock = self._key_lock(key)
        contended = not lock.acquire(blocking=False)
        if contended:
            lock.acquire()          # someone else is computing - wait for them
        try:
            # Re-check: the thread we waited on has likely just filled it.
            entry = self._peek(key)
            if entry is not None:
                if contended:
                    self.stats.coalesced += 1
                else:
                    self.stats.hits += 1
                return entry.value, True

            self.stats.misses += 1
            try:
                value = compute()
            except Exception:
                self.stats.errors += 1
                raise

            with self._guard:
                self._store[key] = _Entry(
                    value=value,
                    expires_at=time.time() + float(ttl or self._default_ttl),
                )
                self._prune_locked()
            return value, False
        finally:
            lock.release()

    def invalidate(self, key: str) -> None:
        with self._guard:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._guard:
            self._store.clear()
            self._locks.clear()

    def size(self) -> int:
        with self._guard:
            return len(self._store)


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_GLOBAL: Optional[SharedDecisionCache] = None
_GLOBAL_GUARD = threading.Lock()


def get_decision_cache() -> SharedDecisionCache:
    """The cache every LiveTradingEngine in this process shares."""
    global _GLOBAL
    if _GLOBAL is None:
        with _GLOBAL_GUARD:
            if _GLOBAL is None:
                _GLOBAL = SharedDecisionCache()
    return _GLOBAL
