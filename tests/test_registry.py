"""
Offline tests for engine ownership and portfolio persistence.

The two bugs these guard against are the ones a browser refresh used to
cause: a background trading thread nobody holds a reference to any more, and
a portfolio that silently reverts to empty while that thread keeps trading.
"""
from __future__ import annotations

import json

import pytest

from portfolio.virtual_account import LivePortfolio
from trading.registry import EngineRegistry, RegistryFullError, max_engines


class _FakeEngine:
    """Stands in for LiveTradingEngine — only the bits the registry touches."""

    def __init__(self, running: bool = True, ticker: str = "BTC-USD"):
        self._running = running
        self.stop_calls = 0
        self._ticker = ticker
        self._strategy_mode = "COMMITTEE"

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self.stop_calls += 1
        self._running = False


@pytest.fixture()
def registry() -> EngineRegistry:
    return EngineRegistry()


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #
def test_one_engine_per_account(registry):
    built = []

    def factory():
        built.append(1)
        return _FakeEngine()

    a = registry.get_or_create("user:a", factory)
    b = registry.get_or_create("user:a", factory)
    assert a is b
    assert len(built) == 1, "a refresh must reattach, not start a second bot"


def test_different_accounts_get_different_engines(registry):
    a = registry.get_or_create("user:a", _FakeEngine)
    b = registry.get_or_create("user:b", _FakeEngine)
    assert a is not b
    assert registry.size() == 2


def test_get_does_not_create(registry):
    assert registry.get("user:nobody") is None
    assert registry.size() == 0


def test_stop_removes_and_stops(registry):
    eng = registry.get_or_create("user:a", _FakeEngine)
    assert registry.stop("user:a") is True
    assert eng.stop_calls == 1
    assert registry.get("user:a") is None
    assert registry.stop("user:a") is False


def test_stop_all_stops_everyone(registry):
    engines = [registry.get_or_create(f"user:{i}", _FakeEngine) for i in range(4)]
    assert registry.stop_all() == 4
    assert all(e.stop_calls == 1 for e in engines)
    assert registry.size() == 0


# --------------------------------------------------------------------------- #
# Capacity
# --------------------------------------------------------------------------- #
def test_cap_refuses_rather_than_evicting(registry, monkeypatch):
    """Evicting would silently stop a stranger's trading. Refuse instead."""
    monkeypatch.setenv("BOTTRADE_MAX_LIVE_ENGINES", "2")
    first = registry.get_or_create("user:a", _FakeEngine)
    registry.get_or_create("user:b", _FakeEngine)

    with pytest.raises(RegistryFullError):
        registry.get_or_create("user:c", _FakeEngine)

    assert first.stop_calls == 0, "an existing bot must never be evicted"
    assert registry.get("user:a") is first


def test_existing_account_reattaches_even_when_full(registry, monkeypatch):
    monkeypatch.setenv("BOTTRADE_MAX_LIVE_ENGINES", "1")
    eng = registry.get_or_create("user:a", _FakeEngine)
    assert registry.get_or_create("user:a", _FakeEngine) is eng


def test_idle_engines_are_reaped_to_free_capacity(registry, monkeypatch):
    monkeypatch.setenv("BOTTRADE_MAX_LIVE_ENGINES", "2")
    idle = registry.get_or_create("user:a", lambda: _FakeEngine(running=False))
    registry.get_or_create("user:b", _FakeEngine)

    # user:a stopped trading, so its slot should not block a new user.
    third = registry.get_or_create("user:c", _FakeEngine)
    assert third is not None
    assert registry.get("user:a") is None
    assert idle.stop_calls == 0, "reaping an already-stopped engine is not a stop"


def test_running_count_ignores_stopped_engines(registry):
    registry.get_or_create("user:a", _FakeEngine)
    registry.get_or_create("user:b", lambda: _FakeEngine(running=False))
    assert registry.running_count() == 1
    assert registry.size() == 2


def test_max_engines_reads_env(monkeypatch):
    monkeypatch.setenv("BOTTRADE_MAX_LIVE_ENGINES", "7")
    assert max_engines() == 7
    monkeypatch.setenv("BOTTRADE_MAX_LIVE_ENGINES", "not-a-number")
    from trading.registry import _DEFAULT_MAX_ENGINES
    assert max_engines() == _DEFAULT_MAX_ENGINES
    monkeypatch.setenv("BOTTRADE_MAX_LIVE_ENGINES", "0")
    assert max_engines() == 1


# --------------------------------------------------------------------------- #
# Robustness — a broken engine must not break a page render
# --------------------------------------------------------------------------- #
def test_engine_that_raises_on_is_running_counts_as_stopped(registry):
    class _Broken:
        def is_running(self):
            raise RuntimeError("thread state unreadable")

    registry.get_or_create("user:a", _Broken)
    assert registry.running_count() == 0
    assert registry.reap() == 1


def test_engine_that_raises_on_stop_is_still_forgotten(registry):
    class _Stubborn:
        def is_running(self):
            return True

        def stop(self):
            raise RuntimeError("thread already gone")

    registry.get_or_create("user:a", _Stubborn)
    assert registry.stop("user:a") is True
    assert registry.get("user:a") is None


def test_snapshot_reports_what_is_running(registry):
    registry.get_or_create("user:a", lambda: _FakeEngine(ticker="ETH-USD"))
    snap = registry.snapshot()
    assert snap["held"] == 1 and snap["running"] == 1
    assert snap["accounts"][0]["ticker"] == "ETH-USD"


# --------------------------------------------------------------------------- #
# Portfolio persistence
# --------------------------------------------------------------------------- #
def test_portfolio_survives_a_round_trip(tmp_path):
    port = LivePortfolio(initial_capital=10_000)
    port.update_price("BTC-USD", 50_000)
    port.buy("BTC-USD", 50_000, 0.1, reasoning="test")
    path = tmp_path / "p.json"
    port.save(path)

    restored = LivePortfolio.load(path)
    assert restored.cash == pytest.approx(port.cash)
    assert "BTC-USD" in restored.positions
    assert len(restored.trade_log) == len(port.trade_log)


def test_save_is_atomic_leaving_no_temp_file(tmp_path):
    port = LivePortfolio(initial_capital=5_000)
    path = tmp_path / "p.json"
    port.save(path)
    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(path.read_text(encoding="utf-8"))["initial_capital"] == 5_000


def test_engine_checkpoints_the_portfolio_after_every_cycle(tmp_path):
    """The engine must persist through its own hook, not rely on the UI."""
    from trading.live_engine import LiveTradingEngine

    port = LivePortfolio(initial_capital=1_000)
    eng = LiveTradingEngine(portfolio=port, fetcher=None, retriever=None,
                            engine=None)
    saved = []
    eng.set_persist_callback(lambda p: saved.append(p))
    eng._checkpoint()
    assert saved == [port]


def test_a_failing_checkpoint_does_not_kill_the_bot(tmp_path):
    from trading.live_engine import LiveTradingEngine

    eng = LiveTradingEngine(portfolio=LivePortfolio(initial_capital=1_000),
                            fetcher=None, retriever=None, engine=None)

    def _explode(_p):
        raise OSError("disk full")

    eng.set_persist_callback(_explode)
    eng._checkpoint()          # must not raise
    assert any("save portfolio" in e.message.lower() for e in eng.history())


def test_no_callback_means_no_persistence(tmp_path):
    from trading.live_engine import LiveTradingEngine

    eng = LiveTradingEngine(portfolio=LivePortfolio(initial_capital=1_000),
                            fetcher=None, retriever=None, engine=None)
    eng._checkpoint()          # single-user mode: silently does nothing
