import threading
import time
from dataclasses import replace
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from market_data.protection_quotes import ProtectionQuote
from portfolio.virtual_account import LivePortfolio
from strategy.committee import CommitteeVerdict
from tests.test_entry_research import bars
from tests.test_entry_research import live as _live_fixture
from trading.live_engine import LiveTradingEngine

live = _live_fixture


def quote(price):
    now = time.time()
    return ProtectionQuote(price, now, now)


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for the local monitor")


def test_monitor_closes_while_ai_is_blocked_and_discards_late_buy(live, monkeypatch, tmp_path):
    # Load optional model libraries before timing the concurrency assertions.
    import decision_engine.decision_cache  # noqa: F401

    eng, snap = live(bars())
    price = float(snap.data.Close.iloc[-1])
    eng.portfolio.buy("BTC-USD", price, cash_amount=1000)
    entered, release = threading.Event(), threading.Event()
    market = {"price": price}
    eng._strategy_mode = "AI"
    eng._protection_poll_seconds = .02
    eng._protection_quote_provider = lambda ticker: quote(market["price"])
    eng._retriever = NS(get_relevant_strategies=lambda *a, **k: NS(chunks=[]))
    def model(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return CommitteeVerdict("BUY", .5, 28, 9, 1, 38, True).to_trading_decision("BTC-USD")
    monkeypatch.setattr(eng, "_ai_engine", lambda: NS(evaluate_market=model))
    monkeypatch.setattr(eng, "_shared_decision", lambda *a, **k: k["compute"]())
    saved = tmp_path / "isolated.json"
    eng.set_persist_callback(lambda port: port.save(saved))
    try:
        eng.start()
        assert entered.wait(3)
        market["price"] = price * 1.01
        wait_for(lambda: eng.portfolio.positions["BTC-USD"].profit_protection_armed)
        # State is durable even though the analysis thread has not returned.
        wait_for(lambda: saved.exists() and LivePortfolio.load(saved).positions["BTC-USD"].profit_protection_armed)
        market["price"] = price * 1.004
        wait_for(lambda: not eng.portfolio.positions)
        assert not release.is_set()
        assert eng.portfolio.trade_log[-1].realized_pnl > 0
        assert "Profit protection" in eng.portfolio.trade_log[-1].reasoning
        release.set()
        wait_for(lambda: any("discarded" in event.message for event in eng.history()))
        assert len(eng.portfolio.trade_log) == 2
    finally:
        release.set()
        eng.stop()
        for thread in (eng._thread, eng._protection_thread):
            if thread is not None:
                thread.join(timeout=3)
                assert not thread.is_alive()


def test_protection_precedes_same_candle_dedup_and_buy_block(live):
    eng, snap = live(bars())
    # Keep the quote fresh regardless of the wall clock's five-minute phase.
    snap.data.index += pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=301) - snap.data.index[-1]
    price = float(snap.data.Close.iloc[-1])
    eng.portfolio.buy("BTC-USD", price, cash_amount=1000)
    eng._safety.manual_block("No new entries")
    eng.portfolio.protect_profit("BTC-USD", price * 1.01, observed_at=1., exit_slippage_bps=5.)
    eng._last_signal_bars[("BTC-USD", "5m")] = eng._bar_stamp(snap)
    eng._cycle_once()
    assert not eng.portfolio.positions
    assert eng._committee.calls == 0
    assert "Profit protection" in eng.portfolio.trade_log[-1].reasoning


def test_all_positions_are_monitored_when_active_symbol_is_different(monkeypatch):
    from notifications import NotificationConfig
    monkeypatch.setattr(NotificationConfig, "load", lambda: NotificationConfig())
    port = LivePortfolio(10_000)
    port.buy("LONG", 100., quantity=10.)
    port.open_short("SHORT", 100., cash_amount=1000.)
    prices = {"LONG": 101., "SHORT": 99.}
    eng = LiveTradingEngine(port, None, None, None,
                           protection_quote_provider=lambda ticker: quote(prices[ticker]))
    monkeypatch.setattr(eng, "_emit", lambda *a, **k: None)
    eng._ticker = "UNRELATED"
    eng._protection_once()
    assert all(pos.profit_protection_armed for pos in port.positions.values())
    prices.update(LONG=100.5, SHORT=99.5)
    eng._protection_once()
    assert not port.positions
    assert all(t.realized_pnl > 0 for t in port.trade_log[-2:])


def test_stale_feed_is_reported_without_fabricating_a_fill(live):
    eng, _ = live(bars())
    eng.portfolio.buy("BTC-USD", 100., quantity=10.)
    eng.portfolio.protect_profit("BTC-USD", 101., observed_at=1.)
    eng._protection_quote_provider = lambda ticker: replace(quote(98.), bar_start=time.time() - 600)
    eng._protection_once()
    assert len(eng.portfolio.trade_log) == 1
    assert eng.profit_protection_status()["BTC-USD"]["status"] == "UNAVAILABLE"


def test_stop_during_quote_request_prevents_new_protective_execution(live):
    eng, _ = live(bars())
    eng.portfolio.buy("BTC-USD", 100., quantity=10.)
    eng.portfolio.protect_profit("BTC-USD", 101., observed_at=1.)
    def provider(ticker):
        eng.stop()
        return quote(98.)
    eng._protection_quote_provider = provider
    eng._protection_once()
    assert len(eng.portfolio.trade_log) == 1


@pytest.mark.parametrize("order", ["buy", "sell", "open_short", "cover"])
def test_stale_strategy_order_is_rejected_atomically(order):
    port = LivePortfolio(10_000)
    stale = len(port.trade_log)
    port.buy("OTHER", 100., quantity=1.)
    with pytest.raises(ValueError, match="Portfolio changed"):
        if order in ("buy", "sell"):
            getattr(port, order)("TEST", 100., quantity=1., expected_revision=stale)
        elif order == "open_short":
            port.open_short("TEST", 100., cash_amount=100., expected_revision=stale)
        else:
            port.cover("TEST", 100., expected_revision=stale)
    assert len(port.trade_log) == 1
