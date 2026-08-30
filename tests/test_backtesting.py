"""
Tests for backtesting/backtest_runner.py — the historical simulation engine
that replays the fetch -> RAG -> AI-decide pipeline day-by-day.

Split into three tiers:
  1. Pure logic: _PositionState (open/close/mark-to-market P&L bookkeeping).
  2. Small helpers: _validate_dates, _empty_retrieval, _build_vbt_portfolio
     (vectorbt itself mocked via sys.modules — never installed/imported for
     real here).
  3. A full BacktestRunner.run() integration test with fake fetcher/
     retriever/engine doubles standing in for the real Phase 1-4 pipeline,
     and inter_call_delay=0 so it doesn't sleep. This is the only place the
     BUY/SELL/HOLD state machine, confidence gating, and forced last-bar
     close are exercised end to end.

No real network calls and no real Anthropic API calls anywhere in this file.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from backtesting.backtest_runner import (
    BacktestRunner,
    BacktestResult,
    TradeRecord,
    _PositionState,
    _render_backtest_report,
)
from market_data.fetcher import MACDParams, MarketSnapshot
from rag.retriever import RetrievalResult
from decision_engine.ai_engine import TradingDecision


# --------------------------------------------------------------------------- #
# _PositionState
# --------------------------------------------------------------------------- #
def test_position_state_starts_flat_with_given_cash():
    p = _PositionState(cash=10_000.0)
    assert p.in_position is False
    assert p.portfolio_value(123.45) == 10_000.0


def test_open_trade_converts_all_cash_to_shares():
    p = _PositionState(cash=1_000.0)
    p.open_trade(price=100.0, dt=date(2026, 1, 5))
    assert p.shares == pytest.approx(10.0)
    assert p.cash == 0.0
    assert p.in_position is True
    assert p.entry_price == 100.0
    assert p.entry_date == date(2026, 1, 5)


def test_close_trade_computes_pnl_and_counts_a_win():
    p = _PositionState(cash=1_000.0)
    p.open_trade(price=100.0, dt=date(2026, 1, 5))
    pnl = p.close_trade(price=110.0)
    assert pnl == pytest.approx(100.0)
    assert p.cash == pytest.approx(1_100.0)
    assert p.shares == 0.0
    assert p.in_position is False
    assert p.trade_count == 1
    assert p.winning_trades == 1


def test_close_trade_a_loss_does_not_increment_winning_trades():
    p = _PositionState(cash=1_000.0)
    p.open_trade(price=100.0, dt=date(2026, 1, 5))
    pnl = p.close_trade(price=90.0)
    assert pnl == pytest.approx(-100.0)
    assert p.trade_count == 1
    assert p.winning_trades == 0


def test_portfolio_value_marks_open_position_to_current_price():
    p = _PositionState(cash=1_000.0)
    p.open_trade(price=100.0, dt=date(2026, 1, 5))
    assert p.portfolio_value(120.0) == pytest.approx(1_200.0)


# --------------------------------------------------------------------------- #
# _validate_dates
# --------------------------------------------------------------------------- #
def test_validate_dates_accepts_strings_and_returns_date_objects():
    s, e = BacktestRunner._validate_dates("2026-01-01", "2026-03-01")
    assert s == date(2026, 1, 1)
    assert e == date(2026, 3, 1)


def test_validate_dates_accepts_date_objects_directly():
    s, e = BacktestRunner._validate_dates(date(2026, 1, 1), date(2026, 3, 1))
    assert (s, e) == (date(2026, 1, 1), date(2026, 3, 1))


def test_validate_dates_rejects_start_not_before_end():
    with pytest.raises(ValueError, match="must be before"):
        BacktestRunner._validate_dates("2026-03-01", "2026-01-01")


def test_validate_dates_rejects_equal_dates():
    with pytest.raises(ValueError, match="must be before"):
        BacktestRunner._validate_dates("2026-01-01", "2026-01-01")


def test_validate_dates_rejects_windows_shorter_than_20_days():
    with pytest.raises(ValueError, match="at least 20 trading days"):
        BacktestRunner._validate_dates("2026-01-01", "2026-01-10")


def test_validate_dates_accepts_exactly_20_days():
    s, e = BacktestRunner._validate_dates("2026-01-01", "2026-01-21")
    assert (e - s).days == 20


# --------------------------------------------------------------------------- #
# _empty_retrieval
# --------------------------------------------------------------------------- #
def test_empty_retrieval_is_a_safe_no_op_result():
    result = BacktestRunner._empty_retrieval("AAPL", snapshot=None)
    assert isinstance(result, RetrievalResult)
    assert result.ticker == "AAPL"
    assert result.chunks == []
    assert result.collection_size == 0


# --------------------------------------------------------------------------- #
# _build_vbt_portfolio — vectorbt mocked via sys.modules
# --------------------------------------------------------------------------- #
def _runner(**kw) -> BacktestRunner:
    return BacktestRunner(fetcher=None, retriever=None, engine=None, **kw)


def test_build_vbt_portfolio_returns_defaults_when_no_entries():
    runner = _runner()
    price = pd.Series([100.0, 101.0, 102.0])
    entries = pd.Series([False, False, False])
    exits = pd.Series([False, False, False])
    portfolio, stats = runner._build_vbt_portfolio(price, entries, exits, 10_000.0)
    assert portfolio is None
    assert stats == {"sharpe_ratio": 0.0, "max_drawdown_pct": 0.0, "total_return_pct": 0.0}


def test_build_vbt_portfolio_returns_defaults_when_vectorbt_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "vectorbt", None)
    runner = _runner()
    price = pd.Series([100.0, 105.0])
    entries = pd.Series([True, False])
    exits = pd.Series([False, True])
    portfolio, stats = runner._build_vbt_portfolio(price, entries, exits, 10_000.0)
    assert portfolio is None
    assert stats["sharpe_ratio"] == 0.0


def _fake_vbt_module(stats: dict, raise_on_build: bool = False):
    """Build a minimal fake `vectorbt` module exposing Portfolio.from_signals."""
    mod = types.ModuleType("vectorbt")

    class _FakePortfolio:
        def stats(self):
            return stats

    class _Portfolio:
        @staticmethod
        def from_signals(**kwargs):
            if raise_on_build:
                raise RuntimeError("boom")
            return _FakePortfolio()

    mod.Portfolio = _Portfolio
    return mod


def test_build_vbt_portfolio_maps_vbt_stats_to_standard_keys(monkeypatch):
    fake_stats = {
        "Sharpe Ratio": 1.234,
        "Max Drawdown [%]": -12.5,
        "Total Return [%]": 42.0,
        "Win Rate [%]": 60.0,
    }
    monkeypatch.setitem(sys.modules, "vectorbt", _fake_vbt_module(fake_stats))
    runner = _runner()
    price = pd.Series([100.0, 105.0])
    entries = pd.Series([True, False])
    exits = pd.Series([False, True])
    portfolio, stats = runner._build_vbt_portfolio(price, entries, exits, 10_000.0)
    assert portfolio is not None
    assert stats["sharpe_ratio"] == pytest.approx(1.234)
    assert stats["max_drawdown_pct"] == pytest.approx(12.5)  # abs() applied
    assert stats["total_return_pct"] == pytest.approx(42.0)
    assert stats["win_rate_pct"] == pytest.approx(60.0)


def test_build_vbt_portfolio_falls_back_to_lowercase_keys(monkeypatch):
    fake_stats = {"sharpe_ratio": 0.5, "max_drawdown_pct": -5.0}
    monkeypatch.setitem(sys.modules, "vectorbt", _fake_vbt_module(fake_stats))
    runner = _runner()
    price = pd.Series([100.0, 105.0])
    entries = pd.Series([True, False])
    exits = pd.Series([False, True])
    _, stats = runner._build_vbt_portfolio(price, entries, exits, 10_000.0)
    assert stats["sharpe_ratio"] == pytest.approx(0.5)
    assert stats["max_drawdown_pct"] == pytest.approx(5.0)


def test_build_vbt_portfolio_returns_defaults_when_construction_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "vectorbt", _fake_vbt_module({}, raise_on_build=True))
    runner = _runner()
    price = pd.Series([100.0, 105.0])
    entries = pd.Series([True, False])
    exits = pd.Series([False, True])
    portfolio, stats = runner._build_vbt_portfolio(price, entries, exits, 10_000.0)
    assert portfolio is None
    assert stats == {"sharpe_ratio": 0.0, "max_drawdown_pct": 0.0, "total_return_pct": 0.0}


# --------------------------------------------------------------------------- #
# TradeRecord / BacktestResult
# --------------------------------------------------------------------------- #
def _sample_result(**overrides) -> BacktestResult:
    defaults = dict(
        ticker="QQQ",
        start_date=date(2026, 1, 5),
        end_date=date(2026, 2, 5),
        initial_capital=10_000.0,
        final_portfolio_value=11_000.0,
        total_return_pct=10.0,
        buy_hold_return_pct=5.0,
        alpha_pct=5.0,
        sharpe_ratio=1.2,
        max_drawdown_pct=8.0,
        win_rate_pct=60.0,
        total_trades=5,
        total_days_simulated=22,
        ai_calls_made=22,
        fallback_count=1,
        trade_log=[
            TradeRecord(
                sim_date=date(2026, 1, 6), action="BUY", exec_price=101.0,
                confidence=0.8, risk_level="LOW", rag_quality="STRONG",
                is_fallback=False, reasoning_snippet="looks good", portfolio_value=10_000.0,
            ),
            TradeRecord(
                sim_date=date(2026, 1, 7), action="HOLD", exec_price=None,
                confidence=0.4, risk_level="MEDIUM", rag_quality="WEAK",
                is_fallback=False, reasoning_snippet="unclear", portfolio_value=10_000.0,
            ),
        ],
        portfolio=None,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def test_backtest_result_repr_includes_key_metrics():
    r = _sample_result()
    text = repr(r)
    assert "QQQ" in text
    assert "return=+10.00%" in text
    assert "trades=5" in text


def test_backtest_result_print_summary_runs_without_error(capsys):
    r = _sample_result()
    r.print_summary()  # delegates to _render_backtest_report
    out = capsys.readouterr().out
    assert "BACKTEST REPORT" in out
    assert "QQQ" in out


def test_render_backtest_report_handles_no_executed_trades(capsys):
    r = _sample_result(trade_log=[
        TradeRecord(
            sim_date=date(2026, 1, 6), action="HOLD", exec_price=None,
            confidence=0.2, risk_level="LOW", rag_quality="NONE",
            is_fallback=True, reasoning_snippet="no signal", portfolio_value=10_000.0,
        ),
    ])
    _render_backtest_report(r)
    out = capsys.readouterr().out
    assert "No Trades Executed" in out or "No BUY or SELL" in out


def test_render_backtest_report_handles_zero_ai_calls(capsys):
    r = _sample_result(ai_calls_made=0, fallback_count=0, trade_log=[])
    _render_backtest_report(r)  # must not raise ZeroDivisionError
    out = capsys.readouterr().out
    assert "BACKTEST REPORT" in out


# --------------------------------------------------------------------------- #
# BacktestRunner.run() — full integration with fake pipeline dependencies
# --------------------------------------------------------------------------- #
def _make_full_snapshot(ticker: str, n_days: int, start: date) -> MarketSnapshot:
    idx = pd.bdate_range(start, periods=n_days)
    close = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)
    df = pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": [1_000_000] * len(idx),
    })
    return MarketSnapshot(
        ticker=ticker,
        data=df,
        sma_periods=(20, 50, 200),
        rsi_period=14,
        macd_params=MACDParams(),
    )


class _FakeFetcher:
    def __init__(self, snapshot: MarketSnapshot):
        self._snapshot = snapshot

    def fetch(self, ticker, start, end):
        return self._snapshot


class _FakeRetriever:
    def get_relevant_strategies(self, snap, top_k):
        return RetrievalResult(
            query="q", ticker=snap.ticker, chunks=[], market_regime={}, collection_size=0,
        )


def _decision(action: str, confidence: float = 0.9, is_fallback: bool = False) -> TradingDecision:
    return TradingDecision(
        action=action,
        confidence_score=confidence,
        reasoning="synthetic test decision",
        risk_level="LOW",
        key_indicators=[],
        attractiveness_score=0.5,
        attractiveness_label="NEUTRAL",
        price_outlook="NEUTRAL",
        rag_context_quality="NONE",
        ticker="QQQ",
        is_fallback=is_fallback,
    )


class _ScriptedEngine:
    """Returns a scripted sequence of decisions, one per call."""
    def __init__(self, decisions: list[TradingDecision]):
        self._decisions = list(decisions)
        self.calls = 0

    def evaluate_market(self, snap, retrieval):
        self.calls += 1
        return self._decisions[min(self.calls - 1, len(self._decisions) - 1)]


def _run_scripted(decisions, n_days=25, min_confidence=0.55):
    start = date(2026, 1, 5)
    snapshot = _make_full_snapshot("QQQ", n_days, start)
    fetcher = _FakeFetcher(snapshot)
    retriever = _FakeRetriever()
    engine = _ScriptedEngine(decisions)
    runner = BacktestRunner(
        fetcher=fetcher, retriever=retriever, engine=engine,
        min_confidence=min_confidence, inter_call_delay=0.0,
        fees=0.0, slippage=0.0,
    )
    sim_dates = list(snapshot.data.index)
    sim_start, sim_end = sim_dates[0].date(), sim_dates[-1].date()
    result = runner.run(
        ticker="QQQ", start_date=sim_start, end_date=sim_end,
        initial_capital=10_000.0, warmup_days=0,
    )
    return result, engine


def test_run_buys_then_sells_and_records_a_completed_trade():
    decisions = [_decision("BUY")] + [_decision("HOLD")] * 5 + [_decision("SELL")] + [_decision("HOLD")] * 20
    result, engine = _run_scripted(decisions)
    assert result.total_trades == 1
    buys = [r for r in result.trade_log if r.action == "BUY"]
    sells = [r for r in result.trade_log if r.action == "SELL"]
    assert len(buys) == 1
    assert len(sells) == 1
    # Price series is monotonically increasing -> the round trip should be a win.
    assert result.win_rate_pct == 100.0


def test_run_never_buys_twice_while_already_in_position():
    decisions = [_decision("BUY")] * 30
    result, engine = _run_scripted(decisions)
    buys = [r for r in result.trade_log if r.action == "BUY"]
    assert len(buys) == 1  # subsequent BUY signals are ignored while in_position


def test_run_ignores_low_confidence_signals_treats_as_hold():
    decisions = [_decision("BUY", confidence=0.1)] * 30
    result, engine = _run_scripted(decisions, min_confidence=0.55)
    assert result.total_trades == 0
    assert all(r.action == "HOLD" for r in result.trade_log)


def test_run_ignores_fallback_decisions_even_if_action_is_buy():
    decisions = [_decision("BUY", confidence=0.9, is_fallback=True)] * 30
    result, engine = _run_scripted(decisions)
    assert result.total_trades == 0
    assert result.fallback_count == len(result.trade_log)


def test_run_force_closes_an_open_position_on_the_last_bar():
    decisions = [_decision("BUY")] + [_decision("HOLD")] * 28
    result, engine = _run_scripted(decisions, n_days=30)
    # No explicit SELL was ever scripted, but the position must be closed by the end.
    assert result.total_trades == 1
    last_rec = result.trade_log[-1]
    assert last_rec.action == "SELL*"


def test_run_counts_ai_calls_and_fallbacks():
    decisions = [_decision("HOLD", is_fallback=True)] * 5 + [_decision("HOLD")] * 20
    result, engine = _run_scripted(decisions)
    assert result.ai_calls_made == len(result.trade_log)
    assert result.fallback_count == 5


def test_run_computes_buy_and_hold_return_from_first_and_last_close():
    decisions = [_decision("HOLD")] * 30
    result, engine = _run_scripted(decisions)
    # Synthetic close series rises by exactly 1.0/day starting at 100 -> B&H > 0.
    assert result.buy_hold_return_pct > 0
    assert result.total_return_pct == 0.0  # never entered a position
    assert result.alpha_pct == pytest.approx(result.total_return_pct - result.buy_hold_return_pct)


def test_run_raises_when_retriever_and_engine_both_blow_up_but_keeps_going():
    class _ExplodingRetriever:
        def get_relevant_strategies(self, snap, top_k):
            raise RuntimeError("retrieval down")

    class _ExplodingEngine:
        def evaluate_market(self, snap, retrieval):
            raise RuntimeError("engine down")

    start = date(2026, 1, 5)
    snapshot = _make_full_snapshot("QQQ", 25, start)
    runner = BacktestRunner(
        fetcher=_FakeFetcher(snapshot),
        retriever=_ExplodingRetriever(),
        engine=_ExplodingEngine(),
        inter_call_delay=0.0,
    )
    sim_dates = list(snapshot.data.index)
    result = runner.run(
        ticker="QQQ", start_date=sim_dates[0].date(), end_date=sim_dates[-1].date(),
        initial_capital=10_000.0, warmup_days=0,
    )
    # Every bar falls back to a HOLD fallback rather than propagating the exception.
    assert result.total_trades == 0
    assert result.fallback_count == len(result.trade_log)
    assert all(r.is_fallback for r in result.trade_log)


def test_run_raises_value_error_for_a_too_short_window():
    snapshot = _make_full_snapshot("QQQ", 25, date(2026, 1, 5))
    runner = BacktestRunner(
        fetcher=_FakeFetcher(snapshot), retriever=_FakeRetriever(),
        engine=_ScriptedEngine([_decision("HOLD")]),
    )
    with pytest.raises(ValueError):
        runner.run("QQQ", start_date="2026-01-05", end_date="2026-01-10")
