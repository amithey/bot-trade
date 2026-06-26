"""Offline tests for the 38-indicator committee — no network, no API key."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.committee import (
    BEAR, BULL, NEUTRAL, CommitteeConfig, IndicatorCommittee,
)
from strategy.committee_backtest import backtest_committee


# ---------------------------------------------------------------------------
# Synthetic OHLCV factory
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 400, trend: float = 0.0, seed: int = 7) -> pd.DataFrame:
    """Random-walk OHLCV with an optional per-bar drift (in %)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=trend / 100.0, scale=0.01, size=n)
    close = 100.0 * np.cumprod(1 + rets)
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.integers(1_000, 100_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


@pytest.fixture(scope="module")
def committee() -> IndicatorCommittee:
    return IndicatorCommittee()


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_exactly_38_agents(committee):
    assert len(committee.agents) == 38
    cats = {a.category for a in committee.agents}
    assert cats == {"Trend", "Momentum", "Volatility", "Volume"}


def test_agent_names_unique(committee):
    names = [a.name for a in committee.agents]
    assert len(names) == len(set(names))


def test_votes_are_ternary(committee):
    votes = committee.vote_matrix(make_ohlcv())
    assert votes.shape[1] == 38
    assert set(np.unique(votes.to_numpy())) <= {BULL, NEUTRAL, BEAR}


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

def test_strong_uptrend_votes_buy(committee):
    df = make_ohlcv(trend=0.8)          # +0.8 %/bar — unambiguous bull run
    v = committee.vote_latest(df)
    assert v.action == "BUY"
    assert v.bulls > v.bears


def test_strong_downtrend_votes_sell(committee):
    df = make_ohlcv(trend=-0.8)
    v = committee.vote_latest(df)
    assert v.action == "SELL"
    assert v.bears > v.bulls


def test_no_lookahead(committee):
    """Votes on bar t must not change when future bars are appended."""
    full = make_ohlcv(n=400)
    head = full.iloc[:300]
    v_head = committee.vote_matrix(head).iloc[-1]
    v_full = committee.vote_matrix(full).iloc[299]
    # PSAR/Supertrend are recursive from series start — identical inputs
    # up to t means identical state at t, so every agent must agree.
    pd.testing.assert_series_equal(v_head, v_full, check_names=False)


def test_verdict_adapts_to_trading_decision(committee):
    v = committee.vote_latest(make_ohlcv(trend=0.8))
    td = v.to_trading_decision("TEST")
    assert td.action == v.action
    assert 0.5 <= td.confidence_score <= 1.0
    assert len(td.reasoning) >= 20


def test_too_few_bars_raises(committee):
    with pytest.raises(ValueError):
        committee.vote_latest(make_ohlcv(n=30))


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

def test_backtest_runs_and_is_consistent():
    df = make_ohlcv(n=500, trend=0.3)
    res = backtest_committee(df, ticker="SYN", interval="1d")
    assert res.bars == 500
    assert set(np.unique(res.position.to_numpy())) <= {0, 1}
    assert (res.equity > 0).all()
    # equity starts at 1.0 (cash) before the warm-up ends
    assert res.equity.iloc[0] == pytest.approx(1.0)
    # reported return matches the curve
    assert res.total_return_pct == pytest.approx(
        (res.equity.iloc[-1] - 1) * 100, abs=0.01)
    # closed trades have full records
    for t in res.trades:
        if t.exit_time is not None:
            assert t.exit_price is not None and t.pnl_pct is not None
            assert t.exit_time >= t.entry_time


def test_backtest_fees_reduce_returns():
    df = make_ohlcv(n=500, trend=0.3)
    free = backtest_committee(df, fee_pct=0.0)
    paid = backtest_committee(df, fee_pct=0.25)
    if free.total_trades > 0:
        assert paid.total_return_pct < free.total_return_pct


def test_optimizer_returns_sorted_grid():
    from strategy.committee_backtest import optimize_committee
    df = make_ohlcv(n=400, trend=0.3)
    cells, best = optimize_committee(df, margins=(4, 8, 12))
    assert len(cells) == 9
    fitness = [c.fitness for c in cells]
    assert fitness == sorted(fitness, reverse=True)
    # best full result must match the winning cell's config outcome
    assert best.total_return_pct == pytest.approx(
        cells[0].total_return_pct, abs=0.01)


def test_hybrid_combine_rules(committee):
    from trading.live_engine import LiveTradingEngine
    bull_v = committee.vote_latest(make_ohlcv(trend=0.8))
    bear_v = committee.vote_latest(make_ohlcv(trend=-0.8))
    assert bull_v.action == "BUY" and bear_v.action == "SELL"

    ai_buy = bull_v.to_trading_decision("T").model_copy(
        update={"action": "BUY", "confidence_score": 0.7})
    ai_sell = bull_v.to_trading_decision("T").model_copy(
        update={"action": "SELL", "confidence_score": 0.9})
    ai_hold = bull_v.to_trading_decision("T").model_copy(
        update={"action": "HOLD", "confidence_score": 0.6})

    # Both agree on BUY → BUY with confidence bonus (clamped at 1.0)
    d = LiveTradingEngine._combine_hybrid("T", bull_v, ai_buy)
    assert d.action == "BUY"
    expected = min(1.0, max(0.7, bull_v.confidence) + 0.08)
    assert d.confidence_score == pytest.approx(expected, abs=0.001)

    # AI objects with SELL against a bullish committee → no panic sell
    d = LiveTradingEngine._combine_hybrid("T", bull_v, ai_sell)
    assert d.action != "SELL"

    # Committee SELL always exits, whatever the AI says
    d = LiveTradingEngine._combine_hybrid("T", bear_v, ai_hold)
    assert d.action == "SELL"


def test_config_thresholds_respected():
    df = make_ohlcv(n=500)
    loose = backtest_committee(df, config=CommitteeConfig(
        enter_score=0.05, exit_score=-0.05))
    strict = backtest_committee(df, config=CommitteeConfig(
        enter_score=0.6, exit_score=-0.6))
    assert strict.total_trades <= loose.total_trades
