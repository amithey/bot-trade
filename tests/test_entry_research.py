"""Causal entry gates and end-to-end paper execution regressions."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from market_data.signal_bars import completed_bars
from market_data.fetcher import MarketSnapshot, MACDParams
from strategy.research import analyze_entry, entry_features
from strategy.committee import CommitteeVerdict
from strategy.committee_backtest import compare_entry_policies, backtest_committee


def bars(down=False, n=260):
    close = np.linspace(130, 100, n) if down else np.linspace(100, 126, n)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("5min")-pd.Timedelta(minutes=5), periods=n, freq="5min")
    df = pd.DataFrame({"Open": close-.1, "High": close+.4, "Low": close-.4,
                       "Close": close, "Volume": 1000.}, index=idx)
    df.loc[idx[-1], "Close"] = close[-2]+.7
    df.loc[idx[-1], "High"] = close[-2]+1.
    df.loc[idx[-1], "Volume"] = 1800.
    return df


def test_balanced_blocks_a_bounce_inside_a_downtrend():
    research = analyze_entry(bars(down=True))
    assert research.regime == "DOWNTREND"
    assert research.setup == "NONE"
    assert not research.entry_allowed


def test_confirmed_trend_breakout_is_eligible():
    research = analyze_entry(bars())
    assert research.regime == "UPTREND"
    assert research.setup == "TREND_BREAKOUT"
    assert research.entry_allowed


def test_micro_scalp_countertrend_permission_is_explicit():
    df = bars(down=True)
    assert not analyze_entry(df).entry_allowed
    assert analyze_entry(df, risk_profile="Micro-Scalp").setup == "SCALP_REVERSAL"
    assert analyze_entry(df, risk_profile="Micro-Scalp").entry_allowed


def test_low_volume_blocks_even_a_breakout():
    df = bars()
    df.loc[df.index[-1], "Volume"] = 100
    assert not analyze_entry(df).entry_allowed


def test_extended_price_is_not_chased():
    df = bars()
    df.loc[df.index[-1], "Close"] += 15
    df.loc[df.index[-1], "High"] += 15
    r = analyze_entry(df)
    assert not r.entry_allowed
    assert not next(c for c in r.checks if c["name"] == "Extension")["passed"]


def test_missing_history_never_qualifies():
    assert not entry_features(bars(n=120)).eligible.any()


def test_entry_research_has_no_future_dependency():
    df = bars(n=500)
    pd.testing.assert_frame_equal(entry_features(df).iloc[:300], entry_features(df.iloc[:300]))


def test_open_candle_is_excluded_and_cannot_change_signal():
    df = bars()
    now = df.index[-1] + pd.Timedelta(minutes=7)
    open_bar = df.iloc[-1:].copy()
    open_bar.index = pd.DatetimeIndex([df.index[-1]+pd.Timedelta(minutes=5)])
    open_bar.loc[:, ["Open", "High", "Low", "Close"]] *= 2
    result = completed_bars(pd.concat([df, open_bar]), "5m", now=now)
    pd.testing.assert_frame_equal(result.data, df, check_freq=False)
    assert result.fresh


def test_stale_bars_are_reported():
    df = bars()
    result = completed_bars(df, "5m", now=df.index[-1]+pd.Timedelta(hours=2))
    assert not result.fresh


@pytest.mark.parametrize("field", ["Close", "High", "Volume"])
def test_invalid_data_is_rejected(field):
    df = bars()
    df.loc[df.index[-1], field] = float("nan")
    with pytest.raises(ValueError):
        completed_bars(df, "5m")


def test_duplicate_indices_are_rejected():
    df = bars()
    with pytest.raises(ValueError):
        completed_bars(pd.concat([df, df.iloc[-1:]]), "5m")


@pytest.fixture
def live(monkeypatch):
    from trading.live_engine import LiveTradingEngine
    from portfolio.virtual_account import LivePortfolio
    from notifications import NotificationConfig
    import market_data.research_context as context
    monkeypatch.setattr(context, "news_context", lambda ticker: {"status":"unavailable","items":[]})
    monkeypatch.setattr(NotificationConfig, "load", lambda: NotificationConfig())
    def build(df, action="BUY"):
        snap = MarketSnapshot("BTC-USD", df, (20,50,200), 14, MACDParams())
        fetcher = NS(fetch_with_fundamentals=lambda *a, **k: snap)
        eng = LiveTradingEngine(LivePortfolio(10000), fetcher, None, None)
        eng._strategy_mode = "COMMITTEE"
        eng._notifier = NS(notify=lambda *a, **k: None)
        eng._reflect_on_sell = lambda *a, **k: None
        class Committee:
            calls = 0
            def vote_latest(self, df, **kwargs):
                self.calls += 1
                return CommitteeVerdict(action, .5 if action == "BUY" else -.5,
                                        28, 9, 1, 38, True)
        eng._committee = Committee()
        return eng, snap
    return build


def test_live_raw_buy_is_replaced_by_explained_hold(live):
    eng, _ = live(bars(down=True))
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    assert eng.snapshot()["last_decision"].action == "HOLD"
    r = eng.snapshot()["last_research"]
    assert r["raw_action"] == "BUY" and r["filtered_action"] == "HOLD"
    assert r["regime"] == "DOWNTREND"


def test_live_entry_passes_and_same_candle_is_not_replayed(live):
    eng, _ = live(bars())
    eng._cycle_once()
    assert eng.portfolio.trade_log[-1].action == "BUY"
    eng._cycle_once()
    assert eng._committee.calls == 1
    assert len(eng.portfolio.trade_log) == 1


def test_stop_loss_remains_active_on_already_analyzed_bar(live):
    eng, snap = live(bars())
    eng._cycle_once()
    original_calls = eng._committee.calls
    row = snap.data.iloc[-1:].copy()
    row.index = pd.DatetimeIndex([snap.data.index[-1] + pd.Timedelta(minutes=5)])
    row.loc[:, ["Open", "High", "Low", "Close"]] *= .96
    snap.data = pd.concat([snap.data, row])
    eng._cycle_once()
    assert eng.portfolio.trade_log[-1].action == "SELL"
    assert "stop-loss" in eng.portfolio.trade_log[-1].reasoning
    assert eng._committee.calls == original_calls


def test_exit_signal_is_not_vetoed_by_entry_rules(live):
    eng, snap = live(bars(down=True), action="SELL")
    eng.portfolio.buy("BTC-USD", float(snap.data.Close.iloc[-1]), cash_amount=500)
    eng._cycle_once()
    assert not eng.portfolio.positions
    assert eng.portfolio.trade_log[-1].action == "SELL"


def test_holdout_uses_only_later_dates_and_equal_evaluation_window():
    df = bars(n=600)
    results = compare_entry_policies(df)
    a, b = results.values()
    assert a.start == b.start == df.index[420]
    assert a.bars == b.bars == 180
    assert a.equity.iloc[0] == b.equity.iloc[0] == 1
    assert all(t.entry_time > a.start for r in results.values() for t in r.trades)


def test_backtest_applies_exact_same_entry_feature_gate():
    df = bars(n=500)
    result = backtest_committee(df, entry_filter=True)
    eligible = entry_features(df).eligible
    for trade in result.trades:
        i = df.index.get_loc(trade.entry_time)
        assert eligible.iloc[i-1]


def test_live_stale_candles_do_not_create_orders(live):
    df = bars()
    df.index = df.index - pd.Timedelta(days=2)
    eng, _ = live(df)
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    assert eng.snapshot()["last_research"]["status"] == "DATA_BLOCKED"


def test_user_stop_during_analysis_cancels_order(live):
    eng, _ = live(bars())
    vote = eng._committee.vote_latest
    def stop_then_vote(*args, **kwargs):
        eng.stop()
        return vote(*args, **kwargs)
    eng._committee.vote_latest = stop_then_vote
    eng._cycle_once()
    assert not eng.portfolio.trade_log


def test_price_gap_after_closed_signal_is_not_chased(live):
    df = bars()
    row = df.iloc[-1:].copy()
    row.index = pd.DatetimeIndex([df.index[-1]+pd.Timedelta(minutes=5)])
    row.loc[:, ["Open", "High", "Low", "Close"]] *= 1.08
    eng, _ = live(pd.concat([df,row]))
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    assert "one ATR" in eng.snapshot()["last_research"]["explanation"]


def test_exchange_timezone_is_converted_before_becoming_naive(monkeypatch):
    import market_data.fetcher as fetcher_module
    df = bars()
    expected = df.index[0].tz_convert("UTC").tz_localize(None)
    df.index = df.index.tz_convert("America/New_York")
    monkeypatch.setattr(fetcher_module.yf, "Ticker", lambda symbol: NS(history=lambda **kwargs: df))
    result = fetcher_module.MarketDataFetcher()._download("TEST",period="5d",interval="5m")
    assert result.index[0] == expected


def test_research_panel_renders_the_reason_for_blocking():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string("""
import streamlit as st
from tests.test_entry_research import bars
from strategy.research import analyze_entry
from dashboard.components import render_entry_research
r = analyze_entry(bars(down=True)).to_dict()
r.update(status="ANALYZED", ticker="BTC-USD", interval="5m", bar_closed_at="test candle",
         explanation="No confirmed entry", raw_action="BUY", filtered_action="HOLD")
render_entry_research(r)
""")
    app.run(timeout=15)
    assert not app.exception
    assert [m.value for m in app.metric] == ["DOWNTREND", "NONE", "Blocked"]
