"""Causal entry gates and end-to-end paper execution regressions."""
from types import SimpleNamespace as NS
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from market_data.signal_bars import completed_bars
from market_data.fetcher import MarketSnapshot, MACDParams
from strategy.research import analyze_entry, entry_features
from strategy.committee import CommitteeVerdict
from strategy.committee_backtest import compare_entry_policies, backtest_committee


def bars(down=False, n=1200):
    # Keep the per-bar slope stable while providing enough hourly history.
    close = np.linspace(100 + 30 * (n - 1) / 259, 100, n) if down else np.linspace(100, 126, n)
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


def short_breakdown_bars():
    df = bars(down=True)
    previous = float(df.Close.iloc[-2])
    df.loc[df.index[-1], ["Open", "High", "Low", "Close", "Volume"]] = [
        previous + 0.1,
        previous + 0.3,
        previous - 0.9,
        previous - 0.6,
        1800.0,
    ]
    return df


def test_downtrend_breakdown_becomes_a_short_setup():
    research = analyze_entry(short_breakdown_bars())
    assert research.regime == "DOWNTREND"
    assert research.setup == "SHORT_TREND_BREAKDOWN"
    assert research.signal_side == "SHORT"
    assert research.entry_allowed


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
    monkeypatch.setattr("market_data.committee_fundamentals.crypto_context", lambda ticker: {"status": "UNAVAILABLE", "metrics": {}})
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
                score = .5 if action == "BUY" else -.5
                return CommitteeVerdict(action, score,
                                        28, 9, 1, 38, True,
                                        category_scores={c: score for c in ("Trend", "Momentum", "Volume")})
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
    eng.portfolio.positions["BTC-USD"].opened_at = datetime.utcnow() - timedelta(hours=9)
    eng._cycle_once()
    assert not eng.portfolio.positions
    assert eng.portfolio.trade_log[-1].action == "SELL"


def test_live_committee_can_open_and_profitably_cover_a_short(live):
    eng, snap = live(short_breakdown_bars(), action="SELL")
    eng._cycle_once()
    assert eng.portfolio.trade_log[-1].action == "SHORT"
    assert eng.portfolio.positions["BTC-USD"].side == "SHORT"
    entry = eng.portfolio.positions["BTC-USD"].avg_entry_price
    eng.portfolio.update_price("BTC-USD", entry * 0.99)
    covered = eng.portfolio.cover("BTC-USD", entry * 0.99)
    assert covered.action == "COVER"
    assert covered.realized_pnl > 0


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


def test_live_does_not_buy_after_open_candle_crashes(live):
    df = bars()
    row = df.iloc[-1:].copy()
    row.index = row.index + pd.Timedelta(minutes=5)
    row.loc[:, ["Open", "High", "Low", "Close"]] *= .97
    eng, _ = live(pd.concat([df, row]))
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    assert "setup invalidated" in eng.snapshot()["last_decision"].reasoning


def test_live_rejects_a_failed_breakout_at_execution(live):
    df = bars()
    row = df.iloc[-1:].copy()
    row.index = row.index + pd.Timedelta(minutes=5)
    level = float(df.High.iloc[-21:-1].max()) - .05
    row.loc[:, ["Open", "High", "Low", "Close"]] = level
    eng, _ = live(pd.concat([df, row]))
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    assert "Breakout failed" in eng.snapshot()["last_decision"].reasoning


def test_research_version_and_evidence_reach_persisted_trade(live):
    eng, _ = live(bars())
    eng._cycle_once()
    report = eng.snapshot()["last_research"]
    assert report["policy_version"] == "research-v3"
    assert report["fundamental_analysis"]["status"] == "NOT_APPLICABLE"
    assert "research-v3" in eng.portfolio.trade_log[-1].reasoning
    assert "closed" in eng.portfolio.trade_log[-1].reasoning


def test_briefing_is_causal_and_does_not_invent_crypto_fundamentals():
    from strategy.briefing import chart_evidence, fundamental_evidence, trade_experience
    df = bars()
    before = chart_evidence(df.iloc[:230])
    df.iloc[230:, df.columns.get_loc("Close")] *= 5
    assert before == chart_evidence(df.iloc[:230])
    assert fundamental_evidence(NS(profit_margin=.5), "BTC-USD")["status"] == "NOT_APPLICABLE"
    assert trade_experience([], "BTC-USD")["exit_fills"] == 0


@pytest.mark.parametrize("mode", ["AI", "HYBRID"])
def test_ai_modes_receive_measured_research_and_playbook(live, monkeypatch, mode):
    eng, _ = live(bars())
    eng._strategy_mode = mode
    eng._retriever = NS(get_relevant_strategies=lambda *a, **k: NS(chunks=[]))
    captured = []
    def evaluate(*args, **kwargs):
        captured.append(kwargs["extra_context"])
        return CommitteeVerdict("BUY", .5, 28, 9, 1, 38, True).to_trading_decision("BTC-USD")
    monkeypatch.setattr(eng, "_ai_engine", lambda: NS(evaluate_market=evaluate))
    monkeypatch.setattr(eng, "_shared_decision", lambda key, **kwargs: kwargs["compute"]())
    eng._cycle_once()
    assert captured and "research-v3" in captured[0]
    assert "failed_resistance_break" in captured[0]
    assert "NOT_APPLICABLE" in captured[0]
    assert "No recorded exits" in captured[0]


@pytest.mark.parametrize("move", [.96, 1.04])
def test_short_entry_rejects_post_signal_price_gap(live, move):
    df = short_breakdown_bars()
    row = df.iloc[-1:].copy()
    row.index = row.index + pd.Timedelta(minutes=5)
    row.loc[:, ["Open", "High", "Low", "Close"]] *= move
    eng, _ = live(pd.concat([df, row]), action="SELL")
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    assert "one ATR" in eng.snapshot()["last_research"]["explanation"]


def test_short_entry_rejects_reclaimed_support(live):
    df = short_breakdown_bars()
    row = df.iloc[-1:].copy()
    row.index = row.index + pd.Timedelta(minutes=5)
    support = float(df.Low.iloc[-21:-1].min())
    row.loc[:, ["Open", "High", "Low", "Close"]] = support + .05
    eng, _ = live(pd.concat([df, row]), action="SELL")
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    assert "Breakdown failed" in eng.snapshot()["last_research"]["explanation"]


def test_ai_buy_cannot_use_a_short_setup_as_long_permission(live, monkeypatch):
    eng, _ = live(short_breakdown_bars())
    eng._strategy_mode = "AI"
    eng._retriever = NS(get_relevant_strategies=lambda *a, **k: NS(chunks=[]))
    decision = CommitteeVerdict("BUY", .5, 28, 9, 1, 38, True).to_trading_decision("BTC-USD")
    monkeypatch.setattr(eng, "_ai_engine", lambda: NS(evaluate_market=lambda *a, **k: decision))
    monkeypatch.setattr(eng, "_shared_decision", lambda key, **kwargs: kwargs["compute"]())
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    assert "direction" in eng.snapshot()["last_research"]["explanation"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., 0.])
def test_execution_entry_check_rejects_invalid_market_values(value):
    from strategy.briefing import execution_entry_check
    report = {"signal_side": "SHORT", "metrics": {"close": 100., "atr": 1.}}
    assert not execution_entry_check(report, value, side="SHORT")[0]


def test_short_cover_is_allowed_when_new_entries_are_blocked(live):
    eng, snap = live(bars(down=True), action="BUY")
    eng.portfolio.open_short("BTC-USD", float(snap.data.Close.iloc[-1]), cash_amount=500)
    eng.portfolio.positions["BTC-USD"].opened_at = datetime.utcnow() - timedelta(hours=9)
    eng._safety.manual_block("No new exposure")
    eng._cycle_once()
    assert not eng.portfolio.positions
    assert eng.portfolio.trade_log[-1].action == "COVER"


@pytest.mark.parametrize("down, expected", [(False, "LONG"), (True, "SHORT")])
def test_hourly_bias_persists_between_hour_boundaries_without_lookahead(down, expected):
    df = bars(down=down, n=1200)
    # An exact hour-aligned sample makes the within-hour regression explicit.
    df.index = pd.date_range("2026-09-01", periods=len(df), freq="5min", tz="UTC")
    features = entry_features(df)
    assert features.higher_timeframe_bias.iloc[-288:].eq(expected).all()
    for cutoff in (1000, 1005, 1010):
        pd.testing.assert_frame_equal(
            features.iloc[:cutoff], entry_features(df.iloc[:cutoff]),
        )


@pytest.mark.parametrize("side, action", [("LONG", "SELL"), ("SHORT", "BUY")])
@pytest.mark.parametrize("age, loss, closes", [(5, .006, False), (45, .001, False), (45, .006, True)])
def test_committee_opposite_vote_obeys_soft_exit_policy(live, side, action, age, loss, closes):
    eng, snap = live(bars(down=True), action=action)
    eng._risk_profile = "Balanced"
    price = float(snap.data.Close.iloc[-1])
    entry = price / (1 - loss if side == "LONG" else 1 + loss)
    opener = eng.portfolio.buy if side == "LONG" else eng.portfolio.open_short
    opener("BTC-USD", entry, cash_amount=500)
    eng.portfolio.positions["BTC-USD"].opened_at = datetime.utcnow() - timedelta(minutes=age)
    eng._cycle_once()
    assert ("BTC-USD" not in eng.portfolio.positions) == closes
    if not closes:
        assert len(eng.portfolio.trade_log) == 1
        assert eng.snapshot()["last_decision"].action == "HOLD"
        assert "Directional exit deferred" in eng.snapshot()["last_decision"].reasoning


def test_live_committee_fails_closed_without_hourly_history(live):
    eng, _ = live(bars(n=260))
    eng._cycle_once()
    assert not eng.portfolio.trade_log
    report = eng.snapshot()["last_research"]
    assert not report["committee_admission"]["allowed"]
    assert "completed-hour" in report["explanation"]


def test_live_committee_applies_risk_budget_and_records_costs(live):
    eng, _ = live(bars())
    eng._cycle_once()
    report = eng.snapshot()["last_research"]
    admission = report["committee_admission"]
    assert report["committee_policy_version"] == "committee-v5-adx25"
    assert admission["allowed"]
    assert eng.portfolio.trade_log[0].gross_value <= 10000 * admission["size_pct"] / 100
    assert "committee-v5-adx25" in eng.portfolio.trade_log[0].reasoning
    assert report["committee_evidence"]["mode"] == "SHADOW"
    assert report["committee_evidence"]["base_allowed"]


@pytest.mark.parametrize("value, allowed", [(19., False), (24.99, False), (25., True), (float("nan"), False)])
def test_live_committee_adx_gate_cannot_be_overruled_by_votes(live, monkeypatch, value, allowed):
    eng, _ = live(bars())
    def fixed_adx(df, period):
        series = pd.Series(value, index=df.index)
        return series, series, series
    monkeypatch.setattr("strategy.committee._adx", fixed_adx)
    eng._cycle_once()
    report = eng.snapshot()["last_research"]
    assert report["committee_admission"]["allowed"] == allowed
    assert bool(eng.portfolio.trade_log) == allowed
    check = next(c for c in report["checks"] if c["name"] == "ADX trend strength")
    assert check["passed"] == allowed
    if not allowed:
        assert not report["committee_evidence"]["candidate_allowed"]


def test_unavailable_candidate_research_cannot_stop_production_entry(live, monkeypatch):
    eng, _ = live(bars())
    def unavailable(*args, **kwargs):
        raise ValueError("Provider unavailable")
    monkeypatch.setattr("strategy.committee_evidence.decision_evidence", unavailable)
    eng._cycle_once()
    assert eng.portfolio.trade_log[0].action == "BUY"
    assert eng.snapshot()["last_research"]["committee_evidence"]["status"] == "UNAVAILABLE"


def test_live_weak_adx_never_blocks_an_existing_position_exit(live, monkeypatch):
    eng, snap = live(bars(), action="SELL")
    eng.portfolio.buy("BTC-USD", float(snap.data.Close.iloc[-1]), cash_amount=500)
    eng.portfolio.positions["BTC-USD"].opened_at = datetime.utcnow() - timedelta(hours=9)
    monkeypatch.setattr("strategy.committee._adx", lambda df, period: (pd.Series(10., index=df.index),) * 3)
    eng._cycle_once()
    assert not eng.portfolio.positions
    assert eng.portfolio.trade_log[-1].action == "SELL"


def test_committee_does_not_pyramid_even_on_aggressive_profile(live):
    eng, snap = live(bars())
    eng._risk_profile = "Aggressive"
    eng.portfolio.buy("BTC-USD", float(snap.data.Close.iloc[-1]) * .999, cash_amount=500)
    eng._cycle_once()
    assert len(eng.portfolio.trade_log) == 1
    assert eng.snapshot()["last_decision"].action == "HOLD"


def test_committee_exit_changes_do_not_delay_ai_exits(live, monkeypatch):
    eng, snap = live(bars(), action="SELL")
    eng._strategy_mode = "AI"
    eng.portfolio.buy("BTC-USD", float(snap.data.Close.iloc[-1]), cash_amount=500)
    eng._retriever = NS(get_relevant_strategies=lambda *a, **k: NS(chunks=[]))
    decision = CommitteeVerdict("SELL", -.5, 9, 28, 1, 38, True).to_trading_decision("BTC-USD")
    monkeypatch.setattr(eng, "_ai_engine", lambda: NS(evaluate_market=lambda *a, **k: decision))
    monkeypatch.setattr(eng, "_shared_decision", lambda *a, **k: k["compute"]())
    eng._cycle_once()
    assert eng.portfolio.trade_log[-1].action == "SELL"
    assert not eng.portfolio.positions
    assert "committee_admission" not in eng.snapshot()["last_research"]
