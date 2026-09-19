from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from strategy.trend_scanner import ENTRY_RULES, GRID, ScannerConfig, plan_day, scanner_features, simulate_portfolio
from trading.trend_scanner_runner import TrendScannerRunner, session_frame


def bars(n=400, drift=.002, seed=1, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, .01, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.004,
                         "Low": np.minimum(open_, close) * .996, "Close": close},
                        index=pd.date_range(start, periods=n, freq="D"))


def row(close=100., atr=2., strength=1., enter=True, valid=True):
    return {"close": close, "atr": atr, "strength": strength, "enter": enter, "valid": valid}


@pytest.mark.parametrize("rule", ENTRY_RULES)
def test_features_never_use_later_bars(rule):
    df = bars()
    full = scanner_features(df, rule)
    for cut in (210, 300, 399):
        assert full.iloc[cut].equals(scanner_features(df.iloc[:cut + 1], rule).iloc[-1])


def test_no_signal_before_minimum_history():
    features = scanner_features(bars(), "donchian_55")
    assert not features.enter.iloc[:199].any() and not features.valid.iloc[:199].any()


def test_grid_is_declared_and_valid():
    assert len(GRID) == 12
    with pytest.raises(ValueError):
        ScannerConfig(entry_rule="magic")


def test_plan_ranks_by_strength_and_respects_free_slots():
    config = ScannerConfig(max_positions=2)
    positions = {"HELD": {"peak_close": 100., "stop": 90.}}
    today = {"HELD": row(), "A": row(strength=.5), "B": row(strength=2.), "C": row(strength=-1.)}
    exits, entries, _ = plan_day(today, positions, 10000., config)
    assert exits == [] and [e["ticker"] for e in entries] == ["B"]


def test_exit_frees_slot_and_trend_break_exits():
    config = ScannerConfig(max_positions=1)
    positions = {"HELD": {"peak_close": 100., "stop": 90.}}
    exits, entries, _ = plan_day({"HELD": row(valid=False), "B": row()}, positions, 10000., config)
    assert exits == ["HELD"] and entries[0]["ticker"] == "B"


def test_stop_only_ratchets_up_and_sizing_follows_risk():
    config = ScannerConfig(stop_atr=3., risk_pct=1., max_weight_pct=25.)
    positions = {"HELD": {"peak_close": 120., "stop": 110.}}
    _, _, stops = plan_day({"HELD": row(close=105., atr=5.)}, positions, 10000., config)
    assert stops["HELD"] == 110.                     # 120 - 15 = 105 would loosen it; keep 110
    _, entries, _ = plan_day({"NEW": row(close=100., atr=2.)}, {}, 10000., config)
    assert entries[0]["weight_pct"] == pytest.approx(1 / .06) and entries[0]["stop"] == 94.   # 1% risk / 6% stop
    _, entries, _ = plan_day({"NEW": row(close=100., atr=.1)}, {}, 10000., config)
    assert entries[0]["weight_pct"] == 25.           # capped


def test_simulation_fills_at_next_open_and_is_profitable_only_from_later_bars():
    df = bars(drift=.004)
    curve, trades = simulate_portfolio({"X": df}, ScannerConfig(entry_rule="tsmom_120"), df.index[0])
    features = scanner_features(df, "tsmom_120")
    first = pd.Timestamp(trades[0]["entry_date"]) if trades else curve.index[curve.invested.gt(0).argmax()]
    signal_day = features.index[features.enter & features.valid][0]
    assert first > signal_day
    assert curve.equity.iloc[0] == 10000.


def test_stock_session_is_not_complete_before_the_close():
    df = bars(n=5, start="2026-09-14")
    df.index = df.index.tz_localize("America/New_York")
    during = datetime(2026, 9, 18, 17, 0, tzinfo=timezone.utc)     # 13:00 New York, Friday 18th
    after = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)      # 17:00 New York
    assert session_frame(df, "SPY", during)[1] is False
    assert session_frame(df, "SPY", after)[1] is True


def test_runner_decides_on_completed_bar_and_fills_next_session(tmp_path):
    full = bars(n=300, drift=.004)
    clock = {"n": 260}

    def fetch(_):
        return full.iloc[:clock["n"]].tz_localize("UTC")

    runner = TrendScannerRunner(tmp_path, universe=["AAA-USD"], fetch=fetch,
                                config=ScannerConfig(entry_rule="tsmom_120", max_positions=1))
    # Find a day where an entry gets planned.
    while not runner.state["pending_entries"] and clock["n"] < 300:
        stamp = full.index[clock["n"] - 1]
        runner.step(now=(stamp + pd.Timedelta(hours=1)).tz_localize("UTC").to_pydatetime())
        assert not runner.portfolio.positions, "must not fill on the bar that produced the decision"
        clock["n"] += 1
    assert runner.state["pending_entries"], "synthetic uptrend should produce an entry"
    decided = runner.state["decided"]["AAA-USD"]
    stamp = full.index[clock["n"] - 1]
    runner.step(now=(stamp + pd.Timedelta(hours=1)).tz_localize("UTC").to_pydatetime())
    position = runner.portfolio.positions["AAA-USD"]
    assert position.avg_entry_price == pytest.approx(full.Open.iloc[clock["n"] - 1] * 1.0005)
    assert str(stamp.date()) > decided
    # A restart resumes the same state.
    again = TrendScannerRunner(tmp_path, universe=["AAA-USD"], fetch=fetch)
    assert "AAA-USD" in again.portfolio.positions and "AAA-USD" in again.state["positions"]


def test_runner_stop_fills_at_worse_of_open_and_stop(tmp_path):
    df = bars(n=260, drift=.003)
    runner = TrendScannerRunner(tmp_path, universe=["AAA-USD"], fetch=lambda _: df.tz_localize("UTC"))
    runner.portfolio.buy("AAA-USD", 100., cash_amount=1000.)
    stop = float(df.Low.iloc[-1]) + 1                # the last session trades through the stop
    runner.state["positions"]["AAA-USD"] = {"peak_close": 100., "stop": stop}
    runner.state["decided"]["AAA-USD"] = str(df.index[-2].date())
    runner.step(now=(df.index[-1] + pd.Timedelta(hours=3)).tz_localize("UTC").to_pydatetime())
    assert "AAA-USD" not in runner.portfolio.positions
    sell = runner.portfolio.trade_log[-1]
    assert sell.price == pytest.approx(min(float(df.Open.iloc[-1]), stop) * (1 - .0005))
