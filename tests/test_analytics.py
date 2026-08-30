"""
Tests for the analytics/ package — performance metrics, equity/drawdown
curves, trade attribution, and holdings-risk (concentration + correlation).

All pure computation over synthetic TradeRecord/DailySnapshot/Position
objects; the one network boundary (yfinance, inside correlation_matrix) is
monkeypatched everywhere so the suite makes no real HTTP calls.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import analytics
from analytics.performance import (
    PerformanceMetrics,
    _to_records_df,
    _to_snapshots_df,
    _annualised_return,
    _longest_underwater_run,
    equity_curve,
    drawdown_series,
    benchmark_equity,
    compute_metrics,
)
from analytics.attribution import (
    pnl_by_ticker,
    pnl_by_weekday,
    pnl_by_hour,
    win_rate_by_ticker,
    trade_durations,
)
from analytics.holdings_risk import concentration, correlation_matrix
from portfolio.virtual_account import Position, TradeRecord, DailySnapshot


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #
def make_trade(
    executed_at: datetime,
    action: str,
    ticker: str = "AAPL",
    quantity: float = 10.0,
    price: float = 100.0,
    realized_pnl: float = 0.0,
    portfolio_value: float = 10_000.0,
) -> TradeRecord:
    gross = quantity * price
    return TradeRecord(
        executed_at=executed_at,
        action=action,
        ticker=ticker,
        quantity=quantity,
        price=price,
        gross_value=gross,
        fee=0.0,
        net_value=gross,
        realized_pnl=realized_pnl,
        cash_after=0.0,
        portfolio_value=portfolio_value,
        reasoning="test",
    )


def make_snapshot(d: date, value: float) -> DailySnapshot:
    return DailySnapshot(snapshot_date=d, portfolio_value=value)


def make_position(ticker: str, quantity: float, avg_entry: float, current: float) -> Position:
    return Position(
        ticker=ticker, quantity=quantity, avg_entry_price=avg_entry,
        current_price=current,
    )


D0 = datetime(2026, 1, 5, 9, 30)  # a Monday


# --------------------------------------------------------------------------- #
# analytics/__init__.py — public surface
# --------------------------------------------------------------------------- #
def test_package_reexports_everything_in_all():
    for name in analytics.__all__:
        assert hasattr(analytics, name), f"analytics.{name} missing"


# --------------------------------------------------------------------------- #
# _to_records_df / _to_snapshots_df
# --------------------------------------------------------------------------- #
def test_to_records_df_accepts_trade_record_objects():
    trades = [make_trade(D0, "BUY"), make_trade(D0 + timedelta(days=1), "SELL")]
    df = _to_records_df(trades)
    assert list(df["action"]) == ["BUY", "SELL"]


def test_to_records_df_accepts_plain_dicts():
    d = make_trade(D0, "BUY").to_dict()
    df = _to_records_df([d])
    assert len(df) == 1
    assert isinstance(df.iloc[0]["executed_at"], datetime)


def test_to_records_df_skips_unparseable_rows():
    bad = {"executed_at": "not-a-date", "action": "BUY"}
    good = make_trade(D0, "BUY").to_dict()
    df = _to_records_df([bad, good])
    assert len(df) == 1


def test_to_records_df_skips_unrecognised_items():
    df = _to_records_df([object(), 42, "nope"])
    assert df.empty


def test_to_records_df_empty_input_has_expected_columns():
    df = _to_records_df([])
    assert list(df.columns) == [
        "executed_at", "action", "ticker", "quantity", "price",
        "realized_pnl", "portfolio_value",
    ]


def test_to_records_df_sorts_by_executed_at():
    later = make_trade(D0 + timedelta(days=1), "SELL")
    earlier = make_trade(D0, "BUY")
    df = _to_records_df([later, earlier])
    assert list(df["action"]) == ["BUY", "SELL"]


def test_to_snapshots_df_accepts_objects_and_dicts_and_dedupes():
    snaps = [
        make_snapshot(date(2026, 1, 5), 10_000.0),
        make_snapshot(date(2026, 1, 5), 10_050.0),  # same day, later wins
        {"snapshot_date": "2026-01-06", "portfolio_value": "10100.0"},
    ]
    df = _to_snapshots_df(snaps)
    assert len(df) == 2
    row0 = df[df["snapshot_date"] == date(2026, 1, 5)].iloc[0]
    assert row0["portfolio_value"] == 10_050.0


def test_to_snapshots_df_skips_bad_rows():
    df = _to_snapshots_df([{"snapshot_date": "bad", "portfolio_value": "x"}])
    assert df.empty


# --------------------------------------------------------------------------- #
# equity_curve
# --------------------------------------------------------------------------- #
def test_equity_curve_uses_snapshots_when_present():
    snaps = [make_snapshot(date(2026, 1, 5), 10_000.0),
             make_snapshot(date(2026, 1, 6), 10_500.0)]
    eq = equity_curve([], snaps, initial_capital=10_000.0)
    assert eq.loc[pd.Timestamp("2026-01-06"), "equity"] == 10_500.0


def test_equity_curve_falls_back_to_trade_portfolio_value():
    trades = [make_trade(D0, "BUY", portfolio_value=9_900.0)]
    eq = equity_curve(trades, [], initial_capital=10_000.0)
    assert eq.loc[pd.Timestamp(D0.date()), "equity"] == 9_900.0


def test_equity_curve_empty_returns_single_row_of_initial_capital():
    eq = equity_curve([], [], initial_capital=5_000.0)
    assert len(eq) == 1
    assert eq["equity"].iloc[0] == 5_000.0


def test_equity_curve_forward_fills_gaps():
    snaps = [make_snapshot(date(2026, 1, 5), 10_000.0),
             make_snapshot(date(2026, 1, 9), 11_000.0)]  # Mon .. Fri, gap of days
    eq = equity_curve([], snaps, initial_capital=10_000.0)
    # Every business day between should be forward-filled to 10_000 until the 9th.
    mid = eq.loc[pd.Timestamp("2026-01-07"), "equity"]
    assert mid == 10_000.0


def test_equity_curve_includes_current_value_at_today():
    eq = equity_curve([], [], initial_capital=10_000.0, current_value=12_345.0)
    today = pd.Timestamp(date.today())
    assert eq.loc[today, "equity"] == 12_345.0


def test_equity_curve_weekend_trade_is_kept_via_union():
    saturday = datetime(2026, 1, 10, 12, 0)  # Sat
    trades = [make_trade(saturday, "BUY", portfolio_value=10_200.0)]
    eq = equity_curve(trades, [], initial_capital=10_000.0)
    assert eq.loc[pd.Timestamp(saturday.date()), "equity"] == 10_200.0


# --------------------------------------------------------------------------- #
# drawdown_series
# --------------------------------------------------------------------------- #
def test_drawdown_series_accepts_series_or_dataframe():
    s = pd.Series([100.0, 110.0, 90.0], index=pd.date_range("2026-01-01", periods=3))
    dd_from_series = drawdown_series(s)
    dd_from_df = drawdown_series(pd.DataFrame({"equity": s}))
    pd.testing.assert_frame_equal(dd_from_series, dd_from_df)


def test_drawdown_series_computes_expected_pct():
    s = pd.Series([100.0, 200.0, 100.0])
    dd = drawdown_series(s)
    assert dd["drawdown_pct"].iloc[-1] == pytest.approx(-50.0)
    assert dd["peak"].iloc[-1] == 200.0


def test_longest_underwater_run_counts_consecutive_negative_stretch():
    dd = pd.Series([0.0, -1.0, -2.0, 0.0, -1.0])
    assert _longest_underwater_run(dd) == 2


def test_longest_underwater_run_empty_series_is_zero():
    assert _longest_underwater_run(pd.Series([], dtype=float)) == 0


# --------------------------------------------------------------------------- #
# _annualised_return
# --------------------------------------------------------------------------- #
def test_annualised_return_needs_at_least_two_points():
    s = pd.Series([100.0], index=[pd.Timestamp("2026-01-01")])
    assert _annualised_return(s) == 0.0


def test_annualised_return_zero_when_same_day():
    idx = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")]
    s = pd.Series([100.0, 110.0], index=idx)
    assert _annualised_return(s) == 0.0


def test_annualised_return_negative_100_when_total_loss():
    idx = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01")]
    s = pd.Series([100.0, 0.0], index=idx)
    assert _annualised_return(s) == -100.0


def test_annualised_return_doubling_over_one_year_is_about_100pct():
    idx = [pd.Timestamp("2026-01-01"), pd.Timestamp("2027-01-01")]
    s = pd.Series([100.0, 200.0], index=idx)
    assert _annualised_return(s) == pytest.approx(100.0, abs=1.0)


# --------------------------------------------------------------------------- #
# benchmark_equity
# --------------------------------------------------------------------------- #
class _FakeHistoryTicker:
    def __init__(self, closes: list[float]):
        self._closes = closes

    def history(self, start, end, interval, auto_adjust):
        idx = pd.date_range(start, periods=len(self._closes), freq="D")
        return pd.DataFrame({"Close": self._closes}, index=idx)


def test_benchmark_equity_rescales_to_initial_value(monkeypatch):
    import analytics.performance as perf_mod
    fake_yf = type("FakeYF", (), {
        "Ticker": staticmethod(lambda t: _FakeHistoryTicker([100.0, 110.0, 121.0])),
        "set_tz_cache_location": staticmethod(lambda p: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    out = benchmark_equity("SPY", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), 10_000.0)
    assert out is not None
    assert out["equity"].iloc[0] == pytest.approx(10_000.0)
    assert out["equity"].iloc[-1] == pytest.approx(12_100.0)


def test_benchmark_equity_returns_none_on_import_failure(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "yfinance", None)
    # Setting the module to None makes `import yfinance` raise ImportError.
    out = benchmark_equity("SPY", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), 10_000.0)
    assert out is None


def test_benchmark_equity_returns_none_on_empty_frame(monkeypatch):
    fake_yf = type("FakeYF", (), {
        "Ticker": staticmethod(lambda t: type("T", (), {
            "history": staticmethod(lambda **kw: pd.DataFrame())
        })()),
        "set_tz_cache_location": staticmethod(lambda p: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    out = benchmark_equity("SPY", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), 10_000.0)
    assert out is None


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_on_empty_history_returns_zeros():
    m = compute_metrics([], [], initial_capital=10_000.0)
    assert isinstance(m, PerformanceMetrics)
    assert m.n_trades == 0
    assert m.n_round_trips == 0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.total_return_pct == 0.0


def test_compute_metrics_win_rate_and_profit_factor():
    trades = [
        make_trade(D0, "BUY"),
        make_trade(D0 + timedelta(days=1), "SELL", realized_pnl=100.0, portfolio_value=10_100.0),
        make_trade(D0 + timedelta(days=2), "BUY"),
        make_trade(D0 + timedelta(days=3), "SELL", realized_pnl=-50.0, portfolio_value=10_050.0),
    ]
    m = compute_metrics(trades, [], initial_capital=10_000.0)
    assert m.n_round_trips == 2
    assert m.n_wins == 1
    assert m.n_losses == 1
    assert m.win_rate == 0.5
    assert m.profit_factor == pytest.approx(2.0)
    assert m.expectancy == pytest.approx(25.0)


def test_compute_metrics_force_close_counts_as_a_round_trip():
    trades = [
        make_trade(D0, "BUY"),
        make_trade(D0 + timedelta(days=1), "FORCE_CLOSE", realized_pnl=-10.0),
    ]
    m = compute_metrics(trades, [], initial_capital=10_000.0)
    assert m.n_round_trips == 1
    assert m.n_losses == 1


def test_compute_metrics_profit_factor_is_capped_when_no_losses():
    trades = [
        make_trade(D0, "BUY"),
        make_trade(D0 + timedelta(days=1), "SELL", realized_pnl=100.0),
    ]
    m = compute_metrics(trades, [], initial_capital=10_000.0)
    assert m.profit_factor == 999.0  # inf sentinel


def test_compute_metrics_avg_holding_period_pairs_buy_and_sell():
    trades = [
        make_trade(D0, "BUY"),
        make_trade(D0 + timedelta(days=2), "SELL", realized_pnl=10.0),
    ]
    m = compute_metrics(trades, [], initial_capital=10_000.0)
    assert m.avg_holding_period_days == pytest.approx(2.0)


def test_compute_metrics_total_return_reflects_current_value():
    m = compute_metrics([], [], initial_capital=10_000.0, current_value=11_000.0)
    assert m.total_return_pct == pytest.approx(10.0)


def test_compute_metrics_sharpe_and_sortino_are_zero_with_insufficient_returns():
    snaps = [make_snapshot(date(2026, 1, 5), 10_000.0)]
    m = compute_metrics([], snaps, initial_capital=10_000.0)
    assert m.sharpe == 0.0
    assert m.sortino == 0.0
    assert m.volatility_pct == 0.0


def test_compute_metrics_drawdown_reflects_a_known_dip():
    snaps = [
        make_snapshot(date(2026, 1, 5), 10_000.0),
        make_snapshot(date(2026, 1, 6), 12_000.0),
        make_snapshot(date(2026, 1, 7), 6_000.0),
    ]
    m = compute_metrics([], snaps, initial_capital=10_000.0)
    assert m.max_drawdown_pct == pytest.approx(-50.0)
    assert m.max_drawdown_dollars == pytest.approx(-6_000.0)


def test_compute_metrics_calmar_zero_when_no_drawdown():
    snaps = [make_snapshot(date(2026, 1, 5), 10_000.0),
             make_snapshot(date(2026, 1, 6), 10_100.0)]
    m = compute_metrics([], snaps, initial_capital=10_000.0)
    assert m.calmar == 0.0


def test_performance_metrics_to_dict_roundtrips_all_fields():
    m = compute_metrics([], [], initial_capital=10_000.0)
    d = m.to_dict()
    assert set(d.keys()) == set(m.__dataclass_fields__.keys())


# --------------------------------------------------------------------------- #
# attribution: pnl_by_ticker / win_rate_by_ticker
# --------------------------------------------------------------------------- #
def test_pnl_by_ticker_empty_when_no_sells():
    df = pnl_by_ticker([make_trade(D0, "BUY")])
    assert df.empty
    assert list(df.columns) == ["ticker", "n_trades", "total_pnl", "avg_pnl", "win_rate"]


def test_pnl_by_ticker_aggregates_and_sorts_descending():
    trades = [
        make_trade(D0, "SELL", ticker="AAPL", realized_pnl=50.0),
        make_trade(D0, "SELL", ticker="MSFT", realized_pnl=200.0),
        make_trade(D0, "SELL", ticker="AAPL", realized_pnl=-10.0),
    ]
    df = pnl_by_ticker(trades)
    assert list(df["ticker"]) == ["MSFT", "AAPL"]
    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["n_trades"] == 2
    assert aapl["total_pnl"] == pytest.approx(40.0)
    assert aapl["win_rate"] == pytest.approx(0.5)


def test_win_rate_by_ticker_sorts_by_win_rate_descending():
    trades = [
        make_trade(D0, "SELL", ticker="A", realized_pnl=1.0),
        make_trade(D0, "SELL", ticker="B", realized_pnl=100.0),
        make_trade(D0, "SELL", ticker="B", realized_pnl=-100.0),
    ]
    df = win_rate_by_ticker(trades)
    assert df.iloc[0]["ticker"] == "A"  # 100% win rate vs B's 50%


# --------------------------------------------------------------------------- #
# attribution: pnl_by_weekday
# --------------------------------------------------------------------------- #
def test_pnl_by_weekday_empty_when_no_sells():
    df = pnl_by_weekday([make_trade(D0, "BUY")])
    assert df.empty


def test_pnl_by_weekday_orders_monday_to_sunday_and_fills_zero():
    # D0 is a Monday.
    trades = [make_trade(D0, "SELL", realized_pnl=30.0)]
    df = pnl_by_weekday(trades)
    assert list(df["weekday"]) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert df.iloc[0]["total_pnl"] == 30.0
    assert df.iloc[1]["total_pnl"] == 0.0
    assert df.iloc[1]["n_trades"] == 0


# --------------------------------------------------------------------------- #
# attribution: pnl_by_hour
# --------------------------------------------------------------------------- #
def test_pnl_by_hour_empty_when_no_sells():
    df = pnl_by_hour([make_trade(D0, "BUY")])
    assert df.empty


def test_pnl_by_hour_covers_all_24_hours_and_places_trade_correctly():
    ts = datetime(2026, 1, 5, 14, 30)
    trades = [make_trade(ts, "SELL", realized_pnl=75.0)]
    df = pnl_by_hour(trades)
    assert len(df) == 24
    assert list(df["hour"]) == list(range(24))
    row = df[df["hour"] == 14].iloc[0]
    assert row["total_pnl"] == 75.0
    assert df[df["hour"] == 0].iloc[0]["n_trades"] == 0


# --------------------------------------------------------------------------- #
# attribution: trade_durations
# --------------------------------------------------------------------------- #
def test_trade_durations_empty_input():
    df = trade_durations([])
    assert df.empty
    assert "duration_hours" in df.columns


def test_trade_durations_pairs_buy_with_next_sell():
    entry = D0
    exit_ = D0 + timedelta(hours=5)
    trades = [
        make_trade(entry, "BUY", ticker="AAPL", price=100.0),
        make_trade(exit_, "SELL", ticker="AAPL", price=110.0, realized_pnl=100.0),
    ]
    df = trade_durations(trades)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["duration_hours"] == pytest.approx(5.0)
    assert row["pnl_pct"] == pytest.approx(10.0)
    assert row["realized_pnl"] == 100.0


def test_trade_durations_ignores_a_sell_with_no_matching_open_buy():
    trades = [make_trade(D0, "SELL", ticker="AAPL", realized_pnl=5.0)]
    df = trade_durations(trades)
    assert df.empty


def test_trade_durations_force_close_also_pairs():
    trades = [
        make_trade(D0, "BUY", ticker="TSLA", price=200.0),
        make_trade(D0 + timedelta(hours=1), "FORCE_CLOSE", ticker="TSLA", price=190.0),
    ]
    df = trade_durations(trades)
    assert len(df) == 1
    assert df.iloc[0]["pnl_pct"] == pytest.approx(-5.0)


# --------------------------------------------------------------------------- #
# holdings_risk: concentration
# --------------------------------------------------------------------------- #
def test_concentration_empty_positions():
    df = concentration([])
    assert df.empty
    assert list(df.columns) == ["ticker", "market_value", "weight_pct", "hhi_contrib"]


def test_concentration_weights_and_hhi_sum_to_one_when_single_position():
    positions = [make_position("AAPL", 10, 100.0, 100.0)]  # market_value 1000
    df = concentration(positions)
    assert df.iloc[0]["weight_pct"] == 100.0
    assert df.iloc[0]["hhi_contrib"] == pytest.approx(1.0)


def test_concentration_two_equal_positions_hhi_is_half():
    positions = [
        make_position("AAPL", 10, 100.0, 100.0),   # 1000
        make_position("MSFT", 5, 200.0, 200.0),    # 1000
    ]
    df = concentration(positions)
    assert df["hhi_contrib"].sum() == pytest.approx(0.5)
    assert set(df["weight_pct"]) == {50.0}


def test_concentration_sorted_by_market_value_descending():
    positions = [
        make_position("SMALL", 1, 10.0, 10.0),
        make_position("BIG", 100, 50.0, 50.0),
    ]
    df = concentration(positions)
    assert list(df["ticker"]) == ["BIG", "SMALL"]


def test_concentration_skips_positions_that_error_on_market_value():
    class Bad:
        ticker = "BAD"
        @property
        def market_value(self):
            raise RuntimeError("boom")
    df = concentration([Bad(), make_position("OK", 1, 10.0, 10.0)])
    assert list(df["ticker"]) == ["OK"]


def test_concentration_zero_total_value_gives_zero_weights():
    positions = [make_position("AAPL", 0, 100.0, 0.0)]
    df = concentration(positions)
    assert df.iloc[0]["weight_pct"] == 0.0
    assert df.iloc[0]["hhi_contrib"] == 0.0


# --------------------------------------------------------------------------- #
# holdings_risk: correlation_matrix
# --------------------------------------------------------------------------- #
def test_correlation_matrix_returns_none_with_fewer_than_two_tickers():
    assert correlation_matrix(["AAPL"]) is None
    assert correlation_matrix([]) is None


def test_correlation_matrix_dedupes_and_uppercases_tickers(monkeypatch):
    import analytics.holdings_risk as hr_mod
    seen = {}

    def fake_download(tickers, **kw):
        seen["tickers"] = tickers
        return pd.DataFrame()

    fake_yf = type("FakeYF", (), {
        "download": staticmethod(fake_download),
        "set_tz_cache_location": staticmethod(lambda p: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    correlation_matrix(["aapl", "AAPL", " msft "])
    assert seen["tickers"] == "AAPL MSFT"


def test_correlation_matrix_returns_none_on_download_failure(monkeypatch):
    fake_yf = type("FakeYF", (), {
        "download": staticmethod(lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))),
        "set_tz_cache_location": staticmethod(lambda p: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    assert correlation_matrix(["AAPL", "MSFT"]) is None


def test_correlation_matrix_returns_none_on_empty_frame(monkeypatch):
    fake_yf = type("FakeYF", (), {
        "download": staticmethod(lambda **kw: pd.DataFrame()),
        "set_tz_cache_location": staticmethod(lambda p: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    assert correlation_matrix(["AAPL", "MSFT"]) is None


def _multi_index_download(tickers: list[str], n: int, seed_offset: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    frames = {}
    rng = np.random.default_rng(42)
    for i, tk in enumerate(tickers):
        base = 100.0 + i * 10 + seed_offset
        closes = base + np.cumsum(rng.normal(0, 1, n))
        frames[(tk, "Close")] = closes
        frames[(tk, "Open")] = closes
    cols = pd.MultiIndex.from_tuples(frames.keys())
    return pd.DataFrame(frames.values(), index=cols).T.set_axis(idx)


def test_correlation_matrix_computes_pearson_correlation_from_multiindex(monkeypatch):
    tickers = ["AAPL", "MSFT"]
    df = _multi_index_download(tickers, 40)

    fake_yf = type("FakeYF", (), {
        "download": staticmethod(lambda **kw: df),
        "set_tz_cache_location": staticmethod(lambda p: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    corr = correlation_matrix(tickers)
    assert corr is not None
    assert set(corr.columns) == {"AAPL", "MSFT"}
    assert corr.loc["AAPL", "AAPL"] == pytest.approx(1.0)
    # Symmetric.
    assert corr.loc["AAPL", "MSFT"] == pytest.approx(corr.loc["MSFT", "AAPL"])


def test_correlation_matrix_none_when_too_few_return_rows(monkeypatch):
    tickers = ["AAPL", "MSFT"]
    df = _multi_index_download(tickers, 3)  # only 3 rows -> < 5 return rows

    fake_yf = type("FakeYF", (), {
        "download": staticmethod(lambda **kw: df),
        "set_tz_cache_location": staticmethod(lambda p: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    assert correlation_matrix(tickers) is None


def test_correlation_matrix_none_when_a_ticker_has_no_close_column(monkeypatch):
    # Only one of the two requested tickers actually comes back.
    df = _multi_index_download(["AAPL"], 40)

    fake_yf = type("FakeYF", (), {
        "download": staticmethod(lambda **kw: df),
        "set_tz_cache_location": staticmethod(lambda p: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    assert correlation_matrix(["AAPL", "MSFT"]) is None
