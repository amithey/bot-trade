"""
Tests for market_data/earnings.py — the earnings-date calendar the AI engine
uses to downgrade conviction and shrink position size ahead of a binary
earnings event.

No network: yf.Ticker is monkeypatched everywhere. get_earnings_dates() and
the Ticker.calendar fallback are exercised separately since the source tries
the former first and only falls back to the latter when it fails or returns
nothing.

Dates are built relative to date.today() rather than by freezing the clock -
_query's own isinstance(r, datetime) check on the calendar-fallback path
needs the real datetime class to keep behaving like itself, and swapping the
module's `datetime` symbol for a stand-in breaks that silently.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

import market_data.earnings as earnings_mod
from market_data.earnings import EarningsCalendar, EarningsInfo

TODAY = date.today()


def _future(days: int) -> date:
    return TODAY + timedelta(days=days)


def _past(days: int) -> date:
    return TODAY - timedelta(days=days)


# --------------------------------------------------------------------------- #
# EarningsInfo
# --------------------------------------------------------------------------- #
def test_is_imminent_within_three_days():
    assert EarningsInfo("T", date(2026, 1, 1), 0).is_imminent is True
    assert EarningsInfo("T", date(2026, 1, 1), 3).is_imminent is True
    assert EarningsInfo("T", date(2026, 1, 1), 4).is_imminent is False


def test_is_imminent_false_for_a_past_date():
    assert EarningsInfo("T", date(2026, 1, 1), -1).is_imminent is False


def test_is_near_within_seven_days():
    assert EarningsInfo("T", date(2026, 1, 1), 7).is_near is True
    assert EarningsInfo("T", date(2026, 1, 1), 8).is_near is False


def test_is_imminent_and_near_are_false_with_no_date():
    info = EarningsInfo("T", None, None)
    assert info.is_imminent is False and info.is_near is False


def test_summary_with_no_scheduled_date():
    assert EarningsInfo("T", None, None).summary() == "(no scheduled earnings date)"


def test_summary_today():
    info = EarningsInfo("T", date(2026, 6, 15), 0)
    assert "TODAY" in info.summary()


def test_summary_future():
    info = EarningsInfo("T", date(2026, 6, 20), 5)
    assert "in 5d" in info.summary()


def test_summary_past():
    info = EarningsInfo("T", date(2026, 6, 1), -14)
    assert "14d ago" in info.summary()


# --------------------------------------------------------------------------- #
# EarningsCalendar._query — get_earnings_dates() path
# --------------------------------------------------------------------------- #
def _dates_ticker(dates: list[date]):
    """A yf.Ticker stand-in whose get_earnings_dates() returns *dates*."""
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dates])
    frame = pd.DataFrame({"EPS Estimate": [1.0] * len(dates)}, index=idx)

    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            return frame
        calendar = {}
    return FakeTicker()


def test_query_picks_the_nearest_future_date(monkeypatch):
    dates = [_future(30), _future(90), _future(5)]   # unordered
    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: _dates_ticker(dates))
    info = EarningsCalendar._query("AAPL")
    assert info.next_date == _future(5)
    assert info.days_until == 5


def test_query_returns_none_date_when_only_past_dates_exist(monkeypatch):
    monkeypatch.setattr(earnings_mod.yf, "Ticker",
                        lambda t: _dates_ticker([_past(400)]))
    info = EarningsCalendar._query("AAPL")
    assert info.next_date is None
    assert info.days_until is None


def test_query_falls_back_to_calendar_when_get_earnings_dates_is_empty(monkeypatch):
    future_dt = datetime(_future(10).year, _future(10).month, _future(10).day,
                         tzinfo=timezone.utc)

    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            return pd.DataFrame()
        calendar = {"Earnings Date": [future_dt]}

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    info = EarningsCalendar._query("AAPL")
    assert info.next_date == _future(10)


def test_query_falls_back_to_calendar_when_get_earnings_dates_raises(monkeypatch):
    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            raise RuntimeError("unsupported")
        calendar = {"Earnings Date": [_future(10)]}

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    info = EarningsCalendar._query("AAPL")
    assert info.next_date == _future(10)


def test_query_calendar_fallback_picks_the_earliest_future_date(monkeypatch):
    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            return pd.DataFrame()
        calendar = {"Earnings Date": [_future(90), _future(5)]}

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    info = EarningsCalendar._query("AAPL")
    assert info.next_date == _future(5)


def test_query_calendar_fallback_ignores_past_dates(monkeypatch):
    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            return pd.DataFrame()
        calendar = {"Earnings Date": [_past(10)]}

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    info = EarningsCalendar._query("AAPL")
    assert info.next_date is None


def test_query_handles_a_non_dict_calendar_gracefully(monkeypatch):
    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            return pd.DataFrame()
        calendar = None   # yfinance sometimes returns this for delisted/odd tickers

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    info = EarningsCalendar._query("WEIRD")
    assert info.next_date is None


def test_query_never_raises_when_the_ticker_itself_blows_up(monkeypatch):
    def _explode(t):
        raise RuntimeError("no such ticker")
    monkeypatch.setattr(earnings_mod.yf, "Ticker", _explode)
    info = EarningsCalendar._query("BAD")
    assert info.next_date is None and info.days_until is None
    assert info.ticker == "BAD"


def test_query_does_not_case_fold_the_ticker(monkeypatch):
    """_query itself doesn't uppercase - that normalisation is fetch()'s job
    (tested below), so calling _query directly with lowercase must round-trip
    it unchanged onto the result."""
    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: _dates_ticker([]))
    info = EarningsCalendar._query("aapl")
    assert info.ticker == "aapl"


# --------------------------------------------------------------------------- #
# EarningsCalendar.fetch — caching
# --------------------------------------------------------------------------- #
def test_fetch_uppercases_the_ticker(monkeypatch):
    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            return pd.DataFrame()
        calendar = {}

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    info = EarningsCalendar().fetch("aapl")
    assert info.ticker == "AAPL"


def test_fetch_caches_within_the_ttl(monkeypatch):
    calls = {"n": 0}

    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            calls["n"] += 1
            return pd.DataFrame()
        calendar = {}

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    cal = EarningsCalendar(ttl_seconds=4 * 3600)
    cal.fetch("AAPL")
    cal.fetch("AAPL")
    assert calls["n"] == 1


def test_fetch_refetches_after_the_ttl_expires(monkeypatch):
    calls = {"n": 0}

    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            calls["n"] += 1
            return pd.DataFrame()
        calendar = {}

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    clock = {"t": 1000.0}
    monkeypatch.setattr(earnings_mod.time, "time", lambda: clock["t"])
    cal = EarningsCalendar(ttl_seconds=4 * 3600)
    cal.fetch("AAPL")
    clock["t"] += 4 * 3600 + 1
    cal.fetch("AAPL")
    assert calls["n"] == 2


def test_fetch_caches_per_ticker_independently(monkeypatch):
    calls = {"n": 0}

    class FakeTicker:
        def get_earnings_dates(self, limit=8):
            calls["n"] += 1
            return pd.DataFrame()
        calendar = {}

    monkeypatch.setattr(earnings_mod.yf, "Ticker", lambda t: FakeTicker())
    cal = EarningsCalendar()
    cal.fetch("AAPL")
    cal.fetch("QQQ")
    assert calls["n"] == 2
