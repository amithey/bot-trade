"""
Earnings Calendar — next upcoming earnings date for a ticker.

A stock's reported earnings date is a binary event that can move the
price 5-15% in seconds. The AI engine uses this module to downgrade
BUY conviction and shrink position sizes when earnings are imminent.

Powered by yfinance (`Ticker.calendar` / `Ticker.get_earnings_dates`),
so no extra dependency. Results are cached per-ticker for 4 hours.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import yfinance as yf

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EarningsInfo:
    ticker: str
    next_date: Optional[date]          # date of next scheduled report
    days_until: Optional[int]          # calendar days from today (negative → past)

    @property
    def is_imminent(self) -> bool:
        """True when earnings are within the next 3 calendar days."""
        return self.days_until is not None and 0 <= self.days_until <= 3

    @property
    def is_near(self) -> bool:
        """True when earnings are within the next 7 calendar days."""
        return self.days_until is not None and 0 <= self.days_until <= 7

    def summary(self) -> str:
        if self.next_date is None:
            return "(no scheduled earnings date)"
        if self.days_until is None:
            return f"{self.next_date.isoformat()}"
        if self.days_until < 0:
            return f"{self.next_date.isoformat()} ({-self.days_until}d ago)"
        if self.days_until == 0:
            return f"{self.next_date.isoformat()} (TODAY)"
        return f"{self.next_date.isoformat()} (in {self.days_until}d)"


class EarningsCalendar:
    """Fetches and caches the next earnings date for a ticker."""

    def __init__(self, ttl_seconds: int = 4 * 3600) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, EarningsInfo]] = {}

    def fetch(self, ticker: str) -> EarningsInfo:
        ticker = ticker.upper()
        now = time.time()
        cached = self._cache.get(ticker)
        if cached and (now - cached[0] < self._ttl):
            return cached[1]

        info = self._query(ticker)
        self._cache[ticker] = (now, info)
        return info

    @staticmethod
    def _query(ticker: str) -> EarningsInfo:
        today = datetime.now(timezone.utc).date()
        next_dt: Optional[date] = None
        try:
            tk = yf.Ticker(ticker)
            # Preferred: get_earnings_dates() returns a DataFrame indexed by
            # timezone-aware Timestamps — pick the first future date.
            try:
                df = tk.get_earnings_dates(limit=8)
            except Exception:
                df = None

            if df is not None and not df.empty:
                future = [idx.date() for idx in df.index if idx.date() >= today]
                if future:
                    next_dt = min(future)

            # Fallback: Ticker.calendar dict
            if next_dt is None:
                cal = getattr(tk, "calendar", None)
                if isinstance(cal, dict):
                    raw = cal.get("Earnings Date")
                    if isinstance(raw, list) and raw:
                        candidates = []
                        for r in raw:
                            if isinstance(r, datetime):
                                candidates.append(r.date())
                            elif isinstance(r, date):
                                candidates.append(r)
                        future = [d for d in candidates if d >= today]
                        if future:
                            next_dt = min(future)
        except Exception as exc:
            logger.debug(f"Earnings lookup failed for {ticker}: {exc}")

        days_until = (next_dt - today).days if next_dt is not None else None
        return EarningsInfo(
            ticker=ticker, next_date=next_dt, days_until=days_until,
        )
