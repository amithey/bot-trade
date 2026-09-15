"""Bounded, price-only quotes for the independent paper risk monitor."""
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class ProtectionQuote:
    price: float
    bar_start: float  # UTC Unix seconds, provider's 1-minute candle timestamp
    requested_at: float

    def validate(self, now=None):
        now = datetime.now(timezone.utc).timestamp() if now is None else now
        if not all(isfinite(v) for v in (self.price, self.bar_start, self.requested_at)):
            raise ValueError("Invalid protection quote")
        # One-minute candles can be open. Allow one minute of provider delay.
        if (self.price <= 0 or not -5 <= now - self.bar_start <= 120
                or not -5 <= now - self.requested_at <= 30):
            raise ValueError("Stale or invalid protection quote; no simulated fill")
        return self


def latest_protection_quote(ticker):
    requested = datetime.now(timezone.utc).timestamp()
    # Call the chart feed directly: Ticker.history may initialize timezone,
    # cookie databases and fundamentals with separate, longer requests.
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(ticker, safe="")
    with requests.get(
        url, params={"range": "1d", "interval": "1m", "includePrePost": "false"},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=(2, 5), allow_redirects=False,
    ) as response:
        response.raise_for_status()
        payload = response.json()["chart"]
    if payload.get("error") or not payload.get("result"):
        raise ValueError("No protection quote available")
    result = payload["result"][0]
    stamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0].get("close") or []
    if not stamps or len(stamps) != len(closes) or closes[-1] is None:
        raise ValueError("Incomplete protection quote")
    return ProtectionQuote(
        float(closes[-1]), float(stamps[-1]), requested,
    ).validate()
