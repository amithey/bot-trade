"""Validate execution inputs and separate completed candles from live marks."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalBars:
    data: pd.DataFrame
    bar: str
    fresh: bool
    reason: str


def completed_bars(df: pd.DataFrame, interval: str, *, now=None) -> SignalBars:
    """Provider indices denote bar OPEN times; naive indices are treated as UTC."""
    durations = {"5m": pd.Timedelta(minutes=5), "1d": pd.Timedelta(days=1)}
    if interval not in durations:
        raise ValueError(f"Unsupported live signal interval: {interval}")
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_unique:
        raise ValueError("Market bars need unique datetime indices")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Market bars are out of order")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    values = df[fields].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Market bars contain invalid OHLCV values")
    if ((df.High < df[["Open", "Close", "Low"]].max(axis=1)).any()
            or (df.Low > df[["Open", "Close", "High"]].min(axis=1)).any()):
        raise ValueError("Inconsistent candle high/low")
    clock = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    clock = clock.tz_localize("UTC") if clock.tzinfo is None else clock.tz_convert("UTC")
    index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    duration = durations[interval]
    data = df.loc[index + duration <= clock].copy()
    if data.empty:
        return SignalBars(data, "", False, "No completed candle available")
    close_time = index[index + duration <= clock][-1] + duration
    fresh = clock - close_time <= duration * 3
    return SignalBars(data, close_time.isoformat(), fresh,
                      "Completed candle available" if fresh else "Latest completed candle is stale")
