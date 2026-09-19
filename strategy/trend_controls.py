"""Causal ADX screening and position-specific Chandelier ratcheting.

Research components; not enabled in live COMMITTEE without validation.
Raw Chandelier bands may move backwards. Only the position stop is ratcheted.
"""
from math import isfinite

import pandas as pd

from strategy.committee import _adx, _atr


def trend_control_features(df, *, lookback=22, atr_multiple=3., chandelier_timeframe="5min"):
    if lookback < 2 or not isfinite(atr_multiple) or atr_multiple <= 0:
        raise ValueError("Invalid Chandelier parameters")
    adx, plus_di, minus_di = _adx(df, 14)
    if chandelier_timeframe not in ("5min", "30min", "1h"):
        raise ValueError("Unsupported Chandelier timeframe")
    source = df
    if chandelier_timeframe != "5min":
        # Input timestamps are five-minute bar OPENS. A higher-timeframe
        # value is available only when its last component bar has closed.
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.has_duplicates or not df.index.is_monotonic_increasing:
            raise ValueError("Need unique chronological five-minute candles")
        step = pd.Timedelta(minutes=5)
        if (df.index.asi8 % step.value != 0).any():
            raise ValueError("Candles must align to five-minute boundaries")
        grouped = df.resample(chandelier_timeframe, label="right", closed="left")
        source = grouped.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        required = int(pd.Timedelta(chandelier_timeframe) / step)
        source = source.loc[grouped.Close.count() == required]
    atr = _atr(source, lookback)
    bands = pd.DataFrame({
        "chandelier_long": source.High.rolling(lookback, min_periods=lookback).max() - atr_multiple * atr,
        "chandelier_short": source.Low.rolling(lookback, min_periods=lookback).min() + atr_multiple * atr,
    })
    if chandelier_timeframe != "5min":
        bands = bands.reindex(df.index + pd.Timedelta(minutes=5), method="ffill")
        bands.index = df.index
    return pd.DataFrame({
        "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
        "chandelier_long": bands.chandelier_long,
        "chandelier_short": bands.chandelier_short,
    }, index=df.index)


def ratchet_chandelier(side, previous, candidate):
    if side not in ("LONG", "SHORT"):
        raise ValueError("Unknown position side")
    if not isfinite(candidate) or candidate <= 0:
        return previous
    if previous is None:
        return float(candidate)
    if not isfinite(previous) or previous <= 0:
        raise ValueError("Invalid persisted stop")
    return float(max(previous, candidate) if side == "LONG" else min(previous, candidate))
