"""Pre-declared daily chart-pattern rules for out-of-sample research.

Every rule returns the desired long exposure (0/1) decided at each bar's close,
using only that bar and earlier ones. The caller trades it at the next open.
The variant list is fixed here, before looking at results, so the number of
hypotheses tested is known and can be disclosed. Hypotheses, not forecasts.
"""
import re

import numpy as np
import pandas as pd

from strategy.committee_evidence import chart_features


def _state(enter, exit_, max_hold=None):
    entered, exited = enter.fillna(False).to_numpy(bool), exit_.fillna(False).to_numpy(bool)
    out, on, age = np.zeros(len(entered)), False, 0
    for i in range(len(entered)):
        if on:
            age += 1
            if exited[i] or (max_hold is not None and age >= max_hold):
                on = False
        elif entered[i]:
            on, age = True, 0
        out[i] = on
    return pd.Series(out, index=enter.index)


def _rsi(close, n):
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def donchian(df, entry, exit_):
    """Close above the prior N-day high; leave below the prior M-day low."""
    up = df.Close > df.High.rolling(entry).max().shift()
    down = df.Close < df.Low.rolling(exit_).min().shift()
    return _state(up, down)


def ma_trend(df, fast, slow):
    average = df.Close.rolling
    return ((df.Close > average(slow).mean()) & (average(fast).mean() > average(slow).mean())).astype(float)


def time_series_momentum(df, lookback):
    return (df.Close / df.Close.shift(lookback) - 1 > 0).astype(float)


def rsi2_pullback(df, oversold):
    """Short-term dip inside a long-term uptrend; exit on recovery above the 5-day mean."""
    dip = (df.Close > df.Close.rolling(200).mean()) & (_rsi(df.Close, 2) < oversold)
    return _state(dip, df.Close > df.Close.rolling(5).mean())


def squeeze_breakout(df, percentile):
    """Volatility contraction (Bollinger width in its lowest tail) followed by an upper-band break."""
    mid, spread = df.Close.rolling(20).mean(), df.Close.rolling(20).std()
    width_rank = (4 * spread / mid).rolling(120).rank(pct=True)
    squeezed = width_rank.shift().rolling(5).min() <= percentile
    return _state(squeezed & (df.Close > mid + 2 * spread), df.Close < mid)


def flag_breakout(df):
    """Strong prior run, then a tight 15-day consolidation broken to the upside."""
    run = df.Close.shift(15) / df.Close.shift(75) - 1 > .15
    high, low = df.High.rolling(15).max(), df.Low.rolling(15).min()
    tight = (high - low) / df.Close < .12
    broke = df.Close > high.shift()
    return _state(run & tight.shift() & broke, df.Close < df.Low.rolling(10).min().shift())


def swing_structure(df):
    """Confirmed higher highs and higher lows while price holds its 50-day mean."""
    return ((chart_features(df).structure == "RISING") & (df.Close > df.Close.rolling(50).mean())).astype(float)


def failed_breakdown_reversal(df):
    """A rejected 20-day low inside a long-term uptrend; hold at most ten days."""
    entry = chart_features(df).failed_breakdown & (df.Close > df.Close.rolling(200).mean())
    return _state(entry, df.Close < df.Low.rolling(20).min().shift(), max_hold=10)


VARIANTS = {
    "donchian_20_10": lambda d: donchian(d, 20, 10),
    "donchian_55_20": lambda d: donchian(d, 55, 20),
    "donchian_100_50": lambda d: donchian(d, 100, 50),
    "ma_trend_50_200": lambda d: ma_trend(d, 50, 200),
    "ma_trend_20_100": lambda d: ma_trend(d, 20, 100),
    "ma_trend_10_50": lambda d: ma_trend(d, 10, 50),
    "tsmom_60": lambda d: time_series_momentum(d, 60),
    "tsmom_120": lambda d: time_series_momentum(d, 120),
    "tsmom_252": lambda d: time_series_momentum(d, 252),
    "rsi2_pullback_10": lambda d: rsi2_pullback(d, 10),
    "rsi2_pullback_5": lambda d: rsi2_pullback(d, 5),
    "squeeze_breakout_20": lambda d: squeeze_breakout(d, .20),
    "squeeze_breakout_10": lambda d: squeeze_breakout(d, .10),
    "flag_breakout": flag_breakout,
    "swing_structure": swing_structure,
    "failed_breakdown_reversal": failed_breakdown_reversal,
}
FAMILY = {name: re.sub(r"(_\d+)+$", "", name) for name in VARIANTS}
