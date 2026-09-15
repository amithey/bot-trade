"""Causal intraday market research shared by live trading and backtests.

The policy separates market regime, trade setup, confirmation, and
invalidation. All rolling levels are shifted where required so a signal
never sees a future candle. The output is evidence, not a promise of profit.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / period, adjust=False, min_periods=period,
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, adjust=False, min_periods=period,
    ).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask((loss == 0) & (gain > 0), 100.0)
    result = result.mask((gain == 0) & (loss > 0), 0.0)
    result = result.mask((gain == 0) & (loss == 0), 50.0)
    return result.where(loss.notna() & gain.notna())


def entry_features(
    df: pd.DataFrame,
    risk_profile: str = "Balanced",
) -> pd.DataFrame:
    """Build a causal, regime-aware intraday setup matrix.

    The old policy required an exact 20-bar breakout or one textbook
    pullback candle. This version recognizes four long setups and gives
    faster signals a tighter extension limit. Risk profiles alter
    confirmation thresholds, but never bypass data quality.
    """
    close = df.Close.astype(float)
    open_ = df.Open.astype(float)
    high = df.High.astype(float)
    low = df.Low.astype(float)
    volume = df.Volume.astype(float)

    ema8 = close.ewm(span=8, adjust=False, min_periods=8).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(),
         (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    prior_volume = volume.rolling(20, min_periods=20).mean().shift()
    volume_ratio = volume / prior_volume.replace(0, np.nan)
    rsi = _rsi(close)
    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    prior_high_20 = high.rolling(20, min_periods=20).max().shift()
    prior_low_20 = low.rolling(20, min_periods=20).min().shift()
    prior_high_6 = high.rolling(6, min_periods=6).max().shift()

    span = (high - low).replace(0, np.nan)
    close_location = ((close - low) / span).fillna(0.5)
    body_atr = (close - open_).abs() / atr.replace(0, np.nan)
    extension = (close - ema21) / atr.replace(0, np.nan)
    trend_strength = (ema21 - ema50) / atr.replace(0, np.nan)
    median_bar = (
        df.index.to_series().diff().median()
        if isinstance(df.index, pd.DatetimeIndex) else pd.NaT
    )
    if pd.notna(median_bar) and median_bar <= pd.Timedelta(minutes=15):
        # Use the prior fully completed hourly candle. Shifting before
        # forward-filling prevents early five-minute rows from seeing the
        # rest of their hour.
        hourly_close = close.resample("1h").last().shift()
        hourly_fast = hourly_close.ewm(
            span=20, adjust=False, min_periods=20,
        ).mean()
        hourly_slow = hourly_close.ewm(
            span=50, adjust=False, min_periods=50,
        ).mean()
        htf_fast = hourly_fast.reindex(df.index, method="ffill")
        htf_slow = hourly_slow.reindex(df.index, method="ffill")
    else:
        htf_fast, htf_slow = ema21, ema50
    htf_bias = pd.Series("UNKNOWN", index=df.index)
    htf_bias.loc[
        (htf_fast > htf_slow) & (htf_slow > htf_slow.shift(3))
    ] = "LONG"
    htf_bias.loc[
        (htf_fast < htf_slow) & (htf_slow < htf_slow.shift(3))
    ] = "SHORT"
    rising_trend = (
        (close > ema50) & (ema21 > ema50) & (ema50 > ema50.shift(6))
    )
    falling_trend = (
        (close < ema50) & (ema21 < ema50) & (ema50 < ema50.shift(6))
    )
    regime = pd.Series("RANGE", index=df.index)
    regime.loc[rising_trend] = "UPTREND"
    regime.loc[falling_trend] = "DOWNTREND"

    bullish_candle = (close > open_) & (close_location >= 0.58)
    bearish_candle = (close < open_) & (close_location <= 0.42)
    momentum_improving = (
        (macd_hist > macd_hist.shift()) & (rsi > rsi.shift())
    )
    momentum_weakening = (
        (macd_hist < macd_hist.shift()) & (rsi < rsi.shift())
    )
    recent_pullback = (
        low.rolling(4, min_periods=4).min() <= ema21 + 0.20 * atr
    )

    breakout = (
        regime.ne("DOWNTREND")
        & (close > prior_high_20)
        & bullish_candle
        & rsi.between(52, 100)
    )
    pullback = (
        regime.eq("UPTREND")
        & recent_pullback
        & (close.shift() <= ema8.shift())
        & (close > ema8)
        & bullish_candle
        & momentum_improving
        & rsi.between(44, 70)
    )
    continuation = (
        regime.eq("UPTREND")
        & (close > prior_high_6)
        & (ema8 > ema21)
        & (macd_hist > 0)
        & momentum_improving
        & rsi.between(50, 72)
    )
    range_reversal = (
        regime.eq("RANGE")
        & (low <= bb_lower)
        & (close > bb_lower)
        & bullish_candle
        & (rsi.shift() < 42)
        & (rsi > rsi.shift())
    )
    countertrend = (
        regime.eq("DOWNTREND")
        & (low <= prior_low_20)
        & (close > prior_low_20)
        & bullish_candle
        & momentum_improving
        & (rsi.shift() < 35)
    )
    short_breakdown = (
        regime.ne("UPTREND")
        & (close < prior_low_20)
        & bearish_candle
        & rsi.between(0, 48)
    )
    recent_rally = (
        high.rolling(4, min_periods=4).max() >= ema21 - 0.20 * atr
    )
    rally_rejection = (
        regime.eq("DOWNTREND")
        & recent_rally
        & (close.shift() >= ema8.shift())
        & (close < ema8)
        & bearish_candle
        & momentum_weakening
        & rsi.between(30, 56)
    )
    range_short = (
        regime.eq("RANGE")
        & (high >= bb_upper)
        & (close < bb_upper)
        & bearish_candle
        & (rsi.shift() > 58)
        & (rsi < rsi.shift())
    )

    setup = pd.Series("NONE", index=df.index)
    setup.loc[range_reversal] = "RANGE_REVERSAL"
    if risk_profile in ("Aggressive", "Micro-Scalp"):
        setup.loc[continuation] = "MOMENTUM_CONTINUATION"
    setup.loc[pullback] = "TREND_PULLBACK"
    setup.loc[breakout] = "TREND_BREAKOUT"
    if risk_profile in ("Aggressive", "Micro-Scalp"):
        setup.loc[countertrend] = "COUNTERTREND_REVERSAL"
    if risk_profile == "Micro-Scalp":
        setup.loc[countertrend] = "SCALP_REVERSAL"
    setup.loc[range_short] = "SHORT_RANGE_REVERSAL"
    setup.loc[rally_rejection] = "SHORT_RALLY_REJECTION"
    setup.loc[short_breakdown] = "SHORT_TREND_BREAKDOWN"

    volume_floor = {
        "Conservative": 1.00,
        "Balanced": 0.75,
        "Aggressive": 0.60,
        "Micro-Scalp": 0.65,
    }.get(risk_profile, 0.75)
    extension_cap = {
        "Conservative": 1.25,
        "Balanced": 2.00,
        "Aggressive": 2.25,
        "Micro-Scalp": 1.50,
    }.get(risk_profile, 2.00)
    # A decisive candle can compensate for average volume. Requiring >1x
    # volume on every five-minute bar caused long signal droughts.
    participation = (
        (volume_ratio >= volume_floor)
        | ((volume_ratio >= 0.50) & (body_atr >= 0.55))
    )
    not_exhausted = extension.between(-2.0, extension_cap)
    short_candidate = setup.str.startswith("SHORT_")
    macro_ok = (
        (risk_profile != "Conservative")
        | ((~short_candidate) & (close >= sma200))
        | (short_candidate & (close <= sma200))
    )
    ready = (
        sma200.notna() & atr.gt(0) & volume_ratio.notna() & rsi.notna()
    )
    timeframe_aligned = (
        htf_bias.eq("UNKNOWN")
        | (short_candidate & htf_bias.eq("SHORT"))
        | ((~short_candidate) & htf_bias.eq("LONG"))
        | (risk_profile in ("Aggressive", "Micro-Scalp"))
    )
    eligible = (
        ready
        & setup.ne("NONE")
        & participation
        & not_exhausted
        & macro_ok
        & timeframe_aligned
    )

    required_score = pd.Series(np.inf, index=df.index)
    required_score.loc[setup.eq("TREND_BREAKOUT")] = 0.08
    required_score.loc[setup.eq("TREND_PULLBACK")] = 0.10
    required_score.loc[setup.eq("MOMENTUM_CONTINUATION")] = 0.12
    required_score.loc[setup.eq("RANGE_REVERSAL")] = -0.08
    required_score.loc[
        setup.isin(["COUNTERTREND_REVERSAL", "SCALP_REVERSAL"])
    ] = -0.05
    required_score.loc[setup.eq("SHORT_TREND_BREAKDOWN")] = -0.08
    required_score.loc[setup.eq("SHORT_RALLY_REJECTION")] = -0.10
    required_score.loc[setup.eq("SHORT_RANGE_REVERSAL")] = 0.08
    signal_confidence = pd.Series(0.0, index=df.index)
    signal_confidence.loc[setup.eq("TREND_BREAKOUT")] = 0.68
    signal_confidence.loc[setup.eq("TREND_PULLBACK")] = 0.64
    signal_confidence.loc[setup.eq("MOMENTUM_CONTINUATION")] = 0.62
    signal_confidence.loc[setup.eq("RANGE_REVERSAL")] = 0.60
    signal_confidence.loc[
        setup.isin(["COUNTERTREND_REVERSAL", "SCALP_REVERSAL"])
    ] = 0.58
    signal_confidence.loc[setup.eq("SHORT_TREND_BREAKDOWN")] = 0.68
    signal_confidence.loc[setup.eq("SHORT_RALLY_REJECTION")] = 0.64
    signal_confidence.loc[setup.eq("SHORT_RANGE_REVERSAL")] = 0.60
    signal_confidence += (volume_ratio >= 1.20).astype(float) * 0.06
    signal_confidence += (body_atr >= 0.75).astype(float) * 0.04
    signal_confidence = signal_confidence.clip(upper=0.82)

    # Completed-candle exits supplement committee exits. Hard stops still
    # run continuously in the live engine.
    trend_break = (
        (close < ema21)
        & (ema8 < ema21)
        & (ema8.shift() >= ema21.shift())
        & (macd_hist < 0)
        & (close_location < 0.45)
    )
    range_target = (
        regime.eq("RANGE")
        & (high >= bb_upper)
        & (close < bb_upper)
        & (rsi >= 62)
        & (close_location < 0.50)
    )
    exhaustion = (
        (rsi >= 76)
        & (close_location < 0.35)
        & (macd_hist < macd_hist.shift())
    )
    exit_setup = pd.Series("NONE", index=df.index)
    exit_setup.loc[trend_break] = "MOMENTUM_FAILURE"
    exit_setup.loc[range_target] = "RANGE_TARGET_REJECTION"
    exit_setup.loc[exhaustion] = "EXHAUSTION_REVERSAL"
    short_trend_break = (
        (close > ema21)
        & (ema8 > ema21)
        & (ema8.shift() <= ema21.shift())
        & (macd_hist > 0)
        & (close_location > 0.55)
    )
    short_range_target = (
        regime.eq("RANGE")
        & (low <= bb_lower)
        & (close > bb_lower)
        & (rsi <= 38)
        & (close_location > 0.50)
    )
    short_exhaustion = (
        (rsi <= 24)
        & (close_location > 0.65)
        & (macd_hist > macd_hist.shift())
    )
    short_exit_setup = pd.Series("NONE", index=df.index)
    short_exit_setup.loc[short_trend_break] = "SHORT_MOMENTUM_FAILURE"
    short_exit_setup.loc[short_range_target] = "SHORT_RANGE_TARGET"
    short_exit_setup.loc[short_exhaustion] = "SHORT_EXHAUSTION_REVERSAL"

    return pd.DataFrame(
        {
            "ready": ready,
            "regime": regime,
            "setup": setup,
            "eligible": eligible,
            "required_committee_score": required_score,
            "signal_confidence": signal_confidence,
            "exit_setup": exit_setup,
            "exit_recommended": exit_setup.ne("NONE"),
            "short_exit_setup": short_exit_setup,
            "short_exit_recommended": short_exit_setup.ne("NONE"),
            "signal_side": np.where(
                setup.str.startswith("SHORT_"), "SHORT",
                np.where(setup.ne("NONE"), "LONG", "NONE"),
            ),
            "close": close,
            "ema8": ema8,
            "ema20": ema21,
            "ema21": ema21,
            "ema50": ema50,
            "sma50": ema50,
            "sma200": sma200,
            "atr": atr,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "volume_ratio": volume_ratio,
            "body_atr": body_atr,
            "extension_atr": extension,
            "trend_strength_atr": trend_strength,
            "htf_fast": htf_fast,
            "htf_slow": htf_slow,
            "higher_timeframe_bias": htf_bias,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "support": prior_low_20,
            "resistance": prior_high_20,
        },
        index=df.index,
    )


@dataclass(frozen=True)
class EntryResearch:
    regime: str
    setup: str
    entry_allowed: bool
    checks: list[dict]
    metrics: dict
    bar: str
    exit_setup: str = "NONE"
    exit_recommended: bool = False
    short_exit_setup: str = "NONE"
    short_exit_recommended: bool = False
    signal_side: str = "NONE"
    timeframe_bias: str = "UNKNOWN"
    coverage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def explanation(self) -> str:
        failed = [
            check["detail"] for check in self.checks if not check["passed"]
        ]
        if failed:
            return "; ".join(failed)
        return (
            f"{self.setup}: regime, trigger, participation and "
            "price-location checks passed"
        )


def analyze_entry(
    df: pd.DataFrame,
    *,
    risk_profile: str = "Balanced",
    fundamentals=None,
) -> EntryResearch:
    f = entry_features(df, risk_profile).iloc[-1]
    volume_floor = {
        "Conservative": 1.00,
        "Balanced": 0.75,
        "Aggressive": 0.60,
        "Micro-Scalp": 0.65,
    }.get(risk_profile, 0.75)
    cap = {
        "Conservative": 1.25,
        "Balanced": 2.00,
        "Aggressive": 2.25,
        "Micro-Scalp": 1.50,
    }.get(risk_profile, 2.00)
    participation = bool(
        f.volume_ratio >= volume_floor
        or (f.volume_ratio >= 0.50 and f.body_atr >= 0.55)
    )
    signal_side = (
        "SHORT" if str(f.setup).startswith("SHORT_")
        else "LONG" if f.setup != "NONE"
        else "NONE"
    )
    macro_ok = (
        risk_profile != "Conservative"
        or (signal_side == "LONG" and bool(f.close >= f.sma200))
        or (signal_side == "SHORT" and bool(f.close <= f.sma200))
    )
    timeframe_ok = (
        f.higher_timeframe_bias == "UNKNOWN"
        or f.higher_timeframe_bias == signal_side
        or risk_profile in ("Aggressive", "Micro-Scalp")
    )
    checks = [
        {
            "name": "History",
            "passed": bool(f.ready),
            "detail": (
                "Need 200 completed bars with valid volatility, "
                "momentum and volume"
            ),
        },
        {
            "name": "Intraday setup",
            "passed": f.setup != "NONE",
            "detail": (
                "No confirmed long/short breakout, pullback, continuation "
                "or range reversal"
            ),
        },
        {
            "name": "Participation",
            "passed": participation,
            "detail": (
                f"Need at least {volume_floor:g}x normal volume or a "
                "decisive candle body"
            ),
        },
        {
            "name": "Extension",
            "passed": bool(-2.0 <= f.extension_atr <= cap),
            "detail": (
                f"Price is outside the safe -2 to +{cap:g} ATR zone "
                "around EMA21"
            ),
        },
        {
            "name": "Macro trend",
            "passed": macro_ok,
            "detail": (
                "Conservative entries must align with the 200-bar trend"
            ),
        },
        {
            "name": "Hourly trend",
            "passed": bool(timeframe_ok),
            "detail": (
                f"{signal_side} setup conflicts with completed-hour "
                f"{f.higher_timeframe_bias} bias"
            ),
        },
    ]
    metric_names = (
        "close",
        "ema8",
        "ema21",
        "ema50",
        "sma200",
        "atr",
        "rsi",
        "macd_hist",
        "volume_ratio",
        "body_atr",
        "extension_atr",
        "trend_strength_atr",
        "htf_fast",
        "htf_slow",
        "required_committee_score",
        "signal_confidence",
        "bb_upper",
        "bb_lower",
        "support",
        "resistance",
    )
    metrics = {
        key: (
            float(f[key])
            if pd.notna(f[key]) and np.isfinite(f[key])
            else None
        )
        for key in metric_names
    }
    return EntryResearch(
        str(f.regime),
        str(f.setup),
        bool(f.eligible),
        checks,
        metrics,
        str(df.index[-1]),
        str(f.exit_setup),
        bool(f.exit_recommended),
        str(f.short_exit_setup),
        bool(f.short_exit_recommended),
        signal_side,
        str(f.higher_timeframe_bias),
        {
            "technical": "Completed five-minute candles",
            "fundamentals": (
                "Available as slow-moving context"
                if fundamentals is not None
                else "Unavailable / not applicable"
            ),
            "news": (
                "Context for AI modes; never a stand-alone order trigger"
            ),
        },
    )
