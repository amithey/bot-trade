"""Causal entry research shared by the live gate and historical experiments.

These rules are hypotheses, not calibrated return forecasts. They classify
trend continuation, a pullback recovery, and a separately labelled scalp.
Every rolling reference to a prior range is shifted to exclude today's bar.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


def entry_features(df: pd.DataFrame, risk_profile: str = "Balanced") -> pd.DataFrame:
    close = df.Close.astype(float)
    fast = close.ewm(span=20, adjust=False, min_periods=20).mean()
    medium = close.rolling(50, min_periods=50).mean()
    slow = close.rolling(200, min_periods=200).mean()
    tr = pd.concat([df.High - df.Low, (df.High - close.shift()).abs(),
                    (df.Low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    prior_volume = df.Volume.rolling(20, min_periods=20).mean().shift()
    volume_ratio = df.Volume / prior_volume.replace(0, np.nan)
    trend = (close > slow) & (medium > slow) & (medium > medium.shift(5))
    falling = (close < slow) & (medium < medium.shift(5))
    breakout = close > df.High.rolling(20, min_periods=20).max().shift()
    pullback = ((df.Low.shift() <= fast.shift()) & (close > fast)
                & (close > df.High.shift()))
    extension = (close - fast) / atr.replace(0, np.nan)
    # Scalp mode is explicit: a short-term reversal may oppose the slow trend.
    scalp = ((close > df.High.shift()) & (close.shift() < fast.shift())
             & (close > df.Open) & (volume_ratio >= 1.2))
    volume_floor = 1.1 if risk_profile == "Conservative" else 1.0
    max_extension = 1.5 if risk_profile == "Conservative" else 2.0
    ready = slow.notna() & atr.gt(0) & volume_ratio.notna()
    setup = pd.Series("NONE", index=df.index)
    setup.loc[trend & pullback] = "TREND_PULLBACK"
    setup.loc[trend & breakout] = "TREND_BREAKOUT"
    if risk_profile == "Micro-Scalp":
        setup.loc[scalp] = "SCALP_REVERSAL"
    eligible = (ready & setup.ne("NONE") & volume_ratio.ge(volume_floor)
                & extension.le(max_extension))
    return pd.DataFrame({
        "ready": ready, "regime": np.where(trend, "UPTREND", np.where(falling, "DOWNTREND", "MIXED")),
        "setup": setup, "eligible": eligible, "close": close,
        "ema20": fast, "sma50": medium, "sma200": slow, "atr": atr,
        "volume_ratio": volume_ratio, "extension_atr": extension,
        "support": df.Low.rolling(20, min_periods=20).min().shift(),
        "resistance": df.High.rolling(20, min_periods=20).max().shift(),
    }, index=df.index)


@dataclass(frozen=True)
class EntryResearch:
    regime: str
    setup: str
    entry_allowed: bool
    checks: list[dict]
    metrics: dict
    bar: str
    coverage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def explanation(self) -> str:
        failed = [c["detail"] for c in self.checks if not c["passed"]]
        return "; ".join(failed) if failed else f"{self.setup}: trend, trigger and volume checks passed"


def analyze_entry(df: pd.DataFrame, *, risk_profile="Balanced", fundamentals=None) -> EntryResearch:
    f = entry_features(df, risk_profile).iloc[-1]
    volume_floor = 1.1 if risk_profile == "Conservative" else 1.0
    cap = 1.5 if risk_profile == "Conservative" else 2.0
    checks = [
        {"name": "History", "passed": bool(f.ready), "detail": "Need 200 bars, valid ATR and prior volume"},
        {"name": "Entry setup", "passed": f.setup != "NONE",
         "detail": "No confirmed trend pullback or breakout" if risk_profile != "Micro-Scalp" else "No confirmed trend setup or scalp reversal"},
        {"name": "Volume", "passed": bool(f.volume_ratio >= volume_floor),
         "detail": f"Volume must be at least {volume_floor:g}x the prior 20-bar average"},
        {"name": "Extension", "passed": bool(f.extension_atr <= cap),
         "detail": f"Price must be within {cap:g} ATR above EMA20; do not chase an extended move"},
    ]
    metrics = {k: float(f[k]) if pd.notna(f[k]) and np.isfinite(f[k]) else None
               for k in ("close", "ema20", "sma50", "sma200", "atr", "volume_ratio", "extension_atr", "support", "resistance")}
    return EntryResearch(str(f.regime), str(f.setup), bool(f.eligible), checks, metrics,
                         str(df.index[-1]), {"technical": "Completed candles",
                         "fundamentals": "Available as context" if fundamentals is not None else "Unavailable / not applicable",
                         "news": "Not used by the deterministic entry gate"})
