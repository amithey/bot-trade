"""Shared feature extraction.

Every downstream ML model (regime, anomaly, journal, forecaster) agrees
on the *same* feature vector here so we don't end up with N different
representations of the same market state.

The feature vector is deliberately low-dimensional and regime-agnostic:
- log returns at multiple horizons (momentum)
- volatility proxies (ATR / rolling std)
- trend alignment (MA spreads)
- mean-reversion (distance from VWAP / BBands)
- volume regime (current vs 20-bar mean)
- RSI, MACD histogram

All values are normalized so that a scalar scaler is optional — most
models here accept raw features.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_20",
    "atr_pct",
    "sma20_vs_sma50",
    "sma50_vs_sma200",
    "px_vs_sma20",
    "bb_position",
    "vwap_deviation",
    "rsi_centered",
    "macd_hist_norm",
    "volume_zscore",
    "range_pct",
]

N_FEATURES = len(FEATURE_NAMES)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0, None) else 0.0


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 14-dim feature frame aligned to ``df.index``.

    Expects ``df`` to have the enriched columns produced by
    :class:`market_data.fetcher.MarketDataFetcher` (SMA_20/50/200, RSI_14,
    MACD*, BB_*, ATR_14, VWAP_20).

    NaN rows at the head are dropped.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float).replace(0, np.nan)

    log_ret = np.log(close / close.shift(1))

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = log_ret
    out["ret_5"] = np.log(close / close.shift(5))
    out["ret_20"] = np.log(close / close.shift(20))
    out["vol_20"] = log_ret.rolling(20).std()

    atr = df.get("ATR_14")
    out["atr_pct"] = (atr / close) if atr is not None else (high - low) / close

    sma20 = df.get("SMA_20")
    sma50 = df.get("SMA_50")
    sma200 = df.get("SMA_200")
    out["sma20_vs_sma50"] = ((sma20 - sma50) / sma50) if (sma20 is not None and sma50 is not None) else 0.0
    out["sma50_vs_sma200"] = ((sma50 - sma200) / sma200) if (sma50 is not None and sma200 is not None) else 0.0
    out["px_vs_sma20"] = ((close - sma20) / sma20) if sma20 is not None else 0.0

    bb_up = df.get("BB_Upper_20")
    bb_lo = df.get("BB_Lower_20")
    if bb_up is not None and bb_lo is not None:
        width = (bb_up - bb_lo).replace(0, np.nan)
        out["bb_position"] = (close - bb_lo) / width  # 0 = lower, 1 = upper
    else:
        out["bb_position"] = 0.5

    vwap = df.get("VWAP_20")
    out["vwap_deviation"] = ((close - vwap) / vwap) if vwap is not None else 0.0

    rsi = df.get("RSI_14")
    out["rsi_centered"] = ((rsi - 50.0) / 50.0) if rsi is not None else 0.0

    macd_hist = df.get("MACD_Histogram")
    if macd_hist is not None:
        scale = macd_hist.abs().rolling(60).mean().replace(0, np.nan)
        out["macd_hist_norm"] = macd_hist / scale
    else:
        out["macd_hist_norm"] = 0.0

    # Volume z-score over 20 bars
    v_mean = vol.rolling(20).mean()
    v_std = vol.rolling(20).std().replace(0, np.nan)
    out["volume_zscore"] = (vol - v_mean) / v_std

    # Intra-bar range vs price
    out["range_pct"] = (high - low) / close

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out[FEATURE_NAMES]


def extract_latest(df: pd.DataFrame) -> np.ndarray:
    """1-D feature vector for the most recent bar. Always length N_FEATURES."""
    feats = extract_features(df)
    if feats.empty:
        return np.zeros(N_FEATURES, dtype=float)
    return feats.iloc[-1].to_numpy(dtype=float)


def batch_matrix(df: pd.DataFrame, *, min_rows: int = 30) -> np.ndarray:
    """(n_samples, n_features) matrix, dropping the warm-up head."""
    feats = extract_features(df)
    if len(feats) < min_rows:
        return feats.to_numpy(dtype=float)
    return feats.iloc[min_rows:].to_numpy(dtype=float)
