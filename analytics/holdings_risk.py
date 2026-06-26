"""Holdings risk: concentration of open positions and pairwise return
correlations between currently held tickers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def concentration(positions: Iterable) -> pd.DataFrame:
    """Return DataFrame(ticker, market_value, weight_pct, hhi_contrib).

    *positions* must be an iterable of ``Position`` objects (with
    ``.ticker``, ``.market_value``). The ``hhi_contrib`` column is the
    individual ticker's Herfindahl contribution (weight²); the sum of
    that column is the portfolio HHI ∈ [0, 1] — closer to 1 = more
    concentrated.
    """
    rows: list[dict] = []
    for p in positions:
        try:
            mv = float(p.market_value)
        except Exception:
            continue
        rows.append({"ticker": p.ticker, "market_value": round(mv, 2)})
    if not rows:
        return pd.DataFrame(columns=["ticker", "market_value", "weight_pct", "hhi_contrib"])

    df = pd.DataFrame(rows)
    total = float(df["market_value"].sum())
    if total <= 0:
        df["weight_pct"] = 0.0
        df["hhi_contrib"] = 0.0
    else:
        df["weight_pct"] = (df["market_value"] / total * 100.0).round(2)
        df["hhi_contrib"] = ((df["market_value"] / total) ** 2).round(4)
    return df.sort_values("market_value", ascending=False).reset_index(drop=True)


def correlation_matrix(
    tickers: list[str],
    lookback_days: int = 90,
) -> Optional[pd.DataFrame]:
    """Pull *lookback_days* of daily closes for each ticker and return the
    Pearson correlation matrix of daily returns. Returns None if fewer
    than 2 valid tickers.
    """
    tickers = sorted({t.strip().upper() for t in tickers if t and t.strip()})
    if len(tickers) < 2:
        return None

    try:
        import yfinance as yf
        cache = Path("data/yfinance_cache")
        cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache))

        # Bulk download — single network round trip
        df = yf.download(
            tickers=" ".join(tickers),
            period=f"{max(lookback_days, 30)}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as exc:
        logger.debug(f"correlation_matrix: yf.download failed: {exc}")
        return None

    if df is None or df.empty:
        return None

    closes = pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        for tk in tickers:
            try:
                closes[tk] = df[tk]["Close"]
            except KeyError:
                continue
    else:
        # Single-ticker case (yfinance returns flat columns)
        if "Close" in df.columns and len(tickers) == 1:
            closes[tickers[0]] = df["Close"]

    closes = closes.dropna(axis=1, how="all")
    if closes.shape[1] < 2:
        return None

    rets = closes.pct_change().dropna(how="all")
    if len(rets) < 5:
        return None
    corr = rets.corr().round(3)
    # Order rows/cols by descending mean correlation for nicer heatmaps
    order = corr.mean().sort_values(ascending=False).index.tolist()
    return corr.loc[order, order]
