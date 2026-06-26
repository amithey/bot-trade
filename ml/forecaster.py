"""Short-horizon return forecaster.

A lightweight statistical forecaster — no TensorFlow / Prophet
dependency needed. Uses an EWMA-drift estimate combined with
volatility-scaled confidence bands. Good enough as a *sanity check*:
if the technical signal screams BUY but the forecast distribution
suggests a 10% downside tail, the engine downgrades confidence.

Returns
-------
ForecastResult with:
    - point: expected log-return over the horizon (1=1 bar ahead)
    - lo_95, hi_95: 95% confidence bounds on the expected return
    - bias: "up" / "down" / "flat"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    horizon: int            # bars ahead
    point_return: float     # expected log return
    lo_95: float
    hi_95: float
    bias: str               # "up" | "down" | "flat"
    sigma: float            # estimated per-bar volatility


def forecast(df: pd.DataFrame, horizon: int = 5,
             ewm_span: int = 20) -> ForecastResult:
    """Return an EWMA-drift forecast over ``horizon`` bars.

    Model: r_{t+1..t+h} ~ Normal( μ * h , σ * sqrt(h) )
    where
        μ = EWMA mean of log-returns (span=ewm_span)
        σ = EWMA std  of log-returns (span=ewm_span)
    """
    if df is None or len(df) < max(30, ewm_span):
        return ForecastResult(horizon=horizon, point_return=0.0,
                              lo_95=0.0, hi_95=0.0, bias="flat", sigma=0.0)

    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()
    mu = float(log_ret.ewm(span=ewm_span, adjust=False).mean().iloc[-1])
    sigma = float(log_ret.ewm(span=ewm_span, adjust=False).std().iloc[-1])
    if not np.isfinite(sigma) or sigma == 0:
        sigma = float(log_ret.std())

    point = mu * horizon
    band = 1.96 * sigma * np.sqrt(horizon)

    # Bias: only call up/down if point return exceeds 0.25σ of the horizon
    thresh = 0.25 * sigma * np.sqrt(horizon)
    if point > thresh:
        bias = "up"
    elif point < -thresh:
        bias = "down"
    else:
        bias = "flat"

    return ForecastResult(
        horizon=horizon,
        point_return=float(point),
        lo_95=float(point - band),
        hi_95=float(point + band),
        bias=bias,
        sigma=float(sigma),
    )
