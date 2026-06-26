"""Slippage modeling — adverse fill simulation.

In real trading, your fill price is rarely the last close. The wider the
spread + the more volatile the bar, the worse it gets. This module
applies a small adverse adjustment to the requested execution price so
backtests + paper trading don't overstate edge.

Model
-----
``effective_price = price × (1 + sign × slippage_bps / 10_000)``

where ``sign = +1`` for BUYs (you pay slightly more) and ``-1`` for
SELLs (you receive slightly less). Slippage scales with realized
volatility when an ATR% is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SlippageConfig:
    base_bps: float = 5.0          # always applied (0.05%)
    atr_multiplier: float = 0.30   # extra bps = atr_pct(of price) * 100 * mult
    max_bps: float = 50.0          # hard ceiling

    @classmethod
    def for_profile(cls, risk_profile: str) -> "SlippageConfig":
        if risk_profile == "Micro-Scalp":
            return cls(base_bps=2.0, atr_multiplier=0.15, max_bps=20.0)
        if risk_profile == "Aggressive":
            return cls(base_bps=8.0, atr_multiplier=0.45, max_bps=80.0)
        if risk_profile == "Conservative":
            return cls(base_bps=3.0, atr_multiplier=0.25, max_bps=30.0)
        return cls()  # Balanced


def apply_slippage(
    price: float,
    side: str,
    *,
    cfg: Optional[SlippageConfig] = None,
    atr_pct: Optional[float] = None,
) -> tuple[float, float]:
    """Return (effective_price, applied_bps).

    Parameters
    ----------
    price : float
        Reference / last-close price.
    side : "BUY" | "SELL" | "FORCE_CLOSE"
        BUYs pay more, SELLs receive less.
    cfg : SlippageConfig
        Defaults to a balanced profile.
    atr_pct : float, optional
        Bar's ATR as a percent of price (e.g. 1.8 for 1.8%). When
        provided, adds ``atr_pct * atr_multiplier`` bps on top.
    """
    cfg = cfg or SlippageConfig()
    bps = cfg.base_bps
    if atr_pct is not None and atr_pct > 0:
        bps += float(atr_pct) * cfg.atr_multiplier
    bps = min(bps, cfg.max_bps)
    sign = -1.0 if side.upper() in ("SELL", "FORCE_CLOSE") else 1.0
    eff = float(price) * (1.0 + sign * bps / 10_000.0)
    return eff, float(bps)
