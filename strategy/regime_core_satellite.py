"""Core/Satellite Portfolio Regime Management.

Maintains a core baseline exposure (40%-50%) to broad market leaders/ETFs
when the macro regime is Bullish (Price >= SMA200), preventing the bot from
sitting 100% in cash while the market rallies. When the regime turns Bearish
(Price < SMA200), Core rotates to cash for defense.
The remaining 50%-60% acts as active Satellite capital for trend/momentum breakouts.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class CoreSatelliteConfig:
    core_target_pct: float = 45.0          # 40% - 50% target core allocation
    satellite_target_pct: float = 55.0     # 50% - 60% active trading satellite
    benchmark_ticker: str = "SPY"          # Benchmark proxy (SPY, QQQ, BTC-USD)
    sma_period: int = 200                  # Days for macro trend regime
    rebalance_tolerance_pct: float = 5.0   # Band to avoid excessive rebalancing trades

    def __post_init__(self):
        if not (0 < self.core_target_pct < 100):
            raise ValueError("Core target must be between 0% and 100%")
        if not (self.sma_period > 0):
            raise ValueError("SMA period must be positive")


@dataclass(frozen=True)
class RegimeState:
    benchmark: str
    price: float
    sma200: float
    is_bullish: bool
    pct_from_sma: float
    regime: str       # "BULLISH" | "BEARISH"
    reason: str


@dataclass(frozen=True)
class CoreSatellitePlan:
    regime_state: RegimeState
    total_equity: float
    target_core_pct: float
    target_core_value: float
    target_satellite_pct: float
    target_satellite_value: float
    status_summary: str


def detect_regime(df: pd.DataFrame, benchmark: str = "SPY", sma_period: int = 200) -> RegimeState:
    """Detect macro market regime from daily price series using SMA200."""
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return RegimeState(
            benchmark=benchmark, price=0.0, sma200=0.0, is_bullish=False,
            pct_from_sma=0.0, regime="UNKNOWN", reason="No market data available"
        )
    close = df["Close"].dropna()
    if len(close) < sma_period:
        # Fall back to available period or EMA if fewer bars
        sma = close.rolling(min(len(close), 50)).mean().iloc[-1]
    else:
        sma = close.rolling(sma_period).mean().iloc[-1]

    current_price = float(close.iloc[-1])
    sma_val = float(sma)
    if not (isfinite(current_price) and isfinite(sma_val) and sma_val > 0):
        return RegimeState(
            benchmark=benchmark, price=current_price, sma200=0.0, is_bullish=False,
            pct_from_sma=0.0, regime="UNKNOWN", reason="Non-finite price or SMA"
        )

    pct_diff = (current_price / sma_val - 1.0) * 100.0
    is_bull = current_price >= sma_val
    regime_label = "BULLISH" if is_bull else "BEARISH"
    reason = (
        f"{benchmark} @ ${current_price:,.2f} is {abs(pct_diff):.1f}% "
        f"{'above' if is_bull else 'below'} SMA{sma_period} (${sma_val:,.2f})"
    )
    return RegimeState(
        benchmark=benchmark, price=current_price, sma200=sma_val,
        is_bullish=is_bull, pct_from_sma=round(pct_diff, 2),
        regime=regime_label, reason=reason
    )


def plan_core_satellite(
    total_equity: float,
    benchmark_df: pd.DataFrame,
    config: Optional[CoreSatelliteConfig] = None,
) -> CoreSatellitePlan:
    """Calculate target Core and Satellite allocations based on macro regime."""
    cfg = config or CoreSatelliteConfig()
    regime = detect_regime(benchmark_df, benchmark=cfg.benchmark_ticker, sma_period=cfg.sma_period)

    if regime.is_bullish:
        core_pct = cfg.core_target_pct
        sat_pct = 100.0 - core_pct
        summary = (
            f"Bullish Macro Regime ({regime.reason}): Core baseline active ({core_pct:.0f}%) "
            f"to capture market beta and prevent runaway underperformance; "
            f"Satellite ({sat_pct:.0f}%) allocated for active breakout alpha."
        )
    else:
        core_pct = 0.0
        sat_pct = 100.0
        summary = (
            f"Bearish/Defensive Macro Regime ({regime.reason}): Core rotated to 100% cash defense; "
            f"Selective active trading only."
        )

    core_val = total_equity * (core_pct / 100.0)
    sat_val = total_equity * (sat_pct / 100.0)

    return CoreSatellitePlan(
        regime_state=regime,
        total_equity=total_equity,
        target_core_pct=core_pct,
        target_core_value=core_val,
        target_satellite_pct=sat_pct,
        target_satellite_value=sat_val,
        status_summary=summary,
    )


def calculate_core_rebalance_action(
    portfolio,
    benchmark_df: pd.DataFrame,
    config: Optional[CoreSatelliteConfig] = None,
) -> dict:
    """Determine whether a buy or sell rebalance order is needed for the Core holding."""
    cfg = config or CoreSatelliteConfig()
    equity = portfolio.get_total_value()
    plan = plan_core_satellite(equity, benchmark_df, config=cfg)
    regime = plan.regime_state

    current_pos = portfolio.positions.get(cfg.benchmark_ticker)
    current_val = float(current_pos.market_value) if current_pos else 0.0
    target_val = plan.target_core_value

    diff_val = target_val - current_val
    diff_pct = abs(diff_val) / equity * 100.0 if equity > 0 else 0.0

    if diff_pct < cfg.rebalance_tolerance_pct:
        return {
            "action": "HOLD",
            "ticker": cfg.benchmark_ticker,
            "target_value": target_val,
            "current_value": current_val,
            "rebalance_cash": 0.0,
            "plan": plan,
            "reason": f"Core allocation within tolerance ({diff_pct:.1f}% deviation < {cfg.rebalance_tolerance_pct}%)",
        }

    if diff_val > 0:
        # Need to add to core
        spend = min(diff_val, portfolio.cash / (1 + portfolio.fee_rate))
        return {
            "action": "BUY",
            "ticker": cfg.benchmark_ticker,
            "target_value": target_val,
            "current_value": current_val,
            "rebalance_cash": max(0.0, spend),
            "plan": plan,
            "reason": f"Regime {regime.regime}: Rebalance Core to {plan.target_core_pct:.0f}% (add ${spend:,.2f})",
        }
    else:
        # Need to trim or liquidate core
        trim_cash = abs(diff_val)
        return {
            "action": "SELL",
            "ticker": cfg.benchmark_ticker,
            "target_value": target_val,
            "current_value": current_val,
            "rebalance_cash": trim_cash,
            "plan": plan,
            "reason": f"Regime {regime.regime}: Trim/Liquidate Core to {plan.target_core_pct:.0f}%",
        }
