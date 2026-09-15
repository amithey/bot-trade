"""Deterministic, fee-aware trailing exits for paper long and short positions.

All percentages use entry notional as denominator. Defaults are an initial
test policy, not calibrated return forecasts. Only observed quotes set peaks.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ProfitProtectionConfig:
    activation_net_pct: float = 0.10
    retain_fraction: float = 0.50
    minimum_net_pct: float = 0.02

    def __post_init__(self):
        if not all(isfinite(v) for v in (
            self.activation_net_pct, self.retain_fraction, self.minimum_net_pct,
        )) or not (
            0 < self.minimum_net_pct < self.activation_net_pct
            and 0 < self.retain_fraction < 1
        ):
            raise ValueError("Invalid profit protection policy")


@dataclass(frozen=True)
class ProfitProtectionAssessment:
    best_price: float
    stop_price: float | None
    armed: bool
    should_exit: bool
    estimated_net_pct: float
    peak_net_pct: float
    estimated_fill: float


def assess_profit_protection(*, side, entry_price, quantity, entry_fees,
                             price, best_price=None, stop_price=None,
                             armed=False, exit_pending=False, fee_rate=0.001,
                             exit_slippage_bps=0., config=None):
    cfg = config or ProfitProtectionConfig()
    inputs = (entry_price, quantity, entry_fees, price, fee_rate, exit_slippage_bps)
    if side not in ("LONG", "SHORT") or not all(isfinite(v) for v in inputs):
        raise ValueError("Invalid profit protection input")
    if (min(entry_price, quantity, price) <= 0 or entry_fees < 0
            or not 0 <= fee_rate < .1 or not 0 <= exit_slippage_bps < 10000):
        raise ValueError("Invalid price, size or execution cost")
    for v in (best_price, stop_price):
        if v is not None and (not isfinite(v) or v <= 0):
            raise ValueError("Invalid persisted profit protection price")
    short = side == "SHORT"
    best = (min if short else max)(best_price or entry_price, price)
    slip = exit_slippage_bps / 10000
    fill_factor = 1 + slip if short else 1 - slip
    net_factor = fill_factor * (1 + fee_rate if short else 1 - fee_rate)
    paid_per_unit = entry_fees / quantity

    def net_pct(mark):
        pnl = (entry_price - mark * net_factor if short
               else mark * net_factor - entry_price) - paid_per_unit
        return pnl / entry_price * 100

    peak_net, current_net = net_pct(best), net_pct(price)
    armed = armed or peak_net >= cfg.activation_net_pct
    if armed:
        locked_net = max(cfg.minimum_net_pct, peak_net * cfg.retain_fraction)
        profit_per_unit = entry_price * locked_net / 100
        candidate = ((entry_price - paid_per_unit - profit_per_unit) / net_factor
                     if short else
                     (entry_price + paid_per_unit + profit_per_unit) / net_factor)
        if not isfinite(candidate) or candidate <= 0:
            raise ValueError("Profit protection stop is not a valid price")
        stop_price = candidate if stop_price is None else (
            min(stop_price, candidate) if short else max(stop_price, candidate)
        )
    hit = armed and stop_price is not None and (
        price >= stop_price if short else price <= stop_price
    )
    return ProfitProtectionAssessment(
        best, stop_price, armed, bool(exit_pending or hit), current_net,
        peak_net, price * fill_factor,
    )
