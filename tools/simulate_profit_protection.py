"""Offline demo: python -m tools.simulate_profit_protection.

Synthetic paths test mechanics, not expected returns. No market requests,
model calls or saved customer accounts are used.
"""
import json
from dataclasses import asdict

from portfolio.virtual_account import LivePortfolio
from risk.profit_protection import ProfitProtectionConfig


def run_scenario(side, prices):
    port = LivePortfolio(10_000, fee_rate=.001)
    if side == "LONG":
        port.buy("DEMO", 100., quantity=10.)
    else:
        port.open_short("DEMO", 100., cash_amount=1000.)
    observations = []
    for step, price in enumerate(prices, 1):
        assessment, trade = port.protect_profit(
            "DEMO", price, observed_at=float(step), exit_slippage_bps=5.,
        )
        observations.append({"step": step, "quote": price, **asdict(assessment)})
        if trade:
            return {"side": side, "observations": observations,
                    "exit_fill": trade.price, "net_pnl": trade.realized_pnl,
                    "reason": trade.reasoning}
    return {"side": side, "observations": observations, "exit_fill": None, "net_pnl": None}


def main():
    cases = {
        "long_reversal": run_scenario("LONG", [100.1, 100.4, 101., 102., 101.5, 101.]),
        "short_reversal": run_scenario("SHORT", [99.9, 99.6, 99., 98., 98.5, 99.]),
        "gap_through_stop": run_scenario("LONG", [101., 98.]),
    }
    print(json.dumps({"synthetic": True, "policy": asdict(ProfitProtectionConfig()),
                      "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
