"""Run the paper trend scanner once, or repeatedly with --loop-minutes.

This loop is the only writer of the scanner state files; the dashboard page
only reads them. Core/Satellite regime allocation is on when
BOTTRADE_CORE_SATELLITE is 1/true (or --core-satellite), off otherwise.
"""
import argparse
import json
import os
import time
from pathlib import Path

from trading.trend_scanner_runner import TrendScannerRunner


def status(runner):
    portfolio = runner.portfolio
    value = portfolio.get_total_value()
    lines = [f"Equity ${value:,.2f} (start ${portfolio.initial_capital:,.2f}, {100 * (value / portfolio.initial_capital - 1):+.2f}%), "
             f"cash ${portfolio.cash:,.2f}"]
    for ticker, position in portfolio.positions.items():
        stop = runner.state["positions"].get(ticker, {}).get("stop")
        lines.append(f"  {ticker:<8} {position.quantity:.6g} @ {position.avg_entry_price:,.4f} now {position.current_price:,.4f} "
                     f"({position.unrealized_pnl_pct:+.2f}%) stop {stop:,.4f}" if stop else f"  {ticker}")
    for kind, orders in (("exit", runner.state["pending_exits"]), ("entry", runner.state["pending_entries"])):
        for ticker in orders:
            lines.append(f"  pending {kind}: {ticker}")
    return "\n".join(lines)


def env_flag(name):
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("data/trend_scanner_paper"))
    parser.add_argument("--loop-minutes", type=float, default=0.)
    parser.add_argument("--core-satellite", action=argparse.BooleanOptionalAction,
                        default=env_flag("BOTTRADE_CORE_SATELLITE"),
                        help="Keep a SPY core while SPY is above its 200-day average (default: BOTTRADE_CORE_SATELLITE)")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    runner = TrendScannerRunner(args.state_dir, core_satellite=args.core_satellite)
    while True:
        step = runner.step()
        print(json.dumps(step), flush=True)
        for event in runner.state["events"][-10:]:
            if event["at"] == step["at"]:
                print(f"  {event['kind']:<14} {event['ticker']:<8} {event['detail']}")
        print(status(runner), flush=True)
        if args.loop_minutes <= 0:
            break
        time.sleep(args.loop_minutes * 60)


if __name__ == "__main__":
    main()
