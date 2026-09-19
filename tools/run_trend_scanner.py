"""Run the paper trend scanner once, or repeatedly with --loop-minutes."""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("data/trend_scanner_paper"))
    parser.add_argument("--loop-minutes", type=float, default=0.)
    args = parser.parse_args()
    runner = TrendScannerRunner(args.state_dir)
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
