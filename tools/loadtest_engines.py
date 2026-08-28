"""
Measure how many live bots this process can actually carry.

``BOTTRADE_MAX_LIVE_ENGINES`` defaults to 25, which was an educated guess.
This turns it into a number backed by measurement.

What it measures
----------------
The CPU-bound half of a trading cycle: the 38-indicator committee vote plus
the risk and portfolio bookkeeping around it. Market data is served from a
pre-fetched frame, and no LLM is called — COMMITTEE mode makes no API
requests, which is exactly what makes this test free to run.

That deliberately excludes network latency. Real cycles also wait on yfinance
and (in AI modes) Anthropic, and those waits release the GIL, so a real
process carries *more* concurrent bots than this measures. Read the result as
a floor, not a ceiling: if the CPU work alone degrades at N, the real limit is
somewhere above N, never below it.

Usage
-----
    python -m tools.loadtest_engines               # sweep 1,5,10,25,50
    python -m tools.loadtest_engines --levels 1 10 40 --cycles 20
    python -m tools.loadtest_engines --ticker ETH-USD

Read the p95 column. Cycle latency that stays flat as concurrency climbs
means headroom; the level where p95 starts climbing steeply is the real
ceiling for this host.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fetch_once(ticker: str):
    """One real fetch, reused by every simulated bot."""
    from market_data.fetcher import MarketDataFetcher
    snap = MarketDataFetcher().fetch_with_fundamentals(
        ticker, period="1y", interval="1d")
    if snap is None or snap.data is None or len(snap.data) == 0:
        raise SystemExit(f"No market data for {ticker}")
    return snap


def _run_one_bot(snap, cycles: int, latencies: list, lock: threading.Lock):
    """Simulate one bot's CPU work for *cycles* iterations."""
    from portfolio.virtual_account import LivePortfolio
    from strategy.committee import IndicatorCommittee

    committee = IndicatorCommittee()
    port = LivePortfolio(initial_capital=10_000)
    mine: list[float] = []
    price = float(snap.data["Close"].iloc[-1])

    for _ in range(cycles):
        t0 = time.perf_counter()
        verdict = committee.vote_latest(snap.data, in_position=False)
        verdict.to_trading_decision(snap.ticker)
        port.update_price(snap.ticker, price)
        port.get_daily_pnl_pct()
        port.get_total_value()
        mine.append((time.perf_counter() - t0) * 1000.0)

    with lock:
        latencies.extend(mine)


def measure(snap, bots: int, cycles: int) -> dict:
    latencies: list[float] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_run_one_bot,
                         args=(snap, cycles, latencies, lock),
                         name=f"bot{i}")
        for i in range(bots)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start

    latencies.sort()
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0
    return {
        "bots":       bots,
        "cycles":     len(latencies),
        "wall_s":     wall,
        "median_ms":  statistics.median(latencies) if latencies else 0.0,
        "p95_ms":     p95,
        "throughput": len(latencies) / wall if wall else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="BTC-USD")
    ap.add_argument("--levels", type=int, nargs="+",
                    default=[1, 5, 10, 25, 50])
    ap.add_argument("--cycles", type=int, default=10,
                    help="cycles each simulated bot runs")
    args = ap.parse_args()

    print(f"Fetching {args.ticker} once (shared by every simulated bot)...")
    snap = _fetch_once(args.ticker)
    print(f"  {len(snap.data)} bars, {len(snap.data.columns)} columns\n")

    print(f"{'bots':>5} {'cycles':>7} {'median ms':>10} {'p95 ms':>9} "
          f"{'cycles/s':>9} {'vs 1 bot':>9}")
    print("-" * 54)

    baseline = None
    for level in args.levels:
        r = measure(snap, level, args.cycles)
        if baseline is None:
            baseline = r["median_ms"] or 1.0
        ratio = r["median_ms"] / baseline
        print(f"{r['bots']:>5} {r['cycles']:>7} {r['median_ms']:>10.1f} "
              f"{r['p95_ms']:>9.1f} {r['throughput']:>9.1f} {ratio:>8.1f}x")

    print(
        "\nInterpretation: this is the CPU-bound half only. Real cycles also\n"
        "wait on network I/O, which releases the GIL, so the true ceiling is\n"
        "higher than the level where these numbers degrade. Set\n"
        "BOTTRADE_MAX_LIVE_ENGINES below the level where p95 starts climbing."
    )


if __name__ == "__main__":
    main()
