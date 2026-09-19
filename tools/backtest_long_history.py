"""Three-year replay of the deployed committee-v5-adx25 policy on Binance 5-minute history.

Same engine, fees and slippage model as evaluate_committee_evidence; only the data span differs.
Binance spot USDT pairs stand in for the USD tickers (a close proxy, not identical prices).
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from strategy.committee import IndicatorCommittee
from strategy.committee_replay import replay_committee
from strategy.research import entry_features
from strategy.trend_controls import trend_control_features

SYMBOLS = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT", "ADA-USD": "ADAUSDT"}
WARMUP = 720


def download(symbol, years, path):
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    end = int(time.time() * 1000)
    cursor = end - int(years * 365.25 * 86400 * 1000)
    rows = []
    while cursor < end:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&startTime={cursor}&limit=1000"
        batch = json.loads(urllib.request.urlopen(url, timeout=30).read())
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 300_000
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms", utc=True)
    df = df.set_index("Datetime").astype(float)
    df = df[~df.index.duplicated()].iloc[:-1]  # drop the still-forming candle
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def run(df, start, slippage):
    features, votes = entry_features(df), IndicatorCommittee().vote_matrix(df)
    return replay_committee(df, enhanced=True, evaluation_start=start, features=features, votes=votes,
                            controls=trend_control_features(df), adx_min=25., slippage_multiplier=slippage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", type=float, default=3.)
    parser.add_argument("--stress", type=float, default=1.)
    parser.add_argument("--full-only", action="store_true", help="skip the per-year slices")
    parser.add_argument("--tickers", nargs="+", default=list(SYMBOLS))
    args = parser.parse_args()
    results = []
    for ticker in args.tickers:
        df = download(SYMBOLS[ticker], args.years, args.data_dir / f"{ticker}.csv")
        # Whole span, then one slice per 365 days (each slice restarts a fresh 10k account).
        spans = [("full", df, WARMUP)]
        year_bars = 365 * 288
        for i in range(0 if args.full_only else int(args.years)):
            lo = max(0, i * year_bars - WARMUP)
            hi = (i + 1) * year_bars
            spans.append((f"year{i + 1}", df.iloc[lo:hi], WARMUP))
        for label, part, start in spans:
            result = run(part, start, args.stress)
            result.update(ticker=ticker, span=label, bars=len(part), slippage_multiplier=args.stress)
            result.pop("trade_log", None)
            results.append(result)
            print(json.dumps({"ticker": ticker, "span": label, "return_pct": round(result["return_pct"], 2),
                              "buy_hold_pct": round(result["buy_hold_pct"], 2), "max_dd": round(result["max_drawdown_pct"], 2), "gross": round(result["gross_pnl"]), "fees": round(result["fees"]),
                              "slippage": round(result["slippage_cost"]), "net": round(result["net_pnl"]),
                              "trades": result["trades"], "pf": result["profit_factor"]}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"results": results}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
