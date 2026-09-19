"""Read-only, per-symbol committee validation across a basket of tickers.

Downloads public five-minute OHLCV via yfinance for each symbol and runs the
same costed, chronological replay used by ``validate_committee`` on each one
independently. This answers one question before any rotation/allocation
logic is built: does the committee signal show any edge on a broader
universe, symbol by symbol, or only reduce losses the way it did on BTC-USD?

No shared capital, no cross-symbol ranking, no order execution. Each symbol
gets its own isolated $10,000 paper run so results are directly comparable.
Never tunes parameters, never touches a live account or portfolio file.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from strategy.committee import IndicatorCommittee
from strategy.committee_replay import replay_committee
from strategy.research import entry_features

DEFAULT_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "QQQ", "SPY",
                    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META"]


def download(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="60d", interval="5m",
                                   auto_adjust=True, actions=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Drop the last row: it can be a still-open candle.
    return df.iloc[:-1] if len(df) else df


def scan_one(ticker: str, profile: str) -> dict:
    df = download(ticker)
    if len(df) < 260:
        raise ValueError(f"Only {len(df)} usable bars (need 200+ warm-up "
                         f"plus an evaluation window)")
    features = entry_features(df, profile)
    votes = IndicatorCommittee().vote_matrix(df)
    start = max(200, int(len(df) * .7))
    baseline = replay_committee(df, enhanced=False, evaluation_start=start,
                               risk_profile=profile, features=features, votes=votes)
    candidate = replay_committee(df, enhanced=True, evaluation_start=start,
                                risk_profile=profile, features=features, votes=votes)
    return {
        "ticker": ticker, "bars": len(df),
        "start": candidate["start"], "end": candidate["end"],
        "baseline_return_pct": baseline["return_pct"],
        "candidate_return_pct": candidate["return_pct"],
        "buy_hold_pct": candidate["buy_hold_pct"],
        "baseline_trades": baseline["trades"], "candidate_trades": candidate["trades"],
        "candidate_win_rate_pct": candidate["win_rate_pct"],
        "candidate_max_drawdown_pct": candidate["max_drawdown_pct"],
        "candidate_profit_factor": candidate["profit_factor"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--profile", default="Balanced",
                        choices=["Balanced", "Conservative", "Aggressive", "Micro-Scalp"])
    parser.add_argument("--output", type=Path,
                        default=Path("data/research_reviews/committee_20260918/universe_scan.json"))
    args = parser.parse_args()

    rows, errors = [], {}
    for ticker in args.tickers:
        try:
            rows.append(scan_one(ticker, args.profile))
            print(f"{ticker}: done")
        except Exception as exc:                               # noqa: BLE001
            errors[ticker] = str(exc)
            print(f"{ticker}: FAILED — {exc}")
        time.sleep(1)  # be polite to the free endpoint between symbols

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"profile": args.profile, "results": rows, "errors": errors,
         "limitations": [
             "Independent per-symbol $10,000 runs, not a shared/rotated "
             "capital pool — this only screens for per-symbol signal, it "
             "is not a basket-rotation backtest",
             "No cross-symbol correlation or exposure-overlap accounting",
             "Same limitations as validate_committee: OHLC-only fills, no "
             "funding/latency/liquidity, single non-independent holdout",
         ]}, indent=2, allow_nan=False), encoding="utf-8")

    if rows:
        table = pd.DataFrame(rows).set_index("ticker")
        pd.set_option("display.width", 140)
        print("\n" + table[[
            "bars", "buy_hold_pct", "baseline_return_pct", "candidate_return_pct",
            "candidate_trades", "candidate_win_rate_pct", "candidate_max_drawdown_pct",
        ]].round(2).to_string())
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
