"""Compare existing v5 intraday with a fixed hourly multi-day candidate."""
import argparse
import json
from pathlib import Path

import pandas as pd

from strategy.committee_replay import replay_committee
from strategy.committee_swing import completed_hourly_bars, swing_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--segment", choices=["early", "late"], default="late")
    parser.add_argument("--stress", type=float, default=1.)
    args = parser.parse_args()
    results = []
    for ticker in ("BTC-USD", "ETH-USD", "META", "NVDA", "SPY", "MSFT"):
        df = pd.read_csv(args.data_dir / f"{ticker}.csv", index_col=0, parse_dates=True)
        if args.segment == "early":
            df = df.iloc[:int(len(df) * .7)]
        hourly = completed_hourly_bars(df)
        # Shared evaluation boundaries with >=200 hourly warm-up candles.
        start = 200 if args.segment == "early" else max(200, int(len(hourly) * .7))
        if start >= len(hourly) - 1:
            raise ValueError(f"Insufficient hourly history: {ticker}")
        first_open = hourly.index[start + 1]
        end_open = hourly.index[-1] + pd.Timedelta(minutes=55)
        aligned = df.loc[:end_open]
        baseline_start = int(aligned.index.searchsorted(first_open)) - 1
        features, votes, controls = swing_inputs(aligned)
        swing = {"features": features, "votes": votes, "controls": controls,
                 "chandelier": True, "trend_holding": True}
        for name, prices, evaluation_start, options in (
            ("v5_intraday", aligned, baseline_start, {}),
            ("hourly_swing", aligned, baseline_start, swing),
            ("hourly_swing_atr", aligned, baseline_start, {**swing, "volatility_stop": True}),
        ):
            result = replay_committee(prices, enhanced=True, evaluation_start=evaluation_start,
                adx_min=25., slippage_multiplier=args.stress, **options)
            result.update(ticker=ticker, variant=name, segment=args.segment, slippage_multiplier=args.stress,
                          trading_from=str(first_open), trading_until=str(end_open + pd.Timedelta(minutes=5)))
            results.append(result)
            print(json.dumps({k: result[k] for k in ("ticker", "variant", "return_pct", "max_drawdown_pct", "trades", "mean_hold_minutes")}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"results": results}, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
