"""Fixed ADX/Chandelier ablation, public data only, no trading/deployment.

python -m tools.compare_trend_controls --download --output-dir data/research_reviews/committee_20260918/adx_chandelier
Downloaded OHLCV is reused on subsequent offline runs. No parameter optimizer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from strategy.committee import IndicatorCommittee
from strategy.committee_replay import replay_committee
from strategy.research import entry_features
from strategy.trend_controls import trend_control_features

VARIANTS = {
    "baseline": {},
    "adx25": {"adx_min": 25.},
    "chandelier": {"chandelier": True},
    "adx25_chandelier": {"adx_min": 25., "chandelier": True},
    "adx20_chandelier_sensitivity": {"adx_min": 20., "chandelier": True},
    "adx25_chandelier_no_time_or_profit_cap": {"adx_min": 25., "chandelier": True, "trend_holding": True},
    "adx25_chandelier_30min": {"adx_min": 25., "chandelier": True, "chandelier_timeframe": "30min"},
    "adx25_chandelier_1h": {"adx_min": 25., "chandelier": True, "chandelier_timeframe": "1h"},
    "adx25_chandelier_1h_trend": {"adx_min": 25., "chandelier": True, "trend_holding": True, "chandelier_timeframe": "1h"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["BTC-USD", "ETH-USD", "META", "NVDA", "SPY", "MSFT"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--stress", type=float, default=1.)
    parser.add_argument("--segment", choices=["early", "late"], default="late")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--result-prefix", default="results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    for ticker in args.tickers:
        path = args.output_dir / (ticker + ".csv")
        try:
            if args.download:
                import yfinance as yf
                yf.set_tz_cache_location(str(args.output_dir / "yf_cache"))
                df = yf.Ticker(ticker).history(period="60d", interval="5m", auto_adjust=True,
                                               actions=False, timeout=20)
                if df.empty:
                    raise ValueError("Provider returned no candles")
                df = df[["Open", "High", "Low", "Close", "Volume"]].iloc[:-1]
                df.index = df.index.tz_convert("UTC")
                df.to_csv(path)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if args.segment == "early":
                df = df.iloc[:int(len(df) * .7)]
            features = entry_features(df)
            votes = IndicatorCommittee().vote_matrix(df)
            control_cache = {}
            start = max(720, int(len(df) * .7)) if args.segment == "late" else 720
            for variant in args.variants:
                settings = dict(VARIANTS[variant])
                timeframe = settings.pop("chandelier_timeframe", "5min")
                if timeframe not in control_cache:
                    control_cache[timeframe] = trend_control_features(df, chandelier_timeframe=timeframe)
                result = replay_committee(df, enhanced=True, evaluation_start=start,
                                          features=features, votes=votes, controls=control_cache[timeframe],
                                          slippage_multiplier=args.stress, **settings)
                result.update(ticker=ticker, variant=variant, slippage_multiplier=args.stress,
                              segment=args.segment, source_sha256=source_hash, chandelier_timeframe=timeframe)
                results.append(result)
                print(json.dumps({k: result[k] for k in ("ticker", "variant", "return_pct", "max_drawdown_pct", "trades", "win_rate_pct", "profit_factor", "mean_hold_minutes")}), flush=True)
                (args.output_dir / f"{args.result_prefix}_{args.segment}_slip{args.stress:g}.json").write_text(
                    json.dumps({"results": results, "errors": errors}, indent=2, allow_nan=False), encoding="utf-8")
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
            print(json.dumps(errors[-1]), flush=True)
    (args.output_dir / f"{args.result_prefix}_{args.segment}_slip{args.stress:g}.json").write_text(
        json.dumps({"results": results, "errors": errors}, indent=2, allow_nan=False), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
