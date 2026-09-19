"""Fixed chart-rule ablation versus deployed v5. No historical fundamentals invented."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from strategy.committee import IndicatorCommittee
from strategy.committee_evidence import chart_features, chart_admission
from strategy.committee_replay import replay_committee
from strategy.research import entry_features
from strategy.trend_controls import trend_control_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment", choices=["early", "late"], default="late")
    parser.add_argument("--stress", type=float, default=1.)
    parser.add_argument("--tickers", nargs="+", default=["BTC-USD", "ETH-USD", "META", "NVDA", "SPY", "MSFT"])
    args = parser.parse_args()
    results = []
    for ticker in args.tickers:
        path = args.data_dir / f"{ticker}.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if args.segment == "early":
            df = df.iloc[:int(len(df) * .7)]
        start = 720 if args.segment == "early" else max(720, int(len(df) * .7))
        features, votes = entry_features(df), IndicatorCommittee().vote_matrix(df)
        chart, controls = chart_features(df), trend_control_features(df)
        for variant in ("v5_adx25", "direction", "efficiency", "structure", "retest"):
            candidate = features.copy()
            if variant != "v5_adx25":
                for stamp in features.index[features.eligible]:
                    candidate.loc[stamp, "eligible"] = chart_admission(chart.loc[stamp], features.loc[stamp, "signal_side"], variant)["allowed"]
            result = replay_committee(df, enhanced=True, evaluation_start=start,
                features=candidate, votes=votes, controls=controls, adx_min=25., slippage_multiplier=args.stress)
            result.update(ticker=ticker, variant=variant, segment=args.segment,
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), slippage_multiplier=args.stress,
                          fundamentals="NOT_TESTED_NO_POINT_IN_TIME_DATA")
            results.append(result)
            print(json.dumps({k: result[k] for k in ("ticker", "variant", "return_pct", "max_drawdown_pct", "trades", "profit_factor")}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"results": results}, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
