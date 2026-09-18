"""Read-only historical validation: python -m tools.validate_committee bars.csv.

CSV must have a timestamp first column and Open/High/Low/Close/Volume columns.
Use completed candles only. Never tunes parameters or changes a live account.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from strategy.committee import IndicatorCommittee
from strategy.committee_replay import replay_committee
from strategy.research import entry_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="Balanced",
                        choices=["Balanced", "Conservative", "Aggressive", "Micro-Scalp"])
    args = parser.parse_args()
    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    features = entry_features(df, args.profile)
    votes = IndicatorCommittee().vote_matrix(df)
    start = max(200, int(len(df) * .7))
    results = []
    for stress in (1., 2.):
        for enhanced in (False, True):
            result = replay_committee(df, enhanced=enhanced, evaluation_start=start,
                                      risk_profile=args.profile, features=features, votes=votes,
                                      slippage_multiplier=stress)
            result["slippage_multiplier"] = stress
            results.append(result)
    output = {"source": str(args.csv), "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
              "holdout_fraction": .3, "profile": args.profile, "results": results,
              "limitations": ["Opening and closing quotes only for trailing protection",
                              "Stop first when both stop and target occur in one bar",
                              "No live liquidity, funding, latency, daily account guards or intrabar quote replay",
                              "A positive single holdout is not sufficient deployment evidence"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps([{k: v for k, v in r.items() if k not in ("trade_log", "blocked")}
                      for r in results], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
