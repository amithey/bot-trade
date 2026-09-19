"""Does arming profit protection earlier stop winners from turning into losers?

Replays deployed committee-v5-adx25 on the long-history 5-minute files with only
the protection activation changed. Current live value is stop/2 = 0.40% net.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from risk.profit_protection import ProfitProtectionConfig
from strategy.committee import IndicatorCommittee
from strategy.committee_replay import replay_committee
from strategy.research import entry_features
from strategy.trend_controls import trend_control_features

VARIANTS = {"live_0.40": None, "arm_0.20": .20, "arm_0.10": .10, "arm_0.05": .05}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.data_dir / f"{args.ticker}.csv", index_col=0, parse_dates=True)
    features, votes, controls = entry_features(df), IndicatorCommittee().vote_matrix(df), trend_control_features(df)
    results = []
    for name, activation in VARIANTS.items():
        config = None if activation is None else ProfitProtectionConfig(activation_net_pct=activation, minimum_net_pct=min(.02, activation / 2))
        r = replay_committee(df, enhanced=True, evaluation_start=720, features=features, votes=votes,
                             controls=controls, adx_min=25., profit_config=config)
        log = r.pop("trade_log")
        reasons = pd.Series([t["reason"] for t in log]).value_counts().to_dict()
        r.update(ticker=args.ticker, variant=name, exit_reasons=reasons)
        results.append(r)
        print(json.dumps({k: (round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in
                          ("ticker", "variant", "return_pct", "gross_pnl", "trades", "win_rate_pct", "exit_reasons")}), flush=True)
        args.output.write_text(json.dumps({"results": results}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
