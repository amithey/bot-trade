"""Read-only: why does the committee replay exit trends early?

For one ticker, runs the same costed replay as validate_committee/
scan_universe, then for every closed trade reports the exit reason and how
the price moved AFTER the exit (in the position's own direction) over the
next 1h/1d/3d of bars. A trade followed by a large favorable continuation is
evidence the exit gave up a real move rather than avoiding a real reversal.

Never tunes parameters, never touches a live account.
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from strategy.committee import IndicatorCommittee
from strategy.committee_replay import replay_committee
from strategy.research import entry_features


def download(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="60d", interval="5m",
                                   auto_adjust=True, actions=False)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.iloc[:-1] if len(df) else df


def forward_move(df: pd.DataFrame, exit_time: str, side: str, bars: int) -> float | None:
    ts = pd.Timestamp(exit_time)
    pos = df.index.searchsorted(ts)
    end = pos + bars
    if pos >= len(df) or end >= len(df):
        return None
    start_px = float(df.Close.iloc[pos])
    end_px = float(df.Close.iloc[end])
    move = (end_px / start_px - 1) * 100
    return move if side == "LONG" else -move


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--profile", default="Balanced")
    args = parser.parse_args()

    df = download(args.ticker)
    features = entry_features(df, args.profile)
    votes = IndicatorCommittee().vote_matrix(df)
    start = max(200, int(len(df) * .7))
    result = replay_committee(df, enhanced=True, evaluation_start=start,
                              risk_profile=args.profile, features=features, votes=votes)

    print(f"{args.ticker}: {result['trades']} trades, return {result['return_pct']:+.2f}%, "
         f"buy&hold {result['buy_hold_pct']:+.2f}%\n")

    horizons = {"+1h (12 bars)": 12, "+1d (78 bars)": 78, "+3d (234 bars)": 234}
    rows = []
    for t in result["trade_log"]:
        row = {"entry": t["entry_time"][:16], "exit": t["exit_time"][:16],
              "side": t["side"], "pnl": round(t["pnl"], 2),
              "reason": t["reason"][:28]}
        for label, bars in horizons.items():
            move = forward_move(df, t["exit_time"], t["side"], bars)
            row[label] = round(move, 2) if move is not None else None
        rows.append(row)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))

    print("\nBy exit reason — mean forward move in the position's own "
         "direction after the exit (positive = the move continued and the "
         "exit gave it up; negative = the exit correctly avoided a reversal):")
    reason_group = table.groupby(table["reason"].str.split(":").str[0])
    print(reason_group[list(horizons.keys())].mean().round(2).to_string())
    print("\nCount of exits by reason:")
    print(table["reason"].str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
