"""Frozen forward observation of baseline versus ADX25; never sends orders.

Initialize from saved five-minute CSVs, then run --update to append closed bars.
Every observation replays the frozen history with open positions marked, not
liquidated. This is a public-data paper simulation, not broker paper execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from strategy.committee import IndicatorCommittee
from strategy.committee_replay import replay_committee
from strategy.research import entry_features
from strategy.trend_controls import trend_control_features

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["BTC-USD", "ETH-USD", "META", "NVDA", "SPY", "MSFT"]
VARIANTS = {"baseline": {}, "adx25": {"adx_min": 25.}}


def fingerprint():
    paths = [Path(__file__)]
    for folder in ("strategy", "risk", "config"):
        paths.extend((ROOT / folder).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_bars(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    validate_bars(df)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def validate_bars(df):
    if df.empty or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Missing, duplicate or unsorted candles")
    values = df[["Open", "High", "Low", "Close", "Volume"]]
    if not np.isfinite(values.to_numpy()).all() or (values.iloc[:, :4] <= 0).any().any():
        raise ValueError("Invalid OHLCV values")
    if (df.High < df[["Open", "Close", "Low"]].max(axis=1)).any() or (df.Low > df[["Open", "Close"]].min(axis=1)).any() or (df.Volume < 0).any():
        raise ValueError("Invalid candle ranges")
    if (df.index.asi8 % pd.Timedelta(minutes=5).value != 0).any():
        raise ValueError("Candles must be aligned five-minute bars")


def append_closed(old, incoming, now, frozen_before=None):
    """Keep observations immutable; reject provider revisions instead of rewriting."""
    incoming = incoming.loc[incoming.index + pd.Timedelta(minutes=5) <= now]
    validate_bars(incoming)
    overlap = old.index.intersection(incoming.index)
    if overlap.empty:
        raise ValueError("No overlap: cannot verify continuity of provider history")
    checked = overlap if frozen_before is None else overlap[overlap >= frozen_before]
    if not np.allclose(old.loc[checked].to_numpy(), incoming.loc[checked].to_numpy(), rtol=1e-8, atol=1e-8):
        raise ValueError("Provider revised previously observed candles; investigation required")
    fresh = incoming.loc[incoming.index > old.index[-1]]
    result = pd.concat([old, fresh])
    validate_bars(result)
    return result


def initialize(directory, source, now):
    directory.mkdir(parents=True, exist_ok=False)
    start = now.ceil("5min")
    seeds = {}
    for ticker in TICKERS:
        path = source / f"{ticker}.csv"
        bars = read_bars(path)
        if len(bars) < 720 or bars.index[-1] + pd.Timedelta(minutes=5) > start:
            raise ValueError(f"Invalid seed window for {ticker}")
        target = directory / path.name
        shutil.copyfile(path, target)
        seed = directory / f"{ticker}.seed.csv"
        shutil.copyfile(path, seed)
        seeds[ticker] = hashlib.sha256(seed.read_bytes()).hexdigest()
    protocol = {
        "version": 1, "created_at": str(now), "start": str(start),
        "code_sha256": fingerprint(), "pandas": pd.__version__, "numpy": np.__version__,
        "tickers": TICKERS, "variants": VARIANTS, "seed_sha256": seeds,
        "fee_rate": .001, "slippage_multipliers": [1., 2.],
        "risk_profile": "Balanced", "user_cap_pct": 30.,
        "minimum_calendar_days": 30, "minimum_candidate_closed_trades": 50,
        "promotion": "manual review only; no automatic deployment",
    }
    protocol_path = directory / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (directory / "protocol.sha256").write_text(hashlib.sha256(protocol_path.read_bytes()).hexdigest(), encoding="utf-8")
    return protocol


def observe(directory, update=False):
    protocol_path = directory / "protocol.json"
    expected = (directory / "protocol.sha256").read_text(encoding="utf-8").strip()
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen protocol modified")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["code_sha256"] != fingerprint() or protocol["pandas"] != pd.__version__ or protocol["numpy"] != np.__version__:
        raise ValueError("Study code/runtime changed; start a separately named study")
    now, start = pd.Timestamp.now(tz="UTC"), pd.Timestamp(protocol["start"])
    results, coverage = [], []
    for ticker in protocol["tickers"]:
        seed = directory / f"{ticker}.seed.csv"
        if hashlib.sha256(seed.read_bytes()).hexdigest() != protocol["seed_sha256"][ticker]:
            raise ValueError(f"Seed modified: {ticker}")
        path = directory / f"{ticker}.csv"
        df = read_bars(path)
        seed_df = read_bars(seed)
        pd.testing.assert_frame_equal(df.loc[seed_df.index], seed_df)
        warmup_revisions = 0
        if update:
            import yfinance as yf
            yf.set_tz_cache_location(str(directory / "yf_cache"))
            incoming = yf.Ticker(ticker).history(period="60d", interval="5m", auto_adjust=True, actions=False, timeout=20)
            if incoming.empty:
                raise ValueError(f"No provider data: {ticker}")
            incoming.index = incoming.index.tz_convert("UTC")
            incoming = incoming[["Open", "High", "Low", "Close", "Volume"]]
            overlap = df.index.intersection(incoming.index)
            warmup = overlap[overlap < start]
            warmup_revisions = int((~np.isclose(df.loc[warmup].to_numpy(), incoming.loc[warmup].to_numpy(), rtol=1e-8, atol=1e-8)).sum())
            # Preserve the original pre-study observations, even if the
            # provider subsequently corrects them. Record the discrepancy.
            # Revisions inside the actual forward period still fail closed.
            df = append_closed(df, incoming, now, frozen_before=start)
            if ticker.endswith("-USD"):
                relevant = df.loc[df.index >= start]
                if not relevant.empty and (relevant.index[0] != start or (relevant.index.to_series().diff().dropna() != pd.Timedelta(minutes=5)).any()):
                    raise ValueError(f"Missing forward crypto candles: {ticker}")
            temporary = path.with_suffix(".tmp")
            df.to_csv(temporary)
            temporary.replace(path)
        coverage.append({"ticker": ticker, "last_closed_bar": str(df.index[-1]), "ignored_pre_study_revision_cells": warmup_revisions, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        first = int(df.index.searchsorted(start))
        if first >= len(df) or first < 201:
            continue
        features, votes = entry_features(df), IndicatorCommittee().vote_matrix(df)
        controls = trend_control_features(df)
        for stress in protocol["slippage_multipliers"]:
            for name, settings in protocol["variants"].items():
                result = replay_committee(df, enhanced=True, evaluation_start=first - 1,
                    features=features, votes=votes, controls=controls, finalize=False,
                    fee_rate=protocol["fee_rate"], slippage_multiplier=stress,
                    risk_profile=protocol["risk_profile"], user_cap_pct=protocol["user_cap_pct"], **settings)
                result.update(ticker=ticker, variant=name, slippage_multiplier=stress)
                results.append(result)
    count = sum(r["trades"] for r in results if r["variant"] == "adx25" and r["slippage_multiplier"] == 1.)
    # Readiness uses observed data through the earliest asset, not elapsed wall
    # time alone. Stale input can never satisfy the calendar coverage floor.
    through = min(pd.Timestamp(c["last_closed_bar"]) + pd.Timedelta(minutes=5) for c in coverage)
    days = max(0., (through - start).total_seconds() / 86400)
    status = "insufficient_forward_evidence"
    if days >= protocol["minimum_calendar_days"] and count >= protocol["minimum_candidate_closed_trades"]:
        status = "ready_for_manual_review_not_approved_for_live"
    report = {"observed_at": str(now), "start": str(start), "status": status,
              "coverage_days": days, "candidate_closed_trades": count,
              "coverage": coverage, "results": results}
    path = directory / f"observation_{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--update", action="store_true", help="Download closed public candles; never send orders")
    args = parser.parse_args()
    if args.initialize_from:
        initialize(args.directory, args.initialize_from, pd.Timestamp.now(tz="UTC"))
    lock = args.directory / ".observation.lock"
    handle = lock.open("x")
    try:
        with handle:
            path, report = observe(args.directory, update=args.update)
            print(json.dumps({"path": str(path), "status": report["status"], "start": report["start"], "candidate_closed_trades": report["candidate_closed_trades"]}))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
