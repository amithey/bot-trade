"""Out-of-sample test of the pre-declared daily chart-pattern rules.

Parameters are chosen per family on data BEFORE --split only; the final window
is read once, afterwards. Signals use closed bars and fill at the next open,
with the bot's 0.1% fee plus 5 bps slippage per side. Cash earns nothing.
Universe is current large caps, ETFs and coins: survivorship bias flatters
long-only rules, so treat positive results as an upper bound.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.pattern_rules import VARIANTS, FAMILY

UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EEM", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA",
            "JPM", "XOM", "JNJ", "WMT", "KO", "INTC", "PFE", "BA", "DIS", "VZ", "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"]
TRAIN_START = "2015-01-01"


def load(ticker, directory):
    path = directory / f"{ticker}.csv"
    if not path.exists():
        import yfinance as yf
        yf.set_tz_cache_location(str(directory / "yf_cache"))
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True, actions=False, timeout=30)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        directory.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def daily_returns(df, signal, cost):
    """Position decided at close t-1 is held from open t; entry/exit costs land on day t."""
    held = signal.shift(1).fillna(0.)
    open_to_open = df.Open.shift(-1) / df.Open - 1
    return (held * open_to_open - cost * held.diff().abs().fillna(held.abs())).iloc[:-1], held.iloc[:-1]


def newey_west_t(x, lags=5):
    x = np.asarray(x, float)
    n = len(x)
    if n < 30:
        return float("nan")
    centered = x - x.mean()
    variance = centered @ centered / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (centered[lag:] @ centered[:-lag]) / n
    return float(x.mean() / np.sqrt(variance / n)) if variance > 0 else float("nan")


def summarise(strategy, benchmark, exposure):
    years = max((strategy.index[-1] - strategy.index[0]).days / 365.25, 1e-9)
    per_year = len(strategy) / years
    curve = (1 + strategy).cumprod()
    bench_curve = (1 + benchmark).cumprod()
    beta = float(np.cov(strategy, benchmark)[0, 1] / benchmark.var()) if benchmark.var() > 0 else 0.
    residual = strategy - beta * benchmark
    return {
        "return_pct": float((curve.iloc[-1] - 1) * 100), "buy_hold_pct": float((bench_curve.iloc[-1] - 1) * 100),
        "sharpe": float(strategy.mean() / strategy.std() * np.sqrt(per_year)) if strategy.std() > 0 else 0.,
        "max_dd_pct": float((curve / curve.cummax() - 1).min() * 100),
        "bh_max_dd_pct": float((bench_curve / bench_curve.cummax() - 1).min() * 100),
        "exposure_pct": float(exposure.mean() * 100), "beta": beta,
        "alpha_pct_per_year": float(residual.mean() * per_year * 100), "alpha_t": newey_west_t(residual),
    }


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def portfolio(returns, held):
    """Equal-weight sleeves, each asset independent; averages whatever assets trade that day."""
    return pd.DataFrame(returns).mean(axis=1, skipna=True), pd.DataFrame(held).mean(axis=1, skipna=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="2023-09-19", help="first day of the untouched test window")
    parser.add_argument("--cost-multiplier", type=float, default=1.)
    parser.add_argument("--tickers", nargs="+", default=UNIVERSE)
    args = parser.parse_args()
    cost = (0.001 + 0.0005) * args.cost_multiplier
    split = pd.Timestamp(args.split)
    frames = {t: load(t, args.data_dir) for t in args.tickers}
    bench = {t: (df.Open.shift(-1) / df.Open - 1).iloc[:-1] for t, df in frames.items()}
    results = []
    for name, rule in VARIANTS.items():
        rets, helds = {}, {}
        for ticker, df in frames.items():
            rets[ticker], helds[ticker] = daily_returns(df, rule(df), cost)
        ret_all, held_all = portfolio(rets, helds)
        bench_all = pd.DataFrame(bench).mean(axis=1, skipna=True)
        row = {"variant": name, "family": FAMILY[name]}
        for label, mask in (("train", (ret_all.index >= TRAIN_START) & (ret_all.index < split)),
                            ("test", ret_all.index >= split)):
            row[label] = summarise(ret_all[mask], bench_all.reindex(ret_all.index)[mask], held_all[mask])
        per_asset = {}
        for ticker in frames:
            mask = rets[ticker].index >= split
            asset = rets[ticker][mask]
            per_asset[ticker] = {"return_pct": float(((1 + asset).prod() - 1) * 100),
                                 "buy_hold_pct": float(((1 + bench[ticker][mask]).prod() - 1) * 100),
                                 "exposure_pct": float(helds[ticker][mask].mean() * 100)}
        row["test_per_asset"] = per_asset
        results.append(row)
    selected = {}
    for family in dict.fromkeys(FAMILY.values()):
        members = [r for r in results if r["family"] == family]
        selected[family] = max(members, key=lambda r: r["train"]["sharpe"])["variant"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean({"split": args.split, "cost_per_side": cost, "variants_tested": len(VARIANTS),
                                             "universe": args.tickers, "selected_on_train": selected, "results": results}),
                                      indent=2, allow_nan=False), encoding="utf-8")
    fmt = "{:<27}{:>8}{:>9}{:>8}{:>8}{:>9}{:>9}{:>8}{:>8}{:>8}{:>9}{:>7}"
    print(fmt.format("variant", "trSharpe", "trAlpha%", "teRet%", "teB&H%", "teSharpe", "teDD%", "B&H DD", "expo%", "alpha%", "alpha t", "sel"))
    for r in results:
        tr, te = r["train"], r["test"]
        print(fmt.format(r["variant"], f"{tr['sharpe']:.2f}", f"{tr['alpha_pct_per_year']:.1f}", f"{te['return_pct']:.1f}",
                         f"{te['buy_hold_pct']:.1f}", f"{te['sharpe']:.2f}", f"{te['max_dd_pct']:.1f}", f"{te['bh_max_dd_pct']:.1f}",
                         f"{te['exposure_pct']:.0f}", f"{te['alpha_pct_per_year']:.1f}", "n/a" if te["alpha_t"] is None or te["alpha_t"] != te["alpha_t"] else f"{te['alpha_t']:.2f}",
                         "*" if selected[r["family"]] == r["variant"] else ""))


if __name__ == "__main__":
    main()
