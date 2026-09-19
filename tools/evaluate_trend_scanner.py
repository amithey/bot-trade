"""Portfolio test of the multi-asset trend scanner against buy-and-hold.

The configuration is chosen by Sharpe on the training window only, then the
test window is reported for that one choice (and, for disclosure, the whole grid).
Note: 2019+ daily data was already inspected in the 2026-09-19 pattern study,
so the test window is not pristine. Universe = today's large caps: survivorship bias.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.trend_scanner import GRID, simulate_portfolio
from tools.evaluate_patterns_daily import UNIVERSE, load, newey_west_t, clean

CRYPTO = [t for t in UNIVERSE if t.endswith("-USD")]


def buy_hold(frames, start, end):
    """Equal dollars at the start into every asset that already trades; never rebalanced."""
    prices = pd.DataFrame({t: df.Close for t, df in frames.items()}).sort_index().ffill()
    prices = prices[(prices.index >= start) & (prices.index <= end)]
    live = prices.columns[prices.iloc[0].notna()]
    return (prices[live] / prices[live].iloc[0]).mean(axis=1) * 10000


def stats(curve, spy_curve):
    daily = curve.pct_change().dropna()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    per_year = len(daily) / years
    result = {"return_pct": (curve.iloc[-1] / curve.iloc[0] - 1) * 100,
              "cagr_pct": ((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1) * 100,
              "sharpe": daily.mean() / daily.std() * np.sqrt(per_year) if daily.std() > 0 else 0.,
              "max_dd_pct": (curve / curve.cummax() - 1).min() * 100}
    if spy_curve is not None:
        # Stocks rest at weekends; compare on SPY trading days only.
        joint = pd.concat([curve, spy_curve], axis=1, keys=["s", "b"]).dropna()
        s, b = joint.s.pct_change().dropna(), joint.b.pct_change().dropna()
        beta = float(np.cov(s, b)[0, 1] / b.var())
        residual = s - beta * b
        result.update(beta=beta, alpha_pct_per_year=float(residual.mean() * 252 * 100), alpha_t=newey_west_t(residual))
    return result


def trade_stats(trades):
    if not trades:
        return {"trades": 0}
    returns = np.array([t["return_pct"] for t in trades])
    hold = [(pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days for t in trades]
    wins, losses = returns[returns > 0], returns[returns <= 0]
    return {"trades": len(trades), "win_rate_pct": 100 * len(wins) / len(returns),
            "avg_win_pct": float(wins.mean()) if len(wins) else 0., "avg_loss_pct": float(losses.mean()) if len(losses) else 0.,
            "median_hold_days": float(np.median(hold)),
            "profit_factor": float(sum(t["pnl"] for t in trades if t["pnl"] > 0) / -sum(t["pnl"] for t in trades if t["pnl"] < 0))
            if any(t["pnl"] < 0 for t in trades) else None}


def run(frames, config, start, end, spy):
    curve, trades = simulate_portfolio(frames, config, start, end)
    return {**stats(curve.equity, spy), **trade_stats(trades),
            "exposure_pct": float((curve.invested / curve.equity).mean() * 100)}, curve, trades


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", nargs=2, default=["2016-01-01", "2021-12-31"])
    parser.add_argument("--test", nargs=2, default=["2022-01-01", "2026-09-18"])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = {t: load(t, args.data_dir) for t in UNIVERSE}
    universes = {"all": frames, "stocks": {t: f for t, f in frames.items() if t not in CRYPTO},
                 "crypto": {t: f for t, f in frames.items() if t in CRYPTO}}
    spy = frames["SPY"].Close
    windows = {"train": [pd.Timestamp(d) for d in args.train], "test": [pd.Timestamp(d) for d in args.test]}
    rows = []
    for config in GRID:
        row = {"config": config.__dict__}
        for label, (start, end) in windows.items():
            row[label], _, _ = run(frames, config, start, end, spy[(spy.index >= start) & (spy.index <= end)])
        rows.append(row)
        print(f"{config.entry_rule:<12} stop {config.stop_atr:.0f}ATR slots {config.max_positions:>2} | "
              f"train sharpe {row['train']['sharpe']:.2f} cagr {row['train']['cagr_pct']:5.1f}% | "
              f"test sharpe {row['test']['sharpe']:.2f} cagr {row['test']['cagr_pct']:5.1f}% dd {row['test']['max_dd_pct']:5.1f}%", flush=True)
    chosen = max(rows, key=lambda r: r["train"]["sharpe"])
    from strategy.trend_scanner import ScannerConfig
    config = ScannerConfig(**chosen["config"])
    report = {"chosen_on_train": chosen["config"], "grid": rows, "universe": UNIVERSE, "detail": {}}
    for name, universe in universes.items():
        for label, (start, end) in windows.items():
            spy_window = spy[(spy.index >= start) & (spy.index <= end)]
            result, curve, trades = run(universe, config, start, end, spy_window)
            report["detail"][f"{name}_{label}"] = {
                "scanner": result,
                "buy_hold_equal": stats(buy_hold(universe, start, end), spy_window),
                "spy": stats(spy_window / spy_window.iloc[0] * 10000, None),
                "exit_reasons": pd.Series([t["reason"] for t in trades]).value_counts().to_dict() if trades else {},
                "by_asset_pnl": pd.DataFrame(trades).groupby("ticker").pnl.sum().round(0).to_dict() if trades else {},
            }
            if name == "all":
                curve.to_csv(args.output.parent / f"trend_scanner_curve_{label}.csv")
                pd.DataFrame(trades).to_csv(args.output.parent / f"trend_scanner_trades_{label}.csv", index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(report), indent=2, default=str), encoding="utf-8")
    print("\nchosen on train:", chosen["config"])
    fmt = "{:<14}{:>11}{:>8}{:>8}{:>9}{:>9}{:>9}{:>8}"
    print(fmt.format("window", "who", "CAGR%", "Sharpe", "MaxDD%", "alpha%", "alpha_t", "trades"))
    for key, d in report["detail"].items():
        for who in ("scanner", "buy_hold_equal", "spy"):
            s = d[who]
            print(fmt.format(key, who[:11], f"{s['cagr_pct']:.1f}", f"{s['sharpe']:.2f}", f"{s['max_dd_pct']:.1f}",
                             f"{s.get('alpha_pct_per_year', float('nan')):.1f}", f"{s.get('alpha_t') or float('nan'):.2f}",
                             str(s.get("trades", ""))))


if __name__ == "__main__":
    main()
