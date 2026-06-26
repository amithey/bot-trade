"""Trade attribution: P&L breakdowns by ticker / weekday / hour, win-rates,
and round-trip durations.

All functions take a portfolio-style trade log (TradeRecord or dict) and
return a tidy DataFrame suitable for direct charting.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from analytics.performance import _to_records_df


_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def pnl_by_ticker(trade_log: Iterable) -> pd.DataFrame:
    """Return DataFrame(ticker, n_trades, total_pnl, avg_pnl, win_rate)
    sorted by total_pnl descending. Only counts SELL/FORCE_CLOSE rows.
    """
    df = _to_records_df(trade_log)
    sells = df[df["action"].isin(["SELL", "FORCE_CLOSE"])].copy()
    if sells.empty:
        return pd.DataFrame(
            columns=["ticker", "n_trades", "total_pnl", "avg_pnl", "win_rate"]
        )
    sells["realized_pnl"] = sells["realized_pnl"].astype(float)
    grp = sells.groupby("ticker", dropna=False)["realized_pnl"]
    out = pd.DataFrame({
        "n_trades": grp.count(),
        "total_pnl": grp.sum().round(2),
        "avg_pnl": grp.mean().round(2),
        "win_rate": (grp.apply(lambda s: (s > 0).mean())).round(3),
    }).reset_index().sort_values("total_pnl", ascending=False)
    return out


def pnl_by_weekday(trade_log: Iterable) -> pd.DataFrame:
    """Return DataFrame(weekday, n_trades, total_pnl, avg_pnl) ordered Mon→Sun."""
    df = _to_records_df(trade_log)
    sells = df[df["action"].isin(["SELL", "FORCE_CLOSE"])].copy()
    if sells.empty:
        return pd.DataFrame(columns=["weekday", "n_trades", "total_pnl", "avg_pnl"])
    sells["realized_pnl"] = sells["realized_pnl"].astype(float)
    sells["weekday_idx"] = sells["executed_at"].apply(
        lambda d: d.weekday() if isinstance(d, datetime) else -1
    )
    sells = sells[sells["weekday_idx"] >= 0]
    grp = sells.groupby("weekday_idx")["realized_pnl"]
    df_out = pd.DataFrame({
        "n_trades": grp.count(),
        "total_pnl": grp.sum().round(2),
        "avg_pnl": grp.mean().round(2),
    }).reset_index()
    df_out["weekday"] = df_out["weekday_idx"].apply(lambda i: _WEEKDAYS[i])
    df_out = df_out.set_index("weekday_idx").reindex(range(7))
    df_out["weekday"] = [_WEEKDAYS[i] for i in df_out.index]
    df_out = df_out.fillna(0)
    return df_out[["weekday", "n_trades", "total_pnl", "avg_pnl"]].reset_index(drop=True)


def pnl_by_hour(trade_log: Iterable) -> pd.DataFrame:
    """Return DataFrame(hour, n_trades, total_pnl) for hours 0..23 (UTC)."""
    df = _to_records_df(trade_log)
    sells = df[df["action"].isin(["SELL", "FORCE_CLOSE"])].copy()
    if sells.empty:
        return pd.DataFrame(columns=["hour", "n_trades", "total_pnl"])
    sells["realized_pnl"] = sells["realized_pnl"].astype(float)
    sells["hour"] = sells["executed_at"].apply(
        lambda d: d.hour if isinstance(d, datetime) else -1
    )
    sells = sells[sells["hour"] >= 0]
    grp = sells.groupby("hour")["realized_pnl"]
    out = pd.DataFrame({
        "n_trades": grp.count(),
        "total_pnl": grp.sum().round(2),
    }).reindex(range(24)).fillna(0).reset_index().rename(columns={"index": "hour"})
    return out


def win_rate_by_ticker(trade_log: Iterable) -> pd.DataFrame:
    """Alias for pnl_by_ticker but ordered by win_rate descending."""
    df = pnl_by_ticker(trade_log)
    return df.sort_values("win_rate", ascending=False)


def trade_durations(trade_log: Iterable) -> pd.DataFrame:
    """Return DataFrame of paired trips: ticker, entry_at, exit_at,
    duration_hours, realized_pnl. Pairs each BUY with the next SELL
    on the same ticker.
    """
    df = _to_records_df(trade_log)
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "entry_at", "exit_at", "duration_hours",
            "entry_price", "exit_price", "realized_pnl", "pnl_pct",
        ])
    rows: list[dict] = []
    open_buys: dict[str, dict] = {}
    for _, r in df.iterrows():
        if r["action"] == "BUY":
            open_buys[r["ticker"]] = r.to_dict()
        elif r["action"] in ("SELL", "FORCE_CLOSE") and r["ticker"] in open_buys:
            entry = open_buys.pop(r["ticker"])
            try:
                dur_h = (r["executed_at"] - entry["executed_at"]).total_seconds() / 3600.0
            except Exception:
                dur_h = 0.0
            ep = float(entry.get("price") or 0)
            xp = float(r.get("price") or 0)
            pnl_pct = ((xp - ep) / ep * 100.0) if ep > 0 else 0.0
            rows.append({
                "ticker": r["ticker"],
                "entry_at": entry["executed_at"],
                "exit_at": r["executed_at"],
                "duration_hours": round(dur_h, 2),
                "entry_price": round(ep, 4),
                "exit_price": round(xp, 4),
                "realized_pnl": round(float(r.get("realized_pnl") or 0), 2),
                "pnl_pct": round(pnl_pct, 3),
            })
    return pd.DataFrame(rows)
