"""
Trend scanner — read-only view of the shared multi-asset paper account.

The scanner runs as its own background process (docker/entrypoint.sh) and
writes its state to data/trend_scanner_paper/. This page only reads it.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import json

import pandas as pd
import streamlit as st

from dashboard._shared import secure_page

st.set_page_config(page_title="Trend Scanner", layout="wide", page_icon=":material/trending_up:")
secure_page()

from dashboard.components import page_header
from portfolio.virtual_account import LivePortfolio
from trading.trend_scanner_runner import CONFIG, UNIVERSE

page_header("Trend scanner", "Daily multi-asset trend following on a shared paper account.",
            section="Trade / Paper strategy")

STATE_DIR = _PROJECT_ROOT / "data" / "trend_scanner_paper"
portfolio_path, state_path = STATE_DIR / "portfolio.json", STATE_DIR / "scanner_state.json"
if not portfolio_path.exists() or not state_path.exists():
    st.info("The scanner has not completed its first run yet.")
    st.stop()

portfolio = LivePortfolio.load(portfolio_path)
state = json.loads(state_path.read_text(encoding="utf-8"))
value = portfolio.get_total_value()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"${value:,.2f}", f"{100 * (value / portfolio.initial_capital - 1):+.2f}%")
c2.metric("Cash", f"${portfolio.cash:,.2f}")
c3.metric("Open positions", f"{len(portfolio.positions)} / {CONFIG.max_positions}")
c4.metric("Last run (UTC)", state.get("last_step", {}).get("at", "never")[:16].replace("T", " "))

errors = state.get("last_step", {}).get("errors") or {}
if errors:
    st.warning("Data unavailable on the last run for: " + ", ".join(sorted(errors)))

st.subheader("Open positions")
rows = [{"Asset": t, "Quantity": p.quantity, "Entry": p.avg_entry_price, "Last": p.current_price,
         "P&L %": p.unrealized_pnl_pct, "Stop": state["positions"].get(t, {}).get("stop"),
         "Opened": p.opened_at.strftime("%Y-%m-%d")} for t, p in portfolio.positions.items()]
st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Asset"]), hide_index=True, width="stretch")

pending = ([{"Order": "BUY at next open", "Asset": t, "Weight %": round(o["weight_pct"], 2), "Stop": o["stop"]}
            for t, o in state.get("pending_entries", {}).items()] +
           [{"Order": "SELL at next open", "Asset": t, "Weight %": None, "Stop": None}
            for t in state.get("pending_exits", {})])
if pending:
    st.subheader("Orders waiting for the next session")
    st.dataframe(pd.DataFrame(pending), hide_index=True, width="stretch")

st.subheader("Activity")
events = list(reversed(state.get("events", [])))[:100]
st.dataframe(pd.DataFrame(events) if events else pd.DataFrame(columns=["at", "kind", "ticker", "detail"]),
             hide_index=True, width="stretch")

with st.expander("How it decides"):
    st.markdown(
        f"- Universe: {', '.join(UNIVERSE)}\n"
        f"- Entry: 10-day average above 50-day, price above 50-day, and a new 20-day high on a completed day; "
        f"strongest trends first, up to {CONFIG.max_positions} positions.\n"
        f"- Size: about {CONFIG.risk_pct:.0f}% of equity at risk to the initial stop, max {CONFIG.max_weight_pct:.0f}% per asset.\n"
        f"- Exit: stop {CONFIG.stop_atr:.0f}×ATR below the highest close since entry (only moves up), "
        "or the trend rule failing at a close. No fixed target, no time limit.\n"
        "- Paper only. Backtest 2022–2026: similar return to buy-and-hold with about half the drawdown; "
        "excess return is not established.")
