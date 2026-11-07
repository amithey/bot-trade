"""
Trend scanner — Multi-asset daily / 4H trend following on a shared paper account.

Operates across a diversified 29-asset universe (equities, ETFs, crypto) with
Chandelier trailing ATR stops, optional Core/Satellite regime allocation,
and on-demand scanning from this page or background automation.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from dashboard._shared import secure_page

st.set_page_config(page_title="Trend Scanner", layout="wide", page_icon=":material/trending_up:")
secure_page()

from dashboard.components import page_header
from portfolio.virtual_account import LivePortfolio
from trading.trend_scanner_runner import CONFIG, UNIVERSE
from strategy.regime_core_satellite import detect_regime, CoreSatelliteConfig

page_header("Trend scanner", "Multi-asset trend following with trailing chandelier stops & Core/Satellite regime protection.",
            section="Trade / Paper strategy")

STATE_DIR = _PROJECT_ROOT / "data" / "trend_scanner_paper"
portfolio_path, state_path = STATE_DIR / "portfolio.json", STATE_DIR / "scanner_state.json"

# This page only reads. The background loop (tools/run_trend_scanner.py) is the single
# writer of the scanner state; a second writer here once overwrote its trades.
if not portfolio_path.exists() or not state_path.exists():
    st.info("The trend scanner has not completed its first run yet. It starts automatically with the server.")
    st.stop()

portfolio = LivePortfolio.load(portfolio_path)
state = json.loads(state_path.read_text(encoding="utf-8"))
value = portfolio.get_total_value()

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"${value:,.2f}", f"{100 * (value / portfolio.initial_capital - 1):+.2f}%")
c2.metric("Cash", f"${portfolio.cash:,.2f}", f"{(portfolio.cash / value * 100):.1f}% unallocated")
c3.metric("Open positions", f"{len(portfolio.positions)} / {CONFIG.max_positions}")
last_step_at = state.get("last_step", {}).get("at", "never")[:16].replace("T", " ")
c4.metric("Last run (UTC)", last_step_at)

last_step = state.get("last_step", {})
st.caption(
    f"Timeframe {last_step.get('interval', '1d')} · "
    f"Core/Satellite {'ON' if last_step.get('core_satellite') else 'OFF'} · "
    f"{last_step.get('universe_count', len(UNIVERSE))} assets · runs automatically every 30 minutes; this page is read-only."
)

errors = last_step.get("errors") or {}
if errors:
    st.warning("Data unavailable on the last run for: " + ", ".join(sorted(errors)))

# ── Positions & Orders ────────────────────────────────────────────────────────
st.subheader("Open positions")
rows = []
for t, p in portfolio.positions.items():
    stop = state.get("positions", {}).get(t, {}).get("stop")
    rows.append({
        "Asset": t,
        "Quantity": f"{p.quantity:.4f}",
        "Entry ($)": f"${p.avg_entry_price:,.2f}",
        "Last ($)": f"${p.current_price:,.2f}",
        "P&L (%)": f"{p.unrealized_pnl_pct:+.2f}%",
        "Stop ($)": f"${stop:,.2f}" if stop else "—",
        "Opened": p.opened_at.strftime("%Y-%m-%d"),
    })
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.info("No active positions currently held. All capital is held in cash waiting for confirmed breakouts.")

pending = ([{"Order": "BUY at next open", "Asset": t, "Weight %": round(o["weight_pct"], 2), "Stop": f"${o['stop']:,.2f}"}
            for t, o in state.get("pending_entries", {}).items()] +
           [{"Order": "SELL at next open", "Asset": t, "Weight %": None, "Stop": None}
            for t in state.get("pending_exits", {})])
if pending:
    st.subheader("Orders waiting for the next session")
    st.dataframe(pd.DataFrame(pending), hide_index=True, use_container_width=True)

st.subheader("Recent activity log")
events = list(reversed(state.get("events", [])))[:50]
if events:
    st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True)
else:
    st.caption("No events logged yet.")

with st.expander("Strategy configuration & architecture"):
    st.markdown(
        f"- **Universe ({len(UNIVERSE)} assets)**: {', '.join(UNIVERSE)}\n"
        f"- **Timeframe**: Daily or 4-Hour completed candles with trailing Chandelier stops.\n"
        f"- **Core/Satellite**: 45% Core index baseline in bull markets (Price >= SMA200); 55% active momentum satellite.\n"
        f"- **Entry Rule**: `{CONFIG.entry_rule}` (10-day MA > 50-day MA, Price > 50-day MA, and 20-day high breakout).\n"
        f"- **Position Sizing**: ~{CONFIG.risk_pct:.0f}% risk budget per asset to stop, max {CONFIG.max_weight_pct:.0f}% per position.\n"
        f"- **Exit Mechanism**: Chandelier trailing stop ({CONFIG.stop_atr:.0f}×ATR below peak close) or trend invalidation. No arbitrary profit cap."
    )
