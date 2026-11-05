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
from trading.trend_scanner_runner import CONFIG, UNIVERSE, TrendScannerRunner
from strategy.regime_core_satellite import detect_regime, CoreSatelliteConfig

page_header("Trend scanner", "Multi-asset trend following with trailing chandelier stops & Core/Satellite regime protection.",
            section="Trade / Paper strategy")

STATE_DIR = _PROJECT_ROOT / "data" / "trend_scanner_paper"
portfolio_path, state_path = STATE_DIR / "portfolio.json", STATE_DIR / "scanner_state.json"

# ── Control panel ─────────────────────────────────────────────────────────────
c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 2, 2])
with c_ctrl1:
    selected_interval = st.selectbox("Execution Timeframe", ["Daily (1d)", "4-Hour (4h)"], index=0)
    interval_code = "4h" if "4h" in selected_interval else "1d"

with c_ctrl2:
    enable_core_sat = st.checkbox(
        "Core/Satellite Regime Protection",
        value=True,
        help="Keeps 45% core baseline in broad market proxy (SPY) when Price > SMA200 so the market never outruns cash, while reserving 55% for active trend breakouts.",
    )

with c_ctrl3:
    st.write("")
    st.write("")
    run_clicked = st.button("⚡ Run Scan / Step Now", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner(f"Running trend scan across {len(UNIVERSE)} assets ({selected_interval})..."):
        runner = TrendScannerRunner(
            STATE_DIR,
            universe=UNIVERSE,
            config=CONFIG,
            interval=interval_code,
            core_satellite=enable_core_sat,
        )
        step_result = runner.step()
        st.success(f"Scan step completed at {step_result.get('at', '')[:19]}! {len(step_result.get('decided', []))} assets evaluated.")
        st.rerun()

# ── Check state existence ─────────────────────────────────────────────────────
if not portfolio_path.exists() or not state_path.exists():
    st.info("The trend scanner has not completed its first run yet. Click 'Run Scan / Step Now' above to initialize and execute the first scan.")
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

errors = state.get("last_step", {}).get("errors") or {}
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
