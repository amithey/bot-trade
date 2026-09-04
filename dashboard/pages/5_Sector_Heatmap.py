"""Sector Heatmap — zero-token snapshot of what's hot and what's not.

Uses yfinance only — no LLM calls, no API budget. Refreshes on button.
Displays:
  • 11 GICS sector ETFs with 1D / 5D / 1M returns
  • Major index quotes (SPY, QQQ, DIA, IWM, BTC, ETH, GLD, TLT)
  • 52-week high/low proximity for each sector
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

# ── sys.path bootstrap (so `from dashboard...` works in direct `streamlit run`)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ────────────────────────────────────────────────────────────────────────────

from dashboard._shared import (
    C_BUY, C_SELL, TEXT, TEXT_DIM,
    secure_page, ensure_profile_in_session,
)

st.set_page_config(page_title="BotTrade - Sector Heatmap", page_icon=":material/grid_view:",
                   layout="wide", initial_sidebar_state="expanded")
secure_page()
ensure_profile_in_session()

st.markdown('<div class="page-title">SECTOR HEATMAP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Real-time snapshot of sector rotation — no AI tokens consumed. '
    'Data is cached for 5 min; click refresh to force reload.</div>',
    unsafe_allow_html=True,
)

# ─── GICS sector ETFs + major benchmarks ────────────────────────────────────
_SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLC":  "Communication Svcs",
}
_BENCHMARKS = {
    "SPY":     "S&P 500",
    "QQQ":     "Nasdaq 100",
    "DIA":     "Dow 30",
    "IWM":     "Russell 2000",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "GLD":     "Gold",
    "TLT":     "20Y Treasuries",
}


@st.cache_data(ttl=300, show_spinner=False)
def _load_snapshot(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Pull ~60 trading days for each symbol, compute perf metrics."""
    import yfinance as yf
    rows = []
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="3mo", interval="1d", auto_adjust=False)
            if hist is None or len(hist) < 2:
                continue
            closes = hist["Close"].dropna()
            last = float(closes.iloc[-1])
            d1 = float(closes.iloc[-2]) if len(closes) >= 2 else last
            d5 = float(closes.iloc[-6]) if len(closes) >= 6 else last
            d20 = float(closes.iloc[-21]) if len(closes) >= 21 else last
            hi52 = float(closes.max())
            lo52 = float(closes.min())
            rows.append({
                "symbol":     sym,
                "price":      last,
                "ret_1d_pct": (last / d1 - 1.0) * 100.0 if d1 else 0.0,
                "ret_5d_pct": (last / d5 - 1.0) * 100.0 if d5 else 0.0,
                "ret_1m_pct": (last / d20 - 1.0) * 100.0 if d20 else 0.0,
                "from_hi_pct": (last / hi52 - 1.0) * 100.0 if hi52 else 0.0,
                "from_lo_pct": (last / lo52 - 1.0) * 100.0 if lo52 else 0.0,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"symbol": sym, "price": None, "error": str(exc)[:80]})
    return pd.DataFrame(rows)


# ─── Control ─────────────────────────────────────────────────────────────────
c_refresh, c_info = st.columns([1, 5])
with c_refresh:
    if st.button("Refresh", use_container_width=True, help="Force reload (clears 5-min cache)"):
        _load_snapshot.clear()
        st.rerun()
with c_info:
    st.caption(f"Last rendered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")


def _color_for(ret: float) -> str:
    """Stock-green / stock-red gradient."""
    if ret is None:
        return TEXT_DIM
    if ret >= 3.0:     return "#00ff88"
    if ret >= 1.0:     return "#5ae49a"
    if ret >= 0.2:     return "#8fd6a8"
    if ret > -0.2:     return TEXT_DIM
    if ret > -1.0:     return "#e89b9b"
    if ret > -3.0:     return "#ff6666"
    return "#ff1e4f"


def _render_grid(df: pd.DataFrame, labels: dict[str, str], *, cols: int = 4) -> None:
    if df.empty:
        st.warning("No data returned — check internet/yfinance.")
        return
    rows = df.to_dict("records")
    for i in range(0, len(rows), cols):
        chunk = rows[i:i + cols]
        columns = st.columns(cols)
        for col, r in zip(columns, chunk):
            sym = r["symbol"]
            name = labels.get(sym, sym)
            price = r.get("price")
            if price is None:
                col.markdown(
                    f'<div class="bt-panel" style="text-align:center;opacity:0.5">'
                    f'<div style="font-family:monospace;font-weight:700">{sym}</div>'
                    f'<div style="color:#ff6666;font-size:0.8rem">no data</div>'
                    f'</div>', unsafe_allow_html=True,
                )
                continue
            d1 = r.get("ret_1d_pct") or 0.0
            d5 = r.get("ret_5d_pct") or 0.0
            dm = r.get("ret_1m_pct") or 0.0
            hi = r.get("from_hi_pct") or 0.0
            c1 = _color_for(d1)
            col.markdown(
                f'<div class="bt-panel" style="text-align:left">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline">'
                f'<span style="font-family:monospace;font-weight:700;color:{TEXT}">{sym}</span>'
                f'<span style="color:{TEXT_DIM};font-size:0.7rem">{name}</span>'
                f'</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:{c1};margin:4px 0">'
                f'{d1:+.2f}%'
                f'</div>'
                f'<div style="color:{TEXT_DIM};font-size:0.78rem">'
                f'5D <span style="color:{_color_for(d5)}">{d5:+.1f}%</span> · '
                f'1M <span style="color:{_color_for(dm)}">{dm:+.1f}%</span>'
                f'</div>'
                f'<div style="color:{TEXT_DIM};font-size:0.72rem;margin-top:2px">'
                f'${price:.2f} · {hi:+.1f}% from 3M hi'
                f'</div>'
                f'</div>', unsafe_allow_html=True,
            )


# ─── Sectors ─────────────────────────────────────────────────────────────────
st.markdown('<div class="bt-section-title">GICS SECTOR ROTATION</div>',
            unsafe_allow_html=True)
sectors_df = _load_snapshot(tuple(_SECTOR_ETFS.keys()))
_render_grid(sectors_df, _SECTOR_ETFS, cols=4)

# Leader / laggard summary
st.markdown("<br>", unsafe_allow_html=True)
if not sectors_df.empty and "ret_1d_pct" in sectors_df.columns:
    ranked = sectors_df.dropna(subset=["ret_1d_pct"]).sort_values(
        "ret_1d_pct", ascending=False)
    if len(ranked) >= 2:
        lead = ranked.iloc[0]
        lag = ranked.iloc[-1]
        lead_name = _SECTOR_ETFS.get(lead['symbol'], lead['symbol'])
        lag_name = _SECTOR_ETFS.get(lag['symbol'], lag['symbol'])
        lcol, rcol = st.columns(2)
        lcol.markdown(
            f'<div class="bt-panel" style="border-left:4px solid {C_BUY}">'
            f'<div style="color:{TEXT_DIM};font-size:0.75rem;letter-spacing:1px">'
            f'LEADER (1D)</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{C_BUY}">'
            f'{lead["symbol"]} · {lead_name} '
            f'<span style="color:{TEXT}">({lead["ret_1d_pct"]:+.2f}%)</span>'
            f'</div></div>', unsafe_allow_html=True,
        )
        rcol.markdown(
            f'<div class="bt-panel" style="border-left:4px solid {C_SELL}">'
            f'<div style="color:{TEXT_DIM};font-size:0.75rem;letter-spacing:1px">'
            f'LAGGARD (1D)</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{C_SELL}">'
            f'{lag["symbol"]} · {lag_name} '
            f'<span style="color:{TEXT}">({lag["ret_1d_pct"]:+.2f}%)</span>'
            f'</div></div>', unsafe_allow_html=True,
        )

st.markdown("---")

# ─── Benchmarks ──────────────────────────────────────────────────────────────
st.markdown('<div class="bt-section-title">MAJOR BENCHMARKS</div>',
            unsafe_allow_html=True)
bench_df = _load_snapshot(tuple(_BENCHMARKS.keys()))
_render_grid(bench_df, _BENCHMARKS, cols=4)

st.markdown("---")
st.caption(
    "Interpretation: When XLK & XLY lead while XLP & XLU lag, the market is typically "
    "in risk-on mode (growth rally). Flip that — defensives leading — and the "
    "tape is rotating defensive."
)
