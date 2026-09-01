"""
Committee Lab — backtest the 38-indicator voting strategy on any asset.

Runs entirely offline (no Claude API calls), so a 5-year daily backtest
finishes in about a second. Crypto and stocks alike.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard._shared import (
    BORDER, C_BUY, C_SELL, CYAN, DEFAULT_TICKERS, GRID, TEXT, TEXT_DIM,
    TEXT_HI, secure_page,
)

st.set_page_config(page_title="Committee Lab", layout="wide",
                   page_icon=":material/how_to_vote:")
secure_page()

st.markdown('<div class="page-title">🗳 Committee Lab</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">38 indicators · one vote per window · long-only, '
    'exits to cash · crypto &amp; stocks · zero API cost</div>',
    unsafe_allow_html=True)

# ── Controls ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns([1.8, 1.1, 1.3, 1.3, 1.3, 1.0],
                                    gap="small")
with c1:
    opts = list(dict.fromkeys(DEFAULT_TICKERS)) + ["Custom"]
    sel = st.selectbox("ASSET", opts, index=0)
    if sel == "Custom":
        ticker = st.text_input("", placeholder="e.g. TSLA / SOL-USD",
                               label_visibility="collapsed").upper().strip()
    else:
        ticker = sel
with c2:
    interval = st.selectbox("BARS", ["1d", "1h"], index=0,
                            help="Hourly history only goes back ~2 years "
                                 "(Yahoo limit). Daily goes back decades.")
with c3:
    years = st.select_slider("LOOKBACK", options=[0.5, 1, 2, 3, 5, 8],
                             value=3, format_func=lambda y: f"{y} yr")
with c4:
    enter_votes = st.slider("ENTER MARGIN", 2, 16, 6, step=2,
                            help="How many net bull votes (of 38) are needed "
                                 "to go long.")
with c5:
    exit_votes = st.slider("EXIT MARGIN", 2, 16, 6, step=2,
                           help="How many net bear votes force an exit "
                                "to cash.")
with c6:
    fee_pct = st.select_slider("FEE/SIDE", options=[0.0, 0.05, 0.1, 0.25],
                               value=0.1, format_func=lambda f: f"{f:.2f}%")

b1, b2, _sp = st.columns([1.2, 1.6, 5.2], gap="small")
run = b1.button("RUN BACKTEST", type="primary")
optimize = b2.button("⚡ AUTO-OPTIMIZE", help="Grid-search all entry/exit "
                     "margin combinations and pick the best risk-adjusted "
                     "setup (return − ½·|drawdown|). Takes a few seconds.")

st.markdown("")


@st.cache_data(ttl=600, show_spinner=False)
def _fetch(tk: str, days: int, iv: str) -> pd.DataFrame:
    from strategy.committee_backtest import fetch_history
    return fetch_history(tk, days, iv)


if run:
    if not ticker:
        st.warning("Pick an asset first.")
        st.stop()

    from strategy.committee import CommitteeConfig
    from strategy.committee_backtest import backtest_committee

    cfg = CommitteeConfig(
        enter_score=enter_votes / 38.0,
        exit_score=-exit_votes / 38.0,
    )

    with st.spinner(f"Fetching {ticker} and replaying the committee…"):
        try:
            df = _fetch(ticker, int(years * 365), interval)
            res = backtest_committee(df, ticker=ticker, interval=interval,
                                     config=cfg, fee_pct=fee_pct)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    st.session_state["_committee_result"] = res
    st.session_state.pop("_committee_grid", None)

if optimize:
    if not ticker:
        st.warning("Pick an asset first.")
        st.stop()

    from strategy.committee_backtest import optimize_committee

    with st.spinner(f"Optimizing {ticker} — replaying 49 committee "
                    f"configurations…"):
        try:
            df = _fetch(ticker, int(years * 365), interval)
            cells, best_res = optimize_committee(
                df, ticker=ticker, interval=interval, fee_pct=fee_pct)
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
            st.stop()

    st.session_state["_committee_result"] = best_res
    st.session_state["_committee_grid"] = cells

grid = st.session_state.get("_committee_grid")
if grid:
    best = grid[0]
    st.success(f"Best setup for this window: **enter at +{best.enter_votes} "
               f"net votes / exit at −{best.exit_votes}** → "
               f"{best.total_return_pct:+.1f}% return, "
               f"{best.max_drawdown_pct:.1f}% max drawdown, "
               f"{best.total_trades} trades. Chart below uses this setup.")

    margins = sorted({c.enter_votes for c in grid})
    z = [[next((c.fitness for c in grid
                if c.enter_votes == ev and c.exit_votes == xv), None)
          for ev in margins] for xv in margins]
    heat = go.Figure(go.Heatmap(
        z=z, x=[f"+{m}" for m in margins], y=[f"−{m}" for m in margins],
        colorscale=[[0, "#3d1220"], [0.5, "#0b0f14"], [1, "#0a3d2a"]],
        colorbar=dict(title=dict(text="fitness", font=dict(size=10)),
                      tickfont=dict(size=9)),
        hovertemplate=("enter %{x} / exit %{y}<br>"
                       "fitness %{z:.1f}<extra></extra>"),
    ))
    heat.update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#020406",
        margin=dict(l=12, r=12, t=26, b=12),
        title=dict(text="Risk-adjusted fitness by entry / exit margin",
                   font=dict(size=12, color=TEXT_DIM)),
        font=dict(family="Inter, Segoe UI, sans-serif", size=11, color=TEXT),
        xaxis=dict(title="enter margin (net bull votes)"),
        yaxis=dict(title="exit margin (net bear votes)"),
    )
    c_heat, c_top = st.columns([1.4, 1], gap="small")
    c_heat.plotly_chart(heat, use_container_width=True)
    with c_top:
        top_rows = [{
            "Enter": f"+{c.enter_votes}", "Exit": f"−{c.exit_votes}",
            "Return": f"{c.total_return_pct:+.1f}%",
            "Max DD": f"{c.max_drawdown_pct:.1f}%",
            "Trades": c.total_trades,
            "Fitness": c.fitness,
        } for c in grid[:8]]
        st.dataframe(pd.DataFrame(top_rows), hide_index=True,
                     use_container_width=True, height=290)

res = st.session_state.get("_committee_result")
if res is None:
    st.info("Choose an asset and press RUN BACKTEST. The committee replays "
            "every historical bar — 38 indicators vote, the bot goes long on "
            "a bull majority and steps out to cash on a bear majority.")
    st.stop()

# ── KPI strip ───────────────────────────────────────────────────────────────
_pos = res.total_return_pct >= res.buy_hold_return_pct


def _kpi(label: str, value: str, color: str = TEXT_HI, sub: str = "") -> str:
    sub_html = (f'<span style="color:{TEXT_DIM};font-size:.66rem;'
                f'margin-left:.35rem;font-family:monospace;">{sub}</span>'
                if sub else "")
    return (f'<div class="kpi-item"><span class="kpi-label">{label}</span>'
            f'<span class="kpi-value" style="color:{color};">{value}</span>'
            f'{sub_html}</div>')


st.markdown(
    '<div class="kpi-strip">'
    + _kpi("Committee", f"{res.total_return_pct:+.1f}%",
           C_BUY if res.total_return_pct >= 0 else C_SELL)
    + _kpi("Buy & Hold", f"{res.buy_hold_return_pct:+.1f}%",
           TEXT_HI)
    + _kpi("Alpha", f"{res.alpha_pct:+.1f}pp",
           C_BUY if res.alpha_pct >= 0 else C_SELL)
    + _kpi("Max DD", f"{res.max_drawdown_pct:.1f}%", C_SELL,
           f"B&H {res.bh_max_drawdown_pct:.1f}%")
    + _kpi("DD Edge", f"{-res.drawdown_edge_pp:+.1f}pp",
           C_BUY if res.drawdown_edge_pp > 0 else TEXT_HI,
           "shallower" if res.drawdown_edge_pp > 0 else "")
    + _kpi("Capture", f"▲{res.upside_captured_pct:.0f}%",
           TEXT_HI, f"▼{res.downside_captured_pct:.0f}%")
    + _kpi("Trades", str(res.total_trades), TEXT_HI,
           f"win {res.win_rate_pct:.0f}%")
    + _kpi("In Market", f"{res.time_in_market_pct:.0f}%", CYAN,
           f"fees {res.fees_paid_pct:.1f}%")
    + "</div>",
    unsafe_allow_html=True,
)

# ── Equity overlay chart (green = bot, gray = buy & hold) ──────────────────
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.58, 0.22, 0.20],
    subplot_titles=("", "Committee net score (38 voters)", "Long / Cash"),
)

fig.add_trace(go.Scatter(
    x=res.buy_hold.index, y=(res.buy_hold - 1) * 100,
    name="Buy & Hold (no touch)",
    line=dict(color="#5a6b7d", width=1.4),
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=res.equity.index, y=(res.equity - 1) * 100,
    name="Committee bot",
    line=dict(color=C_BUY, width=2.0),
), row=1, col=1)

# Trade markers on the equity curve
entries = [t for t in res.trades]
if entries:
    eq_pct = (res.equity - 1) * 100
    ent_x = [t.entry_time for t in entries]
    fig.add_trace(go.Scatter(
        x=ent_x, y=[float(eq_pct.asof(x)) for x in ent_x],
        mode="markers", name="Enter long",
        marker=dict(symbol="triangle-up", size=9, color=C_BUY,
                    line=dict(width=1, color="white")),
    ), row=1, col=1)
    ex = [(t.exit_time, t.pnl_pct) for t in entries if t.exit_time is not None]
    if ex:
        fig.add_trace(go.Scatter(
            x=[x for x, _ in ex],
            y=[float(eq_pct.asof(x)) for x, _ in ex],
            mode="markers", name="Exit to cash",
            marker=dict(symbol="triangle-down", size=9, color=C_SELL,
                        line=dict(width=1, color="white")),
            customdata=[p for _, p in ex],
            hovertemplate="Exit · trade P&L %{customdata:+.1f}%<extra></extra>",
        ), row=1, col=1)

# Score subplot
fig.add_trace(go.Scatter(
    x=res.score.index, y=res.score, name="Net score",
    line=dict(color="#ab47bc", width=1.1), showlegend=False,
), row=2, col=1)
fig.add_hline(y=0, line_color="#2a3a4c", line_width=1, row=2, col=1)

# Position strip
fig.add_trace(go.Scatter(
    x=res.position.index, y=res.position, name="Position",
    fill="tozeroy", mode="lines",
    line=dict(color=C_BUY, width=0.6),
    fillcolor="rgba(0,193,118,0.22)", showlegend=False,
), row=3, col=1)

is_crypto = "-USD" in res.ticker.upper()
fig.update_layout(
    template="plotly_dark", height=680,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#020406",
    margin=dict(l=12, r=54, t=28, b=12),
    hovermode="x unified",
    font=dict(family="Inter, Segoe UI, sans-serif", size=11, color=TEXT),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=True, gridcolor=GRID,
               rangebreaks=[] if is_crypto else [dict(bounds=["sat", "mon"])]),
    yaxis=dict(side="right", showgrid=True, gridcolor=GRID,
               ticksuffix="%"),
    yaxis2=dict(side="right", range=[-1, 1], showgrid=True, gridcolor=GRID),
    yaxis3=dict(side="right", range=[-0.05, 1.05], showticklabels=False),
)
for ann in fig.layout.annotations:
    ann.font.size = 10
    ann.font.color = TEXT_DIM
st.plotly_chart(fig, use_container_width=True)

# ── Trades table ────────────────────────────────────────────────────────────
closed = [t for t in res.trades if t.exit_time is not None]
open_t = [t for t in res.trades if t.exit_time is None]
with st.expander(f"Trade log — {len(res.trades)} trades "
                 f"({len(closed)} closed, {len(open_t)} open)",
                 expanded=False):
    if res.trades:
        rows = [{
            "Entry":  t.entry_time.strftime("%Y-%m-%d %H:%M"),
            "Exit":   (t.exit_time.strftime("%Y-%m-%d %H:%M")
                       if t.exit_time is not None else "— open —"),
            "Entry $": f"{t.entry_price:,.2f}",
            "Exit $":  (f"{t.exit_price:,.2f}"
                        if t.exit_price is not None else "—"),
            "P&L %":   (f"{t.pnl_pct:+.2f}%"
                        if t.pnl_pct is not None else "—"),
        } for t in reversed(res.trades)]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True, height=320)
    else:
        st.caption("The committee never reached an entry majority in this "
                   "window. Lower the ENTER MARGIN and try again.")

st.caption(
    f"{res.ticker} · {res.bars} bars ({res.interval}) · "
    f"{res.start:%Y-%m-%d} → {res.end:%Y-%m-%d} · decided at close, "
    f"filled next open, {fee_pct if run else 0.1}% taker fee per side. "
    "Past performance does not guarantee future results."
)
