"""Analytics — performance metrics, equity curve, drawdown, attribution.

Pulls everything from the in-session ``LivePortfolio`` (no LLM, no API).
KPI strip → equity curve vs benchmark → drawdown → attribution heatmaps
→ holdings risk (concentration + correlation matrix).
"""
from __future__ import annotations

# ── sys.path bootstrap ─────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import (
    BG_PANEL, BORDER, C_BUY, C_HOLD, C_SELL, CYAN, GRID, TEXT, TEXT_DIM,
    secure_page, ensure_portfolio_in_session, ensure_profile_in_session,
)

st.set_page_config(page_title="BotTrade - Analytics",
                   page_icon=":material/analytics:",
                   layout="wide", initial_sidebar_state="expanded")
secure_page()
ensure_profile_in_session()
ensure_portfolio_in_session()

from analytics import (
    compute_metrics, equity_curve, drawdown_series, benchmark_equity,
    pnl_by_ticker, pnl_by_weekday, pnl_by_hour, trade_durations,
    concentration, correlation_matrix,
)

port = st.session_state["portfolio"]

st.markdown('<div class="page-title">ANALYTICS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Quantitative performance — Sharpe / Sortino / Calmar, '
    'equity vs benchmark, drawdown, P&amp;L attribution, holdings risk. '
    'No LLM tokens consumed.</div>',
    unsafe_allow_html=True,
)

# ─── Controls ──────────────────────────────────────────────────────────────
c_bench, c_corr, c_info = st.columns([1, 1, 3])
with c_bench:
    benchmark = st.selectbox(
        "Benchmark",
        options=["SPY", "QQQ", "DIA", "IWM", "BTC-USD", "None"],
        index=0,
        key="anl_bench",
    )
with c_corr:
    corr_lookback = st.selectbox(
        "Corr. lookback",
        options=[30, 60, 90, 180],
        index=2,
        key="anl_corr_lb",
    )
with c_info:
    st.caption(
        f"Computed from {len(port.trade_log)} trade(s) · "
        f"Rendered {datetime.now().strftime('%H:%M:%S')}"
    )

st.markdown("---")


# ─── No data guard ─────────────────────────────────────────────────────────
if not port.trade_log:
    st.info(
        "No trades have been executed yet. Run the live engine on the "
        "**Portfolio** page to start populating the trade log — analytics "
        "will appear here as soon as the first round trip closes."
    )
    st.stop()


# ─── Compute everything ────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Crunching analytics…")
def _crunch(
    n_trades: int,
    last_trade_iso: str,
    initial_capital: float,
    current_value: float,
    benchmark_ticker: str,
    corr_lb: int,
):
    """Cache key includes a portfolio fingerprint so the cache invalidates
    automatically when a new trade lands."""
    trade_log = list(port.trade_log)
    snaps = list(port._daily_snapshots)  # noqa: SLF001
    pos = list(port.positions.values())

    metrics = compute_metrics(trade_log, snaps, initial_capital, current_value)
    eq = equity_curve(trade_log, snaps, initial_capital, current_value)
    dd = drawdown_series(eq)

    bench = None
    if benchmark_ticker and benchmark_ticker != "None" and len(eq) >= 2:
        bench = benchmark_equity(
            benchmark_ticker,
            eq.index.min(), eq.index.max(),
            float(eq["equity"].iloc[0]),
        )

    by_ticker = pnl_by_ticker(trade_log)
    by_wd = pnl_by_weekday(trade_log)
    by_hr = pnl_by_hour(trade_log)
    durations = trade_durations(trade_log)
    conc = concentration(pos)

    corr = None
    if pos:
        corr = correlation_matrix([p.ticker for p in pos], lookback_days=corr_lb)

    return metrics, eq, dd, bench, by_ticker, by_wd, by_hr, durations, conc, corr


_fingerprint_iso = (
    port.trade_log[-1].executed_at.isoformat() if port.trade_log else ""
)
metrics, eq, dd, bench, by_ticker, by_wd, by_hr, durations, conc, corr = _crunch(
    n_trades=len(port.trade_log),
    last_trade_iso=_fingerprint_iso,
    initial_capital=port.initial_capital,
    current_value=port.get_total_value(),
    benchmark_ticker=benchmark,
    corr_lb=corr_lookback,
)


# ─── KPI strip ─────────────────────────────────────────────────────────────
def _kpi_color(v: float, *, neutral_zero: bool = False) -> str:
    if abs(v) < 1e-9:
        return TEXT_DIM if neutral_zero else TEXT
    return C_BUY if v > 0 else C_SELL


def _kpi(label: str, value: str, color: str = TEXT, sub: str = "") -> str:
    sub_html = f'<span class="kpi-sub" style="color:{TEXT_DIM}">{sub}</span>' \
               if sub else ""
    return (
        f'<div class="kpi-item">'
        f'<span class="kpi-label">{label}</span> '
        f'<span class="kpi-value" style="color:{color}">{value}</span>'
        f'{sub_html}'
        f'</div>'
    )


kpi_html = '<div class="kpi-strip">'
kpi_html += _kpi(
    "Total Return",
    f"{metrics.total_return_pct:+.2f}%",
    _kpi_color(metrics.total_return_pct),
    sub=f"CAGR {metrics.cagr_pct:+.1f}%",
)
kpi_html += _kpi(
    "Sharpe",
    f"{metrics.sharpe:+.2f}",
    _kpi_color(metrics.sharpe, neutral_zero=True),
    sub=f"Sortino {metrics.sortino:+.2f}",
)
kpi_html += _kpi(
    "Max DD",
    f"{metrics.max_drawdown_pct:.2f}%",
    C_SELL if metrics.max_drawdown_pct < -1e-9 else TEXT_DIM,
    sub=f"${metrics.max_drawdown_dollars:+,.0f}",
)
kpi_html += _kpi(
    "Calmar",
    f"{metrics.calmar:+.2f}",
    _kpi_color(metrics.calmar, neutral_zero=True),
    sub=f"vol {metrics.volatility_pct:.1f}%",
)
kpi_html += _kpi(
    "Win Rate",
    f"{metrics.win_rate*100:.1f}%",
    C_BUY if metrics.win_rate >= 0.5 else C_SELL,
    sub=f"{metrics.n_wins}/{metrics.n_round_trips}",
)
kpi_html += _kpi(
    "Profit Factor",
    f"{metrics.profit_factor:.2f}" if metrics.profit_factor < 999 else "∞",
    C_BUY if metrics.profit_factor >= 1 else C_SELL,
    sub=f"exp ${metrics.expectancy:+,.0f}",
)
kpi_html += _kpi(
    "Avg Hold",
    f"{metrics.avg_holding_period_days:.2f}d",
    TEXT,
    sub=f"longest DD {metrics.longest_drawdown_days}d",
)
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)


# ─── Equity curve vs benchmark ────────────────────────────────────────────
st.markdown('<div class="bt-section-title">EQUITY CURVE</div>',
            unsafe_allow_html=True)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=eq.index, y=eq["equity"],
    mode="lines", name="Portfolio",
    line=dict(color=CYAN, width=2),
    fill="tozeroy",
    fillcolor="rgba(0,183,255,0.08)",
))
if bench is not None and not bench.empty:
    fig_eq.add_trace(go.Scatter(
        x=bench.index, y=bench["equity"],
        mode="lines",
        name=f"{benchmark} (rebased)",
        line=dict(color=TEXT_DIM, width=1.4, dash="dash"),
    ))
fig_eq.add_hline(
    y=port.initial_capital, line_color="rgba(180,180,180,0.45)",
    line_dash="dot",
    annotation_text="Starting capital", annotation_position="top right",
    annotation_font_color=TEXT_DIM, annotation_font_size=10,
)
fig_eq.update_layout(
    height=320, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
    xaxis=dict(gridcolor=GRID, color=TEXT_DIM),
    yaxis=dict(gridcolor=GRID, color=TEXT_DIM, tickprefix="$"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    hovermode="x unified",
)
st.plotly_chart(fig_eq, use_container_width=True)


# ─── Drawdown + per-trade scatter ─────────────────────────────────────────
c_dd, c_scatter = st.columns(2)

with c_dd:
    st.markdown('<div class="bt-section-title">DRAWDOWN (UNDERWATER)</div>',
                unsafe_allow_html=True)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd["drawdown_pct"],
        mode="lines",
        name="Drawdown %",
        line=dict(color=C_SELL, width=1.6),
        fill="tozeroy",
        fillcolor="rgba(255,77,79,0.18)",
    ))
    fig_dd.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        xaxis=dict(gridcolor=GRID, color=TEXT_DIM),
        yaxis=dict(gridcolor=GRID, color=TEXT_DIM, ticksuffix="%"),
        showlegend=False,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with c_scatter:
    st.markdown('<div class="bt-section-title">ROUND-TRIP P&L</div>',
                unsafe_allow_html=True)
    if not durations.empty:
        colors = [C_BUY if v >= 0 else C_SELL for v in durations["realized_pnl"]]
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=durations["duration_hours"],
            y=durations["realized_pnl"],
            mode="markers",
            marker=dict(
                color=colors, size=10,
                line=dict(color="rgba(0,0,0,0.4)", width=0.6),
            ),
            text=[
                f"{r.ticker} · {r.pnl_pct:+.2f}%<br>{r.entry_at:%Y-%m-%d %H:%M} → {r.exit_at:%Y-%m-%d %H:%M}"
                for r in durations.itertuples()
            ],
            hovertemplate="%{text}<br>Duration: %{x:.1f}h<br>P&L: $%{y:+,.0f}<extra></extra>",
        ))
        fig_sc.add_hline(y=0, line_color=TEXT_DIM, line_dash="dot")
        fig_sc.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
            xaxis=dict(gridcolor=GRID, color=TEXT_DIM, title="Hold time (hours)"),
            yaxis=dict(gridcolor=GRID, color=TEXT_DIM, tickprefix="$",
                       title="Realised P&L"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.caption("No completed round trips yet.")


# ─── Attribution panel ────────────────────────────────────────────────────
st.markdown('<div class="bt-section-title">P&L ATTRIBUTION</div>',
            unsafe_allow_html=True)
c_tk, c_wd, c_hr = st.columns([2, 1, 1])

with c_tk:
    st.caption("By ticker")
    if not by_ticker.empty:
        fig_tk = go.Figure()
        colors = [C_BUY if v >= 0 else C_SELL for v in by_ticker["total_pnl"]]
        fig_tk.add_trace(go.Bar(
            x=by_ticker["ticker"], y=by_ticker["total_pnl"],
            marker_color=colors,
            text=[
                f"${v:+,.0f}<br>{n}× · {wr*100:.0f}% wr"
                for v, n, wr in zip(
                    by_ticker["total_pnl"], by_ticker["n_trades"],
                    by_ticker["win_rate"],
                )
            ],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>P&L: $%{y:+,.0f}<extra></extra>",
        ))
        fig_tk.add_hline(y=0, line_color=TEXT_DIM, line_dash="dot")
        fig_tk.update_layout(
            height=260, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
            xaxis=dict(color=TEXT_DIM),
            yaxis=dict(gridcolor=GRID, color=TEXT_DIM, tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig_tk, use_container_width=True)
    else:
        st.caption("(empty)")

with c_wd:
    st.caption("By weekday (UTC)")
    if not by_wd.empty:
        wd_colors = [C_BUY if v >= 0 else C_SELL for v in by_wd["total_pnl"]]
        fig_wd = go.Figure()
        fig_wd.add_trace(go.Bar(
            x=by_wd["weekday"], y=by_wd["total_pnl"],
            marker_color=wd_colors,
            hovertemplate="<b>%{x}</b><br>P&L: $%{y:+,.0f}<extra></extra>",
        ))
        fig_wd.add_hline(y=0, line_color=TEXT_DIM, line_dash="dot")
        fig_wd.update_layout(
            height=260, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
            xaxis=dict(color=TEXT_DIM),
            yaxis=dict(gridcolor=GRID, color=TEXT_DIM, tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig_wd, use_container_width=True)
    else:
        st.caption("(empty)")

with c_hr:
    st.caption("By hour (UTC)")
    if not by_hr.empty:
        hr_colors = [C_BUY if v >= 0 else C_SELL for v in by_hr["total_pnl"]]
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Bar(
            x=by_hr["hour"], y=by_hr["total_pnl"],
            marker_color=hr_colors,
            hovertemplate="<b>%{x}h</b><br>P&L: $%{y:+,.0f}<extra></extra>",
        ))
        fig_hr.add_hline(y=0, line_color=TEXT_DIM, line_dash="dot")
        fig_hr.update_layout(
            height=260, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
            xaxis=dict(color=TEXT_DIM, dtick=2, title="UTC hour"),
            yaxis=dict(gridcolor=GRID, color=TEXT_DIM, tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig_hr, use_container_width=True)
    else:
        st.caption("(empty)")


# ─── Holdings risk ────────────────────────────────────────────────────────
st.markdown('<div class="bt-section-title">HOLDINGS RISK</div>',
            unsafe_allow_html=True)
c_conc, c_corr = st.columns([1, 1])

with c_conc:
    st.caption("Concentration (open positions)")
    if not conc.empty:
        hhi = float(conc["hhi_contrib"].sum())
        # Map HHI to a label
        if hhi >= 0.50:
            badge = f'<span class="badge badge-red">HIGH · HHI {hhi:.2f}</span>'
        elif hhi >= 0.25:
            badge = f'<span class="badge badge-amber">MODERATE · HHI {hhi:.2f}</span>'
        else:
            badge = f'<span class="badge badge-green">DIVERSIFIED · HHI {hhi:.2f}</span>'
        st.markdown(badge, unsafe_allow_html=True)
        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=conc["ticker"], values=conc["market_value"],
            hole=0.55,
            textinfo="label+percent",
            marker=dict(line=dict(color=BG_PANEL, width=2)),
        ))
        fig_pie.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
            showlegend=False,
            font=dict(color=TEXT),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.caption("No open positions.")

with c_corr:
    st.caption(f"Correlation matrix · {corr_lookback}-day daily returns")
    if corr is not None and not corr.empty:
        fig_h = go.Figure()
        fig_h.add_trace(go.Heatmap(
            z=corr.values,
            x=list(corr.columns), y=list(corr.index),
            colorscale="RdBu", zmin=-1, zmax=1, reversescale=True,
            text=[[f"{v:+.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            hovertemplate="%{y} ↔ %{x}: %{z:+.2f}<extra></extra>",
        ))
        fig_h.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
            xaxis=dict(color=TEXT_DIM),
            yaxis=dict(color=TEXT_DIM, autorange="reversed"),
        )
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.caption("Need ≥ 2 open positions with overlapping price history.")


# ─── Detailed metrics table ──────────────────────────────────────────────
with st.expander("📐 Full metrics table"):
    md = metrics.to_dict()
    df_md = pd.DataFrame({
        "Metric": list(md.keys()),
        "Value": list(md.values()),
    })
    st.dataframe(df_md, use_container_width=True, hide_index=True)


# ─── Round-trip log ───────────────────────────────────────────────────────
with st.expander(f"📒 Round-trip log ({len(durations)} trips)"):
    if not durations.empty:
        view = durations.copy()
        view["entry_at"] = view["entry_at"].dt.strftime("%Y-%m-%d %H:%M")
        view["exit_at"] = view["exit_at"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.caption("No round trips yet.")
