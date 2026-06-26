"""Portfolio page — equity, trade history, stats."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── sys.path bootstrap (so `from dashboard...` works in direct `streamlit run`)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ────────────────────────────────────────────────────────────────────────────

from dashboard._shared import (
    BG, C_BUY, C_SELL, GRID,
    apply_theme, ensure_logs_in_session, ensure_portfolio_in_session,
    ensure_profile_in_session, pump_toasts,
)

st.set_page_config(page_title="BotTrade - Portfolio", page_icon=":material/account_balance:", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()
ensure_profile_in_session()
ensure_portfolio_in_session()
ensure_logs_in_session()
pump_toasts()

st.markdown('<div class="page-title">PORTFOLIO</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Equity, trade history, and performance stats</div>', unsafe_allow_html=True)

port = st.session_state["portfolio"]
summ = port.get_summary()

# ── Headline KPIs ───────────────────────────────────────────────────────────
closed = [t for t in port.trade_log if "SELL" in t.action]
wins   = [t for t in closed if t.realized_pnl > 0]
losses = [t for t in closed if t.realized_pnl < 0]
total_pnl = port.get_realized_pnl() + summ.get("unrealized_pnl", 0.0)

avg_win  = (sum(t.realized_pnl for t in wins)   / len(wins))   if wins   else 0.0
avg_loss = (sum(t.realized_pnl for t in losses) / len(losses)) if losses else 0.0
win_rate = (len(wins) / len(closed) * 100) if closed else 0.0
profit_factor = (sum(t.realized_pnl for t in wins) /
                 abs(sum(t.realized_pnl for t in losses))) if losses else float("inf") if wins else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Account Value", f"${summ['total_value']:,.2f}", f"{port.get_total_return_pct():+.2f}%")
k2.metric("Total P&L", f"${total_pnl:+,.2f}",
          f"realized ${port.get_realized_pnl():+,.2f}")
k3.metric("Cash", f"${summ['cash']:,.2f}")
k4.metric("Win Rate", f"{win_rate:.0f}%" if closed else "—",
          f"{len(wins)}W / {len(losses)}L" if closed else None)
k5.metric("Avg Win / Loss",
          f"${avg_win:+,.0f} / ${avg_loss:+,.0f}" if closed else "—")
k6.metric("Profit Factor",
          f"{profit_factor:.2f}" if closed and losses else ("∞" if wins else "—"))

st.markdown("---")

# ── Safety controls ─────────────────────────────────────────────────────────
_engine = st.session_state.get("_live_engine")
if _engine is not None:
    try:
        _sstat = _engine.get_safety_status()
    except Exception:
        _sstat = None
else:
    _sstat = None

st.markdown('<div class="bt-section-title">🛡 SAFETY CONTROLS</div>',
            unsafe_allow_html=True)
sc1, sc2, sc3, sc4 = st.columns([3, 1, 1, 1])

with sc1:
    if _sstat is None:
        st.markdown(
            '<div class="bt-panel">Live engine not started — safety controls '
            'become active once the trading loop is running.</div>',
            unsafe_allow_html=True,
        )
    else:
        sev = _sstat.severity
        badge_cls = {
            "OK":    "badge-green",
            "WARN":  "badge-amber",
            "BLOCK": "badge-red",
        }.get(sev, "badge-gray")
        sub_bits = []
        if _sstat.consecutive_losses:
            sub_bits.append(f"streak: {_sstat.consecutive_losses}L")
        if _sstat.cooldown_until:
            sub_bits.append(f"cooldown until {_sstat.cooldown_until:%H:%M UTC}")
        if _sstat.round_trips_today:
            sub_bits.append(f"trips today: {_sstat.round_trips_today}")
        sub_line = " · ".join(sub_bits) if sub_bits else ""
        st.markdown(
            f'<div class="bt-panel">'
            f'<span class="badge {badge_cls}">{sev}</span>'
            f'&nbsp;&nbsp;<span style="font-size:0.85rem">{_sstat.reason}</span>'
            + (f'<br><span style="font-size:0.7rem;opacity:.7">{sub_line}</span>'
               if sub_line else "")
            + '</div>',
            unsafe_allow_html=True,
        )

with sc2:
    panic_disabled = _engine is None or not list(port.positions.values())
    if st.button("🛑 PANIC STOP",
                 use_container_width=True,
                 disabled=panic_disabled,
                 help="Force-close every open position at last known price "
                      "and block new BUYs."):
        try:
            res = _engine.panic_stop("UI panic button")
            st.success(
                f"Closed {len(res['closed'])} position(s). "
                + (f"Errors: {len(res['errors'])}" if res['errors'] else "")
            )
        except Exception as exc:
            st.error(f"Panic stop failed: {exc}")
        st.rerun()

with sc3:
    if st.button("Clear Block",
                 use_container_width=True,
                 disabled=_engine is None,
                 help="Lift a manual safety block (e.g. after Panic Stop)."):
        try:
            _engine.clear_panic()
            st.success("Block cleared.")
        except Exception as exc:
            st.error(f"Clear failed: {exc}")
        st.rerun()

with sc4:
    if st.button("Override 30m",
                 use_container_width=True,
                 disabled=_engine is None,
                 help="Temporarily allow BUYs even if a circuit breaker is "
                      "active (30 minutes)."):
        try:
            _engine.safety_override(30)
            st.success("Override active for 30 min.")
        except Exception as exc:
            st.error(f"Override failed: {exc}")
        st.rerun()

st.markdown("---")

# ── Equity curve (cumulative realized P&L) ──────────────────────────────────
st.markdown("##### Equity Curve")
if port.trade_log:
    rows = []
    cum = 0.0
    for t in port.trade_log:
        cum += t.realized_pnl
        rows.append({"ts": t.executed_at, "cum": cum, "action": t.action, "ticker": t.ticker})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["cum"],
        mode="lines+markers", name="Cumulative P&L",
        line=dict(color="#00c9ff", width=2),
        marker=dict(size=6, color=[C_BUY if v >= 0 else C_SELL for v in df["cum"]]),
        hovertemplate="%{x|%H:%M:%S} · $%{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#2a4060", dash="dot", width=1))
    fig.update_layout(
        template="plotly_dark", height=320,
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=10, r=30, t=10, b=10),
        font=dict(family="Courier New, monospace", size=11),
        xaxis=dict(showgrid=True, gridcolor=GRID),
        yaxis=dict(title="Cumulative P&L ($)", side="right",
                   showgrid=True, gridcolor=GRID),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="equity_curve")
else:
    st.caption("No realised trades yet. Equity curve will appear after the first SELL.")

st.markdown("---")

# ── Open positions ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.markdown("##### Open Positions")
    if summ["open_positions"]:
        # Cached sparkline fetch — one yfinance call per ticker, 5 min TTL.
        @st.cache_data(ttl=300, show_spinner=False)
        def _spark(ticker: str):
            try:
                import yfinance as yf
                cache = _Path("data/yfinance_cache")
                cache.mkdir(parents=True, exist_ok=True)
                yf.set_tz_cache_location(str(cache))
                hist = yf.Ticker(ticker).history(period="5d", interval="30m")
                if hist is None or hist.empty:
                    return None
                return list(hist["Close"].astype(float).tail(80))
            except Exception:
                return None

        for p in summ["open_positions"]:
            pnl = p["unrealized_pnl"]
            pnl_pct = p["unrealized_pnl_pct"]
            color = C_BUY if pnl >= 0 else C_SELL
            spark_col, info_col = st.columns([1.2, 2])
            with spark_col:
                series = _spark(p["ticker"])
                if series and len(series) >= 5:
                    s_min = min(series); s_max = max(series)
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(
                        y=series, x=list(range(len(series))),
                        mode="lines",
                        line=dict(color=color, width=1.6),
                        fill="tozeroy",
                        fillcolor=("rgba(0,193,118,0.13)" if pnl >= 0
                                   else "rgba(255,77,79,0.13)"),
                        hoverinfo="skip",
                    ))
                    # Mark entry price as a faint horizontal line
                    fig_s.add_hline(
                        y=p["avg_entry_price"],
                        line=dict(color="rgba(255,255,255,0.35)",
                                  dash="dot", width=1),
                    )
                    fig_s.update_layout(
                        height=70, margin=dict(l=0, r=0, t=2, b=0),
                        paper_bgcolor=BG, plot_bgcolor=BG,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False,
                                   range=[s_min * 0.998, s_max * 1.002]),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_s, use_container_width=True,
                                    key=f"sp_{p['ticker']}",
                                    config={"displayModeBar": False})
                else:
                    st.caption("(no chart)")
            with info_col:
                st.markdown(
                    f'<div style="padding:4px 0;">'
                    f'<div style="font-family:monospace;font-weight:700;'
                    f'font-size:0.95rem;color:#fff;">{p["ticker"]}</div>'
                    f'<div style="font-size:0.7rem;color:#8b98a8;'
                    f'font-family:monospace;">{p["quantity"]:.4f} @ '
                    f'${p["avg_entry_price"]:,.2f} → '
                    f'${p["current_price"]:,.2f}</div>'
                    f'<div style="font-family:monospace;font-weight:700;'
                    f'color:{color};">${pnl:+,.2f} ({pnl_pct:+.2f}%)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div style="border-bottom:1px solid #182840;margin:0 0 6px 0;"></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No open positions.")

with col_right:
    st.markdown("##### Recent Trades")
    if port.trade_log:
        trade_rows = []
        cum = 0.0
        for t in port.trade_log:
            cum += t.realized_pnl
            trade_rows.append({
                "Time":   t.executed_at.strftime("%m-%d %H:%M"),
                "Action": t.action,
                "Ticker": t.ticker,
                "Price":  f"${t.price:,.2f}",
                "Qty":    f"{t.quantity:.4f}",
                "P&L":    f"${t.realized_pnl:+,.2f}",
                "Cumul.": f"${cum:+,.2f}",
            })
        df = pd.DataFrame(list(reversed(trade_rows)))
        st.dataframe(df, hide_index=True, use_container_width=True, height=420)
    else:
        st.caption("No trades yet.")

st.markdown("---")

# ── Trade Reasoning Timeline ───────────────────────────────────────────────
st.markdown("##### 🧭 Trade Reasoning Timeline")
st.caption(
    "Every executed trade with the AI's reasoning and round-trip P&L. "
    "Click a card to expand."
)
if port.trade_log:
    # Pair BUYs with their next SELL on the same ticker for round-trip view
    log = list(port.trade_log)
    pairs: list[tuple] = []
    open_buys: dict = {}
    for r in log:
        if r.action == "BUY":
            open_buys[r.ticker] = r
        elif r.action in ("SELL", "FORCE_CLOSE") and r.ticker in open_buys:
            entry = open_buys.pop(r.ticker)
            pairs.append((entry, r))
    # Standalone (still open) BUYs go last as half-cards
    standalone = [(b, None) for b in open_buys.values()]
    timeline = list(reversed(pairs)) + list(reversed(standalone))

    if not timeline:
        st.caption("No trade pairs to show yet.")
    else:
        # Render newest first, max 25 to keep page snappy
        for entry, exit_r in timeline[:25]:
            ticker = entry.ticker
            entry_t = entry.executed_at.strftime("%Y-%m-%d %H:%M")
            entry_px = entry.price
            entry_reason = (entry.reasoning or "(no reasoning recorded)").strip()
            if exit_r is not None:
                pnl = float(exit_r.realized_pnl)
                exit_t = exit_r.executed_at.strftime("%Y-%m-%d %H:%M")
                dur_h = (exit_r.executed_at - entry.executed_at).total_seconds() / 3600.0
                pnl_pct = ((exit_r.price - entry_px) / entry_px * 100.0) \
                          if entry_px else 0.0
                pnl_color = C_BUY if pnl >= 0 else C_SELL
                emoji = "🟢" if pnl >= 0 else "🔴"
                exit_reason = (exit_r.reasoning or "").strip()
                title = (f"{emoji} {ticker} · ${pnl:+,.2f} ({pnl_pct:+.2f}%) "
                         f"· {entry_t} → {exit_t} ({dur_h:.1f}h)")
                with st.expander(title):
                    cA, cB = st.columns(2)
                    with cA:
                        st.markdown(
                            f'<div style="font-size:0.72rem;color:#8b98a8">ENTRY @ ${entry_px:,.2f}</div>'
                            f'<div style="font-size:0.85rem;color:#d8e1ec;'
                            f'font-family:monospace;white-space:pre-wrap;'
                            f'padding:6px 0;">{entry_reason}</div>',
                            unsafe_allow_html=True,
                        )
                    with cB:
                        st.markdown(
                            f'<div style="font-size:0.72rem;color:#8b98a8">EXIT @ ${exit_r.price:,.2f}'
                            f' · {exit_r.action}</div>'
                            f'<div style="font-size:0.85rem;color:#d8e1ec;'
                            f'font-family:monospace;white-space:pre-wrap;'
                            f'padding:6px 0;">{exit_reason or "(no reasoning recorded)"}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                # Open round trip — only entry side
                title = f"🟦 {ticker} · OPEN · entered {entry_t} @ ${entry_px:,.2f}"
                with st.expander(title):
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#8b98a8">ENTRY REASONING</div>'
                        f'<div style="font-size:0.85rem;color:#d8e1ec;'
                        f'font-family:monospace;white-space:pre-wrap;'
                        f'padding:6px 0;">{entry_reason}</div>',
                        unsafe_allow_html=True,
                    )
        if len(timeline) > 25:
            st.caption(f"Showing 25 most recent of {len(timeline)} round trips.")
else:
    st.caption("No trades yet — the timeline will populate after the first round trip.")
