"""
BotTrade — Live Trading Terminal (Command Center page)

Routed here by dashboard/app.py's st.navigation() call — run the app with
`streamlit run dashboard/app.py`, not this file directly.
"""
from __future__ import annotations

# ── sys.path bootstrap ──────────────────────────────────────────────────────
# Make `from dashboard...` / `from market_data...` / `from decision_engine...`
# work no matter HOW the app is launched (direct `streamlit run`, `python -m
# streamlit run`, PyCharm play button, Railway/Heroku Procfile, Docker, etc).
# We always add the parent of the `dashboard/` folder (= project root).
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# ────────────────────────────────────────────────────────────────────────────

import html as _html
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard._shared import (
    BG_DEEP, BORDER, C_BUY, C_HOLD, C_SELL, CUSTOM_LABEL, CYAN,
    DEFAULT_TICKERS, GRID, RISK_CHOICES, TEXT, TEXT_DIM, TEXT_HI,
    account_id as _account_id,
    secure_page, ensure_event_buffer, ensure_portfolio_in_session,
    ensure_profile_in_session, engine_capacity_message, get_live_engine,
    get_tenant, pump_events, save_portfolio, save_profile,
)
from saas.plans import Funding as _Funding
from saas.pricing import format_usd as _fmt_usd
from saas.pricing import format_usd_md as _fmt_usd_md
from utils.market_logic import get_market_status

# Install crash reporter once per process — captures uncaught exceptions
# from the main thread AND the live-engine background daemon thread.
from utils import crash_reporter as _crash
if not _crash.is_installed():
    def _crash_to_telegram(one_liner: str, _full: str) -> None:
        # Best-effort fan-out to whatever notifier the user configured.
        try:
            from notifications import NotificationConfig, NotificationDispatcher
            disp = NotificationDispatcher(NotificationConfig.load())
            disp.notify("ERROR", f"💥 CRASH — {one_liner}", level="ERROR")
        except Exception:
            pass
    _crash.install(post_callback=_crash_to_telegram)

# ─────────────────────────────────────────────────────────────────────────────
# Page config + theme
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BotTrade — Live",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",
)
secure_page()
ensure_profile_in_session()
ensure_portfolio_in_session()
ensure_event_buffer()

# ── Arriving from the landing page with a plan in mind ───────────────────────
# The pricing buttons link here as ?plan=PRO rather than straight to Stripe:
# a Checkout Session only grants a plan if it carries bottrade_account_id, and
# that is only knowable once someone has signed in. So the intent rides in on
# the URL and gets handed to Settings, which is where checkout actually runs.
#
# Read after secure_page() on purpose — in oidc mode the sign-in redirect
# happens first, and the parameter survives it.
_wanted_plan = (st.query_params.get("plan") or "").strip().upper()
if _wanted_plan:
    st.query_params.clear()   # don't re-trigger on every rerun
    if _wanted_plan in ("PRO", "DESK"):
        st.session_state["_bt_pending_plan"] = _wanted_plan
        st.switch_page("pages/2_Settings.py")

# ─────────────────────────────────────────────────────────────────────────────
# Engine + auto-refresh
#
# Note there is no API-key pre-flight here. A missing ANTHROPIC_API_KEY used
# to stop the app dead; it no longer does. COMMITTEE mode (38 indicators
# voting on every bar) makes no API calls at all, so a visitor with no key
# still gets a complete, working product. Only the LLM modes are gated, and
# the entitlement layer does that per user further down.
# ─────────────────────────────────────────────────────────────────────────────
engine = get_live_engine()
if engine is None:
    # The process is at its live-bot ceiling. Refusing a new bot is the
    # correct outcome — the alternative is evicting someone else's running
    # engine and leaving their stop-loss unwatched.
    st.error(engine_capacity_message() or
             "No live-bot slots are free on this deployment right now.")
    st.caption(
        "Backtesting, the Committee Lab and every analysis page still work — "
        "only the live loop is capped. Raise `BOTTRADE_MAX_LIVE_ENGINES` if "
        "this host has the headroom."
    )
    st.stop()

pump_events()

state = engine.snapshot()

# Refresh rate is derived from the engine's own cycle rather than fixed.
#
# This used to be a flat 2s, chosen for a "live heartbeat" feel and reasoned
# about purely in terms of API cost (the AI cycle is throttled separately, so
# a fast UI tick costs no Claude calls). What that missed is the CPU: every
# tick re-runs this entire script — rebuilding the four-pane Plotly figure and
# re-reading plan, entitlement and usage from the ledger — and plans cap the
# cycle at 300s (Free), 60s (Pro) and 30s (Desk). At 2s that is 15-150 full
# re-renders per actual change, forever, on a shared vCPU the live trading
# loop is already competing for.
#
# Roughly ten refreshes per cycle keeps the countdown to the next cycle
# visibly moving while cutting the idle load by most of that factor. The
# floor keeps it feeling live on the fastest plan; the ceiling keeps a 5
# minute Free-plan cycle from looking frozen.
if engine.is_running():
    _refresh_ms = max(5_000, min(int(state["interval_sec"]) * 100, 15_000))
    st_autorefresh(interval=_refresh_ms, limit=None, key="live_refresh")

port  = st.session_state["portfolio"]

_tenant_state = state.get("tenant") or {}
_plan_badge = _tenant_state.get("plan_name", "Local")
_fund_badge = {
    "BYOK":     "Your API key",
    "PLATFORM": "Trial credit",
    "NONE":     "No API key",
}.get(_tenant_state.get("funding", ""), "Operator key")
_fund_class = "badge-green" if _tenant_state.get("funding") == "BYOK" \
    else ("badge-amber" if _tenant_state.get("funding") == "PLATFORM"
          else "badge-gray")

st.markdown(
    f'<div class="bt-brand">'
    f'<div style="display:flex;align-items:center;gap:.8rem;">'
    f'<div class="bt-brand-mark">BT</div>'
    f'<div><div class="bt-brand-title">BotTrade / Chart workspace</div>'
    f'<div class="bt-brand-sub">Analyze markets. Monitor strategies. Manage exposure.</div></div>'
    f'</div>'
    f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;justify-content:flex-end;">'
    f'<span class="badge badge-blue">{_html.escape(_plan_badge)} plan</span>'
    f'<span class="badge {_fund_class}">{_html.escape(_fund_badge)}</span>'
    f'<span class="badge badge-gray">Paper execution</span>'
    f'<span class="badge badge-amber">Risk controlled</span>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ██  SIDEBAR — strategy guide
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    with st.expander("📖 Strategy Guide — which one to pick?",
                     expanded=False):
        st.markdown("""
**🧠 AI Brain** — Claude reads the chart, your knowledge base (RAG),
news and fundamentals, and decides alone each cycle.
*Smartest single opinion, but can act on one bad read.*
1 API call/cycle.

**🗳 Committee ×38** — 38 technical indicators vote; bull majority
buys, bear majority exits to cash. No AI, no panic sells, fully
deterministic and backtestable in the 🗳 Committee Lab.
*Best discipline, zero API cost — but blind to news.*

**🤝 Hybrid** — the committee votes AND Claude reviews the tally.
BUY needs both to agree; the AI can never panic-sell against a
bullish committee. *Discipline + context check.* 1 call/cycle.

**🪑 Boardroom ×8** — a full hedge-fund desk: chart, fundamentals,
news, quant, macro, risk officer, volume-flow and a contrarian each
study their own briefing in parallel, then a chairman makes the
binding call. *Richest reasoning, most expensive.* ~9 calls/cycle —
use 120s+ intervals.

---
**Quick picks:**
- Trust pure rules / run cheap → **Committee**
- Balanced default → **Hybrid**
- Maximum context before each trade → **Boardroom**
- Test any setup risk-free first → **🗳 Committee Lab** page
        """)

    # Plan + spend at a glance. Anyone running the bot on their own API key
    # should be able to see what it is costing them without hunting for it.
    _side_tenant = get_tenant()
    _side_ent = _side_tenant.entitlement
    _side_usage = _side_tenant.usage()
    st.markdown("---")
    st.markdown(f"**Plan · {_side_ent.plan.name}**")
    _sc1, _sc2 = st.columns(2)
    _sc1.metric("This month", _fmt_usd(_side_usage["cost_usd"]))
    _sc2.metric("Calls saved", f"{_side_usage['calls_saved']:,}",
                help="Decisions reused from the shared cache instead of "
                     "hitting the API.")
    if _side_ent.funding is _Funding.PLATFORM:
        st.caption(f"Trial credit left: "
                   f"{_fmt_usd_md(_side_ent.platform_budget_remaining_usd)}")
    elif _side_ent.funding is _Funding.BYOK:
        st.caption("Running on your own API key.")
    st.page_link("pages/10_Usage_and_Billing.py", label="Usage & billing",
                 icon="🧾")
    st.page_link("pages/2_Settings.py", label="Plan & API key", icon="🔑")

# ─────────────────────────────────────────────────────────────────────────────
# ██  PLAN CONTEXT — decides which strategy modes and intervals are offered
# ─────────────────────────────────────────────────────────────────────────────
_tenant = get_tenant()
_plan_ent = _tenant.entitlement

if not _plan_ent.llm_available:
    with st.sidebar:
        st.caption("Committee mode uses local indicators. Add an API key in Settings to enable AI analysis.")

# ─────────────────────────────────────────────────────────────────────────────
# ██  TOP CONTROL BAR  — single row, all columns share identical structure
# ─────────────────────────────────────────────────────────────────────────────
control_panel, c_start, c_reset = st.columns([6, 1, 1], gap="small")
with control_panel:
    with st.expander("Symbol, strategy & risk controls", expanded=False):
        c_ticker, c_strategy, c_interval = st.columns([2, 2, 1], gap="small")
        c_cap, c_size, c_risk = st.columns(3, gap="small")

        with c_ticker:
            _options = list(dict.fromkeys(
                list(st.session_state.get("watchlist") or []) + DEFAULT_TICKERS
            )) + [CUSTOM_LABEL]
            prev_sel = st.session_state.get("ticker_sel", state["ticker"])
            idx = _options.index(prev_sel) if prev_sel in _options else 0
            sel = st.selectbox("TICKER", _options, index=idx, key="ticker_sel_widget")
            if sel == CUSTOM_LABEL:
                custom = st.text_input("", value=st.session_state.get("ticker_custom", ""),
                                       placeholder="e.g. SOL-USD",
                                       label_visibility="collapsed",
                                       key="ticker_custom_input").upper().strip()
                ticker = custom or state["ticker"] or "BTC-USD"
                st.session_state["ticker_custom"] = ticker
            else:
                ticker = sel
            st.session_state["ticker_sel"] = ticker

        with c_strategy:
            _STRAT_LABELS = {
                "AI": "🧠 AI Brain",
                "COMMITTEE": "🗳 Committee ×38",
                "HYBRID": "🤝 Hybrid",
                "BOARDROOM": "🪑 Boardroom ×8",
            }
            _strat_opts = ["AI", "COMMITTEE", "HYBRID", "BOARDROOM"]
            # Modes the plan does not cover stay visible but locked, so the upgrade
            # path is obvious instead of the option silently vanishing.
            _ent = _tenant.entitlement
            _locked = {m for m in _strat_opts if not _ent.allows(m)}
            prev_strat = st.session_state.get("strategy_mode",
                                              state.get("strategy_mode", "AI"))
            if prev_strat in _locked:
                prev_strat = "COMMITTEE"
            strat_idx = _strat_opts.index(prev_strat) if prev_strat in _strat_opts else 0
            strategy_mode = st.selectbox(
                "STRATEGY", _strat_opts, index=strat_idx,
                format_func=lambda s: (f"🔒 {_STRAT_LABELS.get(s, s)}" if s in _locked
                                       else _STRAT_LABELS.get(s, s)),
                key="strategy_widget",
                help="🔒 needs an API key or a plan upgrade — see Settings. "
                     "Full comparison in the sidebar → 📖 Strategy Guide.",
            )
            if strategy_mode in _locked:
                st.caption(f"🔒 {_ent.lock_reason or 'Not available on your plan.'}")
                strategy_mode = "COMMITTEE"
            st.session_state["strategy_mode"] = strategy_mode

        with c_cap:
            prev_cap = int(st.session_state["starting_capital"])
            cap = st.number_input("CAPITAL ($)", min_value=1000, max_value=10_000_000,
                                  value=prev_cap, step=1000,
                                  key="capital_widget")
            if cap != prev_cap:
                st.session_state["starting_capital"] = cap
                save_profile()

        with c_size:
            prev_ts = int(st.session_state["trade_size_pct"])
            ts = st.slider("TRADE SIZE %", 5, 100, prev_ts, step=5,
                           key="trade_size_widget")
            if ts != prev_ts:
                st.session_state["trade_size_pct"] = ts
                save_profile()

        with c_risk:
            _RISK_LABELS = {
                "Conservative": "Conservative",
                "Balanced":     "Balanced",
                "Aggressive":   "Aggressive",
                "Micro-Scalp":  "Micro-Scalp",
            }
            prev_risk = st.session_state.get("risk_profile", "Balanced")
            risk_idx  = RISK_CHOICES.index(prev_risk) if prev_risk in RISK_CHOICES else 1
            risk = st.selectbox("RISK", RISK_CHOICES, index=risk_idx,
                                format_func=lambda r: _RISK_LABELS.get(r, r),
                                key="risk_widget",
                                help="Controls position exposure, confidence threshold, stop loss and take profit.")
            if risk != prev_risk:
                st.session_state["risk_profile"] = risk
                save_profile()

        with c_interval:
            # Offer only intervals the plan actually permits — a slider that lets you
            # pick 15s and then silently runs at 300s is worse than not offering it.
            # The ladder runs past 300s so every plan, including the slowest, still
            # has a real choice rather than a single pinned value.
            _all_intervals = [15, 30, 60, 120, 300, 600, 900]
            _ivl_opts = [v for v in _all_intervals if v >= _plan_ent.min_interval_sec] \
                or [_plan_ent.min_interval_sec]
            _ivl_prev = int(state["interval_sec"])
            if _ivl_prev not in _ivl_opts:
                _ivl_prev = _ivl_opts[0]
            _ivl_help = (f"{_plan_ent.plan.name} plan cycles no faster than "
                         f"{_plan_ent.min_interval_sec}s."
                         if len(_ivl_opts) < len(_all_intervals) else None)
            if len(_ivl_opts) == 1:
                # select_slider needs a range; a single permitted value gets a
                # read-only control instead of a one-position slider.
                interval = _ivl_opts[0]
                st.selectbox("CYCLE", _ivl_opts, index=0, disabled=True,
                             format_func=lambda v: f"{v}s",
                             key="interval_widget", help=_ivl_help)
            else:
                interval = st.select_slider(
                    "CYCLE", options=_ivl_opts, value=_ivl_prev,
                    key="interval_widget", format_func=lambda v: f"{v}s",
                    help=_ivl_help,
                )
            st.session_state["interval_sec"] = int(interval)

        # Explain the active policy next to the controls that change it.
        from config.user_profile import RISK_ENVELOPES
        _active_envelope = RISK_ENVELOPES[risk]
        _strategy_scope = {
            "COMMITTEE": "Technical indicators only",
            "AI": "Technical, fundamental and knowledge analysis",
            "HYBRID": "Technical consensus with AI context review",
            "BOARDROOM": "Multiple analysts with a final chair review",
        }
        st.caption(
            f"{_strategy_scope[strategy_mode]} · "
            f"Symbol exposure cap {min(ts, _active_envelope.size_max_pct):g}% · "
            f"Stop loss {_active_envelope.stop_loss_pct:g}% · "
            f"Take profit {_active_envelope.take_profit_pct:g}% · "
            f"Minimum signal confidence {_active_envelope.conf_threshold:.0%}"
        )

with c_start:
    if engine.is_running():
        if st.button("STOP", width="stretch",
                     help="Halt the live loop"):
            engine.stop()
            st.rerun()
    else:
        if st.button("START", width="stretch", type="primary",
                     help="Begin live trading loop"):
            engine.set_config(
                ticker=ticker,
                strategy_mode=st.session_state.get("strategy_mode", "AI"),
                interval_sec=int(st.session_state.get("interval_sec", 30)),
                risk_profile=st.session_state.get("risk_profile", "Balanced"),
                trade_size_pct=float(st.session_state.get("trade_size_pct", 20)),
            )
            engine.start()
            st.rerun()

with c_reset:
    if st.button("RESET", width="stretch", help="Reset portfolio"):
        from portfolio.virtual_account import LivePortfolio
        from trading.registry import get_registry
        # Stop and drop the engine first: it holds a reference to the old
        # portfolio and would otherwise keep checkpointing it back over the
        # fresh one. get_live_engine() rebuilds on the next run.
        get_registry().stop(_account_id())
        fresh = LivePortfolio(
            initial_capital=float(st.session_state["starting_capital"]))
        st.session_state["portfolio"] = fresh
        save_portfolio(fresh)
        st.rerun()

# Keep engine config in sync with widgets + settings even while running
engine.set_config(
    ticker=ticker, interval_sec=interval,
    strategy_mode=strategy_mode,
    risk_profile=risk, trade_size_pct=ts,
    daily_target_pct=float(st.session_state.get("daily_target_pct", 0.0)),
    daily_loss_limit_pct=float(st.session_state.get("daily_loss_limit_pct", 5.0)),
)
state = engine.snapshot()

# ─────────────────────────────────────────────────────────────────────────────
# ██  LIVE HEARTBEAT BAR
# ─────────────────────────────────────────────────────────────────────────────
_stage = state["stage"]
_dot_cls = {
    "IDLE": "off", "STOPPED": "off", "SLEEP": "on",
    "FETCH": "on", "INDICATORS": "on", "RAG": "ai",
    "AI": "ai", "DECISION": "on", "RISK": "exec",
    "EXECUTE": "exec", "ERROR": "err",
}.get(_stage, "off")

_live_badge = ('<span class="badge badge-green">● RUNNING</span>'
               if state["running"] else
               '<span class="badge badge-gray">○ STOPPED</span>')

_strat_badge = {
    "COMMITTEE": '<span class="badge badge-blue">🗳 COMMITTEE ×38</span>',
    "HYBRID":    '<span class="badge badge-green">🤝 HYBRID 38+AI</span>',
    "BOARDROOM": '<span class="badge badge-green">🪑 BOARDROOM</span>',
}.get(state.get("strategy_mode"),
      '<span class="badge badge-amber">🧠 AI BRAIN</span>')

_mkt_open, _mkt_label = get_market_status(ticker)
_mkt_cls = "badge-green" if _mkt_open else "badge-red"
_mkt_badge = (f'<span class="badge {_mkt_cls}">'
              f'{_mkt_label}</span>')

_halt_html = ""
if state["halt_reason"]:
    _halt_html = (f'<span class="badge badge-red">'
                  f'ALERT: {_html.escape(state["halt_reason"])}</span>')

_target_html = ""
_target = float(state.get("daily_target_pct") or 0.0)
if _target > 0:
    _daily = port.get_daily_pnl_pct()
    _tcls = "badge-green" if _daily >= 0 else "badge-amber"
    _target_html = (f'<span class="badge {_tcls}">TARGET '
                    f'{_daily:+.2f}% / +{_target:.1f}%</span>')

_next_html = ""
if state["running"] and _stage == "SLEEP" and state.get("next_cycle_at"):
    secs = int((state["next_cycle_at"] - datetime.now()).total_seconds())
    secs = max(0, secs)
    _next_html = (f'<span style="color:{TEXT_DIM};font-size:.72rem;'
                  f'font-family:monospace;">NEXT IN <b style="color:{CYAN};">'
                  f'{secs:02d}s</b></span>')

_cycle_html = (f'<span style="color:{TEXT_DIM};font-size:.72rem;'
               f'font-family:monospace;">CYCLE '
               f'<b style="color:{TEXT};">#{state["cycle_count"]}</b></span>')

_now_html = (f'<span style="color:{TEXT_DIM};font-size:.72rem;'
             f'font-family:monospace;">'
             f'{datetime.now().strftime("%H:%M:%S")}</span>')

st.markdown(
    f'<div class="bt-panel live-bar">'
    f'<span class="pulse-dot {_dot_cls}"></span>'
    f'<span class="stage-pill stage-{_stage}">{_stage}</span>'
    f'<span style="color:{TEXT};font-weight:700;">{_html.escape(state["activity"])}</span>'
    f'<span style="margin-left:auto;display:flex;gap:0.9rem;align-items:center;">'
    f'{_halt_html}{_target_html}{_strat_badge}{_live_badge}{_mkt_badge}{_cycle_html}{_next_html}{_now_html}'
    f'</span></div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ██  KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
summ       = port.get_summary()
closed     = [t for t in port.trade_log if "SELL" in t.action]
wins       = [t for t in closed if t.realized_pnl > 0]
losses     = [t for t in closed if t.realized_pnl < 0]
total_pnl  = port.get_realized_pnl() + summ.get("unrealized_pnl", 0.0)
win_rate   = (len(wins) / len(closed) * 100) if closed else 0.0

# Compact inline KPI strip — no big cards, TradingView-style quote bar
_ret_pct = port.get_total_return_pct()
_ret_col = C_BUY if _ret_pct >= 0 else C_SELL
_pnl_col = C_BUY if total_pnl >= 0 else C_SELL
_price_str = f"${state['last_price']:,.2f}" if state["last_price"] else "—"

def _kpi(label: str, value: str, color: str = TEXT_HI,
         sub: str = "", sub_color: str = TEXT_DIM) -> str:
    sub_html = (f'<span style="color:{sub_color};font-size:.68rem;'
                f'margin-left:.4rem;font-family:monospace;">{sub}</span>'
                if sub else "")
    return (
        f'<div class="kpi-item">'
        f'<span class="kpi-label">{label}</span>'
        f'<span class="kpi-value" style="color:{color};">{value}</span>'
        f'{sub_html}</div>'
    )

_kpi_strip = (
    f'<div class="kpi-strip">'
    + _kpi(ticker, _price_str, CYAN)
    + _kpi("Value", f"${summ['total_value']:,.2f}", TEXT_HI,
           f"{_ret_pct:+.2f}%", _ret_col)
    + _kpi("P&L", f"${total_pnl:+,.2f}", _pnl_col,
           f"R ${port.get_realized_pnl():+,.2f}")
    + _kpi("Cash", f"${summ['cash']:,.2f}")
    + _kpi("Trades", str(len(port.trade_log)), TEXT_HI,
           f"{len(wins)}W/{len(losses)}L" if closed else "")
    + _kpi("Win", f"{win_rate:.0f}%" if closed else "—", TEXT_HI,
           f"{len(closed)} closed" if closed else "")
    + (_kpi("Committee",
            f"{state['last_committee']['score']:+.2f}",
            C_BUY if state["last_committee"]["score"] >= 0 else C_SELL,
            f"{state['last_committee']['bulls']}🐂 "
            f"{state['last_committee']['bears']}🐻")
       if state.get("last_committee") else "")
    + '</div>'
)
st.markdown(_kpi_strip, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ██  MAIN LAYOUT — full-width chart (TradingView style)
# ─────────────────────────────────────────────────────────────────────────────
from dashboard.charts import market_chart
from dashboard.components import render_desk_sidebar, empty_workspace

chart_column, context_column = st.columns([4.3, 1.25], gap="small")
with chart_column:
    df = state.get("last_df")
    _last_bar = str(df.index[-1])[:19] if df is not None and len(df) else "Awaiting data"
    st.markdown(
        f'<div class="quote-bar"><span class="quote-symbol">{_html.escape(ticker)}</span>'
        f'<span class="quote-detail">Candles · Volume · RSI · MACD</span>'
        f'<span class="quote-detail" style="margin-left:auto">Last bar: {_html.escape(_last_bar)}</span></div>',
        unsafe_allow_html=True,
    )
    if df is None or len(df) == 0:
        empty_workspace(
            "Loading market data" if engine.is_running() else "Your market workspace",
            f"Fetching candles for {ticker}. The first analysis may take a few moments."
            if engine.is_running() else
            "Choose a symbol and strategy, then start the bot to see candles, indicators and trade signals.",
        )
    else:
        fig = market_chart(df, ticker, port.trade_log)
        st.plotly_chart(fig, theme=None, width="stretch", key="main_chart",
                        config={"displaylogo": False, "scrollZoom": True,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
with context_column:
    render_desk_sidebar(state, port, st.session_state.get("watchlist") or DEFAULT_TICKERS,
                        risk=risk, cap_pct=min(ts, _active_envelope.size_max_pct))

# ─────────────────────────────────────────────────────────────────────────────
# ██  BOTTOM TABS — Decision · Positions · Trades · Events
# ─────────────────────────────────────────────────────────────────────────────
tab_dec, tab_board, tab_com, tab_eq, tab_pos, tab_tr, tab_log = st.tabs(
    ["Overview", "Analyst desk", "Technical signals", "Equity", "Positions",
     "Orders & trades", "Activity log"]
)

with tab_dec:
    dec = state.get("last_decision")
    _dec_title = {
        "COMMITTEE": "COMMITTEE DECISION",
        "HYBRID":    "HYBRID DECISION",
        "BOARDROOM": "CHAIRMAN'S RULING",
    }.get(state.get("strategy_mode"), "AI DECISION")
    if dec is not None:
        act_cls = {"BUY": "sig-buy", "SELL": "sig-sell",
                   "HOLD": "sig-hold"}.get(dec.action, "sig-hold")
        conf_pct = int(dec.confidence_score * 100)
        conf_color = {"BUY": C_BUY, "SELL": C_SELL,
                      "HOLD": C_HOLD}.get(dec.action, C_HOLD)
        sl = f"{dec.suggested_stop_loss_pct:.1f}%" \
             if dec.suggested_stop_loss_pct else "—"
        tp = f"{dec.suggested_take_profit_pct:.1f}%" \
             if dec.suggested_take_profit_pct else "—"
        attr_score = int(getattr(dec, "attractiveness_score", 0.0) * 100)
        attr_label = getattr(dec, "attractiveness_label", "NEUTRAL")
        outlook = getattr(dec, "price_outlook", "UNKNOWN")
        chips = "".join(
            f'<span style="display:inline-block;background:#1e222d;'
            f'border:1px solid #2a2e39;border-radius:3px;padding:1px 7px;'
            f'margin:1px 2px;font-size:.64rem;color:#9598a1;">'
            f'{_html.escape(i.replace("_"," ").title()[:24])}</span>'
            for i in (dec.key_indicators or [])[:5]
        ) or f'<span style="color:#2a4050;font-size:.66rem;">—</span>'
        reasoning = _html.escape(dec.reasoning).replace("\n", "<br>")

        st.markdown(
            f'<div class="bt-panel">'
            f'<div class="bt-section-title">{_dec_title}</div>'
            f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:.5rem;">'
            f'<div class="{act_cls}" style="font-size:1.9rem;font-weight:900;">'
            f'{dec.action}</div>'
            f'<div style="flex:1;">'
            f'<div style="color:{TEXT_DIM};font-size:.6rem;letter-spacing:.12em;'
            f'text-transform:uppercase;">Confidence</div>'
            f'<div style="background:#131722;border:1px solid #2a2e39;'
            f'border-radius:3px;height:6px;overflow:hidden;margin-top:3px;">'
            f'<div style="height:6px;width:{conf_pct}%;background:{conf_color};"></div>'
            f'</div>'
            f'<div style="color:{TEXT};font-size:.72rem;margin-top:2px;">{conf_pct}%</div>'
            f'</div>'
            f'</div>'
            f'<div style="display:flex;gap:.6rem;font-size:.68rem;margin-bottom:.5rem;">'
            f'<span style="color:{TEXT_DIM};">RISK</span>'
            f'<span style="color:{TEXT};font-weight:700;">{dec.risk_level}</span>'
            f'<span style="color:{TEXT_DIM};margin-left:.6rem;">SL</span>'
            f'<span style="color:{C_SELL};">{sl}</span>'
            f'<span style="color:{TEXT_DIM};margin-left:.6rem;">TP</span>'
            f'<span style="color:{C_BUY};">{tp}</span>'
            f'<span style="color:{TEXT_DIM};margin-left:.6rem;">RAG</span>'
            f'<span style="color:{CYAN};">{dec.rag_context_quality}</span>'
            f'<span style="color:{TEXT_DIM};margin-left:.6rem;">ATTR</span>'
            f'<span style="color:{TEXT};font-weight:700;">{attr_label} {attr_score}%</span>'
            f'<span style="color:{TEXT_DIM};margin-left:.6rem;">OUTLOOK</span>'
            f'<span style="color:{CYAN};">{outlook}</span>'
            f'</div>'
            f'<div style="margin:.4rem 0;">{chips}</div>'
            f'<div style="color:{TEXT_DIM};font-size:.6rem;letter-spacing:.12em;'
            f'text-transform:uppercase;margin-top:.4rem;">Reasoning</div>'
            f'<div style="background:#131722;border:1px solid #2a2e39;'
            f'border-radius:4px;padding:.5rem .7rem;font-size:.72rem;'
            f'line-height:1.55;color:#d1d4dc;max-height:160px;overflow-y:auto;'
            f'margin-top:.3rem;">{reasoning}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bt-panel" style="text-align:center;padding:1.5rem;">'
            f'<div class="bt-section-title">{_dec_title}</div>'
            f'<div style="color:{TEXT_DIM};font-size:.82rem;">'
            f'No analysis yet. Press START to begin.</div></div>',
            unsafe_allow_html=True,
        )

with tab_board:
    board = state.get("last_boardroom")
    if board is None:
        st.markdown(
            f'<div class="bt-panel" style="text-align:center;padding:1.5rem;">'
            f'<div class="bt-section-title">ANALYST BOARDROOM</div>'
            f'<div style="color:{TEXT_DIM};font-size:.82rem;">'
            f'No meeting held yet. Switch STRATEGY to '
            f'<b>🪑 Boardroom ×8</b> and press START.<br>'
            f'A full hedge-fund desk — 📉 chart · 🏦 fundamentals · '
            f'📰 news · 🤖 quant · 🌍 macro · 🛡 risk officer · '
            f'📊 volume-flow · 😈 contrarian — each studies their own '
            f'briefing at the same moment, votes, and the chairman '
            f'makes the binding call. Use a 120s+ cycle in this mode.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        _vote_clr = {"BUY": C_BUY, "SELL": C_SELL,
                     "HOLD": C_HOLD, "ABSTAIN": "#5a6b7d"}
        _members = board["members"]
        _per_row = 4
        _cols = []
        for i in range(0, len(_members), _per_row):
            _row = st.columns(_per_row, gap="small")
            _cols.extend(_row[:len(_members) - i])
        for col, m in zip(_cols, _members):
            clr = _vote_clr.get(m["vote"], C_HOLD)
            conv = int(m["conviction"] * 100)
            with col:
                st.markdown(
                    f'<div class="bt-panel" style="min-height:215px;">'
                    f'<div style="font-size:1.5rem;line-height:1;">'
                    f'{m["emoji"]}</div>'
                    f'<div style="color:{TEXT_HI};font-weight:800;'
                    f'font-size:.84rem;margin-top:.3rem;">'
                    f'{_html.escape(m["name"])}</div>'
                    f'<div style="color:{TEXT_DIM};font-size:.6rem;'
                    f'text-transform:uppercase;letter-spacing:.1em;'
                    f'margin-bottom:.45rem;">{_html.escape(m["role"])}</div>'
                    f'<div style="display:flex;align-items:center;gap:.5rem;'
                    f'margin-bottom:.45rem;">'
                    f'<span style="color:{clr};font-weight:900;'
                    f'font-size:1.05rem;">{m["vote"]}</span>'
                    f'<div style="flex:1;background:#131722;'
                    f'border:1px solid #2a2e39;border-radius:3px;height:5px;">'
                    f'<div style="height:5px;width:{conv}%;'
                    f'background:{clr};"></div></div>'
                    f'<span style="color:{TEXT_DIM};font-family:monospace;'
                    f'font-size:.64rem;">{conv}%</span></div>'
                    f'<div style="color:#d1d4dc;font-size:.68rem;'
                    f'line-height:1.5;max-height:108px;overflow-y:auto;">'
                    f'{_html.escape(m["opinion"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        chair = board["chair"]
        c_clr = _vote_clr.get(chair["action"], C_HOLD)
        t = board["tally"]
        fb_badge = ('<span class="badge badge-amber">MAJORITY FALLBACK'
                    '</span>' if chair.get("is_fallback") else
                    '<span class="badge badge-green">CHAIR RULED</span>')
        chair_reason = _html.escape(chair["reasoning"]).replace("\n", "<br>")
        st.markdown(
            f'<div class="bt-panel" style="border-left:3px solid {c_clr};">'
            f'<div style="display:flex;align-items:center;gap:1rem;'
            f'flex-wrap:wrap;margin-bottom:.4rem;">'
            f'<span style="font-size:1.4rem;">🪑</span>'
            f'<div><div style="color:{TEXT_HI};font-weight:800;'
            f'font-size:.9rem;">{_html.escape(chair["name"])} — Chairman'
            f'</div>'
            f'<div style="color:{TEXT_DIM};font-size:.62rem;'
            f'font-family:monospace;">convened {board["convened_at"]} · '
            f'panel {t.get("BUY", 0)} BUY / {t.get("SELL", 0)} SELL / '
            f'{t.get("HOLD", 0)} HOLD'
            f'{" / " + str(t["ABSTAIN"]) + " ABSTAIN" if t.get("ABSTAIN") else ""}'
            f'</div></div>'
            f'<span style="margin-left:auto;display:flex;gap:.6rem;'
            f'align-items:center;">{fb_badge}'
            f'<span style="color:{c_clr};font-weight:900;'
            f'font-size:1.6rem;">{chair["action"]}</span>'
            f'<span style="color:{TEXT_DIM};font-family:monospace;'
            f'font-size:.8rem;">conf {int(chair["confidence"] * 100)}%'
            f'</span></span></div>'
            f'<div style="background:#131722;border:1px solid #2a2e39;'
            f'border-radius:4px;padding:.5rem .7rem;font-size:.72rem;'
            f'line-height:1.55;color:#d1d4dc;max-height:170px;'
            f'overflow-y:auto;">{chair_reason}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with tab_com:
    com = state.get("last_committee")
    if com is None:
        st.markdown(
            f'<div class="bt-panel" style="text-align:center;padding:1.5rem;">'
            f'<div class="bt-section-title">INDICATOR COMMITTEE</div>'
            f'<div style="color:{TEXT_DIM};font-size:.82rem;">'
            f'No vote yet. Switch STRATEGY to <b>Committee ×38</b> and press '
            f'START — 38 technical indicators will vote every cycle.<br>'
            f'Long-only: a bear majority exits to cash instead of '
            f'panic-selling. Works on crypto and stocks.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        bulls, bears = com["bulls"], com["bears"]
        neutrals, total = com["neutrals"], com["total"]
        score = com["score"]
        act = com["action"]
        act_color = {"BUY": C_BUY, "SELL": C_SELL}.get(act, C_HOLD)
        b_pct = bulls / total * 100
        s_pct = bears / total * 100
        n_pct = 100 - b_pct - s_pct

        # Vote balance bar: bulls | neutral | bears
        vote_bar = (
            f'<div style="display:flex;height:18px;border-radius:4px;'
            f'overflow:hidden;border:1px solid #1a2c3e;">'
            f'<div style="width:{b_pct}%;background:{C_BUY};"></div>'
            f'<div style="width:{n_pct}%;background:#16202c;"></div>'
            f'<div style="width:{s_pct}%;background:{C_SELL};"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:monospace;font-size:.7rem;margin-top:3px;">'
            f'<span style="color:{C_BUY};">🐂 {bulls} BULL</span>'
            f'<span style="color:{TEXT_DIM};">{neutrals} NEUTRAL</span>'
            f'<span style="color:{C_SELL};">{bears} BEAR 🐻</span>'
            f'</div>'
        )

        cat_html = "".join(
            f'<div style="display:flex;align-items:center;gap:.6rem;'
            f'margin:.2rem 0;">'
            f'<span style="color:{TEXT_DIM};font-size:.62rem;width:90px;'
            f'text-transform:uppercase;letter-spacing:.08em;">{cat}</span>'
            f'<div style="flex:1;background:#131722;border:1px solid #2a2e39;'
            f'border-radius:3px;height:8px;position:relative;">'
            f'<div style="position:absolute;left:50%;top:0;height:8px;'
            f'width:{abs(cs) * 50}%;'
            f'{"" if cs >= 0 else "transform:translateX(-100%);"}'
            f'background:{C_BUY if cs >= 0 else C_SELL};"></div>'
            f'</div>'
            f'<span style="color:{C_BUY if cs >= 0 else C_SELL};'
            f'font-family:monospace;font-size:.68rem;width:48px;'
            f'text-align:right;">{cs:+.2f}</span></div>'
            for cat, cs in sorted(com.get("category_scores", {}).items())
        )

        st.markdown(
            f'<div class="bt-panel">'
            f'<div class="bt-section-title">COMMITTEE VERDICT — '
            f'{total} INDICATORS</div>'
            f'<div style="display:flex;gap:1.4rem;align-items:center;'
            f'margin-bottom:.7rem;">'
            f'<div style="font-size:2rem;font-weight:900;color:{act_color};">'
            f'{act}</div>'
            f'<div style="flex:1;">{vote_bar}</div>'
            f'<div style="text-align:right;">'
            f'<div style="color:{TEXT_DIM};font-size:.6rem;'
            f'text-transform:uppercase;letter-spacing:.1em;">Net score</div>'
            f'<div style="color:{act_color};font-family:monospace;'
            f'font-size:1.25rem;font-weight:800;">{score:+.2f}</div>'
            f'</div></div>'
            f'<div style="margin:.6rem 0 .3rem 0;color:{TEXT_DIM};'
            f'font-size:.6rem;text-transform:uppercase;'
            f'letter-spacing:.12em;">By category</div>'
            f'{cat_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Per-agent vote grid, grouped by category
        votes = com.get("votes", [])
        if votes:
            by_cat: dict[str, list] = {}
            for v in votes:
                by_cat.setdefault(v["category"], []).append(v)
            grid_html = ""
            for cat in sorted(by_cat):
                chips = "".join(
                    f'<span style="display:inline-block;padding:2px 9px;'
                    f'margin:2px;border-radius:3px;font-size:.64rem;'
                    f'font-family:monospace;'
                    + (f'background:rgba(0,193,118,.13);color:{C_BUY};'
                       f'border:1px solid rgba(0,193,118,.4);'
                       if v["vote"] > 0 else
                       f'background:rgba(255,77,79,.12);color:{C_SELL};'
                       f'border:1px solid rgba(255,77,79,.4);'
                       if v["vote"] < 0 else
                       f'background:#1e222d;color:#9598a1;'
                       f'border:1px solid #2a2e39;')
                    + f'">{"▲" if v["vote"] > 0 else "▼" if v["vote"] < 0 else "•"} '
                    f'{_html.escape(v["name"])}</span>'
                    for v in by_cat[cat]
                )
                grid_html += (
                    f'<div style="margin-bottom:.45rem;">'
                    f'<span style="color:{TEXT_DIM};font-size:.6rem;'
                    f'text-transform:uppercase;letter-spacing:.1em;'
                    f'margin-right:.5rem;">{cat}</span><br>{chips}</div>'
                )
            st.markdown(
                f'<div class="bt-panel">'
                f'<div class="bt-section-title">INDIVIDUAL VOTES</div>'
                f'{grid_html}</div>',
                unsafe_allow_html=True,
            )

with tab_eq:
    eq_hist = state.get("equity_history") or []
    if len(eq_hist) < 2:
        st.caption("Equity curve builds while the engine runs — one point "
                   "per cycle. Press START and come back in a few minutes.")
    else:
        eq_ts = [p[0] for p in eq_hist]
        eq_val = [p[1] for p in eq_hist]
        start_cap = float(st.session_state["starting_capital"])
        eq_fig = go.Figure()
        eq_fig.add_hline(y=start_cap, line_dash="dot",
                         line_color="#3a4a5c", line_width=1,
                         annotation_text="start",
                         annotation_font_color=TEXT_DIM)
        last_up = eq_val[-1] >= start_cap
        eq_fig.add_trace(go.Scatter(
            x=eq_ts, y=eq_val, name="Equity",
            mode="lines", fill="tozeroy",
            line=dict(color=C_BUY if last_up else C_SELL, width=1.8),
            fillcolor=("rgba(0,193,118,0.10)" if last_up
                       else "rgba(255,77,79,0.10)"),
        ))
        ymin, ymax = min(eq_val), max(eq_val)
        pad = max((ymax - ymin) * 0.15, ymax * 0.001)
        eq_fig.update_layout(
            template="bottrade", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#131722",
            margin=dict(l=12, r=54, t=12, b=12), showlegend=False,
            font=dict(family="Inter, Segoe UI, sans-serif", size=11,
                      color=TEXT),
            xaxis=dict(showgrid=True, gridcolor=GRID),
            yaxis=dict(side="right", showgrid=True, gridcolor=GRID,
                       range=[ymin - pad, ymax + pad],
                       tickprefix="$", tickformat=",.0f"),
        )
        st.plotly_chart(eq_fig, theme=None, width="stretch", key="equity_chart")
        peak = max(eq_val)
        dd = (eq_val[-1] / peak - 1) * 100 if peak else 0.0
        st.caption(f"{len(eq_val)} samples · peak ${peak:,.2f} · "
                   f"current drawdown from peak {dd:+.2f}%")

with tab_pos:
    if summ["open_positions"]:
        rows = []
        for p in summ["open_positions"]:
            pnl = p["unrealized_pnl"]
            rows.append({
                "Ticker": p["ticker"],
                "Qty":    f'{p["quantity"]:.4f}',
                "Entry":  f'${p["avg_entry_price"]:,.2f}',
                "Mark":   f'${p["current_price"]:,.2f}',
                "P&L":    f'${pnl:+,.2f}',
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     width="stretch", height=240)
    else:
        st.caption("No open positions.")

with tab_tr:
    if port.trade_log:
        rows = []
        for t in port.trade_log:
            rows.append({
                "Time":   t.executed_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Act":    t.action,
                "Ticker": t.ticker,
                "Price":  f"${t.price:,.2f}",
                "P&L":    f"${t.realized_pnl:+,.2f}",
                "Reason": (t.reasoning or "")[:120],
            })
        df_tr = pd.DataFrame(list(reversed(rows)))
        st.dataframe(df_tr, hide_index=True, width="stretch",
                     height=360)
        st.download_button(
            "⬇ Export CSV",
            df_tr.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"bottrade_trades_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No trades yet.")

with tab_log:
    events = st.session_state.get("_event_log", [])
    if not events:
        st.caption("No activity yet. The log fills in real-time when the "
                   "engine runs.")
    else:
        rows_html = "".join(
            f'<div class="log-row log-{e.level}">'
            f'<span class="log-ts">{e.ts.strftime("%H:%M:%S")}</span>'
            f'<span class="stage-pill stage-{e.stage.value}" '
            f'style="margin-right:6px;">{e.stage.value}</span>'
            f'<span class="log-lvl">{e.level}</span>'
            f'<span class="log-msg">{_html.escape(e.message)}</span>'
            f'</div>'
            for e in reversed(events[-200:])
        )
        st.markdown(
            f'<div style="max-height:360px;overflow-y:auto;'
            f'background:{BG_DEEP};border:1px solid {BORDER};'
            f'border-radius:5px;">{rows_html}</div>',
            unsafe_allow_html=True,
        )
