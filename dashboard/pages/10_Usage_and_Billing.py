"""Usage & Billing — what the bot spent, on whose key, and what the cache saved.

A hosted trading bot that runs on someone else's API key owes them a bill they
can audit. This page is that bill: month-to-date spend broken down by strategy
mode, model and symbol, plus the shared-decision-cache savings, so the cost of
running a strategy is never a surprise.

Reads the SQLite ledger only — no LLM calls, no market data, nothing to pay for
by opening it.
"""
from __future__ import annotations

# ── sys.path bootstrap ─────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import (
    AMBER, BG_PANEL, BORDER, C_BUY, CYAN, GRID, TEXT, TEXT_DIM,
    apply_theme, ensure_profile_in_session, get_tenant,
)
from decision_engine.decision_cache import get_decision_cache
from saas.ledger import get_ledger
from saas.plans import CALLS_PER_CYCLE, Funding
from saas.pricing import cost_usd, format_usd, format_usd_md
from trading.registry import get_registry

st.set_page_config(page_title="BotTrade - Usage", page_icon=":material/receipt_long:",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()
ensure_profile_in_session()

st.markdown('<div class="page-title">USAGE &amp; BILLING</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Every Claude call this bot made, priced and attributed'
    '</div>', unsafe_allow_html=True)

tenant = get_tenant()
ent = tenant.entitlement
ledger = get_ledger()
usage = tenant.usage()

# ── KPI strip ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Month to date", format_usd(usage["cost_usd"]))
k2.metric("Today", format_usd(usage["today_usd"]))
k3.metric("API calls", f"{usage['calls']:,}")
k4.metric("Calls avoided", f"{usage['calls_saved']:,}",
          help="Decisions served from the shared cache instead of the API.")
k5.metric("Billed to",
          {Funding.BYOK: "Your key",
           Funding.PLATFORM: "Trial credit",
           Funding.NONE: "—"}[ent.funding])

if ent.funding is Funding.PLATFORM:
    left = ent.platform_budget_remaining_usd
    total = ent.plan.platform_budget_usd or 1.0
    st.progress(min(1.0, max(0.0, 1 - left / total)),
                text=f"Trial credit — {format_usd_md(left)} of "
                     f"{format_usd_md(total)} remaining")
elif ent.funding is Funding.NONE:
    st.info(ent.lock_reason or
            "No API funding configured — the bot is running COMMITTEE mode, "
            "which costs nothing.")

st.markdown("---")

# ── What the shared cache is saving ─────────────────────────────────────────
st.markdown("##### Shared decision cache")
st.caption(
    "A verdict on one symbol for one bar is identical for every user looking "
    "at it, so it is computed once and reused. This is what stops the API bill "
    "from scaling with the number of users."
)

stats = get_decision_cache().stats.to_dict()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lookups", f"{stats['lookups']:,}")
c2.metric("Served from cache", f"{stats['calls_saved']:,}")
c3.metric("Hit rate", f"{stats['hit_rate']:.0%}")
c4.metric("Live entries", f"{get_decision_cache().size():,}")

if stats["coalesced"]:
    st.caption(
        f"{stats['coalesced']:,} of those were concurrent requests that "
        f"arrived while the same decision was already being computed — they "
        f"waited for it rather than firing their own call."
    )

st.markdown("---")

# ── Live bot capacity ───────────────────────────────────────────────────────
st.markdown("##### Live bots on this deployment")
st.caption(
    "Each live bot is a background thread owned by the process, not by your "
    "browser tab — a refresh reattaches to the bot already running rather "
    "than starting a second one. When every slot is taken a new bot is "
    "refused; an existing one is never evicted."
)

_reg = get_registry().snapshot()
r1, r2, r3 = st.columns(3)
r1.metric("Running now", f"{_reg['running']}")
r2.metric("Slots held", f"{_reg['held']} / {_reg['max']}")
r3.metric("Free slots", f"{max(0, _reg['max'] - _reg['held'])}")

if _reg["held"] >= _reg["max"]:
    st.warning(
        "Every live-bot slot is in use. New bots will be refused until one "
        "stops. Raise `BOTTRADE_MAX_LIVE_ENGINES` if this host has headroom.",
        icon="🚦",
    )

_mine = next((a for a in _reg["accounts"] if a["account"] == tenant.account_id),
             None)
if _mine:
    st.caption(
        f"Your bot: **{_mine['mode'] or '—'}** on **{_mine['ticker'] or '—'}** · "
        f"{'running' if _mine['running'] else 'stopped'}"
    )

st.markdown("---")

# ── Breakdown ───────────────────────────────────────────────────────────────
st.markdown("##### Where the money went")
account = tenant.account_id
tab_mode, tab_model, tab_ticker = st.tabs(["By strategy", "By model", "By symbol"])


def _render_breakdown(rows: list[dict], label_col: str) -> None:
    if not rows:
        st.caption("No API calls recorded this month.")
        return
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "label": label_col, "calls": "Calls",
        "input_tokens": "Input tokens", "output_tokens": "Output tokens",
        "cost_usd": "Cost",
    })
    fig = go.Figure(go.Bar(
        x=df["Cost"], y=df[label_col], orientation="h",
        marker_color=CYAN,
        hovertemplate="%{y}: $%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(180, 46 * len(df)),
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT, size=12),
        xaxis=dict(gridcolor=GRID, title="USD", zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    df["Cost"] = df["Cost"].map(format_usd)
    st.dataframe(df, use_container_width=True, hide_index=True)


with tab_mode:
    _render_breakdown(ledger.breakdown(account, by="mode"), "Strategy")
with tab_model:
    _render_breakdown(ledger.breakdown(account, by="model"), "Model")
with tab_ticker:
    _render_breakdown(ledger.breakdown(account, by="ticker"), "Symbol")

st.markdown("---")

# ── Cost estimator ──────────────────────────────────────────────────────────
st.markdown("##### What will it cost to run?")
st.caption(
    "Rough forecast at list prices, before any cache sharing. Real spend is "
    "usually well below this — the shared cache means a symbol many people "
    "watch is only priced once per bar."
)

e1, e2, e3 = st.columns(3)
est_mode = e1.selectbox("Strategy mode", list(CALLS_PER_CYCLE.keys()), index=3)
est_interval = e2.number_input("Cycle interval (seconds)", min_value=15,
                               max_value=3600, value=120, step=15)
est_hours = e3.number_input("Hours per day", min_value=1, max_value=24, value=8)

calls_per_cycle = CALLS_PER_CYCLE[est_mode]
cycles_per_day = (est_hours * 3600) / max(1, est_interval)
# Typical boardroom-sized packet: ~4k input, ~400 output per call.
per_call = cost_usd("claude-haiku-4-5", 4_000, 400)
daily = calls_per_cycle * cycles_per_day * per_call

if calls_per_cycle == 0:
    st.success(
        f"**{est_mode}** makes no API calls at all — it runs entirely on local "
        rf"computation. Cost: **\$0.00**, at any interval, on any plan."
    )
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Calls per day", f"{calls_per_cycle * cycles_per_day:,.0f}")
    m2.metric("Per day", format_usd(daily))
    m3.metric("Per 30 days", format_usd(daily * 30))
    st.caption(
        rf"Assumes Haiku 4.5 at \$1 / \$5 per million tokens and ~4k input / "
        f"400 output per call. Heavier models cost proportionally more — "
        f"see `saas/pricing.py` for the rate card."
    )

st.markdown("---")

# ── Recent calls ────────────────────────────────────────────────────────────
st.markdown("##### Recent calls")
rows = ledger.recent(account, limit=100)
if not rows:
    st.caption("Nothing recorded yet. Run the bot in an AI mode to see calls here.")
else:
    df = pd.DataFrame([{
        "Time":    r.ts.replace("T", " ")[:19],
        "Mode":    r.mode or "—",
        "Symbol":  r.ticker or "—",
        "Model":   r.model,
        "In":      r.input_tokens + r.cache_read_tokens,
        "Out":     r.output_tokens,
        "Cost":    format_usd(r.cost_usd),
        "Funding": r.funding,
    } for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True, height=380)
