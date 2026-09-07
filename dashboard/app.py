"""BotTrade router: four primary destinations with progressively disclosed tools.

All routes stay registered for deep links and per-page access checks. The
sidebar presents the daily workflow; specialist tools remain in an expander.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# Started here because this router is the one file every session runs, and
# install() is idempotent. It logs nothing unless the process is actually
# denied the CPU — see the module docstring for the production failure it
# was added to identify.
from utils import stall_watchdog as _stall_watchdog

_stall_watchdog.install()

# Paths are relative to this entrypoint file, per st.Page's own contract.
# Icons and titles are carried over from each page's existing
# st.set_page_config call rather than invented fresh, so the browser-tab
# title a page already sets for itself stays the source of truth and this
# is just where the *sidebar label* — previously just the raw filename —
# gets a real name.
_live = st.Page("pages/0_Live.py", title="Command Center",
                icon=":material/monitoring:", default=True)
_portfolio = st.Page("pages/1_Portfolio.py", title="Portfolio",
                     icon=":material/account_balance:")
_settings = st.Page("pages/2_Settings.py", title="Settings",
                    icon=":material/settings:")
_knowledge = st.Page("pages/3_Knowledge.py", title="Knowledge Base",
                     icon=":material/database:")
_market_research = st.Page("pages/4_Market_Research.py", title="Market Research",
                           icon=":material/manage_search:")
_sector_heatmap = st.Page("pages/5_Sector_Heatmap.py", title="Sector Heatmap",
                          icon=":material/grid_view:")
_watchlist_scanner = st.Page("pages/6_Watchlist_Scanner.py", title="Watchlist Scanner",
                             icon=":material/filter_alt:")
_ml_lab = st.Page("pages/7_ML_Lab.py", title="ML Lab",
                  icon=":material/model_training:")
_analytics = st.Page("pages/8_Analytics.py", title="Analytics",
                     icon=":material/analytics:")
_committee_lab = st.Page("pages/9_Committee_Lab.py", title="Committee Lab",
                         icon=":material/how_to_vote:")
_usage_billing = st.Page("pages/10_Usage_and_Billing.py", title="Usage & Billing",
                         icon=":material/receipt_long:")

# Four sections, each one word, each answering a different question:
#   Trade    — what is the bot doing with my money right now?
#   Research — what does the market look like, before I commit to anything?
#   Lab      — offline tools that consume no API budget: patterns, ML,
#              a multi-analyst debate — none of them place an order.
#   Account  — not trading at all: identity, billing, API key.
# Register every route; expose a small primary navigation surface.
_page = st.navigation([_live, _portfolio, _market_research, _settings,
                       _sector_heatmap, _watchlist_scanner, _knowledge,
                       _ml_lab, _analytics, _committee_lab, _usage_billing], position="hidden")
with st.sidebar:
    st.markdown('<div class="nav-caption">WORKSPACE</div>', unsafe_allow_html=True)
    for page, label in ((_live, "Your agent"), (_portfolio, "Portfolio"),
                        (_market_research, "Research"), (_settings, "Settings")):
        st.page_link(page, label=label, width="stretch")
    with st.expander("Advanced tools", expanded=False):
        for page in (_watchlist_scanner, _sector_heatmap, _knowledge, _analytics,
                     _committee_lab, _ml_lab):
            st.page_link(page, width="stretch")
    st.markdown('<div class="nav-footer">BOTTRADE<br><span>Investment workspace</span></div>', unsafe_allow_html=True)
_page.run()
