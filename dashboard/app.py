"""
BotTrade — entrypoint / router.

Run:  python -X utf8 -m streamlit run dashboard/app.py

This file used to *be* the live trading page (the default/root page under
Streamlit's implicit ``pages/`` auto-discovery). Its actual content now
lives at ``dashboard/pages/0_Live.py``; this file's only job is to declare
every page once, grouped the way a trading desk actually thinks about them
— Trade / Research / Lab / Account — and hand off to whichever one the
visitor picked.

Why this exists at all: the old sidebar was Streamlit's raw default —
every file under ``pages/`` listed flat, in filename order, no grouping, no
icons, labelled straight from the filename ("6_Watchlist_Scanner"). Eleven
items in one undifferentiated list reads as a pile of settings, not a
product with a shape. ``st.navigation()`` with a section mapping is the
supported way to fix that without fighting Streamlit's router by hand.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

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
_page = st.navigation({
    "Trade":    [_live, _portfolio],
    "Research": [_market_research, _sector_heatmap, _watchlist_scanner, _knowledge],
    "Lab":      [_ml_lab, _analytics, _committee_lab],
    "Account":  [_settings, _usage_billing],
})
_page.run()
