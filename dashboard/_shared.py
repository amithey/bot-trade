"""
Shared helpers for the BotTrade dashboard — theme, state bootstrap, and a
singleton live-engine accessor so every page sees the same background thread.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(find_dotenv(usecwd=True), override=True)

from config.user_profile import UserProfile
from portfolio.virtual_account import LivePortfolio

# ── Constants ────────────────────────────────────────────────────────────────
#: Pre-multi-user profile location. Kept only so an existing deployment's
#: settings are adopted once into that account's own file — see
#: ``_load_profile_for``. Nothing writes here any more.
LEGACY_PROFILE_PATH = ROOT / "data" / "user_profile.json"
DEFAULT_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "QQQ", "SPY",
                   "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META"]
RISK_CHOICES    = ["Conservative", "Balanced", "Aggressive", "Micro-Scalp"]
_TICKER_RE      = re.compile(r"^[A-Z0-9^][A-Z0-9.^=-]{0,14}$")
CUSTOM_LABEL    = "Custom"

# TradingView-style sharp dark palette
BG_DEEP   = "#000000"
BG_PANEL  = "#0b0f14"
BG_RAISED = "#111820"
BORDER    = "#202833"
BORDER_HI = "#34404f"
TEXT_DIM  = "#8b98a8"
TEXT      = "#d8e1ec"
TEXT_HI   = "#ffffff"
CYAN      = "#00b7ff"
C_BUY     = "#00c176"
C_SELL    = "#ff4d4f"
C_HOLD    = "#b3bdc9"
AMBER     = "#f5b544"
GRID      = "rgba(54,64,78,0.42)"

# Aliases kept so older pages don't break
BG = BG_DEEP


_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700;800&display=swap');

:root {{
    --font-ui:   'Inter','Segoe UI',system-ui,sans-serif;
    --font-mono: 'JetBrains Mono','Cascadia Mono','SF Mono',monospace;
}}
html, body, [data-testid="stAppViewContainer"] * {{
    -webkit-font-smoothing:antialiased;
    text-rendering:optimizeLegibility;
}}

#MainMenu, footer {{ display:none !important; }}
/* Hide Streamlit Cloud "Deploy" button + status widget — but keep the
   toolbar itself, because it contains the sidebar-collapse chevron. */
div[data-testid="stStatusWidget"],
button[data-testid="stBaseButton-header"] {{ display:none !important; }}
/* Kill only the decorative rainbow strip, not the whole toolbar. */
div[data-testid="stDecoration"] {{ display:none !important; }}
/* Keep header transparent/slim but DON'T hide it — it hosts the
   collapsed-sidebar re-open chevron. */
header[data-testid="stHeader"] {{
    background:transparent !important;
    box-shadow:none !important; z-index:9998 !important;
}}
.block-container {{ max-width:100% !important; padding:2.6rem 1.2rem 0.6rem 3.2rem !important; }}
/* Hide the empty top padding wrapper Streamlit adds above the first block */
div[data-testid="stAppViewContainer"] > section > div:first-child {{ padding-top:0 !important; }}
/* Gutter on the left when sidebar is collapsed so the chevron doesn't
   sit on top of the TICKER column. */
section[data-testid="stMain"] > div:first-child {{ padding-left:1.2rem !important; }}
html, body, [data-testid="stAppViewContainer"] {{ background:{BG_DEEP} !important; }}
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 10% 0%, rgba(0,183,255,0.10), transparent 24rem),
        radial-gradient(circle at 92% 4%, rgba(0,193,118,0.07), transparent 22rem),
        linear-gradient(180deg,#000000 0%, #020406 45%, #000000 100%) !important;
}}
[data-testid="stAppViewContainer"]::before {{
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
    background-size:48px 48px,48px 48px;
    mask-image:linear-gradient(180deg, rgba(0,0,0,0.45), transparent 70%);
}}

/* Sidebar */
[data-testid="stSidebar"] {{ background:#020304 !important;
    border-right:1px solid {BORDER}; }}
[data-testid="stSidebarNav"] a {{
    font-family:var(--font-ui) !important;
    font-size:0.84rem !important; color:#98a6ba !important;
    letter-spacing:.04em;
}}
[data-testid="stSidebarNav"] a:hover {{ color:{TEXT_HI} !important; }}
/* Collapsed-sidebar re-open chevron — pin it to top-left, above everything. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    display:flex !important; visibility:visible !important;
    opacity:1 !important; z-index:99999 !important;
    position:fixed !important; top:0.5rem !important; left:0.5rem !important;
    width:2.2rem !important; height:2.2rem !important;
    align-items:center !important; justify-content:center !important;
    background:{BG_PANEL} !important; border:1px solid {BORDER_HI} !important;
    border-radius:4px !important; color:{TEXT_HI} !important;
    cursor:pointer !important;
}}
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover {{
    border-color:{CYAN} !important; box-shadow:none;
}}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="collapsedControl"] button svg {{
    color:{TEXT_HI} !important; fill:{TEXT_HI} !important;
    width:1.1rem !important; height:1.1rem !important;
}}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {{
    background:transparent !important; border:none !important;
    padding:0 !important; min-height:0 !important; width:100% !important;
    height:100% !important;
}}

/* Inputs */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {{
    background:{BG_PANEL} !important; color:{TEXT} !important;
    border-color:{BORDER} !important;
    font-family:var(--font-ui);
    font-size:0.82rem !important; border-radius:4px;
}}
label, [data-testid="stWidgetLabel"] {{
    font-size:0.60rem !important; color:{TEXT_DIM} !important;
    text-transform:uppercase; letter-spacing:0.10em; font-weight:600 !important;
    min-height:0.9rem !important; line-height:0.9rem !important;
    margin-bottom:0.35rem !important;
}}

/* Uniform 38px widget height across the top control bar — selectbox, number-
   input, slider, and buttons all share the same baseline so nothing sticks
   above or below its neighbour. */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div,
div[data-baseweb="select"] > div {{
    min-height:38px !important;
}}
div[data-testid="stSlider"] {{
    padding-top:10px !important; padding-bottom:0 !important;
}}
div[data-testid="stSlider"] > div {{
    min-height:28px !important;
}}
.stButton > button {{ min-height:38px !important; }}

/* Flatten Streamlit alert boxes (st.info / st.warning / st.success /
   st.error) — their default top accent line creates a stray cyan streak
   that visually bleeds into whatever panel sits directly above. */
div[data-testid="stAlert"],
div[data-testid="stNotification"] {{
    border:1px solid {BORDER} !important;
    border-top:1px solid {BORDER} !important;
    border-radius:6px !important;
    background:{BG_PANEL} !important;
    box-shadow:none !important;
}}
div[data-testid="stAlert"] > div:first-child,
div[data-testid="stNotification"] > div:first-child {{
    border-top:none !important;
}}

/* Vertically center every widget inside its column so that selectboxes,
   sliders and buttons share one baseline even when label heights differ. */
div[data-testid="column"] > div[data-testid="stVerticalBlock"] {{
    justify-content:flex-start !important;
}}

/* KPI cards — compact */
[data-testid="stMetric"] {{
    background:linear-gradient(145deg,rgba(17,24,39,0.96),rgba(23,32,51,0.88));
    border:1px solid {BORDER}; border-top:1px solid {BORDER_HI};
    border-radius:10px; padding:0.55rem 0.9rem !important;
    box-shadow:0 18px 45px rgba(0,0,0,0.25);
}}
[data-testid="stMetricValue"] {{
    font-size:1.05rem !important; font-weight:700 !important;
    font-family:var(--font-mono) !important;
    color:{TEXT_HI} !important; line-height:1.2 !important;
}}
[data-testid="stMetricLabel"] {{
    font-size:0.55rem !important; color:{TEXT_DIM} !important;
    text-transform:uppercase; letter-spacing:0.12em;
}}
[data-testid="stMetricDelta"] {{
    font-size:0.66rem !important;
    font-family:var(--font-mono) !important;
}}

/* Vertical-stack container compression */
div[data-testid="stVerticalBlock"] {{ gap:0.35rem !important; }}

/* Buttons */
.stButton > button {{
    background:#080c11 !important; color:{TEXT_HI} !important;
    border:1px solid {BORDER_HI} !important; border-radius:6px !important;
    font-family:var(--font-mono) !important;
    font-weight:800 !important; font-size:0.74rem !important;
    letter-spacing:.08em; text-transform:uppercase;
    transition:all .15s ease;
    white-space:nowrap !important;
    padding-left:.55rem !important; padding-right:.55rem !important;
}}
.stButton > button:hover {{ border-color:{CYAN} !important;
    color:{TEXT_HI} !important; box-shadow:none; }}
button[kind="primary"] {{
    background:linear-gradient(135deg,#008f5a,#005f3f) !important;
    border-color:{C_BUY} !important; color:{TEXT_HI} !important;
}}

/* Tables */
.stDataFrame {{ font-size:0.76rem !important; }}
.stDataFrame thead th {{
    background:{BG_DEEP} !important; color:{TEXT_DIM} !important;
    text-transform:uppercase; font-size:0.58rem !important;
    letter-spacing:0.08em; border-bottom:1px solid {BORDER} !important;
}}

/* Badges */
.badge {{ display:inline-block; padding:3px 10px; border-radius:4px;
    font-family:var(--font-mono);
    font-weight:700; font-size:0.74rem; letter-spacing:.05em; }}
.badge-green  {{ background:rgba(47,191,113,0.12); color:{C_BUY};
                 border:1px solid rgba(47,191,113,0.36); }}
.badge-amber  {{ background:rgba(216,164,65,0.12); color:{AMBER};
                 border:1px solid rgba(216,164,65,0.35); }}
.badge-blue   {{ background:rgba(90,167,216,0.12); color:{CYAN};
                 border:1px solid rgba(90,167,216,0.35); }}
.badge-gray   {{ background:rgba(58,64,80,0.18); color:#7090a8;
                 border:1px solid #1e3348; }}
.badge-red    {{ background:rgba(216,74,74,0.12); color:{C_SELL};
                 border:1px solid rgba(216,74,74,0.36); }}

/* Panels */
.bt-panel {{
    background:linear-gradient(180deg,rgba(11,15,20,0.98),rgba(6,9,12,0.96));
    border:1px solid {BORDER}; border-top:1px solid {BORDER_HI};
    border-radius:12px;
    padding:0.8rem 1rem; margin-bottom:0.6rem;
    font-family:var(--font-ui);
    box-shadow:0 22px 60px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter:blur(10px);
}}
.bt-section-title {{
    color:{TEXT_HI}; font-family:var(--font-mono);
    font-weight:800; font-size:0.72rem; letter-spacing:.14em;
    text-transform:uppercase; margin:0 0 .6rem 0;
}}
.bt-brand {{
    display:flex; align-items:center; justify-content:space-between;
    gap:1rem; margin:0 0 .75rem 0; padding:.85rem 1rem;
    border:1px solid {BORDER}; border-top:1px solid {BORDER_HI};
    border-radius:14px;
    background:linear-gradient(135deg,rgba(9,13,18,0.98),rgba(2,4,7,0.96));
    box-shadow:0 20px 70px rgba(0,0,0,0.32);
}}
.bt-brand-title {{
    color:{TEXT_HI}; font-family:var(--font-ui);
    font-size:1.1rem; font-weight:850; letter-spacing:.16em;
    text-transform:uppercase; line-height:1;
}}
.bt-brand-sub {{
    color:{TEXT_DIM}; font-family:var(--font-mono);
    font-size:.62rem; letter-spacing:.16em; text-transform:uppercase;
    margin-top:.35rem;
}}
.bt-brand-mark {{
    width:38px; height:38px; border-radius:10px;
    background:linear-gradient(145deg,rgba(90,167,216,.24),rgba(47,191,113,.15));
    border:1px solid rgba(90,167,216,.35);
    display:flex; align-items:center; justify-content:center;
    font-family:var(--font-mono);
    color:{TEXT_HI}; font-weight:900; letter-spacing:.02em;
}}
.bt-command {{
    border:1px solid {BORDER}; border-radius:12px;
    background:linear-gradient(180deg,rgba(8,12,17,.98),rgba(3,5,8,.96));
    padding:.75rem .85rem .55rem .85rem; margin-bottom:.65rem;
}}
.live-bar {{
    display:flex; align-items:center; gap:1.1rem; flex-wrap:wrap;
    padding:.68rem 1rem !important;
}}
.kpi-strip {{
    display:flex; align-items:center; flex-wrap:wrap;
    background:linear-gradient(180deg,rgba(8,12,17,.98),rgba(2,4,7,.96));
    border:1px solid {BORDER}; border-top:1px solid {BORDER_HI};
    border-radius:12px; padding:.35rem .35rem; overflow-x:auto;
    margin-bottom:1rem; box-shadow:0 18px 55px rgba(0,0,0,.26);
}}
.kpi-item {{
    display:flex; align-items:baseline; gap:.45rem;
    padding:.42rem 1rem; border-right:1px solid rgba(39,50,68,.85);
    min-height:36px;
}}
.kpi-label {{
    color:{TEXT_DIM}; font-size:.58rem; letter-spacing:.13em;
    text-transform:uppercase; font-family:var(--font-mono);
}}
.kpi-value {{
    font-size:.94rem; font-weight:800; font-family:var(--font-mono);
}}
.kpi-sub {{
    font-size:.66rem; margin-left:.25rem; font-family:var(--font-mono);
}}
.page-title {{
    color:{TEXT_HI}; font-family:var(--font-ui);
    font-weight:800; font-size:1.35rem; letter-spacing:.08em;
    margin:0 0 .6rem 0;
}}
.page-sub {{
    color:{TEXT_DIM}; font-family:var(--font-mono);
    font-size:0.68rem; letter-spacing:.14em; text-transform:uppercase;
    margin:0 0 0.8rem 0;
}}

/* Pulse (heartbeat) indicator */
@keyframes pulse-dot {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%     {{ opacity:.4; transform:scale(1.18); }}
}}
.pulse-dot {{
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:8px; vertical-align:middle;
    animation:pulse-dot 1.2s ease-in-out infinite;
}}
.pulse-dot.on  {{ background:{C_BUY}; box-shadow:none; }}
.pulse-dot.off {{ background:#2a3c50; animation:none; }}
.pulse-dot.ai  {{ background:{CYAN}; box-shadow:none; }}
.pulse-dot.exec{{ background:{AMBER}; box-shadow:none; }}
.pulse-dot.err {{ background:{C_SELL}; box-shadow:none; }}

/* Stage pills */
.stage-pill {{ display:inline-block; padding:2px 9px; border-radius:4px;
    font-family:var(--font-mono);
    font-size:0.66rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; }}
.stage-IDLE       {{ background:#1a2638; color:#6a8caa; border:1px solid #243753; }}
.stage-FETCH      {{ background:rgba(0,201,255,0.12); color:{CYAN};
                     border:1px solid rgba(0,201,255,0.4); }}
.stage-INDICATORS {{ background:rgba(0,201,255,0.12); color:{CYAN};
                     border:1px solid rgba(0,201,255,0.4); }}
.stage-RAG        {{ background:rgba(90,167,216,0.12); color:{CYAN};
                     border:1px solid rgba(90,167,216,0.35); }}
.stage-AI         {{ background:rgba(243,156,18,0.14); color:{AMBER};
                     border:1px solid rgba(243,156,18,0.45); }}
.stage-DECISION   {{ background:rgba(0,212,170,0.14); color:{C_BUY};
                     border:1px solid rgba(0,212,170,0.4); }}
.stage-RISK       {{ background:rgba(243,156,18,0.10); color:{AMBER};
                     border:1px solid rgba(243,156,18,0.35); }}
.stage-EXECUTE    {{ background:rgba(0,212,170,0.18); color:{C_BUY};
                     border:1px solid rgba(0,212,170,0.5); }}
.stage-SLEEP      {{ background:#0e1828; color:#5a7a98;
                     border:1px solid #1c2c44; }}
.stage-ERROR      {{ background:rgba(255,71,87,0.14); color:{C_SELL};
                     border:1px solid rgba(255,71,87,0.45); }}
.stage-STOPPED    {{ background:#1a1020; color:#b06a80;
                     border:1px solid #3a1f34; }}

/* Decision action colors — TradingView flat, no neon glow */
.sig-buy  {{ color:{C_BUY}; }}
.sig-sell {{ color:{C_SELL}; }}
.sig-hold {{ color:{C_HOLD}; }}

/* Streamlit tab bar — flat TradingView style */
button[data-baseweb="tab"] {{
    background:transparent !important; color:{TEXT_DIM} !important;
    font-size:0.78rem !important; letter-spacing:.06em;
    text-transform:uppercase;
    border-bottom:2px solid transparent !important;
    padding:0.5rem 0.9rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color:{TEXT_HI} !important;
    border-bottom:2px solid {CYAN} !important;
}}
div[data-baseweb="tab-list"] {{
    border-bottom:1px solid {BORDER} !important;
    gap:0 !important;
}}

/* Event log rows */
.log-row {{ font-family:var(--font-mono);
    font-size:0.72rem; padding:2px 8px;
    border-bottom:1px solid #0c1824; }}
.log-ts  {{ color:#3a5a7a; margin-right:6px; }}
.log-lvl {{ display:inline-block; min-width:72px; font-weight:700; margin-right:6px; }}
.log-msg {{ color:{TEXT}; }}
.log-INFO     .log-lvl {{ color:#5a8aaa; }}
.log-DECISION .log-lvl {{ color:{CYAN}; }}
.log-TRADE    .log-lvl {{ color:{C_BUY}; }}
.log-WARN     .log-lvl {{ color:{AMBER}; }}
.log-ERROR    .log-lvl {{ color:{C_SELL}; }}

/* Hide default toggle label wrap glitch */
div[data-testid="stToggle"] label,
label[data-baseweb="checkbox"] * {{
    white-space:nowrap !important; word-break:keep-all !important;
}}

/* ── Professional polish layer ─────────────────────────────────────── */

/* Panels lift subtly on hover */
.bt-panel {{ transition:border-color .18s ease, box-shadow .18s ease; }}
.bt-panel:hover {{ border-color:{BORDER_HI}; }}

/* Inputs get a cyan focus ring instead of the default */
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {{
    border-color:{CYAN} !important;
    box-shadow:0 0 0 1px rgba(0,183,255,.35) !important;
}}
div[data-testid="stTextArea"] textarea {{
    background:{BG_PANEL} !important; color:{TEXT} !important;
    border-color:{BORDER} !important; font-family:var(--font-mono);
    font-size:0.78rem !important; border-radius:4px;
}}

/* Buttons: micro-lift + primary gradient sheen */
.stButton > button {{ transition:all .16s ease !important; }}
.stButton > button:hover {{ transform:translateY(-1px); }}
.stButton > button:active {{ transform:translateY(0); }}
button[kind="primary"]:hover {{
    filter:brightness(1.15);
    border-color:{C_BUY} !important;
}}

/* Tabs: soft hover before selection */
button[data-baseweb="tab"]:hover {{
    color:{TEXT} !important;
    background:rgba(0,183,255,0.04) !important;
}}

/* Dataframe row hover */
.stDataFrame tbody tr:hover td {{
    background:rgba(0,183,255,0.05) !important;
}}

/* KPI items glow their value on hover */
.kpi-item {{ transition:background .15s ease; border-radius:6px; }}
.kpi-item:hover {{ background:rgba(0,183,255,0.045); }}

/* Headings tracking */
h1,h2,h3,h4,h5 {{ font-family:var(--font-ui) !important;
    letter-spacing:.01em !important; }}

/* Expander styling to match panels */
details[data-testid="stExpander"],
div[data-testid="stExpander"] {{
    background:{BG_PANEL} !important;
    border:1px solid {BORDER} !important; border-radius:10px !important;
}}

::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{BG_DEEP}; }}
::-webkit-scrollbar-thumb {{ background:#1a2d40; border-radius:3px; }}
hr {{ border-color:{BORDER} !important; margin:0.4rem 0 !important; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    # Identity gate. Halts the page with a sign-in screen in `oidc` mode, falls
    # back to the legacy shared-password form in `password` mode, and is a
    # no-op locally. See dashboard/_identity.py.
    from dashboard._identity import render_account_chip, require_login
    require_login()
    render_account_chip()


# ── Per-user profile storage ─────────────────────────────────────────────────
def profile_path(account: str | None = None) -> Path:
    """Where this account's trading profile lives.

    Profiles used to be a single ``data/user_profile.json`` for the whole
    deployment, which meant every visitor overwrote everyone else's capital,
    risk profile and watchlist. They are now one file per account.
    """
    from dashboard._identity import account_slug
    return ROOT / "data" / "profiles" / f"{account_slug(account)}.json"


def _load_profile_for(account: str) -> UserProfile:
    """Load an account's profile, adopting the legacy shared file exactly once.

    An existing single-user deployment keeps its settings on the first sign-in
    after the upgrade instead of silently resetting to defaults. The legacy
    file is then renamed, so the *second* person to sign up starts from
    defaults rather than inheriting a stranger's capital and watchlist —
    leaving it in place would recreate the very leak this change removes.
    """
    path = profile_path(account)
    if path.exists():
        return UserProfile.load(path)

    if LEGACY_PROFILE_PATH.exists():
        adopted = UserProfile.load(LEGACY_PROFILE_PATH)
        adopted.save(path)
        try:
            LEGACY_PROFILE_PATH.rename(
                LEGACY_PROFILE_PATH.with_suffix(".json.migrated"))
        except OSError:
            # Losing the rename only means the next new account inherits it
            # too; not worth failing a page render over.
            from utils.logger import get_logger
            get_logger(__name__).warning(
                "Could not retire legacy profile at %s", LEGACY_PROFILE_PATH)
        return adopted

    return UserProfile.load(path)


#: Session-state keys that belong to one person and must never survive a
#: change of account inside the same browser session.
_USER_SCOPED_KEYS = (
    # The engine itself is NOT here — it lives in trading.registry, keyed by
    # account, and outlives any one session on purpose.
    "portfolio", "_event_log", "bot_logs",
    "starting_capital", "trade_size_pct", "risk_profile", "watchlist",
    "daily_target_pct", "daily_loss_limit_pct",
    "ticker_sel", "ticker_custom", "strategy_mode", "interval_sec",
)


def _reset_user_scoped_state() -> None:
    """Wipe one person's session state when a different account signs in.

    Signing out and back in as someone else in the same browser must not hand
    over the previous person's portfolio, watchlist or API key.

    It deliberately does *not* stop the previous account's bot. That engine is
    owned by :mod:`trading.registry` under its own account id, not by this
    session — it is still reachable, still checkpointing to its owner's
    portfolio file, and stopping it here would halt a stranger's trading just
    because someone else signed in on their laptop. Stopping a bot is what the
    STOP button is for.
    """
    for key in (*_USER_SCOPED_KEYS, _USER_KEY_SLOT):
        st.session_state.pop(key, None)
    st.session_state.pop("_profile_loaded", None)
    st.session_state.pop("_engine_capacity_error", None)


# ── Session state bootstrap ──────────────────────────────────────────────────
def ensure_profile_in_session() -> None:
    """Load the current account's profile into session state.

    Reloads whenever the signed-in account changes, so signing out and back in
    as someone else inside one browser session cannot carry the previous
    person's capital or watchlist across.
    """
    from dashboard._identity import account_id as _account_id
    account = _account_id()
    previous = st.session_state.get("_profile_account")
    if previous == account:
        return
    if previous is not None:
        _reset_user_scoped_state()

    p = _load_profile_for(account)
    st.session_state["starting_capital"]     = int(p.capital)
    st.session_state["trade_size_pct"]       = int(p.trade_size_pct)
    st.session_state["risk_profile"]         = p.risk_profile
    st.session_state["watchlist"]            = list(p.watchlist)
    st.session_state["daily_target_pct"]     = float(p.daily_target_pct)
    st.session_state["daily_loss_limit_pct"] = float(p.daily_loss_limit_pct)
    st.session_state["_profile_account"]     = account
    st.session_state["_profile_loaded"]      = True


def save_profile() -> None:
    from dashboard._identity import account_id as _account_id
    account = _account_id()
    existing = _load_profile_for(account)
    UserProfile(
        capital=float(st.session_state.get("starting_capital", 10_000)),
        trade_size_pct=int(st.session_state.get("trade_size_pct", 20)),
        risk_profile=st.session_state.get("risk_profile", "Balanced"),
        watchlist=list(st.session_state.get("watchlist") or DEFAULT_TICKERS),
        daily_target_pct=float(
            st.session_state.get("daily_target_pct", existing.daily_target_pct)
        ),
        daily_loss_limit_pct=float(
            st.session_state.get("daily_loss_limit_pct", existing.daily_loss_limit_pct)
        ),
    ).save(profile_path(account))


@st.cache_data(ttl=900, show_spinner=False)
def validate_ticker_symbol(ticker: str) -> tuple[bool, str]:
    """Return whether Yahoo Finance recognizes the ticker."""
    symbol = ticker.upper().strip()
    if not symbol:
        return False, "Ticker is empty."
    if not _TICKER_RE.match(symbol):
        return False, "Ticker contains unsupported characters."

    try:
        import yfinance as yf
        cache_dir = ROOT / "data" / "yfinance_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
    except Exception as exc:
        return False, f"Could not verify ticker: {exc}"

    if hist is None or hist.empty:
        return False, "No market data was found for this ticker."
    if "Close" not in hist.columns or hist["Close"].dropna().empty:
        return False, "Ticker data has no valid closing prices."
    return True, "Ticker verified."


# ── Per-user portfolio persistence ───────────────────────────────────────────
def portfolio_path(account: str | None = None) -> Path:
    """Where this account's virtual portfolio is stored."""
    from dashboard._identity import account_slug
    return ROOT / "data" / "portfolios" / f"{account_slug(account)}.json"


def save_portfolio(portfolio=None, account: str | None = None) -> None:
    """Write the portfolio to disk. Safe to call from any thread.

    ``LivePortfolio.save`` writes temp-then-rename under its own lock, so a
    save racing a trade cannot produce a half-written file.
    """
    from dashboard._identity import account_id as _account_id
    port = portfolio if portfolio is not None else st.session_state.get("portfolio")
    if port is None:
        return
    port.save(portfolio_path(account or _account_id()))


def ensure_portfolio_in_session() -> None:
    """Put this account's portfolio in session state, restoring it from disk.

    Before this, the portfolio lived only in session state, so a refresh wiped
    every open position and the whole trade history while the background
    engine kept trading against a portfolio object nobody could see any more.
    """
    if st.session_state.get("portfolio") is not None:
        return

    from dashboard._identity import account_id as _account_id
    account = _account_id()
    path = portfolio_path(account)
    if path.exists():
        try:
            st.session_state["portfolio"] = LivePortfolio.load(path)
            return
        except Exception as exc:                               # noqa: BLE001
            # A corrupt or future-schema file must not lock the user out of
            # their own dashboard. Keep it for forensics, start clean.
            from utils.logger import get_logger
            get_logger(__name__).warning(
                "Could not load portfolio %s (%s) — starting a fresh one",
                path, exc)
            try:
                path.rename(path.with_suffix(".json.corrupt"))
            except OSError:
                pass

    st.session_state["portfolio"] = LivePortfolio(
        initial_capital=float(
            st.session_state.get("starting_capital", 10_000)),
    )


# ── Pipeline singleton (shared across pages) ────────────────────────────────
@st.cache_resource(show_spinner="Warming up trading pipeline. First run loads the embedding model.")
def get_pipeline():
    from decision_engine.ai_engine import AITradingEngine
    from market_data.fetcher import MarketDataFetcher
    from rag.retriever import StrategyRetriever
    fetcher   = MarketDataFetcher()
    retriever = StrategyRetriever()
    engine    = AITradingEngine()
    # Eagerly open the Chroma collection + load the sentence-transformer
    # weights so the FIRST trading cycle doesn't block for 5-10s.
    try:
        col = retriever._get_collection()  # noqa: SLF001 — intentional warm-up
        # Force embedding model weights into memory with a tiny query
        col.query(query_texts=["warmup"], n_results=1)
    except Exception:
        pass
    return fetcher, retriever, engine


# ── Tenancy ──────────────────────────────────────────────────────────────────
#: Where a user's own Anthropic key lives for the duration of their session.
#: Session state only — never written to disk, never logged, never rendered
#: unmasked. See ``saas.keyvault``.
_USER_KEY_SLOT = "_bt_user_api_key"


def account_id() -> str:
    """The current visitor's stable account key.

    Resolved by :mod:`dashboard._identity`: a real per-person id when OIDC
    sign-in is configured, a single shared id behind the legacy password gate,
    ``local`` otherwise. Everything per-user — trial budget, profile, ledger
    attribution — keys off this one string.
    """
    from dashboard._identity import account_id as _account_id
    return _account_id()


def get_user_api_key() -> str:
    return st.session_state.get(_USER_KEY_SLOT, "") or ""


def set_user_api_key(key: str) -> None:
    """Store (or clear) the visitor's own key for this session only."""
    from saas import keyvault
    st.session_state[_USER_KEY_SLOT] = keyvault.normalise(key)
    # The engine cache is keyed by API key, so a rotation must not keep
    # serving from an engine bound to the old one.
    eng = current_engine()
    if eng is not None:
        eng.set_tenant(get_tenant())


def get_tenant():
    """The current visitor's commercial context.

    Rebuilt on every rerun — cheap by design; the ledger and engine cache
    behind it are process-wide singletons.
    """
    from saas.tenant import Tenant
    from dashboard._identity import account_slug
    return Tenant(
        account_id=account_id(),
        user_api_key=get_user_api_key(),
        model=None,
        # Same slug the Knowledge page stamps onto ingested chunks, so
        # retrieval matches what this account actually ingested.
        knowledge_owner=account_slug(),
    )


# ── Live engine, owned by the process registry ───────────────────────────────
def get_live_engine():
    """This account's background engine, surviving refreshes and new tabs.

    Ownership lives in :mod:`trading.registry`, not in session state. A
    refresh reattaches to the bot that is already trading instead of dropping
    the only reference to it and starting a second one on the same portfolio.

    Returns ``None`` when the process is at its engine cap; callers should
    surface :func:`engine_capacity_message` rather than crash.
    """
    from trading.registry import RegistryFullError, get_registry

    ensure_portfolio_in_session()
    account = account_id()
    registry = get_registry()

    def _build():
        from trading.live_engine import LiveTradingEngine
        fetcher, retriever, engine = get_pipeline()
        eng = LiveTradingEngine(
            portfolio=st.session_state["portfolio"],
            fetcher=fetcher, retriever=retriever, engine=engine,
            tenant=get_tenant(),
        )
        eng.set_persist_callback(
            lambda port, _acct=account: save_portfolio(port, account=_acct))
        return eng

    try:
        eng = registry.get_or_create(account, _build)
    except RegistryFullError as exc:
        st.session_state["_engine_capacity_error"] = str(exc)
        return None

    st.session_state.pop("_engine_capacity_error", None)
    # The engine outlives the session, so the session must adopt the engine's
    # portfolio rather than the other way round — otherwise the page would
    # render a fresh empty portfolio while the bot trades a different object.
    st.session_state["portfolio"] = eng.portfolio
    # Entitlements move under a running bot (budget spent, key added), so
    # refresh the tenant on every rerun rather than only at construction.
    eng._tenant = get_tenant()              # noqa: SLF001 — same package
    return eng


def engine_capacity_message() -> str:
    """Why :func:`get_live_engine` returned ``None``, if it did."""
    return st.session_state.get("_engine_capacity_error", "")


def current_engine():
    """This account's engine if it already exists, without building one.

    For callers that only want to nudge a running bot (push a config change,
    drain events) and must not spin one up as a side effect.
    """
    from trading.registry import get_registry
    return get_registry().get(account_id())


def ensure_event_buffer() -> None:
    st.session_state.setdefault("_event_log", [])


# Back-compat alias for older pages that used bot_logs
def ensure_logs_in_session() -> None:
    st.session_state.setdefault("bot_logs", [])
    ensure_event_buffer()


def pump_events() -> None:
    """Drain the engine's event queue into the page-local log buffer."""
    ensure_event_buffer()
    eng = current_engine()
    if eng is None:
        return
    new = eng.drain_events()
    if not new:
        return
    buf = st.session_state["_event_log"]
    buf.extend(new)
    if len(buf) > 600:
        del buf[:len(buf) - 600]


def pump_toasts() -> None:
    """Drain the global notifications toast queue and surface each one as
    ``st.toast``. Safe to call on every page render — the queue is
    process-wide so any rerun anywhere picks up the latest events.
    """
    try:
        from notifications.dispatcher import drain_toast_queue
    except Exception:
        return
    items = drain_toast_queue()
    if not items:
        return
    icon_map = {
        "TRADE":    "💸",
        "ERROR":    "⛔",
        "RISK":     "🛡",
        "DECISION": "🤖",
        "REFLECT":  "📘",
        "TEST":     "🔔",
    }
    for it in items:
        cat = it.get("category", "")
        msg = it.get("message", "")
        ticker = it.get("ticker", "")
        prefix = f"{ticker} · " if ticker else ""
        try:
            st.toast(f"{prefix}{msg}", icon=icon_map.get(cat, "🔔"))
        except Exception:
            pass
