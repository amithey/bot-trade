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

# Re-export tokens for existing pages; theme is independent of account state.
from dashboard.theme import (
    _THEME_CSS,
    AMBER,
    BG,
    BG_DEEP,
    BG_PANEL,
    BG_RAISED,
    BORDER,
    BORDER_HI,
    C_BUY,
    C_HOLD,
    C_SELL,
    CYAN,
    GRID,
    TEXT,
    TEXT_DIM,
    TEXT_HI,
)

# Explicit public exports keep Ruff from removing the page-facing theme API.
__all__ = [
    'AMBER',
    'BG',
    'BG_DEEP',
    'BG_PANEL',
    'BG_RAISED',
    'BORDER',
    'BORDER_HI',
    'CUSTOM_LABEL',
    'CYAN',
    'C_BUY',
    'C_HOLD',
    'C_SELL',
    'DEFAULT_TICKERS',
    'GRID',
    'LEGACY_PROFILE_PATH',
    'RISK_CHOICES',
    'ROOT',
    'TEXT',
    'TEXT_DIM',
    'TEXT_HI',
    '_THEME_CSS',
    'account_id',
    'current_engine',
    'engine_capacity_message',
    'ensure_event_buffer',
    'ensure_logs_in_session',
    'ensure_portfolio_in_session',
    'ensure_profile_in_session',
    'get_live_engine',
    'get_pipeline',
    'get_tenant',
    'get_user_api_key',
    'portfolio_path',
    'profile_path',
    'pump_events',
    'pump_toasts',
    'save_portfolio',
    'save_profile',
    'secure_page',
    'set_user_api_key',
    'validate_ticker_symbol',
]



def secure_page() -> None:
    """Enforce access, then style the page. Every page must call this first.

    This is the *only* thing standing between an unauthenticated visitor and a
    page's contents. Streamlit runs each file in pages/ independently, so there
    is no central router to gate — the guarantee is simply that every page
    calls this before it touches data.

    It was called `apply_theme`, which described the cosmetic half and hid the
    half that matters. The risk with that name was specific: a future page that
    wanted data but not styling would have had no reason to call it, and would
    have served an unauthenticated visitor while looking perfectly correct. The
    name now says what skipping it costs.

    Theme CSS is applied *before* the gate, not after. `require_login()` halts
    the script with `st.stop()` when nobody is signed in, and the earlier
    order meant the sign-in screen itself never received the main theme —
    it rendered against bare Streamlit white with only its own small
    stylesheet layered on top, which is why it read as empty rather than as
    part of the same product. Injecting a `<style>` tag carries no data, so
    moving it ahead of the gate costs nothing security-wise while fixing that.

    Halts with a sign-in screen in `oidc` mode, falls back to the shared
    password form in `password` mode, and is a no-op locally.
    See dashboard/_identity.py.
    """
    from dashboard.theme import register_chart_theme
    register_chart_theme()
    # Apply the complete, account-aware theme before rendering authentication
    # or page content. Injecting the base CSS first and the saved appearance
    # later caused a visible old-theme frame on every Streamlit rerun.
    from dashboard.appearance import apply_appearance
    apply_appearance()
    from dashboard._identity import require_login
    require_login()


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
    from dashboard._identity import account_slug
    from saas.tenant import Tenant
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
