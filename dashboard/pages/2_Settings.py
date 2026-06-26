"""Settings page — user profile, watchlist management, API key status."""
from __future__ import annotations

import os
import streamlit as st

# ── sys.path bootstrap (so `from dashboard...` works in direct `streamlit run`)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ────────────────────────────────────────────────────────────────────────────

from dashboard._shared import (
    DEFAULT_TICKERS, RISK_CHOICES,
    apply_theme, ensure_profile_in_session, save_profile,
    validate_ticker_symbol,
)

st.set_page_config(page_title="BotTrade - Settings", page_icon=":material/settings:", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()
ensure_profile_in_session()

st.markdown('<div class="page-title">SETTINGS</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Trading profile, watchlist, and API status</div>', unsafe_allow_html=True)

# ── API key status ──────────────────────────────────────────────────────────
raw_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
key_ok  = bool(raw_key) and raw_key.startswith("sk-ant-") and "PASTE" not in raw_key.upper()
if key_ok:
    st.markdown('<span class="badge badge-green">ANTHROPIC_API_KEY loaded</span>',
                unsafe_allow_html=True)
else:
    st.error("ANTHROPIC_API_KEY missing or invalid. Edit .env at the project root.")

st.markdown("---")

# ── Trading profile ─────────────────────────────────────────────────────────
st.markdown("##### Trading Profile")
c1, c2, c3 = st.columns(3)

with c1:
    prev_cap = int(st.session_state["starting_capital"])
    cap = st.number_input(
        "Starting Capital ($)", min_value=1000, max_value=10_000_000,
        value=prev_cap, step=1000,
        help="The virtual account's initial balance.",
    )
    st.session_state["starting_capital"] = cap

with c2:
    prev_ts = int(st.session_state["trade_size_pct"])
    ts = st.slider(
        "Trade Size (% of equity)", 5, 100, prev_ts, step=5,
        help="Percentage of total equity committed per BUY.",
    )
    st.session_state["trade_size_pct"] = ts

with c3:
    prev_risk = st.session_state.get("risk_profile", "Balanced")
    risk_idx = RISK_CHOICES.index(prev_risk) if prev_risk in RISK_CHOICES else 1
    risk = st.selectbox(
        "Risk Profile", RISK_CHOICES, index=risk_idx,
        help="Conservative = HOLD bias, Aggressive = enters early on divergence",
    )
    st.session_state["risk_profile"] = risk

if (cap, ts, risk) != (prev_cap, prev_ts, prev_risk):
    save_profile()
    st.success("Profile saved.")

with st.expander("What does each risk profile actually do?"):
    st.markdown("""
- **Conservative** — 5–15% per trade, SL 1.5%, TP 2.5%, confidence threshold 0.65.
  Prefers HOLD on ambiguity. No pyramiding.
- **Balanced** — 15–30% per trade, SL 2.5%, TP 5%, confidence threshold 0.55.
  Standard playbook, 2+ confirming signals.
- **Aggressive** — 25–60% per trade, SL 4%, TP 10%, confidence threshold 0.45.
  Enters on early/anticipatory signals, allows pyramiding into winners, counter-trend OK.
    """)

st.markdown("---")

# ── Daily goals & safety ─────────────────────────────────────────────────────
st.markdown("##### Daily Goals & Safety")
st.caption("Automatically halt the bot when a daily target or loss limit is hit. "
           "Set target to 0 to disable — bot will trade all day.")

prev_target = float(st.session_state.get("daily_target_pct", 0.0))
prev_loss   = float(st.session_state.get("daily_loss_limit_pct", 5.0))

cg1, cg2, cg3 = st.columns(3)
with cg1:
    target_on = st.toggle(
        "Daily profit target",
        value=prev_target > 0.0,
        help="When enabled, the bot closes all positions and halts once the "
             "daily P&L reaches this percentage.",
    )
with cg2:
    target_val = st.number_input(
        "Target (%)", min_value=0.1, max_value=20.0,
        value=max(prev_target, 1.5), step=0.1,
        disabled=not target_on,
        help="Example: 1.5 = stop after +1.5% daily P&L",
    )
with cg3:
    loss_limit = st.number_input(
        "Daily loss limit (%)", min_value=0.5, max_value=50.0,
        value=prev_loss, step=0.5,
        help="Force-halt trading if daily P&L drops below this level",
    )

new_target = float(target_val) if target_on else 0.0
if (new_target != prev_target) or (loss_limit != prev_loss):
    st.session_state["daily_target_pct"]     = new_target
    st.session_state["daily_loss_limit_pct"] = float(loss_limit)
    # Persist to profile JSON
    from config.user_profile import UserProfile
    from dashboard._shared import PROFILE_PATH
    prof = UserProfile.load(PROFILE_PATH)
    prof.daily_target_pct     = new_target
    prof.daily_loss_limit_pct = float(loss_limit)
    prof.save(PROFILE_PATH)
    # Also push live to a running engine
    eng = st.session_state.get("_live_engine")
    if eng is not None:
        eng.set_config(daily_target_pct=new_target,
                       daily_loss_limit_pct=float(loss_limit))
    st.success("Goals saved.")

st.markdown("---")

# ── Watchlist manager ───────────────────────────────────────────────────────
st.markdown("##### Watchlist")
st.caption("Tickers that show up in the Live Trading ticker dropdown.")

watchlist: list[str] = list(st.session_state.get("watchlist") or DEFAULT_TICKERS)

col_a, col_b = st.columns([2, 1])

with col_a:
    if watchlist:
        for idx, tk in enumerate(watchlist):
            row_a, row_b = st.columns([5, 1])
            row_a.markdown(f'<div style="padding:6px 10px;background:#0c1420;border:1px solid #182840;'
                           f'border-radius:4px;font-family:monospace;color:#c8ddf0;">{tk}</div>',
                           unsafe_allow_html=True)
            if row_b.button("Remove", key=f"rm_{idx}_{tk}", help=f"Remove {tk}"):
                watchlist.pop(idx)
                st.session_state["watchlist"] = watchlist
                save_profile()
                st.rerun()
    else:
        st.caption("Watchlist empty — add some tickers.")

with col_b:
    new_tk = st.text_input("Add ticker", placeholder="e.g. SOL-USD",
                           key="new_ticker_input").upper().strip()
    if st.button("Add", use_container_width=True) and new_tk:
        ok, msg = validate_ticker_symbol(new_tk)
        if not ok:
            st.error(f"Cannot add {new_tk}: {msg}")
        elif new_tk not in watchlist:
            watchlist.append(new_tk)
            st.session_state["watchlist"] = watchlist
            save_profile()
            st.success(f"Added {new_tk}")
            st.rerun()
        else:
            st.warning(f"{new_tk} is already on the list.")

    if st.button("Reset to defaults", use_container_width=True):
        st.session_state["watchlist"] = list(DEFAULT_TICKERS)
        save_profile()
        st.rerun()

st.markdown("---")

# ── Notifications ──────────────────────────────────────────────────────────
st.markdown("##### 🔔 Notifications")
st.caption(
    "Get pinged when the bot does something noteworthy — trade execution, "
    "circuit breaker, errors. Telegram / generic webhook / in-app toast."
)

from notifications import NotificationConfig, NotificationDispatcher
_ncfg = NotificationConfig.load()

ncol_a, ncol_b = st.columns(2)

with ncol_a:
    st.markdown("**Telegram**")
    tg_on = st.toggle(
        "Enable Telegram",
        value=_ncfg.telegram_enabled,
        key="ntf_tg_on",
        help="Get a chat message on every event you opted in for.",
    )
    tg_token = st.text_input(
        "Bot token",
        value=_ncfg.telegram_bot_token,
        type="password",
        key="ntf_tg_token",
        help="Create a bot via @BotFather, then paste the token here.",
    )
    tg_chat = st.text_input(
        "Chat ID",
        value=_ncfg.telegram_chat_id,
        key="ntf_tg_chat",
        help="Your numeric chat ID. Send /start to your bot, then visit "
             "api.telegram.org/bot<TOKEN>/getUpdates to see it.",
    )

with ncol_b:
    st.markdown("**Generic webhook**")
    wh_on = st.toggle(
        "Enable Webhook",
        value=_ncfg.webhook_enabled,
        key="ntf_wh_on",
        help="POST a JSON payload to any URL on every event. Works with "
             "Discord, Slack incoming webhooks, Zapier, Pipedream, etc.",
    )
    wh_url = st.text_input(
        "Webhook URL",
        value=_ncfg.webhook_url,
        key="ntf_wh_url",
        placeholder="https://hooks.slack.com/services/...",
    )
    st.markdown("**In-app**")
    toast_on = st.toggle(
        "Enable toast pop-ups (this browser)",
        value=_ncfg.toast_enabled,
        key="ntf_toast_on",
    )

st.markdown("**What to notify on**")
fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
with fcol1:
    f_trade = st.toggle("Trades", value=_ncfg.notify_trade,
                        key="ntf_f_trade")
with fcol2:
    f_error = st.toggle("Errors", value=_ncfg.notify_error,
                        key="ntf_f_error")
with fcol3:
    f_risk  = st.toggle("Risk events", value=_ncfg.notify_risk,
                        key="ntf_f_risk")
with fcol4:
    f_dec   = st.toggle("Decisions", value=_ncfg.notify_decision,
                        key="ntf_f_dec",
                        help="Every Claude call — chatty.")
with fcol5:
    f_refl  = st.toggle("Reflections", value=_ncfg.notify_reflect,
                        key="ntf_f_refl",
                        help="Post-trade lessons.")

new_cfg = NotificationConfig(
    telegram_enabled=bool(tg_on),
    telegram_bot_token=tg_token.strip(),
    telegram_chat_id=tg_chat.strip(),
    webhook_enabled=bool(wh_on),
    webhook_url=wh_url.strip(),
    toast_enabled=bool(toast_on),
    notify_trade=bool(f_trade),
    notify_error=bool(f_error),
    notify_risk=bool(f_risk),
    notify_decision=bool(f_dec),
    notify_reflect=bool(f_refl),
)

# Save + push to live engine on any change
if new_cfg.to_dict() != _ncfg.to_dict():
    new_cfg.save()
    eng = st.session_state.get("_live_engine")
    if eng is not None:
        try:
            eng.update_notifier_config(new_cfg)
        except Exception as exc:
            st.warning(f"Engine push failed: {exc}")
    st.success("Notification settings saved.")

# Test buttons
tcol1, tcol2, tcol3 = st.columns([1, 1, 3])
with tcol1:
    if st.button("Test Telegram", use_container_width=True,
                 disabled=not (tg_on and tg_token and tg_chat)):
        ok, msg = NotificationDispatcher(new_cfg).test_telegram()
        (st.success if ok else st.error)(msg)
with tcol2:
    if st.button("Test Webhook", use_container_width=True,
                 disabled=not (wh_on and wh_url)):
        ok, msg = NotificationDispatcher(new_cfg).test_webhook()
        (st.success if ok else st.error)(msg)
with tcol3:
    if st.button("Test toast", use_container_width=True):
        st.toast("BotTrade test toast 🔔", icon="🔔")
