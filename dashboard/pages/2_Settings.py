"""Settings page — user profile, watchlist management, API key status."""
from __future__ import annotations

import json
import os
import streamlit as st
import streamlit.components.v1 as components

# ── sys.path bootstrap (so `from dashboard...` works in direct `streamlit run`)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ────────────────────────────────────────────────────────────────────────────

from dashboard._identity import account_id, auth_mode, current_user, identifies_individuals
from dashboard._shared import (
    DEFAULT_TICKERS, RISK_CHOICES,
    secure_page, current_engine, ensure_profile_in_session, get_tenant,
    get_user_api_key,
    save_profile, set_user_api_key, validate_ticker_symbol,
)
from saas import billing, keyvault
from saas.ledger import get_ledger
from saas.plans import PLANS, Funding
from saas.pricing import format_usd
from config.settings import settings as _settings

st.set_page_config(page_title="BotTrade - Settings", page_icon=":material/settings:", layout="wide",
                   initial_sidebar_state="expanded")
secure_page()
ensure_profile_in_session()

st.markdown('<div class="page-title">SETTINGS</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Plan, API key, trading profile, and watchlist</div>', unsafe_allow_html=True)

# ── Plan + API key ──────────────────────────────────────────────────────────
tenant = get_tenant()
_ledger = get_ledger()
_billing_account = tenant.billing_account_id

# Returning from a Paddle Checkout overlay: the success URL carries
# ?paddle_return=1. Unlike Stripe's redirect, Paddle hands back no
# transaction id to look up — the customer id was already known before
# checkout opened (it's what the overlay was given to begin with), so a
# forced reconciliation finds the new subscription directly. See the
# "no confirm_checkout_session" note in saas/billing.py's module docstring.
_returning_from_checkout = st.query_params.get("paddle_return")
if _returning_from_checkout:
    st.query_params.clear()
    _resolved_plan = billing.sync_subscription_status(
        _ledger, _billing_account, force=True)
    if _resolved_plan != "FREE":
        st.success(f"Payment confirmed — you're now on the "
                   f"{_resolved_plan.title()} plan.")
    else:
        st.info("If you completed payment, it can take a moment to show up "
                "here — use the refresh button below, or reload the page.")
else:
    # Cheap on every load: sync_subscription_status only re-checks Paddle
    # once per BOTTRADE_SYNC_TTL; this is what notices a cancellation or a
    # failed renewal that happened entirely on Paddle's side.
    billing.sync_subscription_status(_ledger, _billing_account)

ent = tenant.entitlement

st.markdown("##### Your Plan & API Key")

pc1, pc2, pc3 = st.columns([1.1, 1, 1.4])
with pc1:
    st.metric("Plan", ent.plan.name,
              help=ent.plan.tagline)
with pc2:
    funding_label = {
        Funding.BYOK:     "Your own key",
        Funding.PLATFORM: "Trial credit",
        Funding.NONE:     "Not funded",
    }[ent.funding]
    st.metric("Tokens billed to", funding_label)
with pc3:
    if ent.funding is Funding.PLATFORM:
        st.metric("Trial credit left",
                  format_usd(ent.platform_budget_remaining_usd),
                  help="Spent on the shared key. Add your own key for unlimited use.")
    else:
        st.metric("Modes unlocked", ", ".join(sorted(ent.allowed_modes)))

if ent.lock_reason:
    st.warning(ent.lock_reason)

st.caption(
    "COMMITTEE mode — 38 technical indicators voting every bar — makes no API "
    "calls and is always available, on every plan, at no cost. Everything that "
    "calls Claude (AI, HYBRID, BOARDROOM) runs on the key below, so those "
    "tokens bill to your own Anthropic account, not ours."
)

kc1, kc2 = st.columns([2.4, 1])
with kc1:
    current = get_user_api_key()
    entered = st.text_input(
        "Anthropic API key",
        value="",
        type="password",
        placeholder=(keyvault.mask(current) if current
                     else "sk-ant-api03-…  (kept in this session only)"),
        help="Get one at console.anthropic.com → API Keys. Stored in your "
             "browser session only — never written to disk, never logged, "
             "never shown back to you in full.",
    )
with kc2:
    st.markdown('<div style="height:1.85rem"></div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    save_key = b1.button("Save", use_container_width=True, type="primary")
    clear_key = b2.button("Remove", use_container_width=True,
                          disabled=not current)

if save_key:
    ok, msg = keyvault.validate_format(entered)
    if not ok:
        st.error(msg)
    else:
        with st.spinner("Verifying key with Anthropic…"):
            live_ok, live_msg = keyvault.verify_live(entered)
        if live_ok:
            set_user_api_key(entered)
            st.success(f"{live_msg} Key saved for this session.")
            st.rerun()
        else:
            st.error(live_msg)

if clear_key:
    set_user_api_key("")
    st.info("Key removed. AI modes are locked until you add one again.")
    st.rerun()

if current:
    st.markdown(
        f'<span class="badge badge-green">Key active — {keyvault.mask(current)}'
        f'</span>', unsafe_allow_html=True)

with st.expander("Compare plans", expanded=not billing.billing_enabled()):
    for plan in PLANS.values():
        price = "Free" if plan.price_usd_month == 0 else \
                f"${plan.price_usd_month:.0f}/mo"
        marker = "  ← current" if plan.id == ent.plan.id else ""
        st.markdown(f"**{plan.name} — {price}**{marker}  \n_{plan.tagline}_")
        for line in plan.highlights:
            st.markdown(f"- {line}")
        st.markdown("")

    if not billing.billing_enabled():
        st.caption(
            "Billing isn't configured on this deployment yet, so plans can't "
            "be purchased here — see DEPLOY.md."
        )

# ── Upgrade / manage subscription ────────────────────────────────────────────
if billing.billing_enabled():
    st.markdown("##### Upgrade or manage your subscription")

    _base = _settings.bottrade_base_url.rstrip("/")
    _success_url = f"{_base}/Settings?paddle_return=1"
    _user_email = (current_user().get("email") or "").strip()

    # Someone who clicked a specific plan on the landing page arrived here via
    # app.py, which parked their choice in session state. Say so, rather than
    # dropping them into a generic list and making them find it again.
    _pending = st.session_state.pop("_bt_pending_plan", None)
    if _pending and _pending in PLANS:
        if _pending == ent.plan.id:
            st.success(f"You're already on {PLANS[_pending].name}.", icon="✅")
        else:
            st.info(f"Continue to **{PLANS[_pending].name}** below.", icon="🛒")

    _purchasable = [p for p in billing.purchasable_plans() if p != ent.plan.id]
    _has_customer = bool(_ledger.get_paddle_customer_id(_billing_account))
    _slots = len(_purchasable) + (1 if _has_customer else 0)

    if not _user_email and _purchasable:
        st.caption(
            "We don't have an email on file for you yet, so checkout can't "
            "open — sign in with an account that has one, or contact support."
        )

    if _slots:
        cols = st.columns(_slots)

        for col, plan_id in zip(cols, _purchasable):
            plan = PLANS[plan_id]
            with col:
                st.markdown(f"**{plan.name}** — ${plan.price_usd_month:.0f}/mo")
                st.caption(plan.tagline)
                # Built fresh on every render, same reasoning as the old
                # Stripe Checkout Session: this is a low-traffic settings
                # page, and simplicity here beats caching a cheap call.
                cfg = billing.checkout_config(
                    _ledger, _billing_account, plan_id,
                    success_url=_success_url, email=_user_email or None,
                )
                if cfg:
                    _btn_id = f"paddle-checkout-btn-{plan_id}"
                    # json.dumps, not an f-string interpolation of the dict:
                    # these become literal JS source, so they must be valid
                    # JS/JSON syntax (double-quoted), not a Python repr.
                    _items_js = json.dumps(
                        [{"priceId": cfg["price_id"], "quantity": 1}])
                    _customer_js = json.dumps({"id": cfg["customer_id"]})
                    _custom_data_js = json.dumps(cfg["custom_data"])
                    _success_url_js = json.dumps(cfg["success_url"])
                    _token_js = json.dumps(cfg["client_token"])
                    _env_js = json.dumps(cfg["environment"])
                    components.html(f"""
                        <script src="https://cdn.paddle.com/paddle/v2/paddle.js"></script>
                        <style>
                          body {{ margin:0; font-family:inherit; }}
                          button#{_btn_id} {{
                            width:100%; padding:0.5rem 1rem; border-radius:8px;
                            border:1px solid transparent; background:#ff4b4b;
                            color:#fff; font-weight:600; font-size:1rem;
                            cursor:pointer;
                          }}
                          button#{_btn_id}:hover {{ background:#e63e3e; }}
                        </style>
                        <button id="{_btn_id}">Upgrade to {plan.name}</button>
                        <script>
                          Paddle.Environment.set({_env_js});
                          Paddle.Initialize({{ token: {_token_js} }});
                          document.getElementById("{_btn_id}").addEventListener("click", function () {{
                            Paddle.Checkout.open({{
                              items: {_items_js},
                              customer: {_customer_js},
                              customData: {_custom_data_js},
                              settings: {{ successUrl: {_success_url_js} }}
                            }});
                          }});
                        </script>
                    """, height=56)
                else:
                    st.button(f"Upgrade to {plan.name}", disabled=True,
                             use_container_width=True,
                             help="Checkout is temporarily unavailable — try again shortly.")

        if _has_customer:
            with cols[-1]:
                st.markdown("**Manage subscription**")
                st.caption("Cancel, update your payment method, or view invoices.")
                portal_url = billing.create_portal_session(_ledger, _billing_account)
                if portal_url:
                    st.link_button("Open billing portal", portal_url,
                                  use_container_width=True)
    else:
        st.caption("You're on the highest available plan.")

    if _has_customer and st.button("Refresh subscription status"):
        billing.sync_subscription_status(_ledger, _billing_account, force=True)
        st.rerun()

# ── Host notes — only relevant to whoever is running this instance ──────────
_platform_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if not _platform_key:
    st.caption(
        "Host note: no `ANTHROPIC_API_KEY` in `.env`, so trial credit is "
        "unavailable on this deployment and every user must bring their own key."
    )
elif not identifies_individuals():
    st.warning(
        f"**Host note — trial credit is not enforceable in `{auth_mode()}` mode.** "
        f"Without per-person sign-in every visitor shares the account "
        f"`{account_id()}`, so the free budget is one pool for the whole "
        f"deployment rather than per user. Configure `[auth]` in "
        f"`.streamlit/secrets.toml` (see DEPLOY.md) before opening the free "
        f"tier to the public, or set `BOTTRADE_FREE_BUDGET_USD=0`.",
        icon="🔓",
    )

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
    # Write the whole profile from session state to this account's own file.
    # Session state is the source of truth here; re-reading the file first
    # would reset capital and watchlist to defaults on a brand-new account.
    save_profile()
    # Also push live to a running engine
    eng = current_engine()
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
    eng = current_engine()
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
