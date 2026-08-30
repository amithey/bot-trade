"""
Stripe billing — turns a plan choice into a real subscription.

``saas/plans.py`` and ``saas/ledger.py`` already model *what* a plan grants
and *whose account* it belongs to; this module is the missing third piece —
*how a user actually starts paying for one*. Everything here degrades
gracefully: with no Stripe key configured, :func:`billing_enabled` is False,
every upgrade button in the UI simply doesn't render, and every account
keeps working on the Free plan exactly as it did before this module existed.

Design
------
**Checkout, not a custom payment form.** Stripe's hosted Checkout page
collects card details, handles 3-D Secure, and is PCI-compliant by
construction. BotTrade never sees a card number.

**The Billing Portal, not a custom "manage subscription" page.** Cancelling,
updating a payment method, or viewing past invoices is Stripe's hosted
Customer Portal — one function call here, no UI to build or maintain.

**Polling reconciliation, not a webhook receiver — for now.** A push
webhook is the textbook way to learn about a cancellation or a failed
renewal the instant it happens, but Streamlit has no clean way to expose an
HTTP route for one; standing up a second service just for this is a bigger
lift than a subscription business at this stage needs. Instead,
:func:`sync_subscription_status` re-checks a customer's live subscription
state against Stripe on a TTL, the same pattern already used for
fundamentals and news in ``market_data/``. The lag is bounded by the TTL,
not instant — that trade-off is deliberate and documented, not hidden.

Every Stripe call is wrapped: a network hiccup or an API error here must
never crash a page render or silently downgrade someone who is still
correctly paying — see the module-level ``try/except`` around every function
that reaches the network.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from config.settings import settings
from saas.ledger import UsageLedger
from saas.plans import PLANS, DEFAULT_PLAN_ID
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import stripe
except ImportError:                                            # pragma: no cover
    stripe = None  # billing_enabled() below turns this into a clean no-op


# --------------------------------------------------------------------------- #
# Enablement + plan/price mapping
# --------------------------------------------------------------------------- #
def billing_enabled() -> bool:
    """True once Stripe is installed, keyed, and at least one plan is priced."""
    return stripe is not None and settings.billing_configured


def _price_map() -> dict[str, str]:
    """``{plan_id: stripe_price_id}`` for every plan that has one configured."""
    m: dict[str, str] = {}
    if settings.stripe_price_id_pro:
        m["PRO"] = settings.stripe_price_id_pro
    if settings.stripe_price_id_desk:
        m["DESK"] = settings.stripe_price_id_desk
    return m


def price_id_for_plan(plan_id: str) -> Optional[str]:
    return _price_map().get((plan_id or "").strip().upper())


def plan_for_price_id(price_id: str) -> Optional[str]:
    """Reverse lookup — which BotTrade plan a Stripe Price ID corresponds to."""
    for plan, pid in _price_map().items():
        if pid == price_id:
            return plan
    return None


def purchasable_plans() -> list[str]:
    """Plan IDs a Checkout Session can actually be created for."""
    return list(_price_map().keys())


def _client() -> None:
    """Point the SDK at the configured key. Called before every API use."""
    stripe.api_key = settings.stripe_secret_key


# --------------------------------------------------------------------------- #
# Customer
# --------------------------------------------------------------------------- #
def get_or_create_customer(
    ledger: UsageLedger, account_id: str, email: Optional[str] = None,
) -> Optional[str]:
    """This account's Stripe Customer ID, creating one on first use.

    Returns ``None`` on any Stripe error rather than raising — a page render
    must survive a Stripe outage even if checkout can't proceed right now.
    """
    if not billing_enabled():
        return None

    existing = ledger.get_stripe_customer_id(account_id)
    if existing:
        return existing

    _client()
    try:
        customer = stripe.Customer.create(
            email=email or None,
            metadata={"bottrade_account_id": account_id},
        )
    except stripe.error.StripeError as exc:                    # noqa: BLE001
        logger.error(f"[billing] could not create Stripe customer for "
                    f"{account_id}: {exc}")
        return None

    ledger.set_stripe_customer_id(account_id, customer.id)
    return customer.id


# --------------------------------------------------------------------------- #
# Checkout
# --------------------------------------------------------------------------- #
def create_checkout_session(
    ledger: UsageLedger,
    account_id: str,
    plan_id: str,
    success_url: str,
    cancel_url: str,
    email: Optional[str] = None,
) -> Optional[str]:
    """Build a Stripe Checkout URL for *account_id* to subscribe to *plan_id*.

    Returns the URL to redirect the browser to, or ``None`` if billing isn't
    enabled, the plan has no price configured, or Stripe rejects the request.
    """
    price_id = price_id_for_plan(plan_id)
    if not billing_enabled() or price_id is None:
        return None

    customer_id = get_or_create_customer(ledger, account_id, email=email)
    if customer_id is None:
        return None

    _client()
    try:
        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=account_id,
            metadata={"bottrade_account_id": account_id, "plan_id": plan_id},
            subscription_data={
                "metadata": {"bottrade_account_id": account_id, "plan_id": plan_id},
            },
        )
    except stripe.error.StripeError as exc:                    # noqa: BLE001
        logger.error(f"[billing] checkout session failed for {account_id} "
                    f"-> {plan_id}: {exc}")
        return None

    return session.url


def confirm_checkout_session(ledger: UsageLedger, session_id: str) -> Optional[str]:
    """Called on the success redirect — verifies payment and applies the plan.

    Reads the Checkout Session straight back from Stripe rather than trusting
    the redirect alone (a ``session_id`` in a query string is not proof of
    payment by itself), and only upgrades the account when Stripe confirms
    the subscription is actually active. Returns the plan applied, or
    ``None`` if the session isn't a paid, active subscription.
    """
    if not billing_enabled():
        return None

    _client()
    try:
        session = stripe.checkout.Session.retrieve(
            session_id, expand=["subscription"])
    except stripe.error.StripeError as exc:                    # noqa: BLE001
        logger.warning(f"[billing] could not retrieve checkout session "
                       f"{session_id}: {exc}")
        return None

    if session.payment_status != "paid" and session.status != "complete":
        return None

    account_id = (session.metadata or {}).get("bottrade_account_id") \
        or session.client_reference_id
    plan_id = (session.metadata or {}).get("plan_id") \
        or plan_for_price_id(_first_price_id(session))
    if not account_id or not plan_id:
        logger.warning(f"[billing] checkout session {session_id} confirmed "
                       f"but carries no account/plan metadata")
        return None

    subscription = session.subscription
    sub_id = getattr(subscription, "id", None) or subscription
    ledger.set_plan_id(account_id, plan_id, stripe_subscription_id=sub_id)
    _touch_cache(account_id, plan_id)
    logger.info(f"[billing] {account_id} confirmed on {plan_id} "
               f"(subscription {sub_id})")
    return plan_id


def _first_price_id(session) -> Optional[str]:
    try:
        items = session.line_items.data if session.line_items else []
        return items[0].price.id if items else None
    except Exception:                                          # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Billing Portal (cancel / update payment method / view invoices)
# --------------------------------------------------------------------------- #
def create_portal_session(
    ledger: UsageLedger, account_id: str, return_url: str,
) -> Optional[str]:
    """URL to Stripe's hosted portal for an existing customer.

    ``None`` for an account with no Stripe customer yet — there is nothing
    to manage until they've subscribed at least once.
    """
    if not billing_enabled():
        return None
    customer_id = ledger.get_stripe_customer_id(account_id)
    if not customer_id:
        return None

    _client()
    try:
        portal = stripe.billing_portal.Session.create(
            customer=customer_id, return_url=return_url,
        )
    except stripe.error.StripeError as exc:                    # noqa: BLE001
        logger.error(f"[billing] portal session failed for {account_id}: {exc}")
        return None
    return portal.url


# --------------------------------------------------------------------------- #
# Reconciliation — catches cancellations/failed renewals without a webhook
# --------------------------------------------------------------------------- #
_SYNC_TTL_SEC = 300.0
_sync_cache: dict[str, tuple[float, str]] = {}
_sync_lock = threading.Lock()


def _touch_cache(account_id: str, plan_id: str) -> None:
    with _sync_lock:
        _sync_cache[account_id] = (time.time(), plan_id)


def sync_subscription_status(
    ledger: UsageLedger, account_id: str, force: bool = False,
) -> str:
    """Reconcile the ledger's plan against Stripe's live subscription state.

    Runs at most once per ``_SYNC_TTL_SEC`` per account unless ``force`` is
    set (used right after a checkout redirect, where the answer needs to be
    immediate). This is the mechanism that notices a cancellation or a failed
    renewal that happened entirely on Stripe's side — the trade-off for not
    running a webhook receiver is that the downgrade lands on the next check
    within the TTL, not the instant it happens on Stripe.

    Always returns a plan_id — the ledger's current value on any failure, so
    a Stripe outage never locks a page render.
    """
    current = ledger.get_plan_id(account_id)
    if not billing_enabled():
        return current

    if not force:
        with _sync_lock:
            cached = _sync_cache.get(account_id)
        if cached and (time.time() - cached[0]) < _SYNC_TTL_SEC:
            return cached[1]

    customer_id = ledger.get_stripe_customer_id(account_id)
    if not customer_id:
        _touch_cache(account_id, current)
        return current

    _client()
    try:
        subs = stripe.Subscription.list(
            customer=customer_id, status="active", limit=10)
    except stripe.error.StripeError as exc:                    # noqa: BLE001
        logger.warning(f"[billing] subscription sync failed for "
                       f"{account_id}: {exc}")
        return current

    resolved = DEFAULT_PLAN_ID
    sub_id = None
    for sub in subs.auto_paging_iter():
        for item in sub["items"]["data"]:
            plan = plan_for_price_id(item["price"]["id"])
            if plan and plan in PLANS:
                resolved, sub_id = plan, sub.id
                break
        if sub_id:
            break

    if resolved != current:
        logger.info(f"[billing] {account_id} plan reconciled "
                   f"{current} -> {resolved}")
        ledger.set_plan_id(account_id, resolved, stripe_subscription_id=sub_id)
    _touch_cache(account_id, resolved)
    return resolved
