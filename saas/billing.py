"""
Paddle billing — turns a plan choice into a real subscription.

``saas/plans.py`` and ``saas/ledger.py`` already model *what* a plan grants
and *whose account* it belongs to; this module is the missing third piece —
*how a user actually starts paying for one*. Everything here degrades
gracefully: with no Paddle key configured, :func:`billing_enabled` is False,
every upgrade button in the UI simply doesn't render, and every account
keeps working on the Free plan exactly as it did before this module existed.

Why Paddle, not Stripe
-----------------------
This was built against Stripe first. Stripe does not support Israel as a
seller's business location — confirmed against the real onboarding flow
(the "Business location" dropdown has no Israel option at all), not just
read about it. Paddle is a merchant of record: Paddle is the seller of
record for every transaction, handles global tax/VAT so BotTrade carries
zero sales-tax liability, and does accept Israel as a business location.

That changes the integration shape in one important way: there is no
Stripe-style hosted Checkout URL to redirect the browser to. Paddle checkout
runs client-side via Paddle.js — the backend's job is to make sure a Paddle
customer exists and hand the frontend a small, non-secret config
(:func:`checkout_config`) to open the overlay with; see
``dashboard/pages/2_Settings.py`` for the embed.

Design
------
**Paddle Checkout (Paddle.js overlay), not a custom payment form.** Paddle's
overlay collects card details, handles 3-D Secure and local payment methods,
and is PCI-compliant by construction. BotTrade never sees a card number.

**The Customer Portal, not a custom "manage subscription" page.**
Cancelling, updating a payment method, or viewing past invoices is Paddle's
hosted portal — one API call here, no UI to build or maintain.

**Polling reconciliation, not a webhook receiver — for now.** A push
webhook is the textbook way to learn about a cancellation or a failed
renewal the instant it happens, but Streamlit has no clean way to expose an
HTTP route for one; standing up a second service just for this is a bigger
lift than a subscription business at this stage needs. Instead,
:func:`sync_subscription_status` re-checks a customer's live subscription
state against Paddle on a TTL, the same pattern already used for
fundamentals and news in ``market_data/`` (and the same trade-off this
module made under Stripe before the migration). The lag is bounded by the
TTL, not instant — that trade-off is deliberate and documented, not hidden.

There is also no Stripe-style ``confirm_checkout_session(session_id)`` here.
Paddle's overlay does not hand the success-return URL a transaction id the
way Stripe's redirect carried ``?session_id=``, so there is nothing to look
up by. Instead, the return from a completed checkout just triggers
:func:`sync_subscription_status` with ``force=True`` — the customer id is
already known (it was handed to the overlay to open checkout in the first
place), so the forced poll finds the new subscription directly.

Every Paddle call is wrapped: a network hiccup or an API error here must
never crash a page render or silently downgrade someone who is still
correctly paying — see the module-level ``try/except`` around every function
that reaches the network.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Optional

from config.settings import settings
from saas.ledger import UsageLedger
from saas.plans import PLANS, DEFAULT_PLAN_ID
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from paddle_billing import Client, Environment, Options
    from paddle_billing.Entities.Shared import CustomData
    from paddle_billing.Entities.Subscriptions.SubscriptionStatus import (
        SubscriptionStatus,
    )
    from paddle_billing.Exceptions.ApiError import ApiError
    from paddle_billing.Resources.CustomerPortalSessions.Operations import (
        CreateCustomerPortalSession,
    )
    from paddle_billing.Resources.Customers.Operations import CreateCustomer
    from paddle_billing.Resources.Subscriptions.Operations import (
        ListSubscriptions,
    )
except ImportError:                                            # pragma: no cover
    Client = None  # billing_enabled() below turns this into a clean no-op


# --------------------------------------------------------------------------- #
# Enablement + plan/price mapping
# --------------------------------------------------------------------------- #
def billing_enabled() -> bool:
    """True once the Paddle SDK is installed, keyed, and a plan is priced."""
    return Client is not None and settings.billing_configured


def _price_map() -> dict[str, str]:
    """``{plan_id: paddle_price_id}`` for every plan that has one configured."""
    m: dict[str, str] = {}
    if settings.paddle_price_id_pro:
        m["PRO"] = settings.paddle_price_id_pro
    if settings.paddle_price_id_desk:
        m["DESK"] = settings.paddle_price_id_desk
    return m


def price_id_for_plan(plan_id: str) -> Optional[str]:
    return _price_map().get((plan_id or "").strip().upper())


def plan_for_price_id(price_id: str) -> Optional[str]:
    """Reverse lookup — which BotTrade plan a Paddle Price ID corresponds to."""
    for plan, pid in _price_map().items():
        if pid == price_id:
            return plan
    return None


def purchasable_plans() -> list[str]:
    """Plan IDs a checkout can actually be opened for."""
    return list(_price_map().keys())


#: Paddle's SDK default is 60s with 3 retries. Reconciliation runs from
#: ``Tenant.plan``, which the background trading loop reads every cycle, so
#: an unreachable Paddle must fail fast rather than stall a trading decision.
#: Retries are capped for the same reason — the TTL cache means a missed
#: check costs at most one interval, while a hung request costs a bar.
_HTTP_TIMEOUT_SEC = 8.0
_MAX_RETRIES = 1

#: `requests`' `timeout=` (and therefore `_HTTP_TIMEOUT_SEC` above) bounds the
#: connect and read phases of a socket that already exists — it does *not*
#: cover DNS resolution. `socket.getaddrinfo()` is a blocking libc call with
#: its own OS-level resolver timeout/retry policy that Python does not let
#: `requests`/`urllib3` interrupt, so a slow or wedged resolver for
#: `*.paddle.com` can stall a call for minutes even though every timeout
#: parameter on the request itself says single-digit seconds. That is exactly
#: the "must fail fast" promise above being broken by something the SDK
#: cannot see, so it is enforced again here from the outside: the whole
#: Paddle call — DNS included — runs on a worker thread and is given a hard
#: wall-clock budget. A worker that blows through it is simply abandoned
#: (Python cannot kill a thread); it dies on its own once the OS-level
#: resolver or connection eventually gives up. Comfortably above
#: `_HTTP_TIMEOUT_SEC * (_MAX_RETRIES + 1)` so it never fires before the
#: SDK's own timeout would have.
_SYNC_HARD_TIMEOUT_SEC = 20.0

_client_instance = None
_client_lock = threading.Lock()
_sync_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="paddle-sync",
)


def _client():
    """The Paddle SDK client, built once and cached.

    Unlike Stripe's SDK (a module-level ``api_key`` you assign to), Paddle's
    ``Client`` is an actual object you construct — closer to a normal HTTP
    client, and it accepts timeout/retry directly in its constructor rather
    than needing a hand-rolled HTTP client swapped in after the fact.
    """
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                env = (Environment.PRODUCTION
                      if settings.paddle_environment.strip().lower() == "production"
                      else Environment.SANDBOX)
                _client_instance = Client(
                    settings.paddle_api_key,
                    options=Options(environment=env),
                    timeout=_HTTP_TIMEOUT_SEC,
                    retry_count=_MAX_RETRIES,
                )
    return _client_instance


# --------------------------------------------------------------------------- #
# Customer
# --------------------------------------------------------------------------- #
def get_or_create_customer(
    ledger: UsageLedger, account_id: str, email: Optional[str] = None,
) -> Optional[str]:
    """This account's Paddle Customer ID, creating one on first use.

    Returns ``None`` on any Paddle error rather than raising — a page render
    must survive a Paddle outage even if checkout can't proceed right now.
    """
    if not billing_enabled():
        return None

    existing = ledger.get_paddle_customer_id(account_id)
    if existing:
        return existing

    # Paddle requires a real email to create a customer — unlike Stripe,
    # which accepts one being absent. Without one there is genuinely nothing
    # to create yet; the caller tries again once an email is available.
    email = (email or "").strip()
    if not email:
        logger.debug(f"[billing] no email yet for {account_id} — "
                    f"customer creation deferred")
        return None

    try:
        customer = _client().customers.create(CreateCustomer(
            email=email,
            custom_data=CustomData({"bottrade_account_id": account_id}),
        ))
    except ApiError as exc:                                    # noqa: BLE001
        logger.error(f"[billing] could not create Paddle customer for "
                    f"{account_id}: {exc}")
        return None

    ledger.set_paddle_customer_id(account_id, customer.id)
    return customer.id


# --------------------------------------------------------------------------- #
# Checkout
# --------------------------------------------------------------------------- #
def checkout_config(
    ledger: UsageLedger,
    account_id: str,
    plan_id: str,
    success_url: str,
    email: Optional[str] = None,
) -> Optional[dict]:
    """Everything the browser needs to open a Paddle Checkout overlay.

    There is no server-created "session" to redirect to the way Stripe's
    ``create_checkout_session`` returned a URL — Paddle Checkout runs
    client-side. This returns a plain dict (client_token, price_id,
    customer_id, custom_data, success_url) that
    ``dashboard/pages/2_Settings.py`` feeds straight into ``Paddle.js``.
    None of it is secret; the client token is explicitly meant to reach the
    browser. Returns ``None`` if billing isn't enabled, the plan has no
    price configured, or a Paddle customer couldn't be created (most often:
    no email known for this account yet).
    """
    price_id = price_id_for_plan(plan_id)
    if not billing_enabled() or price_id is None:
        return None
    if not settings.paddle_client_token:
        logger.warning("[billing] PADDLE_CLIENT_TOKEN is not set — the "
                       "checkout overlay cannot open without it")
        return None

    customer_id = get_or_create_customer(ledger, account_id, email=email)
    if customer_id is None:
        return None

    return {
        "client_token": settings.paddle_client_token,
        "environment": settings.paddle_environment.strip().lower(),
        "price_id": price_id,
        "customer_id": customer_id,
        "custom_data": {"bottrade_account_id": account_id, "plan_id": plan_id},
        "success_url": success_url,
    }


# --------------------------------------------------------------------------- #
# Customer Portal (cancel / update payment method / view invoices)
# --------------------------------------------------------------------------- #
def create_portal_session(
    ledger: UsageLedger, account_id: str, return_url: str = "",
) -> Optional[str]:
    """URL to Paddle's hosted customer portal for an existing customer.

    ``None`` for an account with no Paddle customer yet — there is nothing
    to manage until they've subscribed at least once. ``return_url`` is
    accepted for call-site symmetry with the old Stripe version but unused —
    Paddle's portal session carries no return-URL parameter; it's a
    standalone hosted page the user navigates back from on their own.
    """
    if not billing_enabled():
        return None
    customer_id = ledger.get_paddle_customer_id(account_id)
    if not customer_id:
        return None

    try:
        session = _client().customer_portal_sessions.create(
            customer_id, CreateCustomerPortalSession(),
        )
    except ApiError as exc:                                    # noqa: BLE001
        logger.error(f"[billing] portal session failed for {account_id}: {exc}")
        return None
    return session.urls.general.overview


# --------------------------------------------------------------------------- #
# Reconciliation — catches cancellations/failed renewals without a webhook
# --------------------------------------------------------------------------- #
#: Subscription states that entitle an account to its paid plan.
#: "Trialing" counts — a trial is a real subscription that has not billed
#: yet. "PastDue", "Paused", "Canceled" and "Inactive" deliberately do not.
#:
#: A tuple, not a set: SubscriptionStatus deliberately sets `__hash__ = None`
#: (it compares equal to plain strings like "active", which would make set
#: membership across the two types inconsistent), so it isn't hashable.
_ACTIVE_SUBSCRIPTION_STATES = (
    SubscriptionStatus.Active, SubscriptionStatus.Trialing,
)

_SYNC_TTL_SEC = 300.0
_sync_cache: dict[str, tuple[float, str]] = {}
_sync_lock = threading.Lock()


def _touch_cache(account_id: str, plan_id: str) -> None:
    with _sync_lock:
        _sync_cache[account_id] = (time.time(), plan_id)


def sync_subscription_status(
    ledger: UsageLedger, account_id: str, force: bool = False,
) -> str:
    """Reconcile the ledger's plan against Paddle's live subscription state.

    Runs at most once per ``_SYNC_TTL_SEC`` per account unless ``force`` is
    set (used right after a checkout return, where the answer needs to be
    immediate — see the module docstring on why there is no separate
    "confirm this specific checkout" step). This is the mechanism that
    notices a cancellation or a failed renewal that happened entirely on
    Paddle's side — the trade-off for not running a webhook receiver is that
    the downgrade lands on the next check within the TTL, not the instant it
    happens on Paddle.

    Always returns a plan_id — the ledger's current value on any failure, so
    a Paddle outage never locks a page render.
    """
    current = ledger.get_plan_id(account_id)
    if not billing_enabled():
        return current

    if not force:
        with _sync_lock:
            cached = _sync_cache.get(account_id)
        if cached and (time.time() - cached[0]) < _SYNC_TTL_SEC:
            return cached[1]

    customer_id = ledger.get_paddle_customer_id(account_id)
    if not customer_id:
        _touch_cache(account_id, current)
        return current

    try:
        future = _sync_executor.submit(
            _client().subscriptions.list,
            ListSubscriptions(
                customer_ids=[customer_id],
                statuses=list(_ACTIVE_SUBSCRIPTION_STATES),
            ),
        )
        subs = future.result(timeout=_SYNC_HARD_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        logger.warning(f"[billing] subscription sync timed out for "
                       f"{account_id} after {_SYNC_HARD_TIMEOUT_SEC}s — "
                       f"treating as a Paddle outage")
        _touch_cache(account_id, current)
        return current
    except ApiError as exc:                                    # noqa: BLE001
        logger.warning(f"[billing] subscription sync failed for "
                       f"{account_id}: {exc}")
        return current

    resolved = DEFAULT_PLAN_ID
    sub_id = None
    for sub in subs:
        for item in sub.items:
            plan = plan_for_price_id(item.price.id)
            if plan and plan in PLANS:
                resolved, sub_id = plan, sub.id
                break
        if sub_id:
            break

    if resolved != current:
        logger.info(f"[billing] {account_id} plan reconciled "
                   f"{current} -> {resolved}")
        ledger.set_plan_id(account_id, resolved, paddle_subscription_id=sub_id)
    _touch_cache(account_id, resolved)
    return resolved
