"""
Tests for saas/billing.py — Paddle Checkout, the Customer Portal, and the
TTL-based subscription reconciliation that stands in for a webhook receiver.

No test talks to Paddle's network. ``billing._client()`` is monkeypatched
with a fake exposing exactly the surface billing.py actually calls
(``customers.create``, ``customer_portal_sessions.create``,
``subscriptions.list``), returning objects shaped like the real Paddle SDK
entities (``customer.id``, ``session.urls.general.overview``,
``sub.items[i].price.id``) rather than plain dicts — this is what caught the
Stripe-era ``StripeObject.get()`` regression, and the same discipline
carries over here. The properties worth pinning down: billing degrades to a
clean no-op with nothing configured (never crashes a page render), checkout
config is never handed out without an email, a cancellation on Paddle's side
is actually noticed on the next sync, and a Paddle API error anywhere never
raises into the caller.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from saas import billing
from saas.ledger import UsageLedger


GOOD_KEY_ENV = {
    "PADDLE_API_KEY": "pdl_sdbx_apikey_fake",
    "PADDLE_CLIENT_TOKEN": "test_fake_client_token",
    "PADDLE_PRICE_ID_PRO": "pri_pro_123",
    "PADDLE_PRICE_ID_DESK": "pri_desk_456",
}


@pytest.fixture()
def enabled(monkeypatch):
    """Billing configured with fake keys — settings is a live singleton."""
    for k, v in GOOD_KEY_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(billing.settings, "paddle_api_key", "pdl_sdbx_apikey_fake")
    monkeypatch.setattr(billing.settings, "paddle_client_token", "test_fake_client_token")
    monkeypatch.setattr(billing.settings, "paddle_price_id_pro", "pri_pro_123")
    monkeypatch.setattr(billing.settings, "paddle_price_id_desk", "pri_desk_456")
    monkeypatch.setattr(billing.settings, "paddle_environment", "sandbox")
    yield


@pytest.fixture()
def disabled(monkeypatch):
    monkeypatch.setattr(billing.settings, "paddle_api_key", None)
    monkeypatch.setattr(billing.settings, "paddle_client_token", None)
    monkeypatch.setattr(billing.settings, "paddle_price_id_pro", None)
    monkeypatch.setattr(billing.settings, "paddle_price_id_desk", None)
    yield


@pytest.fixture()
def ledger(tmp_path) -> UsageLedger:
    return UsageLedger(tmp_path / "usage.db")


@pytest.fixture(autouse=True)
def clean_sync_cache():
    billing._sync_cache.clear()
    yield
    billing._sync_cache.clear()


# --------------------------------------------------------------------------- #
# Fakes — shaped like the real Paddle SDK entities, not plain dicts
# --------------------------------------------------------------------------- #
class _FakeCustomer:
    def __init__(self, customer_id: str):
        self.id = customer_id


class _FakeCustomersClient:
    def __init__(self, customer_id: str = "ctm_abc", explode: bool = False):
        self.calls: list = []
        self._customer_id = customer_id
        self._explode = explode

    def create(self, operation):
        self.calls.append(operation)
        if self._explode:
            raise billing.ApiError.__new__(billing.ApiError)
        return _FakeCustomer(self._customer_id)


class _FakePortalUrls:
    def __init__(self, overview: str):
        self.general = SimpleNamespace(overview=overview)


class _FakePortalSession:
    def __init__(self, overview: str):
        self.urls = _FakePortalUrls(overview)


class _FakePortalClient:
    def __init__(self, overview: str = "https://sandbox-customer-portal.paddle.com/x",
                explode: bool = False):
        self.calls: list = []
        self._overview = overview
        self._explode = explode

    def create(self, customer_id, operation):
        self.calls.append((customer_id, operation))
        if self._explode:
            raise billing.ApiError.__new__(billing.ApiError)
        return _FakePortalSession(self._overview)


def _fake_sub(sub_id: str, price_id: str):
    """Stands in for a Paddle Subscription entity: billing.py reads
    ``sub.id`` and, per item, ``item.price.id`` — real attribute access,
    not dict indexing, so a SimpleNamespace tree is what actually exercises
    the same code path the real SDK entity would."""
    return SimpleNamespace(
        id=sub_id, items=[SimpleNamespace(price=SimpleNamespace(id=price_id))],
    )


class _FakeSubscriptionsClient:
    def __init__(self, subs: list | None = None, explode: bool = False):
        self.calls: list = []
        self._subs = subs if subs is not None else []
        self._explode = explode

    def list(self, operation):
        self.calls.append(operation)
        if self._explode:
            raise billing.ApiError.__new__(billing.ApiError)
        return list(self._subs)


class _FakeClient:
    def __init__(self, customers=None, customer_portal_sessions=None, subscriptions=None):
        self.customers = customers or _FakeCustomersClient()
        self.customer_portal_sessions = customer_portal_sessions or _FakePortalClient()
        self.subscriptions = subscriptions or _FakeSubscriptionsClient()


def _patch_client(monkeypatch, client: _FakeClient) -> None:
    monkeypatch.setattr(billing, "_client", lambda: client)


# --------------------------------------------------------------------------- #
# Enablement / plan-price mapping
# --------------------------------------------------------------------------- #
def test_billing_disabled_with_no_key(disabled):
    assert billing.billing_enabled() is False


def test_billing_enabled_with_a_key_and_a_price(enabled):
    assert billing.billing_enabled() is True


def test_billing_disabled_with_a_key_but_no_prices(monkeypatch):
    monkeypatch.setattr(billing.settings, "paddle_api_key", "pdl_sdbx_apikey_fake")
    monkeypatch.setattr(billing.settings, "paddle_price_id_pro", None)
    monkeypatch.setattr(billing.settings, "paddle_price_id_desk", None)
    assert billing.billing_enabled() is False


def test_price_and_plan_mapping_round_trip(enabled):
    assert billing.price_id_for_plan("PRO") == "pri_pro_123"
    assert billing.price_id_for_plan("pro") == "pri_pro_123"   # case-insensitive
    assert billing.plan_for_price_id("pri_desk_456") == "DESK"
    assert billing.plan_for_price_id("pri_unknown") is None


def test_purchasable_plans_reflects_configured_prices(monkeypatch):
    monkeypatch.setattr(billing.settings, "paddle_api_key", "pdl_sdbx_apikey_fake")
    monkeypatch.setattr(billing.settings, "paddle_price_id_pro", "pri_pro_123")
    monkeypatch.setattr(billing.settings, "paddle_price_id_desk", None)
    assert billing.purchasable_plans() == ["PRO"]


# --------------------------------------------------------------------------- #
# Disabled billing is a clean no-op everywhere
# --------------------------------------------------------------------------- #
def test_checkout_config_is_none_when_disabled(disabled, ledger):
    cfg = billing.checkout_config(ledger, "acct1", "PRO", "http://x/ok", email="a@b.com")
    assert cfg is None


def test_portal_session_is_none_when_disabled(disabled, ledger):
    assert billing.create_portal_session(ledger, "acct1") is None


def test_sync_returns_the_current_ledger_plan_when_disabled(disabled, ledger):
    ledger.set_plan_id("acct1", "PRO")
    assert billing.sync_subscription_status(ledger, "acct1") == "PRO"


def test_customer_creation_is_none_when_disabled(disabled, ledger):
    assert billing.get_or_create_customer(ledger, "acct1", email="a@b.com") is None


# --------------------------------------------------------------------------- #
# Customer creation
# --------------------------------------------------------------------------- #
def test_customer_is_created_once_and_then_reused(enabled, ledger, monkeypatch):
    customers = _FakeCustomersClient(customer_id="ctm_abc")
    _patch_client(monkeypatch, _FakeClient(customers=customers))

    first = billing.get_or_create_customer(ledger, "acct1", email="a@b.com")
    second = billing.get_or_create_customer(ledger, "acct1", email="a@b.com")

    assert first == second == "ctm_abc"
    assert len(customers.calls) == 1, "a second call must reuse the persisted customer"
    assert customers.calls[0].email == "a@b.com"
    assert customers.calls[0].custom_data.data["bottrade_account_id"] == "acct1"
    assert ledger.get_paddle_customer_id("acct1") == "ctm_abc"


def test_customer_creation_is_deferred_with_no_email(enabled, ledger, monkeypatch):
    customers = _FakeCustomersClient()
    _patch_client(monkeypatch, _FakeClient(customers=customers))
    assert billing.get_or_create_customer(ledger, "acct1") is None
    assert len(customers.calls) == 0, "Paddle requires an email — no call should be made"


def test_customer_creation_failure_does_not_raise(enabled, ledger, monkeypatch):
    _patch_client(monkeypatch, _FakeClient(customers=_FakeCustomersClient(explode=True)))
    assert billing.get_or_create_customer(ledger, "acct1", email="a@b.com") is None


# --------------------------------------------------------------------------- #
# Checkout
# --------------------------------------------------------------------------- #
def test_checkout_config_uses_the_right_price_and_persists_the_customer(
    enabled, ledger, monkeypatch,
):
    customers = _FakeCustomersClient(customer_id="ctm_abc")
    _patch_client(monkeypatch, _FakeClient(customers=customers))

    cfg = billing.checkout_config(
        ledger, "acct1", "PRO", "http://x/ok", email="a@b.com")

    assert cfg is not None
    assert cfg["customer_id"] == "ctm_abc"
    assert cfg["price_id"] == "pri_pro_123"
    assert cfg["client_token"] == "test_fake_client_token"
    assert cfg["environment"] == "sandbox"
    assert cfg["custom_data"] == {"bottrade_account_id": "acct1", "plan_id": "PRO"}
    assert cfg["success_url"] == "http://x/ok"
    assert ledger.get_paddle_customer_id("acct1") == "ctm_abc"


def test_checkout_config_is_none_for_an_unpriced_plan(enabled, ledger, monkeypatch):
    _patch_client(monkeypatch, _FakeClient())
    assert billing.checkout_config(
        ledger, "acct1", "FREE", "http://x/ok", email="a@b.com") is None


def test_checkout_config_is_none_with_no_email_yet(enabled, ledger, monkeypatch):
    customers = _FakeCustomersClient()
    _patch_client(monkeypatch, _FakeClient(customers=customers))
    assert billing.checkout_config(ledger, "acct1", "PRO", "http://x/ok") is None
    assert len(customers.calls) == 0


def test_checkout_config_is_none_without_a_client_token(enabled, ledger, monkeypatch):
    monkeypatch.setattr(billing.settings, "paddle_client_token", None)
    _patch_client(monkeypatch, _FakeClient())
    assert billing.checkout_config(
        ledger, "acct1", "PRO", "http://x/ok", email="a@b.com") is None


def test_checkout_config_is_none_when_customer_creation_fails(enabled, ledger, monkeypatch):
    _patch_client(monkeypatch, _FakeClient(customers=_FakeCustomersClient(explode=True)))
    assert billing.checkout_config(
        ledger, "acct1", "PRO", "http://x/ok", email="a@b.com") is None


# --------------------------------------------------------------------------- #
# Customer Portal
# --------------------------------------------------------------------------- #
def test_portal_requires_an_existing_customer(enabled, ledger, monkeypatch):
    _patch_client(monkeypatch, _FakeClient())
    assert billing.create_portal_session(ledger, "acct1") is None


def test_portal_session_for_an_existing_customer(enabled, ledger, monkeypatch):
    ledger.set_paddle_customer_id("acct1", "ctm_abc")
    portal = _FakePortalClient(overview="https://sandbox-customer-portal.paddle.com/x")
    _patch_client(monkeypatch, _FakeClient(customer_portal_sessions=portal))

    url = billing.create_portal_session(ledger, "acct1")
    assert url == "https://sandbox-customer-portal.paddle.com/x"
    assert portal.calls[0][0] == "ctm_abc"


def test_portal_session_failure_does_not_raise(enabled, ledger, monkeypatch):
    ledger.set_paddle_customer_id("acct1", "ctm_abc")
    _patch_client(monkeypatch, _FakeClient(
        customer_portal_sessions=_FakePortalClient(explode=True)))
    assert billing.create_portal_session(ledger, "acct1") is None


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
def test_sync_notices_a_live_cancellation(enabled, ledger, monkeypatch):
    ledger.set_paddle_customer_id("acct1", "ctm_abc")
    ledger.set_plan_id("acct1", "PRO", paddle_subscription_id="sub_999")
    _patch_client(monkeypatch, _FakeClient(subscriptions=_FakeSubscriptionsClient(subs=[])))

    plan = billing.sync_subscription_status(ledger, "acct1", force=True)
    assert plan == "FREE"
    assert ledger.get_plan_id("acct1") == "FREE"


def test_sync_confirms_a_still_active_subscription(enabled, ledger, monkeypatch):
    ledger.set_paddle_customer_id("acct1", "ctm_abc")
    ledger.set_plan_id("acct1", "PRO", paddle_subscription_id="sub_999")
    active = [_fake_sub("sub_999", "pri_pro_123")]
    _patch_client(monkeypatch, _FakeClient(subscriptions=_FakeSubscriptionsClient(subs=active)))

    plan = billing.sync_subscription_status(ledger, "acct1", force=True)
    assert plan == "PRO"


def test_sync_is_cached_within_the_ttl(enabled, ledger, monkeypatch):
    ledger.set_paddle_customer_id("acct1", "ctm_abc")
    subs = _FakeSubscriptionsClient(subs=[])
    _patch_client(monkeypatch, _FakeClient(subscriptions=subs))

    billing.sync_subscription_status(ledger, "acct1", force=True)
    billing.sync_subscription_status(ledger, "acct1")   # within TTL, no force
    assert len(subs.calls) == 1


def test_sync_bypasses_the_cache_when_forced(enabled, ledger, monkeypatch):
    ledger.set_paddle_customer_id("acct1", "ctm_abc")
    subs = _FakeSubscriptionsClient(subs=[])
    _patch_client(monkeypatch, _FakeClient(subscriptions=subs))

    billing.sync_subscription_status(ledger, "acct1", force=True)
    billing.sync_subscription_status(ledger, "acct1", force=True)
    assert len(subs.calls) == 2


def test_sync_with_no_customer_yet_stays_on_current_plan(enabled, ledger, monkeypatch):
    _patch_client(monkeypatch, _FakeClient())
    assert billing.sync_subscription_status(ledger, "acct1") == "FREE"


def test_sync_survives_a_paddle_outage(enabled, ledger, monkeypatch):
    ledger.set_paddle_customer_id("acct1", "ctm_abc")
    ledger.set_plan_id("acct1", "PRO")
    _patch_client(monkeypatch, _FakeClient(
        subscriptions=_FakeSubscriptionsClient(explode=True)))

    # Must not raise, and must not downgrade a paying customer on an outage.
    assert billing.sync_subscription_status(ledger, "acct1", force=True) == "PRO"


def test_sync_passes_the_active_and_trialing_states_to_paddle(enabled, ledger, monkeypatch):
    ledger.set_paddle_customer_id("acct1", "ctm_abc")
    subs = _FakeSubscriptionsClient(subs=[])
    _patch_client(monkeypatch, _FakeClient(subscriptions=subs))

    billing.sync_subscription_status(ledger, "acct1", force=True)
    operation = subs.calls[0]
    assert operation.customer_ids == ["ctm_abc"]
    assert billing.SubscriptionStatus.Active in operation.statuses
    assert billing.SubscriptionStatus.Trialing in operation.statuses
