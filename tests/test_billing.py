"""
Tests for saas/billing.py — Stripe Checkout, the Billing Portal, and the
TTL-based subscription reconciliation that stands in for a webhook receiver.

No test talks to Stripe's network. Every ``stripe.*`` call is monkeypatched
with a fake that returns the shape billing.py actually reads. The properties
worth pinning down: billing degrades to a clean no-op with no key configured
(never crashes a page render), an unpaid checkout session never upgrades an
account, a cancellation on Stripe's side is actually noticed on the next
sync, and a Stripe API error anywhere never raises into the caller.
"""
from __future__ import annotations

import time
import types
from types import SimpleNamespace

import pytest

from saas import billing
from saas.ledger import UsageLedger


GOOD_KEY_ENV = {
    "STRIPE_SECRET_KEY": "sk_test_fake",
    "STRIPE_PRICE_ID_PRO": "price_pro_123",
    "STRIPE_PRICE_ID_DESK": "price_desk_456",
}


@pytest.fixture()
def enabled(monkeypatch):
    """Billing configured with fake keys — settings is a live singleton."""
    for k, v in GOOD_KEY_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(billing.settings, "stripe_secret_key", "sk_test_fake")
    monkeypatch.setattr(billing.settings, "stripe_price_id_pro", "price_pro_123")
    monkeypatch.setattr(billing.settings, "stripe_price_id_desk", "price_desk_456")
    yield


@pytest.fixture()
def disabled(monkeypatch):
    monkeypatch.setattr(billing.settings, "stripe_secret_key", None)
    monkeypatch.setattr(billing.settings, "stripe_price_id_pro", None)
    monkeypatch.setattr(billing.settings, "stripe_price_id_desk", None)
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
# Enablement / price mapping
# --------------------------------------------------------------------------- #
def test_billing_disabled_with_no_key(disabled):
    assert billing.billing_enabled() is False


def test_billing_enabled_with_a_key_and_a_price(enabled):
    assert billing.billing_enabled() is True


def test_billing_disabled_with_a_key_but_no_prices(monkeypatch):
    monkeypatch.setattr(billing.settings, "stripe_secret_key", "sk_test_fake")
    monkeypatch.setattr(billing.settings, "stripe_price_id_pro", None)
    monkeypatch.setattr(billing.settings, "stripe_price_id_desk", None)
    assert billing.billing_enabled() is False


def test_price_and_plan_mapping_round_trip(enabled):
    assert billing.price_id_for_plan("PRO") == "price_pro_123"
    assert billing.price_id_for_plan("pro") == "price_pro_123"   # case-insensitive
    assert billing.plan_for_price_id("price_desk_456") == "DESK"
    assert billing.plan_for_price_id("price_unknown") is None


def test_purchasable_plans_reflects_configured_prices(monkeypatch):
    monkeypatch.setattr(billing.settings, "stripe_secret_key", "sk_test_fake")
    monkeypatch.setattr(billing.settings, "stripe_price_id_pro", "price_pro_123")
    monkeypatch.setattr(billing.settings, "stripe_price_id_desk", None)
    assert billing.purchasable_plans() == ["PRO"]


# --------------------------------------------------------------------------- #
# Disabled billing is a clean no-op everywhere
# --------------------------------------------------------------------------- #
def test_checkout_session_is_none_when_disabled(disabled, ledger):
    url = billing.create_checkout_session(
        ledger, "acct1", "PRO", "http://x/ok", "http://x/no")
    assert url is None


def test_portal_session_is_none_when_disabled(disabled, ledger):
    assert billing.create_portal_session(ledger, "acct1", "http://x") is None


def test_sync_returns_the_current_ledger_plan_when_disabled(disabled, ledger):
    ledger.set_plan_id("acct1", "PRO")
    assert billing.sync_subscription_status(ledger, "acct1") == "PRO"


def test_customer_creation_is_none_when_disabled(disabled, ledger):
    assert billing.get_or_create_customer(ledger, "acct1") is None


# --------------------------------------------------------------------------- #
# Customer creation
# --------------------------------------------------------------------------- #
def test_customer_is_created_once_and_then_reused(enabled, ledger, monkeypatch):
    calls = []
    monkeypatch.setattr(
        billing.stripe.Customer, "create",
        lambda **kw: calls.append(kw) or SimpleNamespace(id="cus_abc"))

    first = billing.get_or_create_customer(ledger, "acct1", email="a@b.com")
    second = billing.get_or_create_customer(ledger, "acct1", email="a@b.com")

    assert first == second == "cus_abc"
    assert len(calls) == 1, "a second call must reuse the persisted customer"
    assert calls[0]["email"] == "a@b.com"
    assert calls[0]["metadata"]["bottrade_account_id"] == "acct1"


def test_customer_creation_failure_does_not_raise(enabled, ledger, monkeypatch):
    def _explode(**kw):
        raise billing.stripe.error.StripeError("network down")
    monkeypatch.setattr(billing.stripe.Customer, "create", _explode)
    assert billing.get_or_create_customer(ledger, "acct1") is None


# --------------------------------------------------------------------------- #
# Checkout
# --------------------------------------------------------------------------- #
def test_checkout_session_uses_the_right_price_and_persists_the_customer(
    enabled, ledger, monkeypatch,
):
    monkeypatch.setattr(billing.stripe.Customer, "create",
                        lambda **kw: SimpleNamespace(id="cus_abc"))
    captured = {}

    def fake_create(**kw):
        captured.update(kw)
        return SimpleNamespace(url="https://checkout.stripe.com/xyz")

    monkeypatch.setattr(billing.stripe.checkout.Session, "create", fake_create)

    url = billing.create_checkout_session(
        ledger, "acct1", "PRO", "http://x/ok", "http://x/no", email="a@b.com")

    assert url == "https://checkout.stripe.com/xyz"
    assert captured["customer"] == "cus_abc"
    assert captured["mode"] == "subscription"
    assert captured["line_items"] == [{"price": "price_pro_123", "quantity": 1}]
    assert captured["metadata"]["plan_id"] == "PRO"
    assert ledger.get_stripe_customer_id("acct1") == "cus_abc"


def test_checkout_session_is_none_for_an_unpriced_plan(enabled, ledger):
    assert billing.create_checkout_session(
        ledger, "acct1", "FREE", "http://x/ok", "http://x/no") is None


def test_checkout_session_failure_does_not_raise(enabled, ledger, monkeypatch):
    monkeypatch.setattr(billing.stripe.Customer, "create",
                        lambda **kw: SimpleNamespace(id="cus_abc"))

    def _explode(**kw):
        raise billing.stripe.error.StripeError("card processing down")
    monkeypatch.setattr(billing.stripe.checkout.Session, "create", _explode)

    assert billing.create_checkout_session(
        ledger, "acct1", "PRO", "http://x/ok", "http://x/no") is None


# --------------------------------------------------------------------------- #
# Confirming a checkout
# --------------------------------------------------------------------------- #
def _fake_session(**overrides):
    base = dict(
        payment_status="paid", status="complete",
        metadata={"bottrade_account_id": "acct1", "plan_id": "PRO"},
        client_reference_id="acct1",
        subscription=SimpleNamespace(id="sub_999"),
        line_items=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_confirm_applies_the_plan_on_a_paid_session(enabled, ledger, monkeypatch):
    monkeypatch.setattr(billing.stripe.checkout.Session, "retrieve",
                        lambda *a, **kw: _fake_session())
    plan = billing.confirm_checkout_session(ledger, "cs_test_123")
    assert plan == "PRO"
    assert ledger.get_plan_id("acct1") == "PRO"
    assert ledger.get_stripe_subscription_id("acct1") == "sub_999"


def test_confirm_rejects_an_unpaid_session(enabled, ledger, monkeypatch):
    """A session_id in a URL is not proof of payment on its own."""
    unpaid = _fake_session(payment_status="unpaid", status="open")
    monkeypatch.setattr(billing.stripe.checkout.Session, "retrieve",
                        lambda *a, **kw: unpaid)
    assert billing.confirm_checkout_session(ledger, "cs_bad") is None
    assert ledger.get_plan_id("acct1") == "FREE"


def test_confirm_falls_back_to_client_reference_id(enabled, ledger, monkeypatch):
    session = _fake_session(metadata={}, client_reference_id="acct2")
    monkeypatch.setattr(billing.stripe.checkout.Session, "retrieve",
                        lambda *a, **kw: session)
    # No plan_id in metadata and no line_items to derive it from -> unresolvable.
    assert billing.confirm_checkout_session(ledger, "cs_x") is None


def test_confirm_handles_a_retrieve_failure(enabled, ledger, monkeypatch):
    def _explode(*a, **kw):
        raise billing.stripe.error.StripeError("boom")
    monkeypatch.setattr(billing.stripe.checkout.Session, "retrieve", _explode)
    assert billing.confirm_checkout_session(ledger, "cs_x") is None


# --------------------------------------------------------------------------- #
# Billing Portal
# --------------------------------------------------------------------------- #
def test_portal_requires_an_existing_customer(enabled, ledger):
    assert billing.create_portal_session(ledger, "acct1", "http://x") is None


def test_portal_session_for_an_existing_customer(enabled, ledger, monkeypatch):
    ledger.set_stripe_customer_id("acct1", "cus_abc")
    monkeypatch.setattr(
        billing.stripe.billing_portal.Session, "create",
        lambda **kw: SimpleNamespace(url="https://billing.stripe.com/p/xyz"))
    url = billing.create_portal_session(ledger, "acct1", "http://return")
    assert url == "https://billing.stripe.com/p/xyz"


def test_portal_session_failure_does_not_raise(enabled, ledger, monkeypatch):
    ledger.set_stripe_customer_id("acct1", "cus_abc")

    def _explode(**kw):
        raise billing.stripe.error.StripeError("boom")
    monkeypatch.setattr(billing.stripe.billing_portal.Session, "create", _explode)
    assert billing.create_portal_session(ledger, "acct1", "http://x") is None


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
class _FakeSubscription(dict):
    """Real Stripe subscription objects are dict subclasses (StripeObject),
    so billing.py reads them with `sub["items"]["data"]` — a SimpleNamespace
    can't stand in for that; only something actually subscriptable can."""
    def __init__(self, sub_id: str, items: list[dict]):
        super().__init__(items={"data": items})
        self.id = sub_id


def _sub_list(items: list[dict]):
    """Stand in for stripe.Subscription.list(...)'s ListObject."""
    subs = [_FakeSubscription(it["sub_id"], it["items"]) for it in items]
    return SimpleNamespace(auto_paging_iter=lambda: iter(subs))


def test_sync_notices_a_live_cancellation(enabled, ledger, monkeypatch):
    ledger.set_stripe_customer_id("acct1", "cus_abc")
    ledger.set_plan_id("acct1", "PRO", stripe_subscription_id="sub_999")

    empty = SimpleNamespace(auto_paging_iter=lambda: iter([]))
    monkeypatch.setattr(billing.stripe.Subscription, "list", lambda **kw: empty)

    plan = billing.sync_subscription_status(ledger, "acct1", force=True)
    assert plan == "FREE"
    assert ledger.get_plan_id("acct1") == "FREE"


def test_sync_confirms_a_still_active_subscription(enabled, ledger, monkeypatch):
    ledger.set_stripe_customer_id("acct1", "cus_abc")
    ledger.set_plan_id("acct1", "PRO", stripe_subscription_id="sub_999")
    active = _sub_list([{"sub_id": "sub_999",
                        "items": [{"price": {"id": "price_pro_123"}}]}])
    monkeypatch.setattr(billing.stripe.Subscription, "list", lambda **kw: active)

    plan = billing.sync_subscription_status(ledger, "acct1", force=True)
    assert plan == "PRO"


def test_sync_is_cached_within_the_ttl(enabled, ledger, monkeypatch):
    ledger.set_stripe_customer_id("acct1", "cus_abc")
    calls = []
    empty = SimpleNamespace(auto_paging_iter=lambda: iter([]))
    monkeypatch.setattr(billing.stripe.Subscription, "list",
                        lambda **kw: calls.append(1) or empty)

    billing.sync_subscription_status(ledger, "acct1", force=True)
    billing.sync_subscription_status(ledger, "acct1")   # within TTL, no force
    assert len(calls) == 1


def test_sync_bypasses_the_cache_when_forced(enabled, ledger, monkeypatch):
    ledger.set_stripe_customer_id("acct1", "cus_abc")
    calls = []
    empty = SimpleNamespace(auto_paging_iter=lambda: iter([]))
    monkeypatch.setattr(billing.stripe.Subscription, "list",
                        lambda **kw: calls.append(1) or empty)

    billing.sync_subscription_status(ledger, "acct1", force=True)
    billing.sync_subscription_status(ledger, "acct1", force=True)
    assert len(calls) == 2


def test_sync_with_no_customer_yet_stays_on_current_plan(enabled, ledger):
    assert billing.sync_subscription_status(ledger, "acct1") == "FREE"


def test_sync_survives_a_stripe_outage(enabled, ledger, monkeypatch):
    ledger.set_stripe_customer_id("acct1", "cus_abc")
    ledger.set_plan_id("acct1", "PRO")

    def _explode(**kw):
        raise billing.stripe.error.StripeError("down")
    monkeypatch.setattr(billing.stripe.Subscription, "list", _explode)

    # Must not raise, and must not downgrade a paying customer on an outage.
    assert billing.sync_subscription_status(ledger, "acct1", force=True) == "PRO"
