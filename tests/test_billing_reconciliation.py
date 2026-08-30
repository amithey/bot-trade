"""
Tests for plan reconciliation — the mechanism that notices a subscription
that was cancelled or whose renewal failed entirely on Stripe's side.

The bug these cover: sync_subscription_status was called from exactly one
place, dashboard/pages/2_Settings.py, while Tenant.plan (read by the sidebar,
the billing page and every cycle of the background trading loop) read the
ledger raw. A lapsed subscriber therefore kept their entitlements — including
the platform budget that spends the operator's own Anthropic credit — for as
long as they avoided the one page that would have downgraded them.
"""
from __future__ import annotations

import pytest

from saas.ledger import UsageLedger
from saas.plans import DEFAULT_PLAN_ID
from saas.tenant import Tenant


@pytest.fixture
def ledger(tmp_path):
    return UsageLedger(db_path=tmp_path / "usage.db")


# --------------------------------------------------------------------------- #
# Tenant.plan reconciles
# --------------------------------------------------------------------------- #
def test_plan_calls_reconciliation(monkeypatch, ledger):
    calls = []

    def _fake_sync(led, account_id, force=False):
        calls.append(account_id)
        return led.get_plan_id(account_id)

    import saas.billing as billing_mod
    monkeypatch.setattr(billing_mod, "sync_subscription_status", _fake_sync)

    t = Tenant(account_id="user:a@b.com", ledger=ledger)
    t.plan
    assert calls == ["user:a@b.com"]


def test_plan_reflects_a_downgrade_stripe_reports(monkeypatch, ledger):
    """The regression: ledger says PRO, Stripe says the subscription is gone."""
    ledger.ensure_account("user:a@b.com", "PRO")
    ledger.set_plan_id("user:a@b.com", "PRO")

    import saas.billing as billing_mod
    monkeypatch.setattr(
        billing_mod, "sync_subscription_status",
        lambda led, account_id, force=False: DEFAULT_PLAN_ID,
    )

    t = Tenant(account_id="user:a@b.com", ledger=ledger)
    assert t.plan.id == DEFAULT_PLAN_ID, (
        "a cancelled subscription must lose its entitlements without the user "
        "having to visit the Settings page"
    )


def test_plan_reconciles_against_the_person_not_the_active_key(monkeypatch, ledger):
    """Billing identity follows billing_account_id, never the BYOK fingerprint."""
    seen = []

    import saas.billing as billing_mod
    monkeypatch.setattr(
        billing_mod, "sync_subscription_status",
        lambda led, account_id, force=False: (seen.append(account_id)
                                              or led.get_plan_id(account_id)),
    )

    t = Tenant(account_id="user:a@b.com",
               user_api_key="sk-ant-" + "x" * 40, ledger=ledger)
    t.plan
    assert seen == ["user:a@b.com"]
    assert not any(s.startswith("byok_") for s in seen)


# --------------------------------------------------------------------------- #
# Reconciliation must never break a render or a trading cycle
# --------------------------------------------------------------------------- #
def test_stripe_failure_falls_back_to_the_stored_plan(monkeypatch, ledger):
    ledger.ensure_account("acct", "PRO")
    ledger.set_plan_id("acct", "PRO")

    import saas.billing as billing_mod

    def _explode(led, account_id, force=False):
        raise RuntimeError("stripe unreachable")

    monkeypatch.setattr(billing_mod, "sync_subscription_status", _explode)

    t = Tenant(account_id="acct", ledger=ledger)
    assert t.plan.id == "PRO", "an outage must degrade to the last known plan"


def test_reconciliation_can_be_switched_off(monkeypatch, ledger):
    """Callers that must not touch the network (tests, offline tools)."""
    called = []

    import saas.billing as billing_mod
    monkeypatch.setattr(
        billing_mod, "sync_subscription_status",
        lambda *a, **kw: called.append(1) or DEFAULT_PLAN_ID,
    )

    t = Tenant(account_id="acct", ledger=ledger, reconcile_billing=False)
    t.plan
    assert called == []


def test_entitlement_goes_through_the_reconciled_plan(monkeypatch, ledger):
    """entitlement reads self.plan, so the downgrade must reach it too."""
    ledger.ensure_account("acct", "PRO")
    ledger.set_plan_id("acct", "PRO")

    import saas.billing as billing_mod
    monkeypatch.setattr(
        billing_mod, "sync_subscription_status",
        lambda led, account_id, force=False: DEFAULT_PLAN_ID,
    )

    t = Tenant(account_id="acct", ledger=ledger)
    assert t.entitlement.plan.id == DEFAULT_PLAN_ID


# --------------------------------------------------------------------------- #
# The Stripe HTTP client is bounded
# --------------------------------------------------------------------------- #
def test_stripe_client_sets_a_short_timeout(monkeypatch):
    """Reconciliation runs on the trading loop's path, so Stripe's 80-second
    default would let one slow request stall a whole trading decision."""
    import stripe
    import saas.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_http_configured", False)
    monkeypatch.setattr(billing_mod.settings, "stripe_secret_key",
                        "sk_test_dummy", raising=False)
    billing_mod._client()

    assert billing_mod._HTTP_TIMEOUT_SEC <= 15
    assert stripe.max_network_retries <= 2
    timeout = getattr(stripe.default_http_client, "_timeout", None)
    assert timeout == billing_mod._HTTP_TIMEOUT_SEC


def test_http_client_is_configured_once(monkeypatch):
    import saas.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_http_configured", False)
    monkeypatch.setattr(billing_mod.settings, "stripe_secret_key",
                        "sk_test_dummy", raising=False)
    billing_mod._client()
    first = billing_mod.stripe.default_http_client
    billing_mod._client()
    assert billing_mod.stripe.default_http_client is first
