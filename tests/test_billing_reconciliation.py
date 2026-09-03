"""
Tests for plan reconciliation — the mechanism that notices a subscription
that was cancelled or whose renewal failed entirely on Paddle's side.

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


def test_plan_reflects_a_downgrade_paddle_reports(monkeypatch, ledger):
    """The regression: ledger says PRO, Paddle says the subscription is gone."""
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
def test_paddle_failure_falls_back_to_the_stored_plan(monkeypatch, ledger):
    ledger.ensure_account("acct", "PRO")
    ledger.set_plan_id("acct", "PRO")

    import saas.billing as billing_mod

    def _explode(led, account_id, force=False):
        raise RuntimeError("paddle unreachable")

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
# The Paddle HTTP client is bounded
# --------------------------------------------------------------------------- #
def test_paddle_client_sets_a_short_timeout(monkeypatch):
    """Reconciliation runs on the trading loop's path, so the SDK's own
    default timeout/retry policy would let one slow request stall a whole
    trading decision — billing.py must override both explicitly."""
    import saas.billing as billing_mod

    captured = {}

    class _FakeSdkClient:
        def __init__(self, api_key, options=None, timeout=None, retry_count=None):
            captured["api_key"] = api_key
            captured["timeout"] = timeout
            captured["retry_count"] = retry_count

    monkeypatch.setattr(billing_mod, "Client", _FakeSdkClient)
    monkeypatch.setattr(billing_mod, "_client_instance", None)
    monkeypatch.setattr(billing_mod.settings, "paddle_api_key",
                        "pdl_test_dummy", raising=False)
    monkeypatch.setattr(billing_mod.settings, "paddle_environment",
                        "sandbox", raising=False)

    billing_mod._client()

    assert billing_mod._HTTP_TIMEOUT_SEC <= 15
    assert captured["timeout"] == billing_mod._HTTP_TIMEOUT_SEC
    assert captured["retry_count"] == billing_mod._MAX_RETRIES
    assert captured["api_key"] == "pdl_test_dummy"


def test_client_is_configured_once(monkeypatch):
    """The Paddle SDK client is a real object to construct (unlike Stripe's
    module-level api_key assignment) — it must be built once and cached,
    not reconstructed on every call."""
    import saas.billing as billing_mod

    calls = []

    class _FakeSdkClient:
        def __init__(self, *a, **kw):
            calls.append(1)

    monkeypatch.setattr(billing_mod, "Client", _FakeSdkClient)
    monkeypatch.setattr(billing_mod, "_client_instance", None)
    monkeypatch.setattr(billing_mod.settings, "paddle_api_key",
                        "pdl_test_dummy", raising=False)
    monkeypatch.setattr(billing_mod.settings, "paddle_environment",
                        "sandbox", raising=False)

    first = billing_mod._client()
    second = billing_mod._client()
    assert first is second
    assert len(calls) == 1
