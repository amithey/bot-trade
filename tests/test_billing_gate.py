"""
Tests for the payment gate in saas.billing.confirm_checkout_session.

Two defects are covered here.

The gate itself read:

    if session.payment_status != "paid" and session.status != "complete":
        return None

`and` only rejects when *both* are false, so a session that reached
status="complete" carrying payment_status="unpaid" was granted a paid plan.
That is a real Stripe state, not a hypothetical one: a card that clears the
Checkout form but fails on the first invoice leaves exactly that, and the
user still lands on the success URL.

Separately, the function expands the session with its subscription and the
docstring claimed it only upgrades "when Stripe confirms the subscription is
actually active" — but it read nothing from that object except the id.
"""
from __future__ import annotations

import pytest

import saas.billing as billing
from saas.ledger import UsageLedger


@pytest.fixture
def ledger(tmp_path):
    return UsageLedger(db_path=tmp_path / "usage.db")


class _Metadata:
    """Mimics a real Stripe StripeObject: `.to_dict()`, deliberately no `.get()`."""

    def __init__(self, data):
        self._data = dict(data)

    def to_dict(self):
        return dict(self._data)


class _Subscription:
    def __init__(self, sub_id="sub_123", status="active"):
        self.id = sub_id
        self.status = status


class _Session:
    def __init__(self, status="complete", payment_status="paid",
                 subscription=None, account="acct", plan="PRO"):
        self.status = status
        self.payment_status = payment_status
        self.subscription = (subscription if subscription is not None
                             else _Subscription())
        self.client_reference_id = account
        self.metadata = _Metadata({"bottrade_account_id": account,
                                   "plan_id": plan})


@pytest.fixture
def stripe_ok(monkeypatch):
    """Billing enabled, and Session.retrieve returns whatever the test sets."""
    box = {}

    monkeypatch.setattr(billing, "billing_enabled", lambda: True)
    monkeypatch.setattr(billing, "_client", lambda: None)
    monkeypatch.setattr(billing, "_touch_cache", lambda *a, **kw: None)

    class _Retriever:
        @staticmethod
        def retrieve(session_id, expand=None):
            return box["session"]

    class _Checkout:
        Session = _Retriever

    monkeypatch.setattr(billing.stripe, "checkout", _Checkout)
    return box


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def test_a_genuinely_paid_session_grants_the_plan(ledger, stripe_ok):
    stripe_ok["session"] = _Session(status="complete", payment_status="paid")
    assert billing.confirm_checkout_session(ledger, "cs_1") == "PRO"
    assert ledger.get_plan_id("acct") == "PRO"


def test_complete_but_unpaid_is_refused(ledger, stripe_ok):
    """The regression. `and` used to let this through."""
    stripe_ok["session"] = _Session(status="complete", payment_status="unpaid")
    assert billing.confirm_checkout_session(ledger, "cs_1") is None
    assert ledger.get_plan_id("acct") != "PRO"


def test_paid_but_session_not_complete_is_refused(ledger, stripe_ok):
    stripe_ok["session"] = _Session(status="open", payment_status="paid")
    assert billing.confirm_checkout_session(ledger, "cs_1") is None


def test_open_and_unpaid_is_refused(ledger, stripe_ok):
    stripe_ok["session"] = _Session(status="open", payment_status="unpaid")
    assert billing.confirm_checkout_session(ledger, "cs_1") is None


def test_no_payment_required_is_allowed(ledger, stripe_ok):
    """A full-coverage coupon or a trial that charges nothing today is a
    legitimate zero-cost subscription."""
    stripe_ok["session"] = _Session(status="complete",
                                    payment_status="no_payment_required")
    assert billing.confirm_checkout_session(ledger, "cs_1") == "PRO"


# --------------------------------------------------------------------------- #
# Subscription state
# --------------------------------------------------------------------------- #
def test_incomplete_subscription_is_refused(ledger, stripe_ok):
    """First invoice failed — the session can still say complete."""
    stripe_ok["session"] = _Session(
        subscription=_Subscription(status="incomplete"))
    assert billing.confirm_checkout_session(ledger, "cs_1") is None
    assert ledger.get_plan_id("acct") != "PRO"


@pytest.mark.parametrize("state", ["past_due", "unpaid", "canceled",
                                    "incomplete_expired"])
def test_non_entitling_subscription_states_are_refused(ledger, stripe_ok, state):
    stripe_ok["session"] = _Session(subscription=_Subscription(status=state))
    assert billing.confirm_checkout_session(ledger, "cs_1") is None


def test_trialing_subscription_is_allowed(ledger, stripe_ok):
    """A trial is a real subscription that simply has not billed yet."""
    stripe_ok["session"] = _Session(subscription=_Subscription(status="trialing"))
    assert billing.confirm_checkout_session(ledger, "cs_1") == "PRO"


def test_a_subscription_without_a_status_field_still_works(ledger, stripe_ok):
    """Defensive: an SDK shape that carries no status must not hard-fail a
    genuinely paid checkout."""
    class _Bare:
        id = "sub_bare"

    stripe_ok["session"] = _Session(subscription=_Bare())
    assert billing.confirm_checkout_session(ledger, "cs_1") == "PRO"


# --------------------------------------------------------------------------- #
# The checkout and reconciliation paths agree
# --------------------------------------------------------------------------- #
def test_entitling_states_match_what_reconciliation_accepts():
    """sync_subscription_status lists only status="active" subscriptions.

    If the checkout gate admitted a state reconciliation then strips, a user
    would be upgraded on the success page and downgraded minutes later.
    """
    assert "active" in billing._ACTIVE_SUBSCRIPTION_STATES
    assert not (billing._ACTIVE_SUBSCRIPTION_STATES
                & {"incomplete", "past_due", "unpaid", "canceled"})
