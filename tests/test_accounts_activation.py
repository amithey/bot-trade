"""Activation must not let self-asserted email claim a Google portfolio."""
import streamlit as st
from dashboard import _accounts, _identity


def test_email_password_identity_does_not_claim_google_identity(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    monkeypatch.setattr(_identity, "auth_mode", lambda: "accounts")
    monkeypatch.setattr(_identity, "is_logged_in", lambda: False)
    monkeypatch.setattr(_accounts, "current_email", lambda: "owner@example.com")
    assert _identity.account_id() == "account:owner@example.com"


def test_existing_google_identity_survives_accounts_activation(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    monkeypatch.setattr(_identity, "auth_mode", lambda: "accounts")
    monkeypatch.setattr(_identity, "is_logged_in", lambda: True)
    monkeypatch.setattr(_identity, "current_user", lambda: {"email": "owner@example.com"})
    assert _identity.account_id() == "user:owner@example.com"


def test_cached_email_cannot_outlive_revoked_session(monkeypatch):
    from types import SimpleNamespace
    state = {"_bt_account_email": "owner@example.com", "_bt_account_token": "revoked"}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(_accounts, "store", lambda: SimpleNamespace(resolve_session=lambda token: None))
    assert _accounts.current_email() is None
    assert "_bt_account_email" not in state
