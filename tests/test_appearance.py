"""Appearance is selected before page rendering without caching a false identity."""
from dashboard import _accounts, _identity, appearance


def test_appearance_path_does_not_cache_anonymous_account(monkeypatch, tmp_path):
    monkeypatch.setattr(_identity, "auth_mode", lambda: "accounts")
    monkeypatch.setattr(_identity, "is_logged_in", lambda: False)
    monkeypatch.setattr(_accounts, "current_email", lambda: None)
    monkeypatch.setattr("dashboard._shared.ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(_identity, "account_slug", lambda identity=None: calls.append(identity) or "anonymous")

    assert appearance._path() == tmp_path / "data/profiles/anonymous.appearance.json"
    assert calls == ["account:unknown"]


def test_existing_provider_keeps_its_appearance_namespace(monkeypatch, tmp_path):
    monkeypatch.setattr(_identity, "auth_mode", lambda: "accounts")
    monkeypatch.setattr(_identity, "is_logged_in", lambda: True)
    monkeypatch.setattr(_identity, "current_user", lambda: {"email": "Trader@Example.com"})
    monkeypatch.setattr("dashboard._shared.ROOT", tmp_path)
    monkeypatch.setattr(_identity, "account_slug", lambda identity=None: identity.replace(":", "_"))

    assert appearance._path().name == "user_trader@example.com.appearance.json"
