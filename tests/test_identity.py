"""
Offline tests for per-user identity and per-account profile storage.

These cover the two failure modes that made the hosted app unsafe for more
than one person:

* one visitor's trading profile overwriting everyone else's;
* a trial budget keyed to something that resets on refresh.

Streamlit is imported but never runs a script here, so ``st.session_state``
behaves as a plain dict-like object and ``st.secrets`` raises exactly as it
does in a deployment with no secrets file — which is the condition the
identity module has to survive.
"""
from __future__ import annotations

import json

import pytest
import streamlit as st

from config.user_profile import UserProfile
from dashboard import _identity


@pytest.fixture(autouse=True)
def clean_session(monkeypatch):
    """Fresh session state and no ambient auth config for every test."""
    try:
        st.session_state.clear()
    except Exception:                                          # noqa: BLE001
        pass
    monkeypatch.delenv("BOTTRADE_AUTH_PASSWORD_HASH", raising=False)
    yield
    try:
        st.session_state.clear()
    except Exception:                                          # noqa: BLE001
        pass


def _fake_user(monkeypatch, **claims):
    """Pretend Streamlit OIDC is configured and someone is signed in."""
    monkeypatch.setattr(_identity, "_auth_secrets",
                        lambda: {"client_id": "x", "cookie_secret": "y"})
    monkeypatch.setattr(_identity, "is_logged_in", lambda: True)
    monkeypatch.setattr(_identity, "current_user", lambda: dict(claims))


# --------------------------------------------------------------------------- #
# Mode detection
# --------------------------------------------------------------------------- #
def test_no_config_is_open_mode():
    assert _identity.auth_mode() == "open"
    assert _identity.account_id() == "local"


def test_missing_secrets_file_does_not_raise():
    """st.secrets raises StreamlitSecretNotFoundError with no secrets.toml."""
    assert _identity.oidc_configured() is False
    assert _identity.is_logged_in() is False


def test_is_logged_in_survives_missing_auth_attribute():
    """st.user.is_logged_in raises AttributeError when [auth] is absent."""
    with pytest.raises(AttributeError):
        _ = st.user.is_logged_in           # the raw attribute really does raise
    assert _identity.is_logged_in() is False   # ...but our wrapper does not


def test_password_gate_is_detected(monkeypatch):
    monkeypatch.setenv("BOTTRADE_AUTH_PASSWORD_HASH", "abc123")
    assert _identity.auth_mode() == "password"
    assert _identity.account_id() == "shared"


def test_oidc_wins_over_password_gate(monkeypatch):
    monkeypatch.setenv("BOTTRADE_AUTH_PASSWORD_HASH", "abc123")
    _fake_user(monkeypatch, email="a@b.com")
    assert _identity.auth_mode() == "oidc"


def test_only_oidc_identifies_individuals(monkeypatch):
    assert _identity.identifies_individuals() is False
    monkeypatch.setenv("BOTTRADE_AUTH_PASSWORD_HASH", "abc123")
    assert _identity.identifies_individuals() is False
    _fake_user(monkeypatch, email="a@b.com")
    assert _identity.identifies_individuals() is True


def test_named_providers_are_listed(monkeypatch):
    monkeypatch.setattr(_identity, "_auth_secrets", lambda: {
        "redirect_uri": "http://x/oauth2callback",
        "cookie_secret": "s",
        "google": {"client_id": "g"},
        "auth0": {"client_id": "a"},
    })
    assert sorted(_identity.oidc_providers()) == ["auth0", "google"]


def test_single_provider_config_lists_no_named_providers(monkeypatch):
    monkeypatch.setattr(_identity, "_auth_secrets", lambda: {
        "redirect_uri": "http://x", "cookie_secret": "s",
        "client_id": "c", "client_secret": "k",
    })
    assert _identity.oidc_providers() == []


# --------------------------------------------------------------------------- #
# Account id
# --------------------------------------------------------------------------- #
def test_email_becomes_the_account_id(monkeypatch):
    _fake_user(monkeypatch, email="Trader@Example.COM", name="Trader")
    assert _identity.account_id() == "user:trader@example.com"


def test_account_id_is_stable_across_calls(monkeypatch):
    _fake_user(monkeypatch, email="a@b.com")
    assert _identity.account_id() == _identity.account_id()


def test_account_id_falls_back_to_sub_without_email(monkeypatch):
    _fake_user(monkeypatch, sub="1234567890")
    assert _identity.account_id() == "sub:1234567890"


def test_two_users_get_different_account_ids(monkeypatch):
    _fake_user(monkeypatch, email="a@b.com")
    a = _identity.account_id()
    _identity.forget_identity()
    _fake_user(monkeypatch, email="c@d.com")
    assert _identity.account_id() != a


# --------------------------------------------------------------------------- #
# Slugs
# --------------------------------------------------------------------------- #
def test_slug_is_filesystem_safe():
    slug = _identity.account_slug("user:trader@example.com")
    assert "/" not in slug and ":" not in slug and "@" not in slug
    assert slug == _identity.account_slug("user:trader@example.com")


def test_slugs_that_read_alike_still_differ():
    """Punctuation collapses to '_', so the hash suffix does the separating."""
    a = _identity.account_slug("user:a+b@x.com")
    b = _identity.account_slug("user:a_b@x.com")
    assert a != b


def test_long_account_ids_stay_bounded():
    assert len(_identity.account_slug("user:" + "x" * 500)) < 80


# --------------------------------------------------------------------------- #
# Per-account profile storage
# --------------------------------------------------------------------------- #
def test_profiles_are_written_per_account(monkeypatch, tmp_path):
    from dashboard import _shared
    monkeypatch.setattr(_shared, "ROOT", tmp_path)

    _fake_user(monkeypatch, email="a@b.com")
    path_a = _shared.profile_path()
    _identity.forget_identity()
    _fake_user(monkeypatch, email="c@d.com")
    path_b = _shared.profile_path()

    assert path_a != path_b
    assert path_a.parent == tmp_path / "data" / "profiles"


def test_one_users_profile_does_not_overwrite_anothers(monkeypatch, tmp_path):
    """The exact bug this change exists to fix."""
    from dashboard import _shared
    monkeypatch.setattr(_shared, "ROOT", tmp_path)

    _fake_user(monkeypatch, email="a@b.com")
    UserProfile(capital=50_000, watchlist=["BTC-USD"]).save(_shared.profile_path())

    _identity.forget_identity()
    _fake_user(monkeypatch, email="c@d.com")
    UserProfile(capital=1_000, watchlist=["TSLA"]).save(_shared.profile_path())

    _identity.forget_identity()
    _fake_user(monkeypatch, email="a@b.com")
    reloaded = UserProfile.load(_shared.profile_path())
    assert reloaded.capital == 50_000
    assert reloaded.watchlist == ["BTC-USD"]


def test_legacy_single_user_profile_is_adopted_once(monkeypatch, tmp_path):
    """An existing deployment keeps its settings after the upgrade."""
    from dashboard import _shared
    monkeypatch.setattr(_shared, "ROOT", tmp_path)
    legacy = tmp_path / "data" / "user_profile.json"
    monkeypatch.setattr(_shared, "LEGACY_PROFILE_PATH", legacy)
    UserProfile(capital=77_000, risk_profile="Aggressive").save(legacy)

    _fake_user(monkeypatch, email="a@b.com")
    adopted = _shared._load_profile_for(_identity.account_id())
    assert adopted.capital == 77_000
    assert adopted.risk_profile == "Aggressive"
    assert _shared.profile_path().exists()


def test_legacy_adoption_does_not_leak_to_a_second_account(monkeypatch, tmp_path):
    """Adoption is a one-time migration, not a shared default forever.

    The first account through the door inherits the old settings; a genuinely
    new user must start from defaults, not from a stranger's capital.
    """
    from dashboard import _shared
    monkeypatch.setattr(_shared, "ROOT", tmp_path)
    legacy = tmp_path / "data" / "user_profile.json"
    monkeypatch.setattr(_shared, "LEGACY_PROFILE_PATH", legacy)
    UserProfile(capital=77_000).save(legacy)

    _fake_user(monkeypatch, email="first@b.com")
    assert _shared._load_profile_for(_identity.account_id()).capital == 77_000

    _identity.forget_identity()
    _fake_user(monkeypatch, email="second@b.com")
    second = _shared._load_profile_for(_identity.account_id())
    assert second.capital == UserProfile().capital
    assert not legacy.exists(), "legacy file should be retired after adoption"


def test_first_account_keeps_its_adopted_profile_on_reload(monkeypatch, tmp_path):
    """Retiring the legacy file must not cost the first user their settings."""
    from dashboard import _shared
    monkeypatch.setattr(_shared, "ROOT", tmp_path)
    legacy = tmp_path / "data" / "user_profile.json"
    monkeypatch.setattr(_shared, "LEGACY_PROFILE_PATH", legacy)
    UserProfile(capital=77_000, risk_profile="Aggressive").save(legacy)

    _fake_user(monkeypatch, email="first@b.com")
    _shared._load_profile_for(_identity.account_id())          # adopt + retire
    again = _shared._load_profile_for(_identity.account_id())  # reload
    assert again.capital == 77_000
    assert again.risk_profile == "Aggressive"


def test_profile_round_trips_through_json(monkeypatch, tmp_path):
    from dashboard import _shared
    monkeypatch.setattr(_shared, "ROOT", tmp_path)
    _fake_user(monkeypatch, email="a@b.com")
    p = _shared.profile_path()
    UserProfile(capital=12_345, trade_size_pct=35,
                risk_profile="Micro-Scalp", watchlist=["SOL-USD"]).save(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["capital"] == 12_345
    assert UserProfile.load(p).risk_profile == "Micro-Scalp"


# --------------------------------------------------------------------------- #
# Session isolation
# --------------------------------------------------------------------------- #
def test_switching_account_clears_the_previous_users_session_data():
    from dashboard import _shared
    st.session_state["portfolio"] = object()
    st.session_state[_shared._USER_KEY_SLOT] = "sk-ant-api03-" + "a" * 40
    st.session_state["watchlist"] = ["BTC-USD"]
    st.session_state["starting_capital"] = 999_999

    _shared._reset_user_scoped_state()

    assert "portfolio" not in st.session_state
    assert _shared._USER_KEY_SLOT not in st.session_state, \
        "one person's API key must never reach the next person to sign in"
    assert "watchlist" not in st.session_state
    assert "starting_capital" not in st.session_state


def test_the_engine_is_not_session_scoped():
    """The engine lives in the registry, so the reset must not try to own it.

    Listing it among the session-scoped keys would be a claim that a refresh
    can drop it — exactly the orphaned-thread bug the registry removes.
    """
    from dashboard import _shared
    assert "_live_engine" not in _shared._USER_SCOPED_KEYS
