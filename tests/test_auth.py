"""
Tests for the shared-password gate.

The gate protects a deployment that is reachable from the internet, so the
properties worth pinning down are the ones an attacker would probe: how
expensive a guess is, whether the comparison leaks, whether the same password
produces the same stored value twice, and whether the lockout can be reset by
simply opening a new tab.
"""
from __future__ import annotations

import hashlib
import time

import pytest

from dashboard import _auth


@pytest.fixture(autouse=True)
def clean_throttle(monkeypatch):
    monkeypatch.delenv("BOTTRADE_AUTH_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("BOTTRADE_AUTH_HASH_SALT", raising=False)
    _auth._reset_throttle_for_tests()
    yield
    _auth._reset_throttle_for_tests()


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def test_hash_round_trips():
    h = _auth.make_hash("correct horse battery staple")
    assert _auth.verify_password("correct horse battery staple", h)
    assert not _auth.verify_password("Correct horse battery staple", h)
    assert not _auth.verify_password("", h)


def test_hash_is_self_describing():
    h = _auth.make_hash("pw")
    algo, iterations, salt, dk = h.split("$")
    assert algo == "pbkdf2_sha256"
    assert int(iterations) == _auth._PBKDF2_ITERATIONS >= 600_000
    assert salt and dk


def test_same_password_hashes_differently_every_time():
    """A random per-hash salt is what makes rainbow tables useless."""
    a = _auth.make_hash("pw")
    b = _auth.make_hash("pw")
    assert a != b
    assert _auth.verify_password("pw", a)
    assert _auth.verify_password("pw", b)


def test_iteration_count_is_honoured_from_the_stored_hash():
    """Old hashes keep verifying after the default iteration count is raised."""
    h = _auth.make_hash("pw", iterations=1000)
    assert "$1000$" in h
    assert _auth.verify_password("pw", h)


def test_pbkdf2_is_costly_enough_to_matter():
    """A single SHA-256 is ~microseconds; the whole point is that this is not."""
    h = _auth.make_hash("pw")
    start = time.perf_counter()
    _auth.verify_password("pw", h)
    assert time.perf_counter() - start > 0.01


def test_malformed_hash_is_rejected_not_crashed():
    for bad in ("pbkdf2_sha256$notanint$xx$yy", "pbkdf2_sha256$1000$!!$!!",
                "pbkdf2_sha256$", "garbage", ""):
        assert _auth.verify_password("pw", bad) is False


# --------------------------------------------------------------------------- #
# Legacy compatibility
# --------------------------------------------------------------------------- #
def test_legacy_sha256_hash_still_verifies():
    """Existing deployments must not be locked out by the upgrade."""
    legacy = hashlib.sha256(b"hunter2").hexdigest()
    assert _auth.verify_password("hunter2", legacy)
    assert not _auth.verify_password("wrong", legacy)


def test_legacy_salted_hash_still_verifies(monkeypatch):
    monkeypatch.setenv("BOTTRADE_AUTH_HASH_SALT", "MYSALT")
    legacy = hashlib.sha256(b"MYSALT" + b"hunter2").hexdigest()
    assert _auth.verify_password("hunter2", legacy)


def test_legacy_format_is_flagged():
    assert _auth.hash_is_legacy(hashlib.sha256(b"x").hexdigest())
    assert not _auth.hash_is_legacy(_auth.make_hash("x"))


def test_legacy_salt_is_ignored_by_new_hashes(monkeypatch):
    """PBKDF2 hashes carry their own salt; the env var must not affect them."""
    h = _auth.make_hash("pw")
    monkeypatch.setenv("BOTTRADE_AUTH_HASH_SALT", "irrelevant")
    assert _auth.verify_password("pw", h)


# --------------------------------------------------------------------------- #
# Enablement
# --------------------------------------------------------------------------- #
def test_auth_is_disabled_without_a_hash():
    assert _auth.auth_enabled() is False


def test_auth_is_enabled_with_a_hash(monkeypatch):
    monkeypatch.setenv("BOTTRADE_AUTH_PASSWORD_HASH", _auth.make_hash("pw"))
    assert _auth.auth_enabled() is True


def test_whitespace_around_the_env_hash_is_tolerated(monkeypatch):
    monkeypatch.setenv("BOTTRADE_AUTH_PASSWORD_HASH",
                       "  " + _auth.make_hash("pw") + "\n")
    assert _auth.verify_password("pw")


# --------------------------------------------------------------------------- #
# Throttling
# --------------------------------------------------------------------------- #
def test_lockout_survives_a_new_session():
    """The old counter lived in st.session_state, so a new tab reset it.

    That made the 5-attempt limit meaningless against anyone willing to
    reconnect. The counter is now per process.
    """
    for _ in range(_auth._MAX_ATTEMPTS):
        _auth._register_failure()
    assert _auth._lock_remaining() > 0


def test_failures_below_the_limit_do_not_lock():
    for _ in range(_auth._MAX_ATTEMPTS - 1):
        _auth._register_failure()
    assert _auth._lock_remaining() == 0


def test_a_success_clears_the_counter():
    _auth._register_failure()
    _auth._register_failure()
    _auth._register_success()
    assert _auth._lock_remaining() == 0
    for _ in range(_auth._MAX_ATTEMPTS - 1):
        _auth._register_failure()
    assert _auth._lock_remaining() == 0, "counter should have restarted at zero"
