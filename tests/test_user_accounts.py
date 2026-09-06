"""Tests for saas/passwords.py and saas/user_accounts.py.

The properties worth pinning down here are the ones a reviewer would want
proof of before letting this near real users: a password is never recoverable
from what is stored, a stolen database yields no usable session, a wrong
password is indistinguishable from a missing account, and revocation and
expiry actually take effect.

Every test uses a temp-file database so nothing touches data/users.db.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from saas import passwords
from saas.user_accounts import (
    MAX_FAILED_ATTEMPTS,
    MIN_PASSWORD_LENGTH,
    UserAccounts,
    normalize_email,
    token_digest,
)

GOOD_PW = "correct horse battery"


@pytest.fixture
def accounts(tmp_path):
    return UserAccounts(tmp_path / "users.db")


# --------------------------------------------------------------------------- #
# passwords.py
# --------------------------------------------------------------------------- #
def test_hash_round_trips_and_rejects_a_near_miss():
    h = passwords.hash_password(GOOD_PW)
    assert passwords.verify_password(GOOD_PW, h)
    assert not passwords.verify_password(GOOD_PW.upper(), h)
    assert not passwords.verify_password("", h)


def test_hash_carries_its_own_random_salt():
    a = passwords.hash_password("pw")
    b = passwords.hash_password("pw")
    assert a != b, "identical passwords must not produce identical hashes"
    assert passwords.verify_password("pw", a)
    assert passwords.verify_password("pw", b)


def test_hash_states_its_algorithm_and_cost():
    algo, iterations, _salt, _dk = passwords.hash_password("pw").split("$", 3)
    assert algo == "pbkdf2_sha256"
    assert int(iterations) == passwords.PBKDF2_ITERATIONS >= 600_000


def test_a_malformed_hash_fails_closed_rather_than_raising():
    for junk in ("", "nonsense", "pbkdf2_sha256$notanint$x$y", "a$b$c$d"):
        assert passwords.verify_password("pw", junk) is False


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_register_then_login(accounts):
    ok, _ = accounts.register("a@b.com", GOOD_PW)
    assert ok
    token, _ = accounts.login("a@b.com", GOOD_PW)
    assert token
    assert accounts.resolve_session(token) == "a@b.com"


def test_email_is_normalised_so_one_person_is_one_account(accounts):
    accounts.register("  Amit@Example.COM ", GOOD_PW)
    assert accounts.user_exists("amit@example.com")
    # Registering the same address differently cased must be refused...
    ok, msg = accounts.register("AMIT@example.com", GOOD_PW)
    assert not ok and "already registered" in msg
    # ...and signing in with any casing must reach the same account.
    token, _ = accounts.login("AMIT@EXAMPLE.COM", GOOD_PW)
    assert accounts.resolve_session(token) == "amit@example.com"


def test_registration_rejects_a_short_password(accounts):
    ok, msg = accounts.register("a@b.com", "x" * (MIN_PASSWORD_LENGTH - 1))
    assert not ok and str(MIN_PASSWORD_LENGTH) in msg
    assert not accounts.user_exists("a@b.com")


@pytest.mark.parametrize("bad", ["", "nope", "a@b", "a b@c.com", "@b.com"])
def test_registration_rejects_a_malformed_address(accounts, bad):
    ok, _ = accounts.register(bad, GOOD_PW)
    assert not ok


# --------------------------------------------------------------------------- #
# What the database is actually worth if it leaks
# --------------------------------------------------------------------------- #
def test_the_password_is_not_recoverable_from_the_row(accounts, tmp_path):
    accounts.register("a@b.com", GOOD_PW)
    conn = sqlite3.connect(tmp_path / "users.db")
    stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert GOOD_PW not in stored
    assert stored.startswith("pbkdf2_sha256$")


def test_a_stolen_session_row_cannot_be_replayed(accounts, tmp_path):
    accounts.register("a@b.com", GOOD_PW)
    token, _ = accounts.login("a@b.com", GOOD_PW)

    conn = sqlite3.connect(tmp_path / "users.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT token_hash FROM sessions").fetchone()

    # The raw token exists only in the caller's hands; the row holds a digest.
    assert row["token_hash"] != token
    assert row["token_hash"] == token_digest(token)
    # Presenting what the database stores must not authenticate anyone.
    assert accounts.resolve_session(row["token_hash"]) is None


# --------------------------------------------------------------------------- #
# Login failures
# --------------------------------------------------------------------------- #
def test_wrong_password_and_unknown_account_are_indistinguishable(accounts):
    accounts.register("a@b.com", GOOD_PW)
    _, wrong_pw = accounts.login("a@b.com", "not the password")
    _, no_user = accounts.login("nobody@b.com", GOOD_PW)
    assert wrong_pw == no_user, "the message must not reveal which accounts exist"


def test_repeated_failures_lock_the_address_out(accounts):
    accounts.register("a@b.com", GOOD_PW)
    for _ in range(MAX_FAILED_ATTEMPTS):
        token, _ = accounts.login("a@b.com", "wrong")
        assert token is None

    # Even the correct password is refused while the cooldown is running.
    token, msg = accounts.login("a@b.com", GOOD_PW)
    assert token is None
    assert "Too many attempts" in msg


def test_a_successful_login_clears_the_failure_count(accounts):
    accounts.register("a@b.com", GOOD_PW)
    for _ in range(MAX_FAILED_ATTEMPTS - 1):
        accounts.login("a@b.com", "wrong")
    assert accounts.login("a@b.com", GOOD_PW)[0]
    for _ in range(MAX_FAILED_ATTEMPTS - 1):
        accounts.login("a@b.com", "wrong")
    assert accounts.login("a@b.com", GOOD_PW)[0], "counter should have reset"


# --------------------------------------------------------------------------- #
# Session lifetime
# --------------------------------------------------------------------------- #
def test_logout_revokes_only_that_session(accounts):
    accounts.register("a@b.com", GOOD_PW)
    laptop, _ = accounts.login("a@b.com", GOOD_PW)
    phone, _ = accounts.login("a@b.com", GOOD_PW)

    accounts.logout(laptop)
    assert accounts.resolve_session(laptop) is None
    assert accounts.resolve_session(phone) == "a@b.com", "other devices stay in"


def test_an_expired_session_stops_resolving(accounts, tmp_path):
    accounts.register("a@b.com", GOOD_PW)
    token, _ = accounts.login("a@b.com", GOOD_PW)

    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    conn = sqlite3.connect(tmp_path / "users.db")
    conn.execute("UPDATE sessions SET expires_at = ?", (past,))
    conn.commit()

    assert accounts.resolve_session(token) is None


def test_unknown_and_empty_tokens_resolve_to_nobody(accounts):
    assert accounts.resolve_session("") is None
    assert accounts.resolve_session("not-a-real-token") is None


def test_changing_the_password_signs_every_session_out(accounts):
    accounts.register("a@b.com", GOOD_PW)
    laptop, _ = accounts.login("a@b.com", GOOD_PW)
    phone, _ = accounts.login("a@b.com", GOOD_PW)

    ok, _ = accounts.change_password("a@b.com", "a whole new password")
    assert ok
    assert accounts.resolve_session(laptop) is None
    assert accounts.resolve_session(phone) is None
    assert accounts.login("a@b.com", GOOD_PW)[0] is None
    assert accounts.login("a@b.com", "a whole new password")[0]


def test_change_password_rejects_a_short_one_and_an_unknown_account(accounts):
    accounts.register("a@b.com", GOOD_PW)
    assert not accounts.change_password("a@b.com", "short")[0]
    assert not accounts.change_password("who@b.com", GOOD_PW)[0]
    assert accounts.login("a@b.com", GOOD_PW)[0], "the old password still works"


def test_purge_expired_removes_dead_rows_only(accounts, tmp_path):
    accounts.register("a@b.com", GOOD_PW)
    live, _ = accounts.login("a@b.com", GOOD_PW)
    dead, _ = accounts.login("a@b.com", GOOD_PW)
    accounts.logout(dead)

    assert accounts.purge_expired() == 1
    assert accounts.resolve_session(live) == "a@b.com"


def test_normalize_email_helper():
    assert normalize_email("  A@B.com ") == "a@b.com"
    assert normalize_email("") == ""
    assert normalize_email(None) == ""
