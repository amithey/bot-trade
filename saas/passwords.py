"""Password hashing primitives, shared by every place that stores one.

Extracted from ``dashboard/_auth.py`` when per-user accounts arrived, so the
shared-deployment gate and the per-user account store derive and check
passwords through exactly one implementation. Two copies of security code is
two chances to fix a flaw in only one of them.

Format
------
Self-describing, so a stored hash carries everything needed to verify it and
the cost can be raised later without invalidating existing hashes::

    pbkdf2_sha256$<iterations>$<b64 salt>$<b64 derived key>

The salt is random per hash, never shared and never derived from the
password: two people who choose the same password still get different
stored values, so one cracked hash says nothing about the other, and a
precomputed rainbow table is useless against either.

Deliberately stdlib-only and free of any Streamlit import, so it can be
unit-tested and reused outside the dashboard process.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

#: OWASP-recommended floor for PBKDF2-HMAC-SHA256. Raise it, never lower it —
#: the value is stored inside each hash, so old hashes keep verifying either
#: way and only newly created ones pay the higher cost.
PBKDF2_ITERATIONS = 600_000
PBKDF2_PREFIX = "pbkdf2_sha256"

#: 16 bytes = 128 bits of salt. Longer buys nothing here; shorter starts to
#: make cross-user precomputation worthwhile again.
_SALT_BYTES = 16


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Derive a storable hash for *password* with a fresh random salt."""
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "{}${}${}${}".format(
        PBKDF2_PREFIX,
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def is_pbkdf2_hash(stored: str) -> bool:
    return stored.strip().startswith(PBKDF2_PREFIX + "$")


def verify_password(password: str, stored: str) -> bool:
    """Check *password* against a ``pbkdf2_sha256$...`` hash, in constant time.

    Comparing digests with ``==`` short-circuits on the first differing byte,
    which leaks how much of a guess was right. ``hmac.compare_digest`` does
    not. A malformed or empty hash verifies as False rather than raising —
    a corrupt row must fail closed, not crash the login page.
    """
    stored = (stored or "").strip()
    if not stored or not password or not is_pbkdf2_hash(stored):
        return False
    try:
        _, iterations, b64_salt, b64_dk = stored.split("$", 3)
        salt = base64.b64decode(b64_salt)
        want = base64.b64decode(b64_dk)
        got = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, want)
