"""Email + password accounts, and the sessions that keep someone signed in.

Why this exists
---------------
Before this, a deployment could tell people apart only through OIDC — a
Google button and nothing else — or not at all (one shared password meant
one shared portfolio and one shared budget). Someone who wanted an account
on this product without handing it a Google identity had no way to get one,
and no way to be remembered between visits.

What a stolen row is worth to an attacker
-----------------------------------------
Nothing that can be replayed, which is the whole design goal:

* Passwords are PBKDF2-HMAC-SHA256 with a random per-user salt
  (:mod:`saas.passwords`). The database never sees a password.
* Session tokens are random 256-bit values and only their SHA-256 is
  stored, so a stolen database yields no usable session either — the raw
  token exists only in the holder's browser cookie. That is why lookup is
  by digest rather than by a token column: there is nothing else to
  compare against.

SHA-256 is right for the token specifically, in contrast to the
deliberately slow PBKDF2 used for passwords: a 256-bit random token has no
guessable structure to brute-force, so the reason to make hashing slow does
not apply, and this runs on every page load.

Scope
-----
Registration, login, logout, session resolution and password change.
Deliberately not here: email verification and password reset both need an
outbound mail path this project does not have, so an operator running in
this mode should know that a forgotten password means an operator-side
reset.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from saas.passwords import hash_password, verify_password
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_DB = Path("data") / "users.db"

#: How long a session stays valid. Long enough that "stay signed in" means
#: something, short enough that a token copied off an abandoned laptop does
#: not live forever.
SESSION_TTL_DAYS = 30

#: Matches the floor the shared-password CLI in dashboard/_auth.py already
#: enforces, so the two doors into this app do not disagree.
MIN_PASSWORD_LENGTH = 8

#: Failed logins tolerated per address before a cooldown. Keyed by email
#: rather than by session: a session-scoped counter is no defence at all,
#: since opening a new tab resets it.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60.0

#: Intentionally permissive — a typo check, not an RFC 5322 parser. The only
#: real proof an address works is mail delivered to it.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email          TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    last_login_at  TEXT
);

-- Only the SHA-256 of a token is ever stored; see the module docstring.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    email       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_sessions_email ON sessions (email);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    """Addresses match case-insensitively and without stray spaces.

    Without this, ``Amit@x.com`` and ``amit@x.com`` become two accounts with
    two separate portfolios, and the person owning both would never work out
    why their positions keep vanishing.
    """
    return (email or "").strip().lower()


def token_digest(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


class UserAccounts:
    """Thread-safe SQLite store of accounts and their live sessions."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._path = Path(db_path or _DEFAULT_DB)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, timeout=15.0,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        # Failed-attempt counters live in memory on purpose: a restart
        # clearing them is acceptable, and it keeps a failed login from
        # writing to disk, which is exactly the load a password-spraying
        # attacker would otherwise get to generate for free.
        self._failures: dict[str, tuple[int, float]] = {}

    # -- registration ------------------------------------------------------
    def user_exists(self, email: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM users WHERE email = ?", (normalize_email(email),),
            ).fetchone()
        return row is not None

    def register(self, email: str, password: str) -> tuple[bool, str]:
        """Create an account. Returns ``(ok, message)``.

        The message is meant to be shown verbatim, so it says what is wrong
        with the input rather than refusing generically: at registration the
        address is being chosen, and confirming it is taken tells an attacker
        nothing they could not learn by trying to register it themselves.
        """
        email = normalize_email(email)
        if not _EMAIL_RE.match(email):
            return False, "That does not look like an email address."
        if len(password or "") < MIN_PASSWORD_LENGTH:
            return False, (f"Use at least {MIN_PASSWORD_LENGTH} characters "
                           f"for the password.")
        if self.user_exists(email):
            return False, "That email is already registered — sign in instead."

        with self._lock:
            self._conn.execute(
                "INSERT INTO users (email, password_hash, created_at) "
                "VALUES (?, ?, ?)",
                (email, hash_password(password), _now().isoformat()),
            )
            self._conn.commit()
        logger.info("[accounts] registered %s", email)
        return True, "Account created."

    # -- throttling --------------------------------------------------------
    def _locked_out(self, email: str) -> float:
        """Seconds left in this address's cooldown, 0 when clear."""
        count, until = self._failures.get(email, (0, 0.0))
        if count < MAX_FAILED_ATTEMPTS:
            return 0.0
        remaining = until - time.time()
        if remaining <= 0:
            self._failures.pop(email, None)
            return 0.0
        return remaining

    def _record_failure(self, email: str) -> None:
        count, _ = self._failures.get(email, (0, 0.0))
        self._failures[email] = (count + 1, time.time() + LOCKOUT_SECONDS)

    # -- login / sessions --------------------------------------------------
    def login(self, email: str, password: str) -> tuple[Optional[str], str]:
        """Verify credentials and open a session. Returns ``(token, message)``.

        The failure message never separates "no such account" from "wrong
        password". Saying which would turn this form into a free tool for
        discovering who holds an account here.
        """
        email = normalize_email(email)
        wait = self._locked_out(email)
        if wait > 0:
            return None, f"Too many attempts. Try again in {int(wait) + 1}s."

        with self._lock:
            row = self._conn.execute(
                "SELECT password_hash FROM users WHERE email = ?", (email,),
            ).fetchone()

        if row is None or not verify_password(password, row["password_hash"]):
            self._record_failure(email)
            return None, "Incorrect email or password."

        self._failures.pop(email, None)
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (token_hash, email, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (token_digest(token), email, now.isoformat(),
                 (now + timedelta(days=SESSION_TTL_DAYS)).isoformat()),
            )
            self._conn.execute(
                "UPDATE users SET last_login_at = ? WHERE email = ?",
                (now.isoformat(), email),
            )
            self._conn.commit()
        logger.info("[accounts] %s signed in", email)
        return token, "Signed in."

    def resolve_session(self, token: str) -> Optional[str]:
        """The email a live session token belongs to, or ``None``.

        Runs on every page load, so it stays one indexed primary-key lookup
        on the digest.
        """
        if not token:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT email, expires_at, revoked FROM sessions "
                "WHERE token_hash = ?", (token_digest(token),),
            ).fetchone()
        if row is None or row["revoked"]:
            return None
        try:
            if datetime.fromisoformat(row["expires_at"]) <= _now():
                return None
        except ValueError:
            return None
        return row["email"]

    def logout(self, token: str) -> None:
        """Revoke one session. Other devices stay signed in."""
        if not token:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET revoked = 1 WHERE token_hash = ?",
                (token_digest(token),),
            )
            self._conn.commit()

    def change_password(self, email: str, new_password: str) -> tuple[bool, str]:
        """Set a new password and revoke every existing session for it.

        Revoking is the point: changing a password is what someone does when
        they think a session may be in the wrong hands, so it has to actually
        end those sessions.
        """
        email = normalize_email(email)
        if len(new_password or "") < MIN_PASSWORD_LENGTH:
            return False, (f"Use at least {MIN_PASSWORD_LENGTH} characters "
                           f"for the password.")
        if not self.user_exists(email):
            return False, "No such account."
        with self._lock:
            self._conn.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (hash_password(new_password), email),
            )
            self._conn.execute(
                "UPDATE sessions SET revoked = 1 WHERE email = ?", (email,),
            )
            self._conn.commit()
        return True, "Password changed. Other sessions were signed out."

    def purge_expired(self) -> int:
        """Drop dead session rows. Returns how many went."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE expires_at <= ? OR revoked = 1",
                (_now().isoformat(),),
            )
            self._conn.commit()
            return cur.rowcount or 0
