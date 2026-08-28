"""Shared-password gate for the Streamlit UI.

This is the *access* gate, not the identity layer. Everyone who knows the
password shares one account, one trading profile and one budget — see
:mod:`dashboard._identity`, which treats this as ``password`` mode and warns
where that matters. For per-person accounts configure OIDC instead
(DEPLOY.md section 2b); this gate remains for private deployments that do not
want to stand up an OAuth provider.

Password storage
----------------
Hashes are PBKDF2-HMAC-SHA256 with a random per-deployment salt, in a
self-describing format::

    pbkdf2_sha256$<iterations>$<b64 salt>$<b64 derived key>

Generate one with::

    python -m dashboard._auth

The older bare-SHA-256 format (a 64-character hex digest, optionally salted
via ``BOTTRADE_AUTH_HASH_SALT``) is still accepted so existing deployments
keep working, but it is a single unsalted-by-default round — trivially
brute-forced on a GPU — and :func:`hash_is_legacy` reports it so the UI can
say so. Rotate it.

Configuration
-------------
* ``BOTTRADE_AUTH_PASSWORD_HASH`` — the hash. Unset → auth disabled (local dev).
* ``BOTTRADE_AUTH_HASH_SALT`` — legacy format only; ignored by PBKDF2 hashes,
  which carry their own salt.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import time

import streamlit as st

_SESSION_KEY = "_bt_authed"

#: OWASP-recommended floor for PBKDF2-HMAC-SHA256. Raise it, never lower it —
#: the value is stored in the hash, so old hashes keep verifying either way.
_PBKDF2_ITERATIONS = 600_000
_PBKDF2_PREFIX = "pbkdf2_sha256"
_LEGACY_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

#: Failed attempts are counted per process, not per session. A session-scoped
#: counter is no defence at all: opening a new tab resets it, so an attacker
#: gets unlimited tries. This is coarse — one slow attacker locks out everyone
#: — but for a single shared password that trade is the right way round.
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 60.0
_throttle_lock = threading.Lock()
_failed_attempts = 0
_locked_until = 0.0


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def make_hash(password: str, iterations: int = _PBKDF2_ITERATIONS) -> str:
    """Derive a storable hash for *password*, with a fresh random salt."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "${}${}${}".format(
        _PBKDF2_PREFIX + f"${iterations}",
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    ).lstrip("$")


def _expected_hash() -> str:
    return os.environ.get("BOTTRADE_AUTH_PASSWORD_HASH", "").strip()


def _legacy_salt() -> bytes:
    return os.environ.get("BOTTRADE_AUTH_HASH_SALT", "").encode("utf-8")


def hash_is_legacy(stored: str | None = None) -> bool:
    """True when the configured hash is the old bare-SHA-256 format."""
    value = (stored if stored is not None else _expected_hash()).strip().lower()
    return bool(_LEGACY_HEX_RE.match(value))


def verify_password(password: str, stored: str | None = None) -> bool:
    """Check *password* against the configured hash, in constant time.

    Comparing digests with ``==`` short-circuits on the first differing byte,
    which leaks how much of a guess was right. ``hmac.compare_digest`` does not.
    """
    expected = (stored if stored is not None else _expected_hash()).strip()
    if not expected or not password:
        return False

    if expected.startswith(_PBKDF2_PREFIX + "$"):
        try:
            _, iterations, b64_salt, b64_dk = expected.split("$", 3)
            salt = base64.b64decode(b64_salt)
            want = base64.b64decode(b64_dk)
            got = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, int(iterations))
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(got, want)

    if hash_is_legacy(expected):
        got = hashlib.sha256(
            _legacy_salt() + password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(got, expected.lower())

    return False


def auth_enabled() -> bool:
    """True iff a password hash is configured."""
    return bool(_expected_hash())


def is_authed() -> bool:
    return bool(st.session_state.get(_SESSION_KEY))


def logout() -> None:
    st.session_state[_SESSION_KEY] = False


# --------------------------------------------------------------------------- #
# Throttling
# --------------------------------------------------------------------------- #
def _lock_remaining() -> int:
    with _throttle_lock:
        return max(0, int(_locked_until - time.time()))


def _register_failure() -> int:
    """Count a bad attempt; returns attempts used so far."""
    global _failed_attempts, _locked_until
    with _throttle_lock:
        _failed_attempts += 1
        if _failed_attempts >= _MAX_ATTEMPTS:
            _locked_until = time.time() + _LOCKOUT_SECONDS
            _failed_attempts = 0
            return _MAX_ATTEMPTS
        return _failed_attempts


def _register_success() -> None:
    global _failed_attempts, _locked_until
    with _throttle_lock:
        _failed_attempts = 0
        _locked_until = 0.0


def _reset_throttle_for_tests() -> None:
    _register_success()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
_LOGIN_CSS = """
<style>
.bt-login-wrap { display:flex; align-items:center; justify-content:center;
                 min-height:60vh; }
.bt-login-card { background:linear-gradient(180deg,#0b0f14,#06090c);
                 border:1px solid #202833; border-top:1px solid #34404f;
                 border-radius:14px; padding:1.6rem 1.8rem;
                 box-shadow:0 22px 70px rgba(0,0,0,0.5); min-width:320px; }
.bt-login-title { color:#fff; font-family:'Aptos Display','Segoe UI',sans-serif;
                  font-weight:850; font-size:1.05rem; letter-spacing:.16em;
                  text-transform:uppercase; margin:0 0 .6rem 0; }
.bt-login-sub { color:#8b98a8; font-family:'IBM Plex Mono',monospace;
                font-size:.66rem; letter-spacing:.14em; text-transform:uppercase;
                margin:0 0 1.1rem 0; }
</style>
"""


def require_auth() -> None:
    """Render a login form and halt the page when not authenticated.

    Call near the top of every page (after ``apply_theme``)."""
    if not auth_enabled() or is_authed():
        return

    wait = _lock_remaining()
    if wait:
        st.error(f"Too many failed attempts. Try again in {wait}s.")
        st.stop()

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="bt-login-wrap"><div class="bt-login-card">',
                    unsafe_allow_html=True)
        st.markdown('<div class="bt-login-title">BotTrade — Login</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="bt-login-sub">Authentication required to access '
            'the dashboard.</div>',
            unsafe_allow_html=True,
        )
        with st.form("bt_login", clear_on_submit=True):
            pw = st.text_input("Password", type="password",
                               label_visibility="collapsed",
                               placeholder="Password")
            submit = st.form_submit_button("UNLOCK", use_container_width=True,
                                           type="primary")
        if submit:
            if verify_password(pw):
                _register_success()
                st.session_state[_SESSION_KEY] = True
                st.rerun()
            else:
                used = _register_failure()
                if used >= _MAX_ATTEMPTS:
                    st.error(f"Locked for {int(_LOCKOUT_SECONDS)} seconds.")
                else:
                    st.error(f"Incorrect password ({used}/{_MAX_ATTEMPTS}).")

        if hash_is_legacy():
            st.caption(
                "This deployment still uses the legacy SHA-256 password hash. "
                "Regenerate it with `python -m dashboard._auth`."
            )
        st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()


def render_logout_button() -> None:
    """Optional: render a small logout button in the sidebar of any page."""
    if not auth_enabled() or not is_authed():
        return
    with st.sidebar:
        st.markdown("---")
        if st.button("Log out", use_container_width=True, key="_bt_logout"):
            logout()
            st.rerun()


# --------------------------------------------------------------------------- #
# CLI: python -m dashboard._auth
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import getpass

    pw1 = getpass.getpass("Password: ")
    pw2 = getpass.getpass("Confirm:  ")
    if pw1 != pw2:
        raise SystemExit("Passwords do not match.")
    if len(pw1) < 8:
        raise SystemExit("Use at least 8 characters.")
    print("\nAdd this to your .env (the salt is inside the hash — "
          "BOTTRADE_AUTH_HASH_SALT is not needed):\n")
    print(f"BOTTRADE_AUTH_PASSWORD_HASH={make_hash(pw1)}")
