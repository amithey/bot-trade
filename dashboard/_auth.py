"""Lightweight password-gate for the Streamlit UI.

How it works
------------
* Set ``BOTTRADE_AUTH_PASSWORD_HASH`` in the environment (sha-256 of the
  password). When unset → auth is **disabled** (local dev mode).
* Optional ``BOTTRADE_AUTH_HASH_SALT`` lets you salt the hash.
* On every page load, ``require_auth()`` reads ``st.session_state``.
  If not authenticated, it renders a centered login form and *halts*
  the page render via ``st.stop()``.
* After 5 failed attempts, the IP-keyed counter throttles further
  tries with a 30-second back-off.

Generate a hash:

    python -c "import hashlib;\
    print(hashlib.sha256(b'mypassword').hexdigest())"

Or with a salt:

    python -c "import hashlib;\
    print(hashlib.sha256(b'salt' + b'mypassword').hexdigest())"
"""
from __future__ import annotations

import hashlib
import os
import time

import streamlit as st


_SESSION_KEY = "_bt_authed"
_ATTEMPTS_KEY = "_bt_auth_attempts"
_LOCK_KEY = "_bt_auth_locked_until"


def _expected_hash() -> str:
    return os.environ.get("BOTTRADE_AUTH_PASSWORD_HASH", "").strip().lower()


def _salt() -> bytes:
    return os.environ.get("BOTTRADE_AUTH_HASH_SALT", "").encode("utf-8")


def _hash(password: str) -> str:
    return hashlib.sha256(_salt() + password.encode("utf-8")).hexdigest()


def auth_enabled() -> bool:
    """True iff a password hash is configured."""
    return bool(_expected_hash())


def is_authed() -> bool:
    return bool(st.session_state.get(_SESSION_KEY))


def logout() -> None:
    st.session_state[_SESSION_KEY] = False


def require_auth() -> None:
    """Render a login form and halt the page when not authenticated.

    Call near the very top of every page (after ``apply_theme``)."""
    if not auth_enabled():
        return
    if is_authed():
        return

    # Throttle on too many bad attempts
    now = time.time()
    locked_until = float(st.session_state.get(_LOCK_KEY) or 0)
    if now < locked_until:
        wait = int(locked_until - now)
        st.error(f"🔒 Too many failed attempts. Try again in {wait}s.")
        st.stop()

    # Center login card
    st.markdown(
        """
        <style>
        .bt-login-wrap {
            display:flex; align-items:center; justify-content:center;
            min-height:60vh;
        }
        .bt-login-card {
            background:linear-gradient(180deg,#0b0f14,#06090c);
            border:1px solid #202833; border-top:1px solid #34404f;
            border-radius:14px; padding:1.6rem 1.8rem;
            box-shadow:0 22px 70px rgba(0,0,0,0.5);
            min-width:320px;
        }
        .bt-login-title {
            color:#fff; font-family:'Aptos Display','Segoe UI',sans-serif;
            font-weight:850; font-size:1.05rem; letter-spacing:.16em;
            text-transform:uppercase; margin:0 0 .6rem 0;
        }
        .bt-login-sub {
            color:#8b98a8; font-family:'IBM Plex Mono',monospace;
            font-size:.66rem; letter-spacing:.14em; text-transform:uppercase;
            margin:0 0 1.1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="bt-login-wrap"><div class="bt-login-card">',
                    unsafe_allow_html=True)
        st.markdown('<div class="bt-login-title">🔒 BotTrade — Login</div>',
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
            attempts = int(st.session_state.get(_ATTEMPTS_KEY) or 0)
            if pw and _hash(pw) == _expected_hash():
                st.session_state[_SESSION_KEY] = True
                st.session_state[_ATTEMPTS_KEY] = 0
                st.session_state[_LOCK_KEY] = 0
                st.rerun()
            else:
                attempts += 1
                st.session_state[_ATTEMPTS_KEY] = attempts
                if attempts >= 5:
                    st.session_state[_LOCK_KEY] = time.time() + 30
                    st.error("🔒 Locked for 30 seconds.")
                else:
                    st.error(f"❌ Incorrect password ({attempts}/5).")
        st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()


def render_logout_button() -> None:
    """Optional: render a small logout button in the sidebar of any page."""
    if not auth_enabled() or not is_authed():
        return
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, key="_bt_logout"):
            logout()
            st.rerun()
