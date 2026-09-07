"""The email + password front door: cookie, forms, and who is signed in.

:mod:`saas.user_accounts` owns the accounts themselves and knows nothing
about Streamlit. This module is the other half — it holds the browser
session together and draws the forms — and it is the only place that touches
the cookie component, so swapping that dependency later means editing one
file.

Why a cookie at all
-------------------
``st.session_state`` does not survive a page refresh, so a login kept only
there is lost the moment someone reloads — which is exactly the complaint
this feature exists to answer. Streamlit's own ``st.context.cookies`` is a
read-only mapping, and app code has no first-party way to *set* a cookie, so
writing one needs a component. What the cookie holds is a random session
token, never an email and never anything derived from the password; the
token is only meaningful against the sessions table, which stores just its
SHA-256.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import streamlit as st

from config.settings import settings
from saas.user_accounts import SESSION_TTL_DAYS, UserAccounts

#: Name of the browser cookie carrying the session token.
COOKIE_NAME = "bottrade_session"

_EMAIL_SLOT = "_bt_account_email"
_TOKEN_SLOT = "_bt_account_token"
_PENDING_SLOT = "_bt_cookie_pending"


def enabled() -> bool:
    return bool(settings.auth_accounts_enabled)


@st.cache_resource(show_spinner=False)
def store() -> UserAccounts:
    """One accounts store per process, opened lazily."""
    return UserAccounts()


def _cookie_from_browser() -> Optional[str]:
    """The session cookie as the browser sent it, or ``None``.

    ``st.context.cookies`` is Streamlit's own read-only view of the request
    cookies, so it is populated on the very first run of a session with no
    component, no round trip and no waiting. Reading through the cookie
    *component* instead was the mistake in the first version of this: that
    component only learns the browser's cookies after its own render has
    made it to the frontend and back, so a returning visitor was shown the
    login form on the run where the answer had not arrived yet.
    """
    try:
        return st.context.cookies.get(COOKIE_NAME) or None
    except Exception:                                          # noqa: BLE001
        return None


def _write_token(token: str) -> None:
    """Remember the token and queue the cookie write for the next run.

    Queued rather than written inline because the component only pushes to
    the browser when the run it belongs to finishes, and a login handler ends
    in ``st.rerun()``, which tears that run down first.
    """
    st.session_state[_TOKEN_SLOT] = token
    st.session_state[_PENDING_SLOT] = token


def flush_cookie_writes() -> None:
    """Perform any queued cookie write or delete. Call once per page load.

    Mounts the cookie component only when there is something to write, which
    is a handful of runs in a session's life — signing in, and signing out.
    Reading never needs it.
    """
    pending = st.session_state.pop(_PENDING_SLOT, None)
    if pending is None:
        return
    try:
        import extra_streamlit_components as stx
        mgr = stx.CookieManager(key="bt_cookie_writer")
        if pending == "":
            mgr.delete(COOKIE_NAME, key="bt_cookie_del")
            return
        mgr.set(
            COOKIE_NAME, pending,
            key="bt_cookie_set",
            expires_at=datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS),
            # Sent only over HTTPS in production. `secure` stays off on
            # plain-HTTP localhost, where the browser would drop the cookie
            # and local development could never stay signed in.
            secure=settings.bottrade_base_url.startswith("https://"),
            same_site="lax",
        )
    except Exception:                                          # noqa: BLE001
        # Worst case the login lasts for this session only, which is still a
        # working login — not a reason to fail the page.
        pass


def _read_token() -> Optional[str]:
    token = st.session_state.get(_TOKEN_SLOT)
    if token:
        return token
    token = _cookie_from_browser()
    if token:
        st.session_state[_TOKEN_SLOT] = token
    return token


def _clear_token() -> None:
    st.session_state.pop(_TOKEN_SLOT, None)
    st.session_state.pop(_EMAIL_SLOT, None)
    # An empty string is the queue's "delete this" marker.
    st.session_state[_PENDING_SLOT] = ""


def current_email() -> Optional[str]:
    """The signed-in address, or ``None``.

    Revalidate on every page pass so password changes, expiry and session
    revocation invalidate an already open dashboard too.
    """
    token = _read_token()
    if not token:
        return None
    email = store().resolve_session(token)
    if email:
        st.session_state[_EMAIL_SLOT] = email
    else:
        # Expired or revoked — drop it so the form is shown rather than a
        # half-signed-in page.
        _clear_token()
    return email


def is_signed_in() -> bool:
    return current_email() is not None


def sign_out() -> None:
    token = st.session_state.get(_TOKEN_SLOT)
    if token:
        store().logout(token)
    _clear_token()


# --------------------------------------------------------------------------- #
# Forms
# --------------------------------------------------------------------------- #
def render_forms() -> None:
    """Draw the sign-in / register tabs. Call inside the login card."""
    accounts = store()
    tab_in, tab_up = st.tabs(["Sign in", "Create account"])

    with tab_in:
        with st.form("bt_signin", clear_on_submit=False):
            email = st.text_input("Email", key="bt_in_email",
                                  autocomplete="username")
            password = st.text_input("Password", type="password",
                                     key="bt_in_pw",
                                     autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", width="stretch",
                                              type="primary")
        if submitted:
            token, msg = accounts.login(email, password)
            if token:
                _write_token(token)
                st.session_state[_EMAIL_SLOT] = accounts.resolve_session(token)
                st.rerun()
            else:
                st.error(msg)

    with tab_up:
        with st.form("bt_signup", clear_on_submit=False):
            email = st.text_input("Email", key="bt_up_email",
                                  autocomplete="username")
            password = st.text_input("Password", type="password",
                                     key="bt_up_pw",
                                     autocomplete="new-password",
                                     help="At least 8 characters.")
            confirm = st.text_input("Confirm password", type="password",
                                    key="bt_up_pw2",
                                    autocomplete="new-password")
            submitted = st.form_submit_button("Create account",
                                              width="stretch", type="primary")
        if submitted:
            if password != confirm:
                st.error("The two passwords do not match.")
            else:
                ok, msg = accounts.register(email, password)
                if not ok:
                    st.error(msg)
                else:
                    # Registering signs you straight in; making someone type
                    # the same credentials again immediately serves nothing.
                    token, _ = accounts.login(email, password)
                    if token:
                        _write_token(token)
                        st.session_state[_EMAIL_SLOT] = \
                            accounts.resolve_session(token)
                        st.rerun()
                    st.success(msg)
