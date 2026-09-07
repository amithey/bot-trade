"""
Who is using this deployment?

Everything per-user in BotTrade — the trial budget, the trading profile, the
watchlist, the usage ledger — hangs off one string: the account id.  This
module is the only place that decides what that string is.

Three modes, detected automatically:

``oidc``
    ``[auth]`` is present in ``.streamlit/secrets.toml``, so Streamlit's own
    OIDC support is active.  Real per-person accounts: Streamlit owns the
    login cookie, this module just reads ``st.user``.  Identity survives a
    refresh, a new tab, and a restart — which is what makes a per-user budget
    meaningful.  This is the mode to run in production.

``password``
    Legacy single shared password (``BOTTRADE_AUTH_PASSWORD_HASH``).  Gates
    access but does **not** distinguish people — everyone through that door is
    one account.  Fine for a private deployment; not enough for a free tier,
    and this module says so out loud.

``open``
    No auth configured.  Local development, single user.

Setting up ``oidc`` — see DEPLOY.md for the full walkthrough:

    pip install "Authlib>=1.3.2"

    # .streamlit/secrets.toml
    [auth]
    redirect_uri = "http://localhost:8501/oauth2callback"
    cookie_secret = "<a long random string>"
    client_id = "..."
    client_secret = "..."
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

import streamlit as st

#: Keys inside ``[auth]`` that configure the single default provider. Anything
#: else that is a mapping is treated as a named provider (``[auth.google]``).
_RESERVED_AUTH_KEYS = {
    "redirect_uri", "cookie_secret", "client_id", "client_secret",
    "server_metadata_url", "client_kwargs",
}

_ID_SLOT = "_bt_identity"


# --------------------------------------------------------------------------- #
# Mode detection
# --------------------------------------------------------------------------- #
def _auth_secrets() -> Optional[dict]:
    """The ``[auth]`` block, or ``None``.

    ``st.secrets`` raises ``StreamlitSecretNotFoundError`` when no secrets file
    exists at all, which is the normal local-dev case — so this never assumes
    the file is there.
    """
    try:
        if "auth" in st.secrets:
            return dict(st.secrets["auth"])
    except Exception:                                          # noqa: BLE001
        return None
    return None


def oidc_configured() -> bool:
    return _auth_secrets() is not None


def oidc_providers() -> list[str]:
    """Named providers (``[auth.google]`` → ``["google"]``).

    Empty list means a single unnamed provider, which ``st.login()`` takes
    with no argument.
    """
    auth = _auth_secrets() or {}
    return [k for k, v in auth.items()
            if k not in _RESERVED_AUTH_KEYS and hasattr(v, "keys")]


def password_gate_enabled() -> bool:
    return bool(os.environ.get("BOTTRADE_AUTH_PASSWORD_HASH", "").strip())


def accounts_enabled() -> bool:
    """True when the built-in email/password account system is switched on."""
    from dashboard import _accounts
    return _accounts.enabled()


def auth_mode() -> str:
    """``"accounts"`` | ``"oidc"`` | ``"password"`` | ``"open"``.

    ``accounts`` enables email/password registration alongside an existing
    OIDC provider. Self-asserted emails use a separate identity namespace;
    switching providers never grants access to another account's data.
    """
    if accounts_enabled():
        return "accounts"
    if oidc_configured():
        return "oidc"
    if password_gate_enabled():
        return "password"
    return "open"


def identifies_individuals() -> bool:
    """True when the mode can tell two people apart.

    The free-tier budget is only enforceable when this is True — everything
    else shares one account, so one user's spend is everyone's spend.
    """
    return auth_mode() in ("oidc", "accounts")


# --------------------------------------------------------------------------- #
# Reading the logged-in user
# --------------------------------------------------------------------------- #
def is_logged_in() -> bool:
    """``st.user.is_logged_in``, but never raising.

    Without ``[auth]`` the attribute does not exist at all (it raises
    ``AttributeError``, it does not return False), so this cannot be a plain
    attribute read.
    """
    if not oidc_configured():
        return False
    try:
        return bool(st.user.is_logged_in)
    except Exception:                                          # noqa: BLE001
        return False


def current_user() -> dict:
    """Claims for the logged-in user: ``email``, ``name``, ``sub``, ``picture``."""
    if not is_logged_in():
        return {}
    try:
        return dict(st.user.to_dict() or {})
    except Exception:                                          # noqa: BLE001
        return {}


def display_name() -> str:
    """Something short to show in the sidebar."""
    mode = auth_mode()
    if mode == "accounts" and not is_logged_in():
        from dashboard import _accounts
        return _accounts.current_email() or "Signed in"
    if mode == "oidc" or (mode == "accounts" and is_logged_in()):
        u = current_user()
        return u.get("name") or u.get("email") or "Signed in"
    if mode == "password":
        return "Shared login"
    return "Local"


def account_id() -> str:
    """The stable per-user key everything else hangs off.

    OIDC accounts are keyed by email — readable in the ledger, and stable
    across providers in a way an opaque ``sub`` is not. ``sub`` is the
    fallback for a provider that returns no email.
    """
    cached = st.session_state.get(_ID_SLOT)
    if cached:
        return cached

    mode = auth_mode()
    if mode == "accounts" and not is_logged_in():
        from dashboard import _accounts
        email = _accounts.current_email()
        # Self-asserted email is not proof of ownership of an OIDC identity.
        # Google users retain their original identity via the provider button.
        ident = f"account:{email}" if email else "account:unknown"
        st.session_state[_ID_SLOT] = ident
        return ident
    if mode == "oidc" or (mode == "accounts" and is_logged_in()):
        u = current_user()
        email = (u.get("email") or "").strip().lower()
        if email:
            ident = f"user:{email}"
        elif u.get("sub"):
            ident = f"sub:{u['sub']}"
        else:
            ident = "user:unknown"
    elif mode == "password":
        # One door, one account. Everyone who knows the password is this user.
        ident = "shared"
    else:
        ident = "local"

    st.session_state[_ID_SLOT] = ident
    return ident


def forget_identity() -> None:
    """Drop the cached identity so the next read re-derives it."""
    st.session_state.pop(_ID_SLOT, None)


def account_slug(ident: Optional[str] = None) -> str:
    """Filesystem-safe form of an account id, for per-user files.

    Readable prefix plus a hash of the full id, so two accounts that slugify
    to the same readable text still get different files.
    """
    raw = ident or account_id()
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")[:48] or "account"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


# --------------------------------------------------------------------------- #
# Login gate
# --------------------------------------------------------------------------- #
_LOGIN_CSS = """
<style>
.st-key-bt_login_card {
    background:var(--secondary-background-color,#1a1d20);
    border:1px solid #2b3034; border-radius:8px;
    padding:2rem; margin-top:2rem; width:100%; min-width:0;
}
.bt-login-mark { width:40px; height:40px; border-radius:8px;
                 background:var(--bt-accent,#62b8a5);
                 border:1px solid var(--bt-accent,#62b8a5);
                 display:flex; align-items:center; justify-content:center;
                 font-family:var(--font-ui,'Segoe UI',sans-serif);
                 color:#fff; font-weight:900; font-size:.9rem;
                 margin:0 0 1.1rem 0; }
.bt-login-title { color:#fff; font-family:var(--font-ui,'Inter','Segoe UI',sans-serif);
                  font-weight:700; font-size:1.7rem; letter-spacing:-.02em;
                  text-transform:uppercase; margin:0 0 .5rem 0; }
.bt-login-sub { color:#8b98a8; font-family:var(--font-ui,'Segoe UI',sans-serif);
                font-size:.9rem; letter-spacing:0; line-height:1.6;
                margin:0 0 1.4rem 0; }
/* A live status line — the same heartbeat language the running dashboard
   uses elsewhere — so the door to the app signals "this is a live system"
   before you're even through it, not a static form. */
@keyframes bt-login-pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%     { opacity:.4; transform:scale(1.2); }
}
.bt-login-status { display:flex; align-items:center; gap:.5rem;
                    margin:1.4rem 0 0 0; padding-top:1rem;
                    border-top:1px solid #1a222c; }
.bt-login-status .dot { width:7px; height:7px; border-radius:50%;
                         background:#00c176;
                         animation:bt-login-pulse 1.6s ease-in-out infinite; }
.bt-login-status .txt { color:#5a7a98;
                         font-family:var(--font-ui,'Segoe UI',sans-serif);
                         font-size:.62rem; letter-spacing:.14em;
                         text-transform:uppercase; }
.bt-login-legal { margin-top:.9rem; text-align:center;
                   font-size:.7rem; color:#4a6178; }
.bt-login-legal a { color:#6a8aa8; text-decoration:none; }
.bt-login-legal a:hover { text-decoration:underline; }
</style>
"""

_PROVIDER_LABELS = {
    "google":    "Continue with Google",
    "microsoft": "Continue with Microsoft",
    "github":    "Continue with GitHub",
    "auth0":     "Continue with Auth0",
    "okta":      "Continue with Okta",
}


def require_login() -> None:
    """Halt the page with a sign-in screen until the visitor is identified.

    In ``password`` mode this delegates to the legacy shared-password gate.
    In ``open`` mode it does nothing.
    """
    mode = auth_mode()

    if mode == "password":
        from dashboard._auth import require_auth
        require_auth()
        return

    if mode == "accounts":
        from dashboard import _accounts
        # Any queued cookie write happens on a run that is allowed to finish;
        # reading the cookie needs nothing but st.context.
        _accounts.flush_cookie_writes()
        if is_logged_in() or _accounts.is_signed_in():
            return
        _render_accounts_card()
        st.stop()

    if mode != "oidc" or is_logged_in():
        return

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid, st.container(key="bt_login_card"):
        st.markdown('<div class="bt-login-mark">BT</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="bt-login-title">BotTrade</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="bt-login-sub">Sign in to keep your portfolio, '
            'watchlist and API key settings across sessions.</div>',
            unsafe_allow_html=True,
        )

        providers = oidc_providers()
        try:
            if providers:
                for name in providers:
                    st.button(
                        _PROVIDER_LABELS.get(name, f"Continue with {name.title()}"),
                        key=f"_bt_login_{name}",
                        width="stretch", type="primary",
                        on_click=st.login, args=(name,),
                    )
            else:
                st.button("Sign in", key="_bt_login",
                          width="stretch", type="primary",
                          on_click=st.login)
        except Exception as exc:                               # noqa: BLE001
            # Almost always a missing Authlib or a malformed [auth] block —
            # say which, rather than showing a button that cannot work.
            st.error(
                f"Sign-in is configured but not working: {type(exc).__name__}. "
                f"Check that `Authlib>=1.3.2` is installed and that the "
                f"`[auth]` block in `.streamlit/secrets.toml` is complete."
            )
        st.markdown(
            '<div class="bt-login-status"><span class="dot"></span>'
            '<span class="txt">System online</span></div>',
            unsafe_allow_html=True,
        )
        # Required by Paddle's domain-approval check: the domain a Checkout
        # overlay opens from must itself link through to these documents.
        # This is the only markup an unauthenticated visitor (including
        # Paddle's approval crawler) ever sees on this domain — the
        # matching footer on the Settings page (saas/billing.py's caller)
        # is behind require_login() and st.stop() above, so it never
        # reaches anyone who isn't already signed in.
        st.markdown(
            '<div class="bt-login-legal">'
            '<a href="https://bottrade-ten.vercel.app/terms.html" target="_blank">Terms of Service</a> · '
            '<a href="https://bottrade-ten.vercel.app/privacy.html" target="_blank">Privacy Policy</a> · '
            '<a href="https://bottrade-ten.vercel.app/refunds.html" target="_blank">Refund Policy</a>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.stop()


def _render_accounts_card() -> None:
    """The email + password front door, in the same card as the OIDC one.

    Kept visually identical to the OIDC gate on purpose: which sign-in
    mechanism a deployment happens to use is not something a visitor should
    have to notice, and the legal links below are required on whatever page
    an unauthenticated visitor lands on — Paddle's domain check looks for
    them, and in this mode this card is that page.
    """
    from dashboard import _accounts

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid, st.container(key="bt_login_card"):
        st.markdown('<div class="bt-login-mark">BT</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="bt-login-title">BotTrade</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="bt-login-sub">Sign in to keep your portfolio, '
            'watchlist and API key settings across sessions.</div>',
            unsafe_allow_html=True,
        )

        if oidc_configured():
            providers = oidc_providers()
            for name in providers:
                st.button(_PROVIDER_LABELS.get(name, f"Continue with {name.title()}"),
                          key=f"_bt_accounts_oidc_{name}", width="stretch",
                          on_click=st.login, args=(name,))
            if not providers:
                st.button("Continue with existing provider", key="_bt_accounts_oidc",
                          width="stretch", on_click=st.login)
            st.caption("Already use Google? Continue with the same provider to access your existing portfolio. Email/password creates a separate account.")
            st.divider()
        _accounts.render_forms()

        st.markdown(
            '<div class="bt-login-status"><span class="dot"></span>'
            '<span class="txt">System online</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="bt-login-legal">'
            '<a href="https://bottrade-ten.vercel.app/terms.html" target="_blank">Terms of Service</a> · '
            '<a href="https://bottrade-ten.vercel.app/privacy.html" target="_blank">Privacy Policy</a> · '
            '<a href="https://bottrade-ten.vercel.app/refunds.html" target="_blank">Refund Policy</a>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_account_chip() -> None:
    """Account settings identity block with a sign-out control."""
    mode = auth_mode()
    if mode == "open":
        return

    with st.container():
        if mode == "accounts" and not is_logged_in():
            from dashboard import _accounts
            st.caption(f"Signed in as **{display_name()}**")
            if st.button("Sign out", width="stretch",
                         key="_bt_signout_accounts"):
                _accounts.sign_out()
                forget_identity()
                st.rerun()
        elif mode in ("oidc", "accounts") and is_logged_in():
            st.caption(f"Signed in as **{display_name()}**")
            if st.button("Sign out", width="stretch",
                         key="_bt_signout"):
                if mode == "accounts":
                    from dashboard import _accounts
                    _accounts.sign_out()
                forget_identity()
                st.logout()
        elif mode == "password":
            from dashboard._auth import hash_is_legacy, logout
            st.caption(
                "Shared password login — everyone who signs in shares one "
                "account and one budget."
            )
            if hash_is_legacy():
                st.caption(
                    ":orange[Legacy SHA-256 password hash — regenerate with "
                    "`python -m dashboard._auth`.]"
                )
            if st.button("Log out", width="stretch",
                         key="_bt_pw_logout"):
                forget_identity()
                logout()
                st.rerun()
