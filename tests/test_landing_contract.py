"""
The landing page and the app have to agree, and nothing else checks it.

The pricing buttons deliberately do NOT go straight to a payment processor.
A Paddle customer is only created for an account that carries
`bottrade_account_id`, and that is only knowable once someone has signed
in — so a payment link opened from a static page would have no account to
attach the subscription to, leaving a customer charged and entitled to
nothing.

Instead the buttons carry ?plan=<ID> into the app, which hands it to
Settings. That is a contract spread across three files with no type system
holding it together: an HTML data attribute, a query-parameter read, and a
session-state key. These tests are what keeps the halves in step.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_INDEX = _ROOT / "landing" / "index.html"
# dashboard/app.py is the st.navigation() router now — the plan-redirect
# logic this file's tests check for actually lives on the default page,
# dashboard/pages/0_Live.py, which is what app.py hands off to on load.
_APP = _ROOT / "dashboard" / "pages" / "0_Live.py"
_SETTINGS = _ROOT / "dashboard" / "pages" / "2_Settings.py"

_SESSION_KEY = "_bt_pending_plan"


@pytest.fixture(scope="module")
def html() -> str:
    return _INDEX.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The landing page must not sell directly
# --------------------------------------------------------------------------- #
def test_no_direct_payment_links_on_the_landing_page(html):
    """A payment started here cannot carry an account id, so there'd be
    nothing to attach the resulting subscription to."""
    markers = ("buy.stripe.com", "checkout.stripe.com", "price_",
              "buy.paddle.com", "checkout.paddle.com", "pri_")
    for marker in markers:
        assert marker not in html, (
            f"{marker!r} found on the landing page: a payment begun before "
            f"sign-in has no bottrade_account_id to attach"
        )


def test_every_pricing_cta_is_tagged_with_a_plan(html):
    plans = set(re.findall(r'data-plan="([A-Z]+)"', html))
    assert plans == {"FREE", "PRO", "DESK"}, plans


def test_ctas_keep_a_mailto_fallback_in_the_markup(html):
    """The mailto href is the anchor's real `href` regardless of what
    APP_URL is set to — the script only overwrites it client-side once the
    page has loaded, so a visitor with JS disabled (or a scraper, or a
    slow connection reading the page before the script runs) always sees a
    working link, never a dead one."""
    for m in re.finditer(r'data-plan="[A-Z]+"', html):
        anchor = html[max(0, m.start() - 400):m.end()]
        assert "href=" in anchor and "mailto:" in anchor


def test_app_url_is_either_unset_or_a_real_https_origin(html):
    """Before deploy this is `""`; after deploy it must be the app's actual
    origin with no path or trailing slash, since the script builds on it
    with `new URL(APP_URL)` and appends `?plan=`."""
    m = re.search(r'const APP_URL = "([^"]*)";', html)
    assert m, "APP_URL assignment not found"
    value = m.group(1)
    if value:
        assert re.fullmatch(r"https://[a-z0-9.-]+\.fly\.dev", value), value


def test_the_support_address_is_real(html):
    assert "bottrade.app" not in html, "placeholder address is back"
    assert "heymans.amit@gmail.com" in html


@pytest.mark.parametrize("page", ["index.html", "terms.html", "privacy.html",
                                   "refunds.html"])
def test_no_placeholder_address_anywhere_in_the_landing_site(page):
    text = (_ROOT / "landing" / page).read_text(encoding="utf-8")
    assert "bottrade.app" not in text


# --------------------------------------------------------------------------- #
# The app picks the parameter up
# --------------------------------------------------------------------------- #
def test_app_reads_the_plan_query_parameter():
    src = _APP.read_text(encoding="utf-8")
    assert 'st.query_params.get("plan")' in src


def test_app_reads_the_plan_only_after_the_access_gate():
    """In oidc mode the sign-in redirect happens inside secure_page(); reading
    the parameter before it would lose the visitor's choice."""
    src = _APP.read_text(encoding="utf-8")
    assert src.index("secure_page()") < src.index('st.query_params.get("plan")')


def test_app_clears_the_parameter_so_it_does_not_refire():
    src = _APP.read_text(encoding="utf-8")
    tail = src.split('st.query_params.get("plan")', 1)[1][:600]
    assert "st.query_params.clear()" in tail


def test_app_hands_the_choice_over_on_the_agreed_key():
    src = _APP.read_text(encoding="utf-8")
    assert _SESSION_KEY in src


def test_settings_consumes_the_same_key():
    """Both halves must name the identical key — a typo here fails silently,
    with the visitor simply landing on a generic page."""
    assert _SESSION_KEY in _SETTINGS.read_text(encoding="utf-8")


def test_settings_pops_rather_than_reads_the_key():
    """Left in place it would re-announce on every rerun of the page."""
    src = _SETTINGS.read_text(encoding="utf-8")
    assert f'st.session_state.pop("{_SESSION_KEY}"' in src


def test_only_purchasable_plans_trigger_the_redirect():
    """?plan=FREE needs no checkout, and an unknown value must not redirect."""
    src = _APP.read_text(encoding="utf-8")
    tail = src.split('st.query_params.get("plan")', 1)[1][:600]
    assert '("PRO", "DESK")' in tail
