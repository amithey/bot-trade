"""
Every dashboard page must enforce access before it touches data.

Streamlit runs each file in pages/ independently — there is no central router
to gate, so the guarantee is only that every page calls secure_page() first.
That makes it exactly the kind of invariant that holds until someone adds a
page and forgets, which no other test would notice: the new page would work
perfectly, and serve an unauthenticated visitor.

The function used to be called `apply_theme`, which described the cosmetic
half and hid the half that matters.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PAGES = sorted((_ROOT / "dashboard" / "pages").glob("*.py"))


def _module_level_calls(path: Path) -> list[str]:
    """Names of functions called at module level, in source order.

    Pages are scripts, so the gate is a top-level statement rather than
    something inside a function — that is what makes this checkable.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name):
                calls.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                calls.append(fn.attr)
    return calls


def test_there_are_pages_to_check():
    """Guard the guard: a glob that silently matches nothing proves nothing."""
    assert _PAGES, "no dashboard pages found — this test would pass vacuously"


@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.name)
def test_every_page_calls_secure_page(page):
    assert "secure_page" in _module_level_calls(page), (
        f"{page.name} never calls secure_page() at module level — it would "
        f"render for an unauthenticated visitor"
    )


@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.name)
def test_the_gate_comes_before_any_data_access(page):
    """Order matters: gating after loading a portfolio still leaks it."""
    calls = _module_level_calls(page)
    gate = calls.index("secure_page")
    for risky in ("ensure_profile_in_session", "ensure_portfolio_in_session",
                  "get_tenant", "get_ledger", "get_live_engine"):
        if risky in calls:
            assert calls.index(risky) > gate, (
                f"{page.name} calls {risky}() before secure_page()"
            )


def test_the_gate_actually_requires_login():
    """The name is only worth anything if the body still enforces."""
    src = (_ROOT / "dashboard" / "_shared.py").read_text(encoding="utf-8")
    body = src.split("def secure_page()", 1)[1].split("\ndef ", 1)[0]
    assert "require_login()" in body


def test_the_old_cosmetic_name_is_gone():
    """`apply_theme` is not kept as an alias on purpose.

    An alias would let code call the theming half without the gate and still
    look correct. Removing it means a missed call site fails loudly instead.
    """
    offenders = [
        p.relative_to(_ROOT)
        for p in _ROOT.rglob("*.py")
        if "__pycache__" not in str(p) and ".git" not in str(p)
        and p.name != Path(__file__).name
        and "apply_theme" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"apply_theme still referenced in: {offenders}"
