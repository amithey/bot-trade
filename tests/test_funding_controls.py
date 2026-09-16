import json
from types import SimpleNamespace as NS

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import _shared as shared
from dashboard import funding
from dashboard.funding import change_funds
from notifications import NotificationConfig
from portfolio.virtual_account import LivePortfolio
from trading.live_engine import LiveTradingEngine


@pytest.fixture
def wallet(monkeypatch, tmp_path):
    state = {}
    streamlit = NS(session_state=state, warning=lambda *a: None)
    monkeypatch.setattr(shared, "st", streamlit)
    monkeypatch.setattr(funding, "st", streamlit)
    monkeypatch.setattr(shared, "account_id", lambda: "funding-test")
    monkeypatch.setattr(shared, "portfolio_path", lambda account: tmp_path / f"{account}.json")
    monkeypatch.setattr(shared, "save_profile", lambda: None)
    monkeypatch.setattr(NotificationConfig, "load", lambda: NotificationConfig())
    port = LivePortfolio(10_000)
    port.buy("TEST", 100., quantity=10.)
    port.sell("TEST", 90.)
    engine = LiveTradingEngine(port, None, None, None)
    monkeypatch.setattr(shared, "current_engine", lambda: engine)
    state["portfolio"] = port
    return state, engine, tmp_path


def test_funding_updates_live_engine_and_survives_reload(wallet):
    state, engine, folder = wallet
    original = engine.portfolio
    before = original.cash
    change_funds("Deposit", 5000.)
    assert state["portfolio"] is engine.portfolio is original
    assert original.cash == before + 5000.
    restored = LivePortfolio.load(folder / "funding-test.json")
    assert restored.cash == original.cash
    change_funds("Withdraw", 2000.)
    assert original.cash == before + 3000.


def test_reset_requires_confirmation_then_archives_and_replaces(wallet):
    state, engine, folder = wallet
    original = engine.portfolio
    with pytest.raises(ValueError, match="Confirm"):
        change_funds("Reset account", 25_000.)
    change_funds("Reset account", 25_000., confirmed=True)
    assert engine.portfolio is state["portfolio"]
    assert engine.portfolio is not original
    assert engine.portfolio.cash == 25_000.
    assert engine.portfolio.trade_log == []
    assert engine.portfolio.get_realized_pnl() == 0.
    archives = list((folder / "archive").glob("*.json"))
    assert len(archives) == 1
    assert LivePortfolio.load(archives[0]).cash == original.cash
    assert len(LivePortfolio.load(archives[0]).trade_log) == 2


@pytest.mark.parametrize("condition", ["running", "unwinding", "position"])
def test_reset_cannot_drop_active_positions_or_workers(wallet, monkeypatch, condition):
    _, engine, folder = wallet
    original = engine.portfolio
    if condition == "running":
        monkeypatch.setattr(engine, "is_running", lambda: True)
    elif condition == "unwinding":
        engine._protection_thread = NS(is_alive=lambda: True)
    else:
        engine.portfolio.buy("TEST", 100., quantity=1.)
    with pytest.raises(ValueError):
        change_funds("Reset account", 25_000., confirmed=True)
    assert engine.portfolio is original
    assert not (folder / "archive").exists()


def test_reset_save_failure_preserves_active_wallet(wallet, monkeypatch):
    state, engine, _ = wallet
    original = engine.portfolio
    monkeypatch.setattr(shared, "save_portfolio", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        change_funds("Reset account", 25_000., confirmed=True)
    assert state["portfolio"] is engine.portfolio is original


def test_sessions_without_engine_share_wallet_and_adopt_reset(wallet, monkeypatch):
    state, engine, _ = wallet
    shared.ensure_portfolio_in_session()
    original = engine.portfolio
    monkeypatch.setattr(shared, "current_engine", lambda: None)
    state["portfolio"] = LivePortfolio(1.)  # stale second browser session
    shared.ensure_portfolio_in_session()
    assert state["portfolio"] is original
    change_funds("Reset account", 20_000., confirmed=True)
    replacement = state["portfolio"]
    state["portfolio"] = original
    shared.ensure_portfolio_in_session()
    assert state["portfolio"] is replacement


def test_form_deposit_is_applied_once_and_invalid_withdrawal_is_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "account_id", lambda: "ui-funding-test")
    monkeypatch.setattr(shared, "current_engine", lambda: None)
    monkeypatch.setattr(shared, "portfolio_path", lambda account: tmp_path / "wallet.json")
    monkeypatch.setattr(shared, "save_profile", lambda: None)
    app = AppTest.from_string('''
import streamlit as st
from dashboard.funding import render_funding_controls
st.session_state.setdefault("starting_capital", 10000)
render_funding_controls("test_wallet")
''')
    app.run(timeout=20)
    assert not app.exception
    app.number_input[0].set_value(5000.)
    app.button[0].click().run()
    raw = json.loads((tmp_path / "wallet.json").read_text())
    assert raw["cash"] == 15_000
    app.run()
    assert len(json.loads((tmp_path / "wallet.json").read_text())["capital_changes"]) == 1
    app.radio[0].set_value("Withdraw").run()
    app.number_input[0].set_value(20_000.)
    app.button[0].click().run()
    assert app.error
    assert json.loads((tmp_path / "wallet.json").read_text())["cash"] == 15_000
