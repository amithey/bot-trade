import json

import pytest

from portfolio.virtual_account import LivePortfolio
from tools.audit_portfolio import audit_portfolio


def test_audit_separates_gross_profit_from_net_loss(tmp_path):
    port = LivePortfolio(10_000)
    port.buy("LONG", 100, quantity=10)
    port.sell("LONG", 100.15)
    port.open_short("SHORT", 100, cash_amount=1000)
    port.cover("SHORT", 99.85)
    port.buy("OPEN", 100, quantity=10)
    path = tmp_path / "p.json"
    port.save(path)
    raw = json.loads(path.read_text())
    before = json.dumps(raw)
    report = audit_portfolio(raw)
    assert json.dumps(raw) == before
    assert report["exit_fills"] == 2
    assert report["gross_realized_before_commissions"] == pytest.approx(3.)
    assert report["closed_trade_commissions"] == pytest.approx(4.)
    assert report["net_realized_pnl"] == pytest.approx(-1.)
    assert report["net_open_pnl"] == -1
    assert report["all_paid_commissions"] == pytest.approx(5.)
    assert report["equity_reconciliation_error"] == pytest.approx(0., abs=1e-8)
    assert report["net_losing_exit_fills"] == 2
    assert len(report["by_symbol"]) == 2


def test_audit_marks_missing_trades_and_detects_unexplained_cash(tmp_path):
    port = LivePortfolio(10_000)
    path = tmp_path / "p.json"
    port.save(path)
    raw = json.loads(path.read_text())
    result = audit_portfolio(raw)
    assert result["mean_net_pnl_per_exit_fill"] is None
    assert "No recorded trades" in result["limitations"][0]
    raw["cash"] -= 20
    result = audit_portfolio(raw)
    assert result["equity_reconciliation_error"] == -20
    assert "ACCOUNTING MISMATCH" in result["limitations"][0]
