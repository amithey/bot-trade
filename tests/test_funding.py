import json
import threading
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from analytics.performance import equity_curve
from portfolio.virtual_account import DailySnapshot, LivePortfolio
from tools.audit_portfolio import audit_portfolio


def losing_account():
    port = LivePortfolio(200_000)
    port.buy("TEST", 100., quantity=10.)
    port.sell("TEST", 90.)
    return port


def test_deposit_and_withdraw_preserve_pnl_and_journal(tmp_path):
    port = losing_account()
    before = (port.cash, port.get_realized_pnl(), port.get_daily_pnl(), port.trade_log)
    target = tmp_path / "wallet.json"
    port.transfer_cash(50_000., persist=lambda p: p.save(target))
    assert port.cash == pytest.approx(before[0] + 50_000)
    assert port.initial_capital == 250_000
    assert port.get_realized_pnl() == before[1]
    assert port.get_daily_pnl() == pytest.approx(before[2])
    assert port.get_daily_pnl_pct() == pytest.approx(before[2] / 200_000 * 100)
    assert port.trade_log == before[3]
    assert port.get_total_return_pct() == pytest.approx(before[1] / 250_000 * 100)
    restored = LivePortfolio.load(target)
    assert restored.cash == port.cash
    assert restored.execution_revision == port.execution_revision
    assert len(restored._capital_changes) == 1
    restored.transfer_cash(-20_000.)
    assert restored.initial_capital == 230_000
    assert restored.get_daily_pnl() == pytest.approx(before[2])
    restored.save(target)
    report = audit_portfolio(json.loads(target.read_text()))
    assert report["equity_reconciliation_error"] == pytest.approx(0., abs=1e-8)


def test_transfer_preserves_open_position_and_armed_stop():
    port = LivePortfolio(10_000)
    port.buy("TEST", 100., quantity=10.)
    port.protect_profit("TEST", 101., observed_at=1.)
    before = port.positions["TEST"]
    port.transfer_cash(2000.)
    assert port.positions["TEST"] == before
    assert port.positions["TEST"].profit_protection_armed


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), -float("inf"), 0., -20_000.])
def test_invalid_transfer_leaves_account_unchanged(amount):
    port = LivePortfolio(10_000)
    before = port.get_summary()
    with pytest.raises(ValueError):
        port.transfer_cash(amount)
    assert port.get_summary() == before
    assert not port._capital_changes


def test_withdrawal_cannot_spend_money_locked_in_positions():
    port = LivePortfolio(10_000)
    port.buy("TEST", 100., quantity=90.)
    with pytest.raises(ValueError, match="available cash"):
        port.transfer_cash(-2000.)
    assert port.initial_capital == 10_000


def test_save_failure_rolls_back_transfer():
    port = losing_account()
    before = port.get_summary()
    snapshots = list(port._daily_snapshots)
    def fail(_):
        raise OSError("disk full")
    with pytest.raises(OSError):
        port.transfer_cash(5000., persist=fail)
    assert port.get_summary() == before
    assert port._daily_snapshots == snapshots
    assert not port._capital_changes


def test_transfer_invalidates_inflight_strategy_order_and_quote():
    port = LivePortfolio(10_000)
    port.buy("TEST", 100., quantity=10.)
    revision = port.execution_revision
    port.transfer_cash(1000.)
    with pytest.raises(ValueError, match="Portfolio changed"):
        port.buy("TEST", 100., quantity=1., expected_revision=revision)
    assert port.protect_profit("TEST", 110., observed_at=1., expected_revision=revision) == (None, None)


def test_simultaneous_deposits_are_not_lost():
    port = LivePortfolio(10_000)
    threads = [threading.Thread(target=port.transfer_cash, args=(1000.,)) for _ in range(8)]
    for worker in threads:
        worker.start()
    for worker in threads:
        worker.join()
    assert port.cash == port.initial_capital == 18_000
    assert len(port._capital_changes) == 8


def test_funding_does_not_create_historical_performance_or_rewrite_records():
    port = LivePortfolio(10_000, fee_rate=0.)
    port.buy("TEST", 100., quantity=1.)
    old_time = datetime.utcnow() - timedelta(days=2)  # noqa: DTZ003 - journal uses naive UTC
    port._trade_log[0] = replace(port._trade_log[0], executed_at=old_time)
    port._daily_snapshots = [DailySnapshot(date.today() - timedelta(days=2), 10_000, 10_000)]  # noqa: DTZ011 - snapshots use local calendar dates
    port.transfer_cash(5000.)
    trades, snapshots = port.performance_history()
    eq = equity_curve(trades, snapshots, port.initial_capital, port.get_total_value())
    assert list(eq.equity.unique()) == [15_000.]
    assert port.trade_log[0].portfolio_value == 10_000
    assert port._daily_snapshots[0].portfolio_value == 10_000
    assert port.get_daily_pnl() == 0.


def test_schema_four_load_then_transfer(tmp_path):
    path = tmp_path / "old.json"
    port = losing_account()
    port.save(path)
    raw = json.loads(path.read_text())
    raw["schema_version"] = 4
    raw.pop("capital_changes")
    for snapshot in raw["daily_snapshots"]:
        snapshot.pop("capital_base")
    path.write_text(json.dumps(raw))
    loaded = LivePortfolio.load(path)
    before = loaded.get_daily_pnl()
    loaded.transfer_cash(50_000.)
    assert loaded.get_daily_pnl() == pytest.approx(before)
    assert loaded.performance_history()[1][0].portfolio_value == 250_000


def test_withdrawal_after_intraday_profit_cannot_flip_daily_risk_negative():
    port = LivePortfolio(1000., fee_rate=0.)
    port.buy("TEST", 100., quantity=5.)
    port.sell("TEST", 40.)
    port._daily_snapshots = [DailySnapshot(date.today(), 700., 1000.)]  # noqa: DTZ011
    port.buy("TEST", 100., quantity=1.)
    port.sell("TEST", 600.)
    daily_pct = port.get_daily_pnl_pct()
    port.transfer_cash(-999.)
    assert port.get_daily_pnl() == 500.
    assert port.get_daily_pnl_pct() == daily_pct > 0
