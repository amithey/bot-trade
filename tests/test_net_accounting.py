"""Net trade outcomes must reconcile to cash, including after restart."""
import json

import pytest

from analytics.attribution import (
    pnl_by_hour,
    pnl_by_ticker,
    pnl_by_weekday,
    trade_durations,
)
from analytics.performance import compute_metrics
from portfolio.virtual_account import LivePortfolio
from risk.safety import SafetyConfig, SafetyController
from strategy.briefing import trade_experience


def assert_reconciles(port):
    assert port.get_realized_pnl() + port.get_unrealized_pnl() == pytest.approx(
        port.get_total_value() - port.initial_capital, abs=1e-8,
    )


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_small_gross_winner_is_net_loser(side):
    port = LivePortfolio(10_000, fee_rate=.001)
    if side == "LONG":
        port.buy("TEST", 100, quantity=10)
        tr = port.sell("TEST", 100.15)
    else:
        port.open_short("TEST", 100, cash_amount=1000)
        tr = port.cover("TEST", 99.85)
    assert tr.realized_pnl < 0  # +$1.50 price gain cannot pay ~$2 commissions
    assert_reconciles(port)
    status = SafetyController(SafetyConfig(max_consecutive_losses=1)).check(
        port.trade_log, initial_capital=port.initial_capital,
    )
    assert status.is_blocked


def test_partial_sales_and_pyramiding_allocate_actual_entry_fees(tmp_path):
    port = LivePortfolio(10_000, fee_rate=.001)
    port.buy("TEST", 100, quantity=10)
    port.fee_rate = .002  # use actual historical commissions, not current rate
    port.buy("TEST", 110, quantity=10)
    assert port.positions["TEST"].entry_fees == pytest.approx(3.2)
    first = port.sell("TEST", 120, quantity=5)
    assert first.realized_pnl == pytest.approx(75 - 1.2 - .8)
    assert port.positions["TEST"].entry_fees == pytest.approx(2.4)
    assert_reconciles(port)
    path = tmp_path / "portfolio.json"
    port.save(path)
    port = LivePortfolio.load(path)
    assert_reconciles(port)
    final = port.sell("TEST", 100)
    assert final.realized_pnl == pytest.approx(-75 - 3 - 2.4)
    assert_reconciles(port)


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_force_close_includes_entry_fee_and_open_marks_stay_gross(side):
    port = LivePortfolio(10_000, fee_rate=.001)
    if side == "LONG":
        port.buy("TEST", 100, quantity=10)
    else:
        port.open_short("TEST", 100, cash_amount=1000)
    assert port.positions["TEST"].unrealized_pnl_pct == 0
    assert port.get_unrealized_pnl() == -1
    assert_reconciles(port)
    tr = port.force_close("TEST", 100)
    assert tr.realized_pnl == -2
    assert_reconciles(port)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_migration_preserves_cash_and_does_not_double_charge(tmp_path, version):
    port = LivePortfolio(10_000, fee_rate=.001)
    port.buy("TEST", 100, quantity=10)
    port.sell("TEST", 100.15, quantity=4)
    port.open_short("SHORT", 100, cash_amount=1000)
    port.cover("SHORT", 99.85)
    path = tmp_path / "portfolio.json"
    port.save(path)
    raw = json.loads(path.read_text())
    # Reproduce the old representation: exits excluded allocated entry fees.
    raw["schema_version"] = version
    raw["realized_pnl"] += 1.4
    raw["trade_log"][1]["realized_pnl"] += .4
    raw["trade_log"][3]["realized_pnl"] += 1
    for pos in raw["positions"].values():
        pos.pop("entry_fees")
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    migrated = LivePortfolio.load(path)
    assert path.read_bytes() == before  # load never overwrites the source
    assert migrated.cash == port.cash
    assert migrated.get_realized_pnl() == pytest.approx(port.get_realized_pnl())
    assert migrated.positions["TEST"].entry_fees == pytest.approx(.6)
    assert_reconciles(migrated)
    migrated.save(path)
    restarted = LivePortfolio.load(path)
    assert restarted.get_realized_pnl() == migrated.get_realized_pnl()
    restarted.sell("TEST", 100)
    assert_reconciles(restarted)


def test_incomplete_legacy_history_is_rejected(tmp_path):
    port = LivePortfolio(10_000)
    port.buy("TEST", 100, quantity=10)
    path = tmp_path / "portfolio.json"
    port.save(path)
    raw = json.loads(path.read_text())
    raw["schema_version"] = 2
    raw["trade_log"] = []
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="journal and positions disagree"):
        LivePortfolio.load(path)


def test_short_closures_reach_all_outcome_analytics():
    port = LivePortfolio(10_000)
    port.open_short("TEST", 100, cash_amount=1000)
    port.cover("TEST", 99.85)
    metrics = compute_metrics(port.trade_log, [], 10_000, port.get_total_value())
    assert metrics.n_round_trips == 1
    assert metrics.n_losses == 1
    for group in (pnl_by_ticker, pnl_by_hour, pnl_by_weekday):
        assert group(port.trade_log)["total_pnl"].sum() == pytest.approx(-.5)
    durations = trade_durations(port.trade_log)
    assert len(durations) == 1
    assert durations.iloc[0]["pnl_pct"] == pytest.approx(.15)  # gross, directional
    assert trade_experience(port.trade_log, "TEST")["losing_exit_fills"] == 1


def test_short_losses_trigger_cooldown_and_daily_exit_cap():
    port = LivePortfolio(10_000)
    port.open_short("TEST", 100, cash_amount=1000)
    port.cover("TEST", 110)
    now = port.trade_log[-1].executed_at
    for cfg in (
        SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=.5),
        SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=0, max_round_trips_per_day=1),
    ):
        assert SafetyController(cfg).check(port.trade_log, initial_capital=10_000, now=now).is_blocked


@pytest.mark.parametrize("panic", [False, True])
def test_emergency_liquidation_closes_both_sides(panic, monkeypatch):
    from notifications import NotificationConfig
    from trading.live_engine import LiveTradingEngine
    monkeypatch.setattr(NotificationConfig, "load", lambda: NotificationConfig())
    port = LivePortfolio(10_000)
    port.buy("LONG", 100, quantity=10)
    port.open_short("SHORT", 100, cash_amount=1000)
    engine = LiveTradingEngine(port, None, None, None)
    monkeypatch.setattr(engine, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_reflect_on_sell", lambda *a, **k: None)
    if panic:
        engine.panic_stop()
        assert engine._stop_flag.is_set()
    else:
        engine._liquidate_all("Daily loss limit hit")
    assert not port.positions
    assert [t.action for t in port.trade_log[-2:]] == ["FORCE_CLOSE", "FORCE_CLOSE"]
    assert_reconciles(port)


def test_unloadable_portfolio_is_preserved_and_never_reset(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import dashboard._identity as identity
    import dashboard._shared as shared
    path = tmp_path / "portfolio.json"
    original = b'{"schema_version": 999}'
    path.write_bytes(original)
    state, errors, stops = {}, [], []
    monkeypatch.setattr(identity, "account_id", lambda: "test")
    monkeypatch.setattr(shared, "portfolio_path", lambda account: path)
    monkeypatch.setattr(shared, "st", SimpleNamespace(
        session_state=state, error=errors.append, stop=lambda: stops.append(True),
    ))
    shared.ensure_portfolio_in_session()
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert "portfolio" not in state
    assert errors and stops
