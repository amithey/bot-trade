from datetime import datetime, timedelta

from strategy.trade_management import assess_intraday_exit


NOW = datetime(2026, 9, 8, 12, 0)


def assess(age_minutes, pnl, signal=True, profile="Balanced"):
    return assess_intraday_exit(
        risk_profile=profile,
        opened_at=NOW - timedelta(minutes=age_minutes),
        now=NOW,
        pnl_pct=pnl,
        signal_exit=signal,
    )


def test_hard_risk_layer_can_ignore_soft_confirmation_window():
    result = assess(10, -0.5)
    assert not result.should_exit
    assert "confirmation window" in result.reason


def test_signal_locks_a_meaningful_gain_after_confirmation():
    assert assess(45, 0.3).should_exit


def test_signal_cuts_a_meaningful_loss_after_confirmation():
    assert assess(45, -0.5).should_exit


def test_small_fee_sized_move_does_not_churn_position():
    result = assess(45, 0.1)
    assert not result.should_exit
    assert "fees" in result.reason


def test_maximum_hold_closes_without_a_signal():
    result = assess(361, 0.0, signal=False)
    assert result.should_exit
    assert "Maximum intraday hold" in result.reason


def test_micro_scalp_uses_shorter_holding_window():
    assert assess(121, 0.0, signal=False, profile="Micro-Scalp").should_exit


def test_committee_profit_exit_accounts_for_round_trip_costs():
    args = dict(risk_profile="Balanced", opened_at=NOW - timedelta(minutes=45),
                now=NOW, signal_exit=True, round_trip_cost_pct=.3)
    assert not assess_intraday_exit(pnl_pct=.3, **args).should_exit
    assert assess_intraday_exit(pnl_pct=.6, **args).should_exit
    assert assess_intraday_exit(pnl_pct=-.5, **args).should_exit
