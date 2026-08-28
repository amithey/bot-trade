"""
Tests for risk/ — circuit breakers, cooldowns, and slippage.

This is the code that stops losses when something goes wrong: it blocks a
BUY after a losing streak, forces a cooldown after a bad trade, caps trades
per day, and models the adverse fill a real order would get. It shipped with
zero test coverage, which is the wrong module to leave untested — a bug here
is invisible until it lets a bot keep trading exactly when it shouldn't.

SafetyController is deliberately stateless: every check derives its answer
from the trade log plus the current wall-clock, so the tests build a trade
log directly rather than driving a full LivePortfolio.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from risk.safety import SafetyConfig, SafetyController, SafetyStatus
from risk.slippage import SlippageConfig, apply_slippage


def _trade(action: str, pnl: float = 0.0, at: datetime | None = None):
    """A minimal stand-in for TradeRecord — SafetyController only reads
    .action, .executed_at and .realized_pnl via getattr."""
    return SimpleNamespace(
        action=action,
        executed_at=at or datetime(2026, 1, 1, tzinfo=timezone.utc),
        realized_pnl=pnl,
    )


NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def _at(days_ago: float = 0, hours_ago: float = 0) -> datetime:
    return NOW - timedelta(days=days_ago, hours=hours_ago)


# --------------------------------------------------------------------------- #
# SafetyConfig.for_profile
# --------------------------------------------------------------------------- #
def test_profiles_produce_distinct_configs():
    profiles = ["Conservative", "Balanced", "Aggressive", "Micro-Scalp"]
    configs = {p: SafetyConfig.for_profile(p) for p in profiles}
    # No two profiles should collapse to an identical risk posture.
    seen = set()
    for cfg in configs.values():
        key = (cfg.max_consecutive_losses, cfg.tilt_loss_pct, cfg.cooldown_minutes)
        assert key not in seen, "two profiles produced identical thresholds"
        seen.add(key)


def test_unknown_profile_falls_back_to_balanced():
    assert SafetyConfig.for_profile("Nonexistent") == SafetyConfig.for_profile("Balanced")


def test_conservative_is_stricter_than_aggressive():
    cons = SafetyConfig.for_profile("Conservative")
    aggr = SafetyConfig.for_profile("Aggressive")
    assert cons.max_consecutive_losses < aggr.max_consecutive_losses
    assert cons.tilt_loss_pct < aggr.tilt_loss_pct
    assert cons.cooldown_minutes > aggr.cooldown_minutes


# --------------------------------------------------------------------------- #
# Clean slate
# --------------------------------------------------------------------------- #
def test_empty_trade_log_allows_buying():
    status = SafetyController().check([], initial_capital=10_000, now=NOW)
    assert status.allow_buy is True
    assert status.severity == "OK"
    assert status.is_blocked is False


# --------------------------------------------------------------------------- #
# Consecutive-loss circuit breaker
# --------------------------------------------------------------------------- #
def test_circuit_breaker_trips_at_the_configured_count():
    cfg = SafetyConfig(max_consecutive_losses=3, tilt_loss_pct=0)
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=i)) for i in range(3, 0, -1)]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is False
    assert status.severity == "BLOCK"
    assert "Circuit breaker" in status.reason
    assert status.consecutive_losses == 3


def test_one_loss_short_of_the_breaker_only_warns():
    cfg = SafetyConfig(max_consecutive_losses=3, tilt_loss_pct=0)
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=i)) for i in range(2, 0, -1)]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is True
    assert status.severity == "WARN"
    assert status.consecutive_losses == 2


def test_a_single_loss_is_neither_blocked_nor_warned_at_a_high_threshold():
    cfg = SafetyConfig(max_consecutive_losses=3, tilt_loss_pct=0)
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=1))]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.severity == "OK"


def test_a_winning_trade_resets_the_streak():
    """Only the losses since the last WIN count."""
    cfg = SafetyConfig(max_consecutive_losses=3, tilt_loss_pct=0)
    log = [
        _trade("SELL", pnl=-500, at=_at(hours_ago=10)),
        _trade("SELL", pnl=-500, at=_at(hours_ago=9)),
        _trade("SELL", pnl=+50, at=_at(hours_ago=5)),     # breaks the streak
        _trade("SELL", pnl=-10, at=_at(hours_ago=1)),
    ]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.consecutive_losses == 1
    assert status.allow_buy is True


def test_zero_disables_the_circuit_breaker_entirely():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=0)
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=i)) for i in range(10, 0, -1)]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is True
    assert status.severity == "OK", "disabled breaker must not even warn"


def test_force_close_counts_as_a_loss_for_the_breaker():
    """A stop-loss forced exit is still a loss for streak-counting purposes."""
    cfg = SafetyConfig(max_consecutive_losses=2, tilt_loss_pct=0)
    log = [
        _trade("FORCE_CLOSE", pnl=-100, at=_at(hours_ago=2)),
        _trade("FORCE_CLOSE", pnl=-100, at=_at(hours_ago=1)),
    ]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.is_blocked is True


def test_buy_records_do_not_count_as_wins_or_losses():
    cfg = SafetyConfig(max_consecutive_losses=2, tilt_loss_pct=0)
    log = [
        _trade("SELL", pnl=-100, at=_at(hours_ago=3)),
        _trade("BUY", pnl=0, at=_at(hours_ago=2)),
        _trade("SELL", pnl=-100, at=_at(hours_ago=1)),
    ]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.consecutive_losses == 2
    assert status.is_blocked is True


# --------------------------------------------------------------------------- #
# Tilt cooldown
# --------------------------------------------------------------------------- #
def test_a_big_loss_triggers_a_cooldown():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=3.0,
                       cooldown_minutes=30.0)
    log = [_trade("SELL", pnl=-500, at=_at(hours_ago=0.1))]   # -5% of 10k
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.is_blocked is True
    assert "cooldown" in status.reason.lower()
    assert status.cooldown_until is not None


def test_cooldown_expires_after_the_configured_minutes():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=3.0,
                       cooldown_minutes=30.0)
    sell_at = NOW - timedelta(minutes=31)
    log = [_trade("SELL", pnl=-500, at=sell_at)]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is True


def test_a_loss_under_the_tilt_threshold_does_not_cool_down():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=5.0,
                       cooldown_minutes=30.0)
    log = [_trade("SELL", pnl=-200, at=_at(hours_ago=0.1))]   # -2% of 10k
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is True


def test_tilt_loss_pct_zero_disables_the_cooldown():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=0)
    log = [_trade("SELL", pnl=-9_000, at=_at(hours_ago=0.1))]  # -90%!
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is True


def test_cooldown_looks_only_at_the_most_recent_sell():
    """An old big loss must not re-arm the cooldown once it has expired."""
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=3.0,
                       cooldown_minutes=30.0)
    log = [
        _trade("SELL", pnl=-500, at=NOW - timedelta(hours=2)),   # old, expired
        _trade("SELL", pnl=-50,  at=NOW - timedelta(minutes=5)), # recent, small
    ]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is True


def test_zero_initial_capital_does_not_crash_the_tilt_check():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=3.0)
    log = [_trade("SELL", pnl=-500, at=_at(hours_ago=0.1))]
    status = SafetyController(cfg).check(log, initial_capital=0, now=NOW)
    assert status.allow_buy is True   # cannot divide by zero capital; no crash


# --------------------------------------------------------------------------- #
# Daily round-trip cap
# --------------------------------------------------------------------------- #
def test_daily_cap_blocks_once_reached():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=0,
                       max_round_trips_per_day=3)
    log = [_trade("SELL", pnl=10, at=NOW - timedelta(hours=i)) for i in range(3)]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.is_blocked is True
    assert "Daily trade cap" in status.reason
    assert status.round_trips_today == 3


def test_daily_cap_zero_disables_it():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=0,
                       max_round_trips_per_day=0)
    log = [_trade("SELL", pnl=10, at=NOW - timedelta(hours=i)) for i in range(50)]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is True


def test_daily_cap_resets_across_midnight():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=0,
                       max_round_trips_per_day=2)
    yesterday = NOW - timedelta(days=1)
    log = [_trade("SELL", pnl=10, at=yesterday) for _ in range(5)]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.round_trips_today == 0
    assert status.allow_buy is True


def test_daily_cap_counts_force_closes_too():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=0,
                       max_round_trips_per_day=1)
    log = [_trade("FORCE_CLOSE", pnl=-10, at=NOW - timedelta(hours=1))]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert status.is_blocked is True


# --------------------------------------------------------------------------- #
# Precedence between simultaneous triggers
# --------------------------------------------------------------------------- #
def test_circuit_breaker_message_wins_when_multiple_triggers_fire():
    """Only one reason should reach the UI even when several are true."""
    cfg = SafetyConfig(max_consecutive_losses=2, tilt_loss_pct=1.0,
                       cooldown_minutes=60, max_round_trips_per_day=1)
    log = [
        _trade("SELL", pnl=-500, at=NOW - timedelta(hours=2)),
        _trade("SELL", pnl=-500, at=NOW - timedelta(hours=1)),
    ]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert "Circuit breaker" in status.reason


def test_daily_cap_wins_over_cooldown_when_breaker_is_off():
    cfg = SafetyConfig(max_consecutive_losses=0, tilt_loss_pct=1.0,
                       cooldown_minutes=60, max_round_trips_per_day=1)
    log = [_trade("SELL", pnl=-500, at=NOW - timedelta(minutes=5))]
    status = SafetyController(cfg).check(log, initial_capital=10_000, now=NOW)
    assert "Daily trade cap" in status.reason


# --------------------------------------------------------------------------- #
# Manual block (panic-stop) and override ("I know what I'm doing")
# --------------------------------------------------------------------------- #
def test_manual_block_stops_everything():
    ctl = SafetyController()
    ctl.manual_block("panic stop")
    status = ctl.check([], initial_capital=10_000, now=NOW)
    assert status.is_blocked is True
    assert "Manual block" in status.reason


def test_manual_block_cannot_be_lifted_by_override():
    """The panic-stop button must outrank the 'I know what I'm doing' toggle.

    Both are user-triggered, but they exist for opposite reasons: override
    exists to relax an automatic gate the user judges too cautious; manual
    block exists as the emergency stop. If override could clear it, panic
    stop would not actually stop anything.
    """
    ctl = SafetyController()
    ctl.manual_block("panic stop")
    ctl.override_until(NOW + timedelta(hours=1))
    status = ctl.check([], initial_capital=10_000, now=NOW)
    assert status.is_blocked is True


def test_override_lifts_the_circuit_breaker():
    cfg = SafetyConfig(max_consecutive_losses=1, tilt_loss_pct=0)
    ctl = SafetyController(cfg)
    ctl.override_until(NOW + timedelta(hours=1))
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=1))]
    status = ctl.check(log, initial_capital=10_000, now=NOW)
    assert status.allow_buy is True
    # The WARN message must say the override is the reason BUYs are still
    # open here — "one more triggers it" would be wrong once the streak has
    # already reached the threshold.
    assert "override" in status.reason.lower()
    assert "one more" not in status.reason.lower()


def test_override_warn_message_differs_below_vs_at_the_threshold():
    """One below the cap: normal 'one more' warning, override or not."""
    cfg = SafetyConfig(max_consecutive_losses=3, tilt_loss_pct=0)
    ctl = SafetyController(cfg)
    ctl.override_until(NOW + timedelta(hours=1))
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=i)) for i in range(2, 0, -1)]
    status = ctl.check(log, initial_capital=10_000, now=NOW)
    assert status.severity == "WARN"
    assert "one more" in status.reason.lower()
    assert "override" not in status.reason.lower()


def test_expired_override_no_longer_applies():
    cfg = SafetyConfig(max_consecutive_losses=1, tilt_loss_pct=0)
    ctl = SafetyController(cfg)
    ctl.override_until(NOW - timedelta(minutes=1))    # already in the past
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=1))]
    status = ctl.check(log, initial_capital=10_000, now=NOW)
    assert status.is_blocked is True


def test_clearing_override_reinstates_the_gate():
    cfg = SafetyConfig(max_consecutive_losses=1, tilt_loss_pct=0)
    ctl = SafetyController(cfg)
    ctl.override_until(NOW + timedelta(hours=1))
    ctl.override_until(None)
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=1))]
    status = ctl.check(log, initial_capital=10_000, now=NOW)
    assert status.is_blocked is True


def test_clearing_manual_block_reinstates_normal_checks():
    ctl = SafetyController()
    ctl.manual_block("panic stop")
    ctl.manual_block(None)
    status = ctl.check([], initial_capital=10_000, now=NOW)
    assert status.allow_buy is True


# --------------------------------------------------------------------------- #
# Config swap mid-flight
# --------------------------------------------------------------------------- #
def test_set_config_takes_effect_immediately():
    ctl = SafetyController(SafetyConfig(max_consecutive_losses=0))
    log = [_trade("SELL", pnl=-10, at=_at(hours_ago=1))]
    assert ctl.check(log, initial_capital=10_000, now=NOW).allow_buy is True
    ctl.set_config(SafetyConfig(max_consecutive_losses=1, tilt_loss_pct=0))
    assert ctl.check(log, initial_capital=10_000, now=NOW).is_blocked is True


def test_status_serialises_to_a_plain_dict():
    status = SafetyController().check([], initial_capital=10_000, now=NOW)
    d = status.to_dict()
    assert d["allow_buy"] is True
    assert d["severity"] == "OK"
    assert d["cooldown_until"] is None


# --------------------------------------------------------------------------- #
# Position-size sanity
# --------------------------------------------------------------------------- #
def test_a_normal_buy_within_cash_is_allowed():
    ok, _ = SafetyController.validate_buy_size(1_000, available_cash=5_000)
    assert ok is True


def test_a_buy_exceeding_cash_is_rejected():
    ok, reason = SafetyController.validate_buy_size(6_000, available_cash=5_000)
    assert ok is False
    assert "exceeds available cash" in reason


def test_fee_is_accounted_for_at_the_boundary():
    # 5000 / 1.001 ~= 4995.00 is the largest buy that leaves room for the fee.
    ok, _ = SafetyController.validate_buy_size(4_990, available_cash=5_000,
                                               fee_rate=0.001)
    assert ok is True
    ok, _ = SafetyController.validate_buy_size(4_999, available_cash=5_000,
                                               fee_rate=0.001)
    assert ok is False


@pytest.mark.parametrize("amount", [0, -1, -100])
def test_non_positive_amounts_are_rejected(amount):
    ok, reason = SafetyController.validate_buy_size(amount, available_cash=5_000)
    assert ok is False
    assert "non-positive" in reason


# --------------------------------------------------------------------------- #
# Slippage
# --------------------------------------------------------------------------- #
def test_buy_pays_more_than_the_reference_price():
    eff, bps = apply_slippage(100.0, "BUY")
    assert eff > 100.0
    assert bps > 0


def test_sell_receives_less_than_the_reference_price():
    eff, _ = apply_slippage(100.0, "SELL")
    assert eff < 100.0


def test_force_close_slips_the_same_direction_as_sell():
    sell_eff, sell_bps = apply_slippage(100.0, "SELL")
    fc_eff, fc_bps = apply_slippage(100.0, "FORCE_CLOSE")
    assert fc_eff == pytest.approx(sell_eff)
    assert fc_bps == pytest.approx(sell_bps)


def test_side_is_case_insensitive():
    eff_upper, _ = apply_slippage(100.0, "BUY")
    eff_lower, _ = apply_slippage(100.0, "buy")
    assert eff_upper == pytest.approx(eff_lower)


def test_higher_volatility_widens_the_slip():
    cfg = SlippageConfig(base_bps=5.0, atr_multiplier=0.3, max_bps=100.0)
    _, calm = apply_slippage(100.0, "BUY", cfg=cfg, atr_pct=0.5)
    _, wild = apply_slippage(100.0, "BUY", cfg=cfg, atr_pct=5.0)
    assert wild > calm


def test_slippage_is_capped_at_max_bps():
    cfg = SlippageConfig(base_bps=5.0, atr_multiplier=1.0, max_bps=10.0)
    _, bps = apply_slippage(100.0, "BUY", cfg=cfg, atr_pct=1000.0)
    assert bps == 10.0


def test_zero_or_negative_atr_adds_nothing():
    cfg = SlippageConfig(base_bps=5.0, atr_multiplier=0.3)
    _, bps_none = apply_slippage(100.0, "BUY", cfg=cfg, atr_pct=None)
    _, bps_zero = apply_slippage(100.0, "BUY", cfg=cfg, atr_pct=0.0)
    _, bps_neg = apply_slippage(100.0, "BUY", cfg=cfg, atr_pct=-5.0)
    assert bps_none == bps_zero == bps_neg == cfg.base_bps


def test_slippage_math_matches_the_documented_formula():
    cfg = SlippageConfig(base_bps=10.0, atr_multiplier=0.0, max_bps=100.0)
    eff, bps = apply_slippage(200.0, "BUY", cfg=cfg)
    assert bps == 10.0
    assert eff == pytest.approx(200.0 * 1.001)


def test_slippage_profiles_are_distinct():
    profiles = ["Conservative", "Balanced", "Aggressive", "Micro-Scalp"]
    seen = set()
    for p in profiles:
        cfg = SlippageConfig.for_profile(p)
        key = (cfg.base_bps, cfg.atr_multiplier, cfg.max_bps)
        assert key not in seen
        seen.add(key)


def test_unknown_slippage_profile_falls_back_to_balanced():
    assert SlippageConfig.for_profile("???") == SlippageConfig.for_profile("Balanced")


def test_micro_scalp_slips_less_than_aggressive():
    """Micro-scalp trades small and fast; it should assume the tightest fills."""
    micro = SlippageConfig.for_profile("Micro-Scalp")
    aggr = SlippageConfig.for_profile("Aggressive")
    assert micro.base_bps < aggr.base_bps
    assert micro.max_bps < aggr.max_bps
