"""Committee-only admission rules. Fixed research rules, not profit forecasts.

The vote is confirmation, never the trading thesis. Corporate ratios are not
used to value crypto; the policy uses completed price/volume evidence and costs.
"""
from dataclasses import dataclass, asdict, replace
from math import isfinite

VERSION = "committee-v4"
TREND_SETUPS = {
    "LONG": {"TREND_BREAKOUT", "TREND_PULLBACK", "MOMENTUM_CONTINUATION"},
    "SHORT": {"SHORT_TREND_BREAKDOWN", "SHORT_RALLY_REJECTION"},
}
COOLDOWN_MINUTES = 15
RISK_BUDGET_PCT = 0.25


def committee_envelope(envelope):
    """Keep the chosen stop/exposure caps; target at least 2.5x gross risk."""
    return replace(envelope, take_profit_pct=max(envelope.take_profit_pct,
                                                2.5 * envelope.stop_loss_pct),
                   allow_pyramiding=False)


@dataclass(frozen=True)
class CommitteeAdmission:
    allowed: bool
    reason: str
    checks: list[dict]
    size_pct: float
    round_trip_cost_pct: float
    net_reward_risk: float

    def to_dict(self):
        return asdict(self)


def assess_committee_entry(report, *, score, quorum, categories, fee_rate,
                           slippage_bps, stop_loss_pct, take_profit_pct,
                           minutes_since_exit=None):
    """Fail closed on missing evidence. Percent inputs are percentage points.

    Six ATR is a volatility screening budget, NOT an expected return.
    Target/stop distances are the actual profile envelope used by execution.
    """
    side = report.get("signal_side", "NONE")
    sign = -1 if side == "SHORT" else 1
    metrics = report.get("metrics", {})
    inputs = (score, fee_rate, slippage_bps, stop_loss_pct, take_profit_pct,
              metrics.get("close"), metrics.get("atr"))
    valid = all(isinstance(v, (int, float)) and isfinite(v) for v in inputs)
    valid = valid and (0 <= fee_rate < .1 and 0 <= slippage_bps <= 1000
                      and stop_loss_pct > 0 and take_profit_pct > 0
                      and metrics["close"] > 0 and metrics["atr"] > 0)
    if not valid:
        return CommitteeAdmission(False, "Missing or invalid committee cost/price evidence",
                                  [], 0., 0., 0.)
    cost = 2 * (fee_rate * 100 + slippage_bps / 100)
    net_rr = (take_profit_pct - cost) / (stop_loss_pct + cost)
    volatility_budget = 6 * metrics["atr"] / metrics["close"] * 100
    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("Confirmed setup", report.get("entry_allowed", False),
          report.get("explanation", "Entry research must pass"))
    check("Trend playbook", report.get("setup") in TREND_SETUPS.get(side, set())
          and report.get("regime") == ("DOWNTREND" if side == "SHORT" else "UPTREND"),
          "Committee trades confirmed trends, not range fades or countertrend guesses")
    check("Hourly confirmation", report.get("timeframe_bias") == side and side != "NONE",
          f"Need completed-hour {side} confirmation; observed {report.get('timeframe_bias', 'UNKNOWN')}")
    check("Committee quorum", quorum and sign * score >= .16,
          f"Need quorum and directional score >= 0.16; observed {sign * score:+.3f}")
    for category in ("Trend", "Momentum", "Volume"):
        value = categories.get(category)
        passed = isinstance(value, (int, float)) and isfinite(value) and sign * value > 0
        check(f"{category} confirmation", passed,
              f"{category} group must confirm {side}; observed {value}")
    check("Costs versus movement", volatility_budget >= 2 * cost,
          f"Six-ATR movement budget {volatility_budget:.3f}% versus round-trip cost {cost:.3f}% (need 2x)")
    check("Net reward/risk", net_rr >= 1,
          f"Target after costs / stop plus costs = {net_rr:.2f}; need >= 1.00")
    cooldown_ok = minutes_since_exit is None or (
        isfinite(minutes_since_exit) and minutes_since_exit >= COOLDOWN_MINUTES)
    check("Re-entry cooldown", cooldown_ok,
          f"Wait {COOLDOWN_MINUTES} minutes after an exit before opening new exposure")
    failures = [c["detail"] for c in checks if not c["passed"]]
    size = min(100., 100 * RISK_BUDGET_PCT / (stop_loss_pct + cost))
    return CommitteeAdmission(not failures, "; ".join(failures) if failures else
                              "Trend, hourly direction, participation and net trade economics confirmed",
                              checks, size, cost, net_rr)


def committee_profit_config(base, stop_loss_pct):
    """Allow a trend trade more room than the generic tiny-profit scalp default."""
    from risk.profit_protection import ProfitProtectionConfig
    return ProfitProtectionConfig(
        activation_net_pct=max(base.activation_net_pct, stop_loss_pct / 2),
        retain_fraction=base.retain_fraction,
        minimum_net_pct=base.minimum_net_pct,
    )
