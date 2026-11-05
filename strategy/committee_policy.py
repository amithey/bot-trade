"""Committee admission rules with continuous composite setup scoring and dynamic sizing.

The vote is confirmation, never the trading thesis. Corporate ratios are not
used to value crypto; the policy uses completed price/volume evidence and costs.
"""
import os
from dataclasses import dataclass, asdict, replace
from math import isfinite

VERSION = "committee-v4"
LIVE_VERSION = "committee-v5-adx25"
LIVE_ADX_MIN = 25.0
LIVE_ADX_SOFT_MIN = 20.0
COMPOSITE_SCORE_THRESHOLD = 65.0
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
    composite_score: float = 0.0
    quality_tier: str = "NONE"

    def to_dict(self):
        return asdict(self)


def calculate_composite_setup_score(report, *, score, quorum, categories, adx=None) -> float:
    """Calculate a 0-100 continuous composite setup score.

    Weights:
    - Trend & Regime Playbook: up to 25 pts
    - Timeframe / Hourly Confirmation: up to 15 pts
    - Committee Net Score & Quorum: up to 25 pts
    - Category Breadth (Trend/Momentum/Volume): up to 15 pts
    - ADX / Trend Strength: up to 20 pts
    """
    side = report.get("signal_side", "NONE")
    sign = -1 if side == "SHORT" else 1
    total = 0.0

    # 1. Regime & Setup (25 pts)
    target_regime = "DOWNTREND" if side == "SHORT" else "UPTREND"
    if report.get("regime") == target_regime:
        total += 15.0
    if report.get("setup") in TREND_SETUPS.get(side, set()):
        total += 10.0

    # 2. Hourly confirmation (15 pts)
    bias = report.get("timeframe_bias")
    if bias == side and side != "NONE":
        total += 15.0
    elif bias in (None, "UNKNOWN", "NEUTRAL"):
        total += 5.0

    # 3. Committee Score & Quorum (25 pts)
    if quorum:
        total += 5.0
    dir_score = sign * score if isinstance(score, (int, float)) and isfinite(score) else 0.0
    if dir_score >= 0.25:
        total += 20.0
    elif dir_score >= 0.16:
        total += 15.0
    elif dir_score >= 0.10:
        total += 10.0
    elif dir_score > 0.0:
        total += 5.0

    # 4. Category confirmation (15 pts - 5 pts each)
    if isinstance(categories, dict):
        for cat in ("Trend", "Momentum", "Volume"):
            val = categories.get(cat)
            if isinstance(val, (int, float)) and isfinite(val) and sign * val > 0:
                total += 5.0

    # 5. ADX Strength (20 pts)
    if isinstance(adx, (int, float)) and not isinstance(adx, bool) and isfinite(adx):
        if adx >= LIVE_ADX_MIN:
            total += 20.0
        elif adx >= LIVE_ADX_SOFT_MIN:
            total += 10.0 + 10.0 * (adx - LIVE_ADX_SOFT_MIN) / (LIVE_ADX_MIN - LIVE_ADX_SOFT_MIN)
        elif adx >= 15.0:
            total += 10.0 * (adx - 15.0) / (LIVE_ADX_SOFT_MIN - 15.0)

    return min(100.0, max(0.0, total))


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
                                  [], 0., 0., 0., 0.0, "NONE")
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
    base_size = min(100., 100 * RISK_BUDGET_PCT / (stop_loss_pct + cost))

    adx_val = metrics.get("adx")
    composite_score = calculate_composite_setup_score(
        report, score=score, quorum=quorum, categories=categories, adx=adx_val,
    )
    if composite_score >= 85.0:
        quality_tier = "HIGH_CONVICTION"
    elif composite_score >= 70.0:
        quality_tier = "STANDARD"
    elif composite_score >= 60.0:
        quality_tier = "MODERATE"
    else:
        quality_tier = "WEAK"

    return CommitteeAdmission(not failures, "; ".join(failures) if failures else
                              "Trend, hourly direction, participation and net trade economics confirmed",
                              checks, base_size, cost, net_rr,
                              composite_score=round(composite_score, 1),
                              quality_tier=quality_tier)


def committee_profit_config(base, stop_loss_pct):
    """Allow a trend trade more room than the generic tiny-profit scalp default."""
    from risk.profit_protection import ProfitProtectionConfig
    return ProfitProtectionConfig(
        activation_net_pct=max(base.activation_net_pct, stop_loss_pct / 2),
        retain_fraction=base.retain_fraction,
        minimum_net_pct=base.minimum_net_pct,
    )


def assess_live_committee_entry(report, *, allow_soft_adx: bool | None = None, **kwargs):
    """Production admission gate with composite setup scoring and dynamic sizing.

    Preserves full compatibility with v5-adx25 while enabling continuous scoring
    and soft ADX floors when allow_soft_adx is enabled.
    """
    if allow_soft_adx is None:
        allow_soft_adx = os.getenv("BOTTRADE_COMPOSITE_SCORING", "0").lower() in ("1", "true")

    admission = assess_committee_entry(report, **kwargs)
    adx = report.get("metrics", {}).get("adx")
    valid = isinstance(adx, (int, float)) and not isinstance(adx, bool) and isfinite(adx) and 0 <= adx <= 100

    composite_score = admission.composite_score
    if not valid:
        passed = False
        detail = "Missing or invalid ADX(14); new committee entry blocked"
    elif adx >= LIVE_ADX_MIN:
        passed = True
        detail = f"ADX(14) {adx:.2f}; need >= {LIVE_ADX_MIN:.0f} on a completed 5-minute candle"
    elif allow_soft_adx and adx >= LIVE_ADX_SOFT_MIN and composite_score >= COMPOSITE_SCORE_THRESHOLD:
        passed = True
        detail = f"ADX(14) {adx:.2f} admitted via composite setup score {composite_score:.1f}/100 (soft floor {LIVE_ADX_SOFT_MIN:.0f})"
    else:
        passed = False
        detail = f"ADX(14) {adx:.2f}; need >= {LIVE_ADX_MIN:.0f} on a completed 5-minute candle"

    checks = admission.checks + [{"name": "ADX trend strength", "passed": bool(passed), "detail": detail}]
    allowed = admission.allowed and passed

    # Dynamic sizing scaling
    if allowed:
        if composite_score >= 85.0:
            size_multiplier = 1.0
            tier = "HIGH_CONVICTION"
        elif composite_score >= 70.0:
            size_multiplier = 0.80
            tier = "STANDARD"
        else:
            size_multiplier = 0.50
            tier = "MODERATE"
        final_size = min(25.0, admission.size_pct * size_multiplier)
    else:
        final_size = 0.0
        tier = admission.quality_tier

    return replace(admission, allowed=allowed, checks=checks,
                   size_pct=final_size,
                   composite_score=round(composite_score, 1),
                   quality_tier=tier,
                   reason=admission.reason + "; " + detail)
