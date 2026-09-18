"""Position-aware intraday exit policy.

Hard stop-loss and take-profit orders remain the first line of defence in the
live engine. This module governs softer completed-candle and model exits so a
five-minute indicator flip does not repeatedly pay a round-trip fee.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExitAssessment:
    should_exit: bool
    reason: str
    age_minutes: float


_HOLD_WINDOWS = {
    "Conservative": (45, 240),
    "Balanced": (30, 360),
    "Aggressive": (20, 480),
    "Micro-Scalp": (15, 120),
}


def assess_intraday_exit(
    *,
    risk_profile: str,
    opened_at: datetime,
    now: datetime,
    pnl_pct: float,
    signal_exit: bool,
    round_trip_cost_pct: float = 0.0,
) -> ExitAssessment:
    """Decide whether a soft exit is actionable.

    A maximum holding window keeps the strategy intraday. Before that limit,
    a completed-candle/model exit is accepted after the minimum observation
    window when it either protects at least 0.25% plus supplied costs or cuts a loss
    that has reached 0.45%. Hard portfolio stops are evaluated separately and
    are never delayed by this function.
    """
    min_minutes, max_minutes = _HOLD_WINDOWS.get(
        risk_profile, _HOLD_WINDOWS["Balanced"],
    )
    age_minutes = max(0.0, (now - opened_at).total_seconds() / 60.0)
    if age_minutes >= max_minutes:
        return ExitAssessment(
            True,
            f"Maximum intraday hold reached ({max_minutes} minutes)",
            age_minutes,
        )
    if not signal_exit:
        return ExitAssessment(False, "No completed-candle exit signal", age_minutes)
    if age_minutes < min_minutes:
        return ExitAssessment(
            False,
            f"Exit signal observed during {min_minutes}-minute confirmation window",
            age_minutes,
        )
    # Committee callers supply their execution-cost budget. Other strategies
    # keep the original threshold through the default argument.
    if pnl_pct >= 0.25 + round_trip_cost_pct:
        return ExitAssessment(
            True,
            f"Exit signal protects a {pnl_pct:+.2f}% gross gain",
            age_minutes,
        )
    if pnl_pct <= -0.45:
        return ExitAssessment(
            True,
            f"Exit signal cuts a weakening trade at {pnl_pct:+.2f}%",
            age_minutes,
        )
    return ExitAssessment(
        False,
        f"Exit signal lacks economic distance from fees ({pnl_pct:+.2f}%)",
        age_minutes,
    )
