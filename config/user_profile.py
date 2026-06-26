"""
User Profile — persistent per-user trading preferences.

Captures: capital, default trade size cap, risk profile, watchlist,
and the optional daily profit target. Injected into the decision engine
so Claude tailors entries / sizing to the user's tolerance.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

RiskProfile = Literal["Conservative", "Balanced", "Aggressive", "Micro-Scalp"]

_DEFAULT_WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD",
                      "QQQ", "SPY", "AAPL", "NVDA", "MSFT"]


@dataclass
class UserProfile:
    capital: float = 10_000.0
    trade_size_pct: int = 20           # ceiling for any single trade
    risk_profile: RiskProfile = "Balanced"
    watchlist: list[str] = field(default_factory=lambda: list(_DEFAULT_WATCHLIST))

    # Optional daily target. 0.0 = disabled → bot trades all day.
    daily_target_pct: float = 0.0      # e.g. 1.5 → stop after +1.5% daily
    daily_loss_limit_pct: float = 5.0  # force-halt if daily P&L drops below this

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "UserProfile":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                capital=float(data.get("capital", 10_000.0)),
                trade_size_pct=int(data.get("trade_size_pct", 20)),
                risk_profile=data.get("risk_profile", "Balanced"),
                watchlist=list(data.get("watchlist") or _DEFAULT_WATCHLIST),
                daily_target_pct=float(data.get("daily_target_pct", 0.0)),
                daily_loss_limit_pct=float(data.get("daily_loss_limit_pct", 5.0)),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return cls()


# ─────────────────────────────────────────────────────────────────────────────
# Risk-profile parameter envelopes
# ─────────────────────────────────────────────────────────────────────────────
# Claude suggests a position size percentage for each decision. The engine
# clamps that suggestion into the profile's envelope and scales by confidence,
# so the three profiles produce *different* behaviors even on the same signal.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RiskEnvelope:
    """Parameter envelope enforced by the engine for a given risk profile."""
    size_min_pct: float        # minimum position size (% of equity)
    size_max_pct: float        # maximum position size (% of equity)
    conf_threshold: float      # min confidence required to enter
    stop_loss_pct: float       # default SL if AI doesn't supply one
    take_profit_pct: float     # default TP if AI doesn't supply one
    allow_pyramiding: bool     # add to winners?


RISK_ENVELOPES: dict[str, RiskEnvelope] = {
    "Conservative": RiskEnvelope(
        size_min_pct=5.0,   size_max_pct=15.0,
        conf_threshold=0.65,
        stop_loss_pct=1.5,  take_profit_pct=2.5,
        allow_pyramiding=False,
    ),
    "Balanced": RiskEnvelope(
        size_min_pct=15.0,  size_max_pct=30.0,
        conf_threshold=0.55,
        stop_loss_pct=2.5,  take_profit_pct=5.0,
        allow_pyramiding=False,
    ),
    "Aggressive": RiskEnvelope(
        size_min_pct=25.0,  size_max_pct=60.0,
        conf_threshold=0.45,
        stop_loss_pct=4.0,  take_profit_pct=10.0,
        allow_pyramiding=True,
    ),
    # Micro-Scalp: many small trades aiming for 0.3-1.5% per round-trip.
    # Sizing stays modest because win rate drives returns, not size. Stops
    # and targets are tight so false breakouts exit fast.
    "Micro-Scalp": RiskEnvelope(
        size_min_pct=8.0,   size_max_pct=25.0,
        conf_threshold=0.50,
        stop_loss_pct=0.6,  take_profit_pct=1.2,
        allow_pyramiding=False,
    ),
}


# Map risk profile -> guidance string injected into Claude's prompt.
RISK_PROFILE_GUIDANCE: dict[str, str] = {
    "Conservative": (
        "User risk profile: CONSERVATIVE — settle for small, safe gains.\n"
        "• Capital preservation beats profit. Prefer HOLD on any ambiguity.\n"
        "• BUY only on textbook setups with 3+ confirming signals.\n"
        "• Cap confidence at 0.80; cap suggested_position_size_pct at 15.\n"
        "• Tight stops (1-2%), modest targets (2-3%). Expect 2:1 R:R minimum.\n"
        "• Avoid BUY in Death-Cross regimes even on oversold bounces.\n"
        "• No pyramiding."
    ),
    "Balanced": (
        "User risk profile: BALANCED — steady growth with measured risk.\n"
        "• Standard playbook rules. BUY on 2+ confirming signals.\n"
        "• Typical stops 2-3%, targets 4-6%, 2:1 R:R minimum.\n"
        "• Suggested_position_size_pct usually 15-30 depending on conviction.\n"
        "• High-conviction setups (conf 0.80+) may use size up to 30%."
    ),
    "Aggressive": (
        "User risk profile: AGGRESSIVE — maximize returns, accept drawdowns.\n"
        "• Willing to enter on early/anticipatory signals (divergence before "
        "confirmation, breakout of 20-bar range, etc.).\n"
        "• Wider stops (3-5%) acceptable for larger targets (6-12%).\n"
        "• Suggested_position_size_pct may go up to 60 for top-conviction "
        "setups (conf 0.80+).\n"
        "• Counter-trend trades OK when setup is clean.\n"
        "• Pyramiding into winners allowed (add on pullback within trend)."
    ),
    "Micro-Scalp": (
        "User risk profile: MICRO-SCALP — many small trades, tight discipline.\n"
        "• Harvest intraday swings of 0.3-1.5% even INSIDE a larger downtrend. "
        "The bot is explicitly expected to go LONG on high-quality bounces "
        "within downtrending regimes and SHORT/SELL on failed rallies within "
        "uptrending regimes — micro-structure matters more than macro bias.\n"
        "• Target: 0.6-1.5% per round-trip. Stops: 0.4-0.8% — if wrong, exit "
        "immediately and wait for the next setup.\n"
        "• Prefer setups with clear short-term triggers: VWAP pullback in a "
        "trending session, range fade at intraday S/R, breakout retest, RSI "
        "oversold bounce at prior support, bullish/bearish engulfing at a "
        "clear level.\n"
        "• Require volume confirmation on every entry. Low-volume scalps are "
        "traps.\n"
        "• suggested_position_size_pct: 8-25%. Size scales with setup quality, "
        "NOT with conviction in a macro direction.\n"
        "• HOLD the majority of cycles. Only enter when a crisp, fresh micro "
        "setup is visible — no 'this might bounce' trades.\n"
        "• No pyramiding. Exit on the first sign the thesis failed.\n"
        "• Do NOT refuse to trade just because the higher-timeframe trend is "
        "against the position — that is the whole point of this profile."
    ),
}
