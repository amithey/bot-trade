"""SafetyController — circuit breakers and cooldowns derived from the
portfolio trade log.

This module is intentionally stateless: every check derives its answer
from the trade log + the current wall-clock. That makes it crash-safe
(survives a Streamlit rerun) and trivially testable.

Checks
------
1. **Consecutive losses** — after N losing round-trips in a row, BUYs are
   blocked until either (a) a manual reset, or (b) the next session/day
   rolls over.

2. **Tilt cooldown** — after a single round-trip that loses more than
   ``tilt_loss_pct`` of capital, BUYs are blocked for
   ``cooldown_minutes`` real-time minutes. This prevents revenge trading
   right after a punch in the gut.

3. **Position-size sanity** — verifies that the requested cash amount
   does not exceed the available cash by more than a tiny tolerance.

4. **Daily-trade cap** — optional ceiling on number of round-trips per
   UTC calendar day; prevents runaway loops during a bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SafetyConfig:
    """All toggles. Defaults are conservative; tune per risk profile."""
    # Consecutive-loss circuit breaker
    max_consecutive_losses: int = 3        # 0 disables
    # Tilt cooldown
    tilt_loss_pct: float = 3.0             # one trip > this % loss → cooldown
    cooldown_minutes: float = 30.0         # cooldown duration after tilt
    # Daily-trade cap (round trips per UTC day)
    max_round_trips_per_day: int = 0       # 0 = disabled
    # Slippage budget — see risk.slippage for actual modeling
    enable_slippage: bool = True

    @classmethod
    def for_profile(cls, risk_profile: str) -> "SafetyConfig":
        """Profile-specific defaults — gentler for Conservative, looser
        for Aggressive."""
        if risk_profile == "Conservative":
            return cls(
                max_consecutive_losses=2, tilt_loss_pct=2.0,
                cooldown_minutes=60.0, max_round_trips_per_day=8,
            )
        if risk_profile == "Aggressive":
            return cls(
                max_consecutive_losses=5, tilt_loss_pct=5.0,
                cooldown_minutes=15.0, max_round_trips_per_day=0,
            )
        if risk_profile == "Micro-Scalp":
            return cls(
                max_consecutive_losses=6, tilt_loss_pct=1.5,
                cooldown_minutes=5.0, max_round_trips_per_day=0,
            )
        # Balanced (default)
        return cls()


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SafetyStatus:
    """Result of a safety check. ``allow_buy`` is the gate; ``reason`` and
    ``severity`` are for logging / UI."""
    allow_buy: bool
    severity: str              # OK / WARN / BLOCK
    reason: str                # human-readable
    consecutive_losses: int = 0
    last_loss_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    round_trips_today: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        return not self.allow_buy

    def to_dict(self) -> dict:
        return {
            "allow_buy":           self.allow_buy,
            "severity":            self.severity,
            "reason":              self.reason,
            "consecutive_losses":  self.consecutive_losses,
            "last_loss_at":        self.last_loss_at.isoformat() if self.last_loss_at else None,
            "cooldown_until":      self.cooldown_until.isoformat() if self.cooldown_until else None,
            "round_trips_today":   self.round_trips_today,
            **self.extra,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────


class SafetyController:
    """Compute safety status from a trade log. Stateless beyond config.

    Parameters
    ----------
    config : SafetyConfig
        Thresholds. May be replaced live via ``set_config``.

    Override hook
    -------------
    Calling ``override_until(dt)`` lets you manually unblock until ``dt``;
    set ``dt=None`` to clear. Useful as a "I know what I'm doing" toggle
    in the UI.
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self._config = config or SafetyConfig()
        self._override_until: Optional[datetime] = None
        self._manual_block_reason: Optional[str] = None

    # ── config ──────────────────────────────────────────────────────────
    @property
    def config(self) -> SafetyConfig:
        return self._config

    def set_config(self, config: SafetyConfig) -> None:
        self._config = config

    def override_until(self, until: Optional[datetime]) -> None:
        self._override_until = until

    def manual_block(self, reason: Optional[str]) -> None:
        """Force a block (e.g. panic stop)."""
        self._manual_block_reason = reason

    # ── core check ──────────────────────────────────────────────────────
    def check(
        self,
        trade_log: Iterable,
        *,
        initial_capital: float,
        now: Optional[datetime] = None,
    ) -> SafetyStatus:
        """Compute a fresh :class:`SafetyStatus`.

        ``trade_log`` is iterable of ``TradeRecord`` (or any object exposing
        ``action``, ``executed_at``, ``realized_pnl``, ``net_value``).
        """
        now = now or datetime.utcnow()
        cfg = self._config

        # Manual override beats everything
        if self._manual_block_reason:
            return SafetyStatus(
                allow_buy=False, severity="BLOCK",
                reason=f"Manual block: {self._manual_block_reason}",
            )

        log = list(trade_log)
        sells = [r for r in log if getattr(r, "action", None) in ("SELL", "FORCE_CLOSE")]

        # --- Consecutive losses ---
        consec = 0
        last_loss_at: Optional[datetime] = None
        for r in reversed(sells):
            pnl = float(getattr(r, "realized_pnl", 0.0) or 0.0)
            if pnl < 0:
                consec += 1
                if last_loss_at is None:
                    last_loss_at = getattr(r, "executed_at", None)
            else:
                break

        # --- Tilt cooldown: did the most recent SELL lose > tilt_loss_pct
        # of *initial_capital*? If so, cooldown_minutes from that timestamp.
        cooldown_until: Optional[datetime] = None
        if sells and cfg.tilt_loss_pct > 0:
            last_sell = sells[-1]
            last_pnl = float(getattr(last_sell, "realized_pnl", 0.0) or 0.0)
            if initial_capital > 0:
                last_pnl_pct = (last_pnl / initial_capital) * 100.0
                if last_pnl_pct <= -abs(cfg.tilt_loss_pct):
                    sell_at = getattr(last_sell, "executed_at", None)
                    if isinstance(sell_at, datetime):
                        cooldown_until = sell_at + timedelta(
                            minutes=cfg.cooldown_minutes
                        )

        # --- Daily cap ---
        today_utc = now.date()
        rt_today = sum(
            1 for r in sells
            if isinstance(getattr(r, "executed_at", None), datetime)
               and r.executed_at.date() == today_utc
        )

        # --- Override ---
        override_active = (
            self._override_until is not None and now < self._override_until
        )

        # --- Decide ---
        # Order: hard blocks first (consec + daily), then cooldown.
        if not override_active and cfg.max_consecutive_losses > 0 \
                and consec >= cfg.max_consecutive_losses:
            return SafetyStatus(
                allow_buy=False, severity="BLOCK",
                reason=(
                    f"Circuit breaker: {consec} consecutive losing trades "
                    f"(cap {cfg.max_consecutive_losses}). Manual reset required."
                ),
                consecutive_losses=consec,
                last_loss_at=last_loss_at,
                cooldown_until=cooldown_until,
                round_trips_today=rt_today,
            )

        if not override_active and cfg.max_round_trips_per_day > 0 \
                and rt_today >= cfg.max_round_trips_per_day:
            return SafetyStatus(
                allow_buy=False, severity="BLOCK",
                reason=(
                    f"Daily trade cap reached ({rt_today}/"
                    f"{cfg.max_round_trips_per_day}). Resumes at UTC midnight."
                ),
                consecutive_losses=consec,
                last_loss_at=last_loss_at,
                cooldown_until=cooldown_until,
                round_trips_today=rt_today,
            )

        if not override_active and cooldown_until is not None and now < cooldown_until:
            mins_left = (cooldown_until - now).total_seconds() / 60.0
            return SafetyStatus(
                allow_buy=False, severity="BLOCK",
                reason=(
                    f"Tilt cooldown active — {mins_left:.1f} min remaining "
                    f"after a > {cfg.tilt_loss_pct:.1f}% loss."
                ),
                consecutive_losses=consec,
                last_loss_at=last_loss_at,
                cooldown_until=cooldown_until,
                round_trips_today=rt_today,
            )

        # WARN bands
        if consec >= max(1, cfg.max_consecutive_losses - 1) \
                and cfg.max_consecutive_losses > 0:
            if override_active and consec >= cfg.max_consecutive_losses:
                # The streak already reached the breaker's threshold; only
                # the override is keeping BUYs open. "one more triggers it"
                # would be wrong here — it already did.
                warn_reason = (
                    f"{consec} consecutive losses would trigger the circuit "
                    f"breaker, but a manual override is active."
                )
            else:
                warn_reason = (
                    f"{consec} consecutive losses — one more triggers the breaker."
                )
            return SafetyStatus(
                allow_buy=True, severity="WARN",
                reason=warn_reason,
                consecutive_losses=consec,
                last_loss_at=last_loss_at,
                cooldown_until=cooldown_until,
                round_trips_today=rt_today,
            )

        # OK
        ok_reason = (
            "Safety controls clear."
            + (" (manual override active)" if override_active else "")
        )
        return SafetyStatus(
            allow_buy=True, severity="OK",
            reason=ok_reason,
            consecutive_losses=consec,
            last_loss_at=last_loss_at,
            cooldown_until=cooldown_until,
            round_trips_today=rt_today,
        )

    # ── Position-size sanity (stateless static helper) ──────────────────
    @staticmethod
    def validate_buy_size(
        cash_amount: float, available_cash: float, fee_rate: float = 0.001,
    ) -> tuple[bool, str]:
        """Return (ok, reason). Block buys that overshoot cash + fees."""
        if cash_amount <= 0:
            return False, f"Buy amount ${cash_amount:.2f} is non-positive."
        max_with_fee = available_cash / (1.0 + fee_rate)
        if cash_amount > max_with_fee + 0.01:
            return False, (
                f"Buy ${cash_amount:.2f} exceeds available cash "
                f"${available_cash:.2f} (after fees ${max_with_fee:.2f})."
            )
        return True, "ok"
