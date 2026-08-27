"""
Plan catalog and entitlement resolution.

The commercial shape of BotTrade: **you supply the platform, the user supplies
the tokens.**  Everything that costs Anthropic money runs on the subscriber's
own API key; the monthly fee buys the platform — the 38-indicator committee,
the analyst boardroom, the knowledge base, backtesting, risk controls, the
dashboard.

That works because one strategy mode is genuinely free to run.  ``COMMITTEE``
is 38 vectorised indicators voting on every bar: no API call, ever.  It is
available on every plan including the free one, so a visitor without a key
still gets a complete, working product rather than a paywall.

Three things vary by plan:

* **Which LLM modes** the user may run (once funding exists).
* **How fast** the loop may cycle — the floor on the polling interval.
* **How many symbols** may run concurrently.

Funding is resolved separately from the plan.  A user is funded if they bring
their own key (``BYOK``) or if the platform still has trial budget left for
them (``PLATFORM``).  With no funding, LLM modes are simply not offered and
the bot falls back to ``COMMITTEE`` — never an error, never a dead end.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# Strategy modes
# --------------------------------------------------------------------------- #
MODE_COMMITTEE = "COMMITTEE"
MODE_AI        = "AI"
MODE_HYBRID    = "HYBRID"
MODE_BOARDROOM = "BOARDROOM"

#: Runs entirely on local computation — costs the operator nothing.
FREE_MODES: frozenset[str] = frozenset({MODE_COMMITTEE})

#: Approximate Claude calls per cycle, used for pre-flight cost estimates.
CALLS_PER_CYCLE: dict[str, int] = {
    MODE_COMMITTEE: 0,
    MODE_AI:        1,
    MODE_HYBRID:    1,
    MODE_BOARDROOM: 9,   # 8 analysts + the chairman
}


class Funding(str, Enum):
    """Who pays for this user's Claude calls."""
    NONE     = "NONE"       # nobody — LLM modes unavailable
    PLATFORM = "PLATFORM"   # operator's key, drawn against a trial budget
    BYOK     = "BYOK"       # the user's own key


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    price_usd_month: float
    #: LLM modes this plan may run *once funded*.  COMMITTEE is always on.
    llm_modes: frozenset[str]
    #: Floor on the live-loop interval, in seconds.
    min_interval_sec: int
    #: Concurrent symbols allowed.
    max_symbols: int
    #: Monthly spend on the *operator's* key this plan gets for free.
    #: 0.0 means the plan is bring-your-own-key only.
    platform_budget_usd: float
    tagline: str
    highlights: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_modes(self) -> frozenset[str]:
        return FREE_MODES | self.llm_modes

    @property
    def is_byok(self) -> bool:
        return self.platform_budget_usd <= 0.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


#: The free tier's taster budget on the operator's key.  Set
#: ``BOTTRADE_FREE_BUDGET_USD=0`` to make the free plan strictly zero-cost.
_FREE_BUDGET = _env_float("BOTTRADE_FREE_BUDGET_USD", 0.25)


PLANS: dict[str, Plan] = {
    "FREE": Plan(
        id="FREE",
        name="Free",
        price_usd_month=0.0,
        llm_modes=frozenset({MODE_AI}),
        min_interval_sec=300,
        max_symbols=1,
        platform_budget_usd=_FREE_BUDGET,
        tagline="The full 38-indicator committee, no card, no API key.",
        highlights=(
            "COMMITTEE mode — 38 technical indicators, unlimited",
            "Backtesting + Committee Lab optimiser",
            "1 live symbol, 5-minute cycle",
            "A small taster of AI mode on us",
        ),
    ),
    "PRO": Plan(
        id="PRO",
        name="Pro",
        price_usd_month=29.0,
        llm_modes=frozenset({MODE_AI, MODE_HYBRID, MODE_BOARDROOM}),
        min_interval_sec=60,
        max_symbols=10,
        platform_budget_usd=0.0,
        tagline="Every strategy mode, running on your own Anthropic key.",
        highlights=(
            "AI, HYBRID and the 9-seat Analyst Boardroom",
            "10 live symbols, 1-minute cycle",
            "Knowledge ingestion — YouTube, PDFs, articles",
            "You hold the API key; tokens bill to your account",
        ),
    ),
    "DESK": Plan(
        id="DESK",
        name="Desk",
        price_usd_month=99.0,
        llm_modes=frozenset({MODE_AI, MODE_HYBRID, MODE_BOARDROOM}),
        min_interval_sec=30,
        max_symbols=50,
        platform_budget_usd=0.0,
        tagline="Desk-scale limits for running a real book.",
        highlights=(
            "50 live symbols, 30-second cycle",
            "Priority shared-decision cache warm-up",
            "Full usage analytics and cost attribution",
            "You hold the API key; tokens bill to your account",
        ),
    ),
}

DEFAULT_PLAN_ID = "FREE"


def get_plan(plan_id: Optional[str]) -> Plan:
    """Resolve a plan id, falling back to FREE for anything unrecognised."""
    return PLANS.get((plan_id or "").strip().upper(), PLANS[DEFAULT_PLAN_ID])


# --------------------------------------------------------------------------- #
# Entitlements
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Entitlement:
    """What a specific user may do right now.

    Produced by :func:`resolve` from the plan plus two live facts: whether the
    user has supplied their own key, and how much platform budget they have
    already burned this month.
    """
    plan: Plan
    funding: Funding
    allowed_modes: frozenset[str]
    min_interval_sec: int
    max_symbols: int
    platform_spent_usd: float
    #: Human-readable explanation of *why* LLM modes are unavailable, if they are.
    lock_reason: str = ""

    @property
    def platform_budget_remaining_usd(self) -> float:
        return max(0.0, self.plan.platform_budget_usd - self.platform_spent_usd)

    @property
    def llm_available(self) -> bool:
        return self.funding is not Funding.NONE

    def allows(self, mode: str) -> bool:
        return (mode or "").strip().upper() in self.allowed_modes

    def coerce_mode(self, mode: str) -> tuple[str, str]:
        """Return ``(effective_mode, note)`` — never raises, never blocks.

        An unavailable mode degrades to ``COMMITTEE`` with an explanation, so a
        user who runs out of budget mid-session keeps trading on the free
        deterministic strategy instead of hitting an error.
        """
        want = (mode or MODE_COMMITTEE).strip().upper()
        if self.allows(want):
            return want, ""
        if not self.llm_available:
            return MODE_COMMITTEE, (
                self.lock_reason
                or f"{want} needs an Anthropic API key — running COMMITTEE instead."
            )
        return MODE_COMMITTEE, (
            f"{want} is not part of the {self.plan.name} plan — "
            f"running COMMITTEE instead."
        )

    def clamp_interval(self, seconds: float) -> tuple[int, str]:
        """Raise a too-fast interval to the plan floor, with a note."""
        s = int(max(1, round(seconds)))
        if s >= self.min_interval_sec:
            return s, ""
        return self.min_interval_sec, (
            f"{self.plan.name} plan cycles no faster than "
            f"{self.min_interval_sec}s — interval raised from {s}s."
        )

    def to_dict(self) -> dict:
        return {
            "plan":             self.plan.id,
            "plan_name":        self.plan.name,
            "funding":          self.funding.value,
            "allowed_modes":    sorted(self.allowed_modes),
            "min_interval_sec": self.min_interval_sec,
            "max_symbols":      self.max_symbols,
            "platform_spent":   round(self.platform_spent_usd, 6),
            "platform_left":    round(self.platform_budget_remaining_usd, 6),
            "lock_reason":      self.lock_reason,
        }


def resolve(
    plan_id: Optional[str],
    has_own_key: bool,
    platform_spent_usd: float = 0.0,
    platform_key_available: bool = True,
) -> Entitlement:
    """Compute the live entitlement for a user.

    Order matters: a user's own key always wins over platform budget, so a
    subscriber who supplies a key never quietly spends the operator's money.

    ``platform_key_available`` guards the other direction — a deployment with
    no operator key must not advertise trial funding it cannot honour.
    """
    plan = get_plan(plan_id)
    spent = max(0.0, float(platform_spent_usd or 0.0))
    remaining = plan.platform_budget_usd - spent

    if has_own_key:
        funding, reason = Funding.BYOK, ""
    elif remaining > 0 and platform_key_available:
        funding, reason = Funding.PLATFORM, ""
    elif remaining > 0 and not platform_key_available:
        funding = Funding.NONE
        reason = (
            "This deployment has no shared API key configured. Add your own "
            "Anthropic API key in Settings to unlock AI modes."
        )
    elif plan.platform_budget_usd > 0:
        funding = Funding.NONE
        reason = (
            f"Your {plan.name} trial budget is used up. Add your own Anthropic "
            f"API key in Settings to keep using AI modes."
        )
    else:
        funding = Funding.NONE
        reason = (
            f"The {plan.name} plan runs on your own Anthropic API key. "
            f"Add one in Settings to unlock AI modes."
        )

    modes = plan.all_modes if funding is not Funding.NONE else FREE_MODES
    return Entitlement(
        plan=plan,
        funding=funding,
        allowed_modes=modes,
        min_interval_sec=plan.min_interval_sec,
        max_symbols=plan.max_symbols,
        platform_spent_usd=spent,
        lock_reason=reason,
    )
