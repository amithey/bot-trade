"""
Tenant — everything the platform knows about one user, in one object.

A ``Tenant`` ties together the four facts that decide what happens when a user
presses START: who they are (account id), what they bought (plan), whose key
funds the calls (:class:`~saas.keyvault.KeyHandle`), and what they have spent
so far (the ledger).  From those it derives an
:class:`~saas.plans.Entitlement` — the allowed strategy modes, the interval
floor, the symbol cap.

It also owns the engine cache.  ``engine_for_key`` hands each distinct API key
its own ``AITradingEngine`` with its own :class:`~saas.meter.UsageMeter`
attached, which is what makes per-account cost attribution work without
threading an account id through every call site.

Identity note
-------------
``account_id`` is supplied by the caller — this module does not invent user
identity.  With BYOK the key fingerprint is a good stable id and is used
automatically.  Without a key, pass whatever id the surrounding app has
(a login, a session id).  A per-session id means the free trial budget resets
when the session does; wire a real login before opening the free tier to the
public internet.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional

from saas import keyvault
from saas.keyvault import KeyHandle
from saas.ledger import UsageLedger, get_ledger
from saas.meter import UsageMeter
from saas.plans import DEFAULT_PLAN_ID, Entitlement, Funding, Plan, get_plan
from saas.plans import resolve as resolve_entitlement
from utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Engine cache — one AITradingEngine per API key
# --------------------------------------------------------------------------- #
_MAX_ENGINES = 64
_ENGINES: "OrderedDict[str, Any]" = OrderedDict()
_ENGINE_GUARD = threading.Lock()


def engine_for_key(
    key: KeyHandle,
    model: Optional[str] = None,
    ledger: Optional[UsageLedger] = None,
):
    """Return the ``AITradingEngine`` that runs on this key, building it once.

    Engines are heavyweight (news feeds, earnings calendar, TTL caches), so
    they are shared by every session using the same key and evicted
    least-recently-used past ``_MAX_ENGINES``.  Returns ``None`` when no key
    is available — callers fall back to COMMITTEE mode.
    """
    if not key.available:
        return None

    cache_key = f"{keyvault.fingerprint(key.key)}::{model or ''}::{key.funding}"
    with _ENGINE_GUARD:
        eng = _ENGINES.get(cache_key)
        if eng is not None:
            _ENGINES.move_to_end(cache_key)
            return eng

    from decision_engine.ai_engine import AITradingEngine

    meter = UsageMeter(
        account_id=key.account_id,
        funding=key.funding,
        default_model=model or "",
        ledger=ledger or get_ledger(),
    )
    eng = AITradingEngine(model=model, api_key=key.key, callbacks=[meter])
    eng.usage_meter = meter          # so callers can read live counters

    with _ENGINE_GUARD:
        _ENGINES[cache_key] = eng
        _ENGINES.move_to_end(cache_key)
        while len(_ENGINES) > _MAX_ENGINES:
            _ENGINES.popitem(last=False)
    logger.debug(f"[tenant] engine built for {key.masked} ({key.funding})")
    return eng


def clear_engine_cache() -> None:
    """Drop every cached engine — used when a key is revoked or rotated."""
    with _ENGINE_GUARD:
        _ENGINES.clear()


# --------------------------------------------------------------------------- #
# Tenant
# --------------------------------------------------------------------------- #
class Tenant:
    """One user's commercial context.

    Cheap to construct and safe to rebuild on every Streamlit rerun — the
    expensive parts (engine, ledger) are process-wide singletons behind it.
    """

    def __init__(
        self,
        account_id: str = "local",
        plan_id: Optional[str] = None,
        user_api_key: Optional[str] = None,
        allow_platform_key: bool = True,
        model: Optional[str] = None,
        ledger: Optional[UsageLedger] = None,
        knowledge_owner: Optional[str] = None,
        reconcile_billing: bool = True,
    ) -> None:
        self._ledger = ledger or get_ledger()
        self._anon_account_id = account_id or "local"
        self._knowledge_owner = (knowledge_owner or "").strip() or None
        #: Whether reading `plan` re-checks Stripe. Off for tests and for
        #: callers that must not make a network call.
        self._reconcile_billing = reconcile_billing
        self._model = model
        self._allow_platform_key = allow_platform_key
        self._user_key = keyvault.normalise(user_api_key)

        self._ledger.ensure_account(self._anon_account_id,
                                    plan_id or DEFAULT_PLAN_ID)
        if plan_id:
            self._ledger.set_plan_id(self._anon_account_id, plan_id)

    # -- identity ----------------------------------------------------------
    @property
    def key(self) -> KeyHandle:
        return keyvault.resolve_key(
            user_key=self._user_key,
            allow_platform=self._allow_platform_key,
            anon_account_id=self._anon_account_id,
        )

    @property
    def account_id(self) -> str:
        """Ledger account for the *current* funding source.

        BYOK spend is attributed to the key's fingerprint; platform-funded
        spend to the app-supplied account id.  Keeping them separate is what
        lets the trial budget be enforced without touching a subscriber's own
        billing history.
        """
        return self.key.account_id

    @property
    def billing_account_id(self) -> str:
        """The account a subscription belongs to — the *person*, not the key.

        ``account_id`` deliberately follows whichever key is funding calls
        right now, so BYOK usage never lands on the operator's ledger rows.
        Billing must NOT follow that: plugging in a different Anthropic key
        must never look like switching to a different paying customer. This
        is the same identity `plan`/`set_plan` already read and write.
        """
        return self._anon_account_id

    @property
    def knowledge_owner(self) -> str:
        """Which account's knowledge chunks this tenant may retrieve.

        Follows the *person*, like `billing_account_id` and for the same
        reason: rotating an Anthropic key must not make a user's own ingested
        documents disappear from their retrieval.

        The dashboard supplies the filesystem-safe slug it already uses for
        per-account files, so this matches exactly what the Knowledge page
        stamped onto the chunks at ingest time. Falling back to the raw
        billing id keeps non-dashboard callers (CLI, tests) coherent.
        """
        return self._knowledge_owner or self.billing_account_id

    @property
    def has_own_key(self) -> bool:
        return bool(self._user_key)

    def set_key(self, api_key: Optional[str]) -> None:
        self._user_key = keyvault.normalise(api_key)

    def clear_key(self) -> None:
        self._user_key = ""

    # -- plan --------------------------------------------------------------
    @property
    def plan(self) -> Plan:
        """The plan this account is actually entitled to right now.

        Reconciles against Stripe before answering. Without this the only
        thing that ever noticed a cancellation or a failed renewal was a visit
        to the Settings page, which is the one page a lapsed subscriber has no
        reason to open — so they kept their entitlements, and kept spending the
        operator's platform budget, indefinitely.

        Cheap to call in a loop: ``sync_subscription_status`` re-checks Stripe
        at most once per account per TTL and returns the ledger's value on any
        failure, so a Stripe outage degrades to the last known plan rather than
        locking a page render or a trading cycle.
        """
        return get_plan(self._synced_plan_id())

    def _synced_plan_id(self) -> str:
        """Ledger plan id, refreshed from Stripe when the TTL has expired."""
        stored = self._ledger.get_plan_id(self._anon_account_id)
        if not self._reconcile_billing:
            return stored
        try:
            from saas import billing
            return billing.sync_subscription_status(
                self._ledger, self.billing_account_id)
        except Exception as exc:                               # noqa: BLE001
            # Entitlement must never be the reason a render or a cycle fails.
            logger.debug(f"[tenant] plan reconciliation skipped: {exc}")
            return stored

    def set_plan(self, plan_id: str) -> None:
        self._ledger.set_plan_id(self._anon_account_id, plan_id)

    # -- entitlement -------------------------------------------------------
    @property
    def entitlement(self) -> Entitlement:
        """Recomputed on every access so budget exhaustion takes effect at once."""
        spent = self._ledger.month_spend(self._anon_account_id, funding="PLATFORM")
        return resolve_entitlement(
            plan_id=self.plan.id,
            has_own_key=self.has_own_key,
            platform_spent_usd=spent,
            platform_key_available=(
                self._allow_platform_key and bool(keyvault.platform_key())
            ),
        )

    @property
    def funding(self) -> Funding:
        return self.entitlement.funding

    # -- engine ------------------------------------------------------------
    def engine(self):
        """The metered AI engine for this tenant, or ``None`` when unfunded."""
        ent = self.entitlement
        if ent.funding is Funding.NONE:
            return None
        return engine_for_key(self.key, model=self._model, ledger=self._ledger)

    # -- usage -------------------------------------------------------------
    def usage(self) -> dict:
        """Month-to-date usage across both funding sources."""
        byok = self._ledger.summary(self.account_id) if self.has_own_key else None
        platform = self._ledger.summary(self._anon_account_id)
        if byok is None:
            return platform
        merged = dict(byok)
        merged["platform_usd"] = platform.get("platform_usd", 0.0)
        merged["calls_saved"] = (byok.get("calls_saved", 0)
                                 + platform.get("calls_saved", 0))
        return merged

    def record_saving(self, mode: str = "", ticker: str = "") -> None:
        self._ledger.record_saving(self.account_id, mode=mode, ticker=ticker)

    # -- reporting ---------------------------------------------------------
    def to_dict(self) -> dict:
        k = self.key
        ent = self.entitlement
        return {
            "account_id":  self.account_id,
            "key_source":  k.source,
            "key_masked":  k.masked,
            **ent.to_dict(),
        }

    def __repr__(self) -> str:
        return (f"Tenant(account={self.account_id!r}, plan={self.plan.id!r}, "
                f"funding={self.funding.value!r})")
