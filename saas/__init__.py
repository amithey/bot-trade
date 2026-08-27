"""
saas — the commercial layer that makes BotTrade safe to host for other people.

Answers one question the single-user bot never had to: *whose tokens is this
call spending?*

* :mod:`saas.plans`     — plan catalog + entitlement resolution (what a user may run)
* :mod:`saas.keyvault`  — bring-your-own-key: validation, masking, fingerprinting
* :mod:`saas.ledger`    — durable per-account record of every call and its cost
* :mod:`saas.meter`     — LangChain callback that prices calls as they happen
* :mod:`saas.pricing`   — model rate card
* :mod:`saas.tenant`    — the per-session object tying all of the above together
"""
from saas.plans import (
    PLANS,
    Entitlement,
    Funding,
    Plan,
    get_plan,
    resolve as resolve_entitlement,
)
from saas.keyvault import KeyHandle, mask, resolve_key, verify_live
from saas.ledger import UsageLedger, get_ledger
from saas.meter import UsageMeter
from saas.pricing import cost_usd, format_usd, rate_for
from saas.tenant import Tenant, engine_for_key

__all__ = [
    "PLANS", "Plan", "Entitlement", "Funding", "get_plan", "resolve_entitlement",
    "KeyHandle", "resolve_key", "verify_live", "mask",
    "UsageLedger", "get_ledger", "UsageMeter",
    "cost_usd", "format_usd", "rate_for",
    "Tenant", "engine_for_key",
]
