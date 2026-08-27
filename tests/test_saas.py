"""
Offline tests for the multi-tenant commercial layer.

No network, no API key, no Streamlit — everything here runs against fakes and
a temp-file ledger.  The invariants under test are the ones that cost real
money if they break:

* an unfunded user can never reach an LLM mode;
* a subscriber's own spend never draws down the operator's trial budget;
* concurrent users on the same bar produce exactly one upstream call.
"""
from __future__ import annotations

import threading
import time

import pytest

from decision_engine.decision_cache import (
    SharedDecisionCache, make_key, pnl_bucket, position_bucket,
)
from saas import keyvault, plans, pricing
from saas.ledger import UsageLedger
from saas.meter import UsageMeter
from saas.plans import Funding
from saas.tenant import Tenant


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def test_rate_resolves_dated_snapshot_ids():
    assert pricing.rate_for("claude-haiku-4-5-20251001") is pricing.RATES["claude-haiku-4-5"]
    assert pricing.rate_for("claude-opus-5").input_per_mtok == 5.00


def test_unknown_model_overestimates_rather_than_billing_zero():
    assert pricing.cost_usd("some-future-model", 1_000_000, 0) > 0


def test_cache_reads_are_an_order_of_magnitude_cheaper_than_fresh_input():
    fresh = pricing.cost_usd("claude-haiku-4-5", input_tokens=1_000_000)
    cached = pricing.cost_usd("claude-haiku-4-5", cache_read_tokens=1_000_000)
    assert cached == pytest.approx(fresh * 0.10)


def test_boardroom_cycle_cost_is_in_the_expected_range():
    # 9 calls, ~4k input / 400 output each, on Haiku.
    cost = 9 * pricing.cost_usd("claude-haiku-4-5", 4_000, 400)
    assert 0.01 < cost < 0.20


# --------------------------------------------------------------------------- #
# Plans and entitlements
# --------------------------------------------------------------------------- #
def test_committee_is_available_on_every_plan():
    for plan in plans.PLANS.values():
        ent = plans.resolve(plan.id, has_own_key=False, platform_spent_usd=1e9)
        assert plans.MODE_COMMITTEE in ent.allowed_modes


def test_committee_costs_the_operator_nothing():
    assert plans.CALLS_PER_CYCLE[plans.MODE_COMMITTEE] == 0


def test_unfunded_user_gets_no_llm_modes():
    ent = plans.resolve("PRO", has_own_key=False)
    assert ent.funding is Funding.NONE
    assert ent.allowed_modes == plans.FREE_MODES
    assert "API key" in ent.lock_reason


def test_own_key_unlocks_the_plan_modes():
    ent = plans.resolve("PRO", has_own_key=True)
    assert ent.funding is Funding.BYOK
    assert plans.MODE_BOARDROOM in ent.allowed_modes


def test_own_key_wins_over_platform_budget():
    """A subscriber with a key must never spend the operator's money."""
    ent = plans.resolve("FREE", has_own_key=True, platform_spent_usd=0.0)
    assert ent.funding is Funding.BYOK


def test_free_trial_budget_runs_out():
    plan = plans.get_plan("FREE")
    ent = plans.resolve("FREE", has_own_key=False,
                        platform_spent_usd=plan.platform_budget_usd + 1)
    assert ent.funding is Funding.NONE
    assert ent.platform_budget_remaining_usd == 0.0


def test_boardroom_is_not_reachable_on_the_free_plan_even_with_a_key():
    ent = plans.resolve("FREE", has_own_key=True)
    assert plans.MODE_BOARDROOM not in ent.allowed_modes


def test_unavailable_mode_degrades_to_committee_instead_of_failing():
    ent = plans.resolve("PRO", has_own_key=False)
    mode, note = ent.coerce_mode("BOARDROOM")
    assert mode == plans.MODE_COMMITTEE
    assert note


def test_allowed_mode_passes_through_untouched():
    ent = plans.resolve("PRO", has_own_key=True)
    assert ent.coerce_mode("BOARDROOM") == ("BOARDROOM", "")


def test_interval_is_clamped_up_to_the_plan_floor():
    ent = plans.resolve("FREE", has_own_key=True)
    secs, note = ent.clamp_interval(5)
    assert secs == plans.get_plan("FREE").min_interval_sec
    assert note
    assert ent.clamp_interval(9999) == (9999, "")


# --------------------------------------------------------------------------- #
# Key handling
# --------------------------------------------------------------------------- #
GOOD_KEY = "sk-ant-api03-" + "a" * 40


def test_key_format_validation():
    assert keyvault.validate_format(GOOD_KEY)[0]
    assert not keyvault.validate_format("")[0]
    assert not keyvault.validate_format("hunter2")[0]
    assert not keyvault.validate_format("sk-ant-short")[0]


def test_mask_never_reveals_the_middle_of_the_key():
    masked = keyvault.mask(GOOD_KEY)
    assert "a" * 10 not in masked
    assert masked.startswith("sk-ant-")


def test_fingerprint_is_stable_and_non_reversible():
    fp = keyvault.fingerprint(GOOD_KEY)
    assert fp == keyvault.fingerprint(GOOD_KEY)
    assert GOOD_KEY not in fp
    assert fp != keyvault.fingerprint(GOOD_KEY + "b")


def test_key_handle_repr_does_not_leak_the_key():
    handle = keyvault.resolve_key(GOOD_KEY)
    assert GOOD_KEY not in repr(handle)
    assert handle.funding == "BYOK"


def test_user_key_beats_platform_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "z" * 40)
    assert keyvault.resolve_key(GOOD_KEY).source == "user"
    assert keyvault.resolve_key(None).source == "platform"
    assert keyvault.resolve_key(None, allow_platform=False).source == "none"


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
@pytest.fixture()
def ledger(tmp_path) -> UsageLedger:
    return UsageLedger(tmp_path / "usage.db")


def test_ledger_records_and_totals_spend(ledger):
    cost = ledger.record("acct", "claude-haiku-4-5", 1_000_000, 0,
                         funding="BYOK", mode="AI", ticker="BTC-USD")
    assert cost == pytest.approx(1.00)
    assert ledger.month_spend("acct") == pytest.approx(1.00)


def test_byok_spend_does_not_draw_down_the_platform_budget(ledger):
    ledger.record("acct", "claude-haiku-4-5", 1_000_000, funding="BYOK")
    assert ledger.month_spend("acct", funding="PLATFORM") == 0.0
    assert ledger.month_spend("acct", funding="BYOK") == pytest.approx(1.00)


def test_ledger_breakdown_groups_by_mode(ledger):
    ledger.record("a", "claude-haiku-4-5", 100_000, mode="AI")
    ledger.record("a", "claude-haiku-4-5", 900_000, mode="BOARDROOM")
    rows = {r["label"]: r for r in ledger.breakdown("a", by="mode")}
    assert rows["BOARDROOM"]["cost_usd"] > rows["AI"]["cost_usd"]


def test_plan_assignment_round_trips(ledger):
    ledger.set_plan_id("a", "PRO")
    assert ledger.get_plan_id("a") == "PRO"
    assert ledger.get_plan_id("never-seen") == "FREE"


def test_savings_are_counted(ledger):
    for _ in range(3):
        ledger.record_saving("a", mode="BOARDROOM", ticker="BTC-USD")
    assert ledger.calls_saved("a") == 3


def test_ledger_is_safe_under_concurrent_writes(ledger):
    def hammer():
        for _ in range(25):
            ledger.record("a", "claude-haiku-4-5", 1000, 100)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ledger.summary("a")["calls"] == 200


# --------------------------------------------------------------------------- #
# Meter
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, usage, model="claude-haiku-4-5"):
        self.usage_metadata = usage
        self.response_metadata = {"model": model}


class _FakeGen:
    def __init__(self, message):
        self.message = message


class _FakeResult:
    def __init__(self, usage, model="claude-haiku-4-5"):
        self.generations = [[_FakeGen(_FakeMessage(usage, model))]]
        self.llm_output = {"model": model}


def test_meter_prices_a_call_from_usage_metadata(ledger):
    meter = UsageMeter("acct", funding="BYOK", ledger=ledger)
    meter.set_context(mode="BOARDROOM", ticker="BTC-USD")
    meter.on_llm_end(_FakeResult({
        "input_tokens": 1_000_000, "output_tokens": 0,
        "input_token_details": {"cache_read": 0, "cache_creation": 0},
    }))
    assert meter.calls == 1
    assert meter.cost_usd == pytest.approx(1.00)
    assert ledger.breakdown("acct", by="mode")[0]["label"] == "BOARDROOM"


def test_meter_reads_raw_anthropic_usage_when_metadata_is_absent(ledger):
    meter = UsageMeter("acct", funding="BYOK", ledger=ledger)
    result = _FakeResult({})
    result.generations = []
    result.llm_output = {"model": "claude-haiku-4-5",
                         "usage": {"input_tokens": 500_000,
                                   "output_tokens": 100_000}}
    meter.on_llm_end(result)
    assert meter.cost_usd == pytest.approx(0.5 + 0.5)


def test_meter_never_raises_into_the_trading_loop(ledger):
    meter = UsageMeter("acct", ledger=ledger)
    meter.on_llm_end(object())          # nothing resembling a result
    meter.on_llm_error(RuntimeError("boom"))
    assert meter.calls == 0


# --------------------------------------------------------------------------- #
# Shared decision cache
# --------------------------------------------------------------------------- #
def test_repeat_lookups_hit_the_cache():
    cache = SharedDecisionCache(default_ttl=60)
    calls = []
    for _ in range(10):
        cache.get_or_compute("k", lambda: calls.append(1) or "v")
    assert len(calls) == 1
    assert cache.stats.calls_saved == 9


def test_concurrent_users_on_one_bar_cost_a_single_call():
    cache = SharedDecisionCache(default_ttl=60)
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.15)
        return "verdict"

    threads = [threading.Thread(target=cache.get_or_compute, args=("k", slow))
               for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert cache.stats.coalesced == 39


def test_a_failed_call_does_not_poison_the_entry():
    cache = SharedDecisionCache(default_ttl=60)
    with pytest.raises(RuntimeError):
        cache.get_or_compute("k", lambda: (_ for _ in ()).throw(RuntimeError("api down")))
    assert cache.get_or_compute("k", lambda: "recovered") == ("recovered", False)


def test_expired_entries_recompute():
    cache = SharedDecisionCache(default_ttl=0.05)
    cache.get_or_compute("k", lambda: "first")
    time.sleep(0.08)
    value, was_cached = cache.get_or_compute("k", lambda: "second")
    assert (value, was_cached) == ("second", False)


def test_a_new_bar_is_a_new_key():
    parts = dict(ticker="BTC-USD", mode="AI", risk="Balanced")
    assert make_key(**parts, bar="14:30") != make_key(**parts, bar="14:35")


def test_key_is_order_independent():
    assert make_key(a=1, b=2) == make_key(b=2, a=1)


def test_position_buckets_share_between_similar_users():
    assert position_bucket(False, None) == position_bucket(False, 5.0)
    assert position_bucket(True, 1.10) == position_bucket(True, 1.15)
    assert position_bucket(True, 1.1) != position_bucket(True, 4.0)
    assert pnl_bucket(1.2) == pnl_bucket(0.9)


def test_cache_evicts_past_its_cap():
    cache = SharedDecisionCache(default_ttl=60, max_entries=5)
    for i in range(20):
        cache.get_or_compute(f"k{i}", lambda: i)
    assert cache.size() <= 5


# --------------------------------------------------------------------------- #
# Tenant
# --------------------------------------------------------------------------- #
def test_tenant_starts_free_and_unfunded_without_a_key(ledger, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = Tenant(account_id="u1", ledger=ledger)
    assert t.plan.id == "FREE"
    assert t.entitlement.funding is Funding.NONE
    assert t.engine() is None


def test_adding_a_key_unlocks_modes_without_touching_the_plan(ledger):
    t = Tenant(account_id="u2", plan_id="PRO", ledger=ledger)
    assert not t.entitlement.llm_available
    t.set_key(GOOD_KEY)
    assert t.entitlement.funding is Funding.BYOK
    assert t.plan.id == "PRO"
    assert plans.MODE_BOARDROOM in t.entitlement.allowed_modes


def test_tenant_account_id_follows_the_funding_source(ledger):
    t = Tenant(account_id="u3", ledger=ledger)
    assert t.account_id == "u3"
    t.set_key(GOOD_KEY)
    assert t.account_id == keyvault.fingerprint(GOOD_KEY)


def test_no_operator_key_means_no_trial_funding_is_advertised(ledger, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ent = plans.resolve("FREE", has_own_key=False, platform_key_available=False)
    assert ent.funding is Funding.NONE
    assert "no shared API key" in ent.lock_reason


def test_exhausting_the_trial_budget_locks_llm_modes(ledger, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "z" * 40)
    t = Tenant(account_id="u4", ledger=ledger)
    assert t.entitlement.funding is Funding.PLATFORM
    budget = plans.get_plan("FREE").platform_budget_usd
    ledger.record("u4", "claude-opus-5", int(budget / 5.0 * 1e6) + 1_000_000,
                  funding="PLATFORM")
    assert t.entitlement.funding is Funding.NONE
    assert t.entitlement.coerce_mode("AI")[0] == plans.MODE_COMMITTEE


# --------------------------------------------------------------------------- #
# LiveTradingEngine gating
# --------------------------------------------------------------------------- #
def _engine_with_tenant(tenant):
    from trading.live_engine import LiveTradingEngine
    return LiveTradingEngine(portfolio=None, fetcher=None, retriever=None,
                             engine=None, tenant=tenant)


def test_free_plan_cannot_start_the_boardroom(ledger, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = Tenant(account_id="e1", plan_id="FREE", ledger=ledger,
               allow_platform_key=False)
    eng = _engine_with_tenant(t)
    eng.set_config(strategy_mode="BOARDROOM", interval_sec=10)
    assert eng._strategy_mode == "COMMITTEE"
    assert eng._interval_sec == plans.get_plan("FREE").min_interval_sec


def test_pro_plan_with_a_key_unlocks_the_boardroom(ledger):
    t = Tenant(account_id="e2", plan_id="PRO", ledger=ledger)
    t.set_key(GOOD_KEY)
    eng = _engine_with_tenant(t)
    eng.set_config(strategy_mode="BOARDROOM", interval_sec=10)
    assert eng._strategy_mode == "BOARDROOM"
    assert eng._interval_sec == plans.get_plan("PRO").min_interval_sec


def test_set_tenant_steps_a_running_bot_down_immediately(ledger):
    t = Tenant(account_id="e3", plan_id="PRO", ledger=ledger)
    t.set_key(GOOD_KEY)
    eng = _engine_with_tenant(t)
    eng.set_config(strategy_mode="BOARDROOM")
    assert eng._strategy_mode == "BOARDROOM"

    t.clear_key()               # subscription lapses / key revoked
    eng.set_tenant(t)
    assert eng._strategy_mode == "COMMITTEE"


def test_single_user_mode_is_ungated(ledger):
    eng = _engine_with_tenant(None)
    eng.set_config(strategy_mode="BOARDROOM", interval_sec=10)
    assert eng._strategy_mode == "BOARDROOM"
    assert eng._interval_sec == 10
