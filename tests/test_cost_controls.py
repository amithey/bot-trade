"""
Tests for the two cost controls that sit on top of the shared decision cache.

The cache collapses repeat cycles *within* one bar. These cover the layer
above it:

* the quiet-market gate — a new bar where nothing actually moved;
* per-seat models — so upgrading the boardroom chairman does not multiply
  the bill by nine.

The safety property that matters most here is the second assertion in
``test_a_quiet_cycle_does_not_re_execute``: skipping must mean "do not act
again", not "re-run the previous verdict", or a stale BUY would pyramid into
the position on every quiet cycle.
"""
from __future__ import annotations

import time

import pytest

from portfolio.virtual_account import LivePortfolio
from trading.live_engine import LiveTradingEngine


class _Verdict:
    def __init__(self, score: float, action: str = "BUY"):
        self.score = score
        self.action = action


def _engine() -> LiveTradingEngine:
    return LiveTradingEngine(portfolio=LivePortfolio(initial_capital=10_000),
                             fetcher=None, retriever=None, engine=None)


# --------------------------------------------------------------------------- #
# Quiet-market gate
# --------------------------------------------------------------------------- #
def test_no_prior_decision_is_never_quiet():
    eng = _engine()
    quiet, _ = eng._is_quiet_since_last_decision(100.0, _Verdict(0.5))
    assert quiet is False


def test_an_unchanged_market_is_quiet():
    eng = _engine()
    eng._last_decision = object()
    eng._remember_ai_context(100.0, _Verdict(0.5))
    quiet, why = eng._is_quiet_since_last_decision(100.02, _Verdict(0.51))
    assert quiet is True
    assert "Quiet market" in why


def test_a_real_price_move_forces_a_fresh_call():
    eng = _engine()
    eng._last_decision = object()
    eng._remember_ai_context(100.0, _Verdict(0.5))
    quiet, _ = eng._is_quiet_since_last_decision(101.0, _Verdict(0.5))
    assert quiet is False


def test_a_committee_swing_forces_a_call_even_at_a_flat_price():
    """A regime change can arrive with no price move at all."""
    eng = _engine()
    eng._last_decision = object()
    eng._remember_ai_context(100.0, _Verdict(0.50))
    quiet, _ = eng._is_quiet_since_last_decision(100.0, _Verdict(0.10))
    assert quiet is False


def test_a_committee_side_flip_forces_a_call():
    eng = _engine()
    eng._last_decision = object()
    eng._remember_ai_context(100.0, _Verdict(0.02, action="BUY"))
    quiet, _ = eng._is_quiet_since_last_decision(100.0, _Verdict(0.01, action="SELL"))
    assert quiet is False


def test_a_stale_decision_is_never_reused():
    """A flat market must not leave the bot coasting on an old opinion."""
    eng = _engine()
    eng._last_decision = object()
    eng._remember_ai_context(100.0, _Verdict(0.5))
    eng._last_ai_context["ts"] = time.time() - eng._max_decision_age_sec - 1
    quiet, _ = eng._is_quiet_since_last_decision(100.0, _Verdict(0.5))
    assert quiet is False


def test_the_gate_can_be_switched_off():
    eng = _engine()
    eng._last_decision = object()
    eng._remember_ai_context(100.0, _Verdict(0.5))
    assert eng._is_quiet_since_last_decision(100.0, _Verdict(0.5))[0] is True
    eng.set_config(quiet_skip=False)
    assert eng._is_quiet_since_last_decision(100.0, _Verdict(0.5))[0] is False


def test_gate_tolerates_a_missing_verdict():
    """AI mode runs no committee, so verdict is None there."""
    eng = _engine()
    eng._last_decision = object()
    eng._remember_ai_context(100.0, None)
    quiet, _ = eng._is_quiet_since_last_decision(100.01, None)
    assert quiet is True


def test_a_zero_previous_price_is_not_treated_as_quiet():
    eng = _engine()
    eng._last_decision = object()
    eng._remember_ai_context(0.0, _Verdict(0.5))
    assert eng._is_quiet_since_last_decision(100.0, _Verdict(0.5))[0] is False


def test_committee_mode_is_exempt_from_the_gate():
    """COMMITTEE makes no API calls, so there is nothing to save by skipping."""
    import inspect
    src = inspect.getsource(LiveTradingEngine._cycle_once)
    gate = src[src.index("Quiet-market gate"):src.index("# ── Execute")]
    assert 'strategy_mode != "COMMITTEE"' in gate


def test_a_quiet_cycle_does_not_re_execute():
    """The gate must return, not feed the stale verdict into execution.

    Re-running an old BUY every quiet cycle would pyramid into the position
    repeatedly. The source must reach `return` before the execute block.
    """
    import inspect
    src = inspect.getsource(LiveTradingEngine._cycle_once)
    gate_at = src.index("Quiet-market gate")
    exec_at = src.index("# ── Execute")
    between = src[gate_at:exec_at]
    assert "\n                return\n" in between
    assert "decision = cached" not in between


def test_quiet_skips_are_counted_in_the_snapshot():
    eng = _engine()
    assert eng.snapshot()["quiet_skips"] == 0


# --------------------------------------------------------------------------- #
# Per-seat models
# --------------------------------------------------------------------------- #
class _FakeLLM:
    def __init__(self, tag):
        self.tag = tag

    def with_structured_output(self, _schema):
        return self


def test_boardroom_defaults_the_chairman_to_the_analyst_model():
    from decision_engine.boardroom import AnalystBoardroom
    llm = _FakeLLM("analyst")
    board = AnalystBoardroom(llm)
    assert board._chair_llm is llm


def test_boardroom_accepts_a_separate_chairman_model():
    from decision_engine.boardroom import AnalystBoardroom
    analysts, chair = _FakeLLM("cheap"), _FakeLLM("strong")
    board = AnalystBoardroom(analysts, chair_llm=chair)
    assert board._llm is analysts
    assert board._chair_llm is chair


def test_make_llm_returns_the_same_object_for_the_same_model(monkeypatch):
    """The single-model case must not allocate a second client."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "a" * 40)
    from decision_engine.ai_engine import AITradingEngine
    eng = AITradingEngine()
    assert eng.make_llm(None) is eng.llm
    assert eng.make_llm(eng._model) is eng.llm
    assert eng.make_llm("") is eng.llm


def test_make_llm_builds_and_caches_a_sibling(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "a" * 40)
    from decision_engine.ai_engine import AITradingEngine
    eng = AITradingEngine()
    other = eng.make_llm("claude-sonnet-5")
    assert other is not eng.llm
    assert eng.make_llm("claude-sonnet-5") is other, "should be cached"


def test_sibling_shares_the_usage_meter(monkeypatch):
    """Cost attribution must survive the model split."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "a" * 40)
    from langchain_core.callbacks import BaseCallbackHandler
    from decision_engine.ai_engine import AITradingEngine

    class _Meter(BaseCallbackHandler):
        pass

    meter = _Meter()
    eng = AITradingEngine(api_key="sk-ant-api03-" + "b" * 40, callbacks=[meter])
    other = eng.make_llm("claude-sonnet-5")
    assert meter in (other.callbacks or []),         "a sibling model must still bill to the same account"


def test_boardroom_cost_split_is_worth_it():
    """Sanity-check the premise: the analysts are where the volume is."""
    from saas.plans import CALLS_PER_CYCLE, MODE_BOARDROOM
    from saas.pricing import cost_usd
    total_calls = CALLS_PER_CYCLE[MODE_BOARDROOM]
    analysts = total_calls - 1
    all_strong = total_calls * cost_usd("claude-sonnet-5", 4_000, 400)
    split = (analysts * cost_usd("claude-haiku-4-5", 4_000, 400)
             + cost_usd("claude-sonnet-5", 4_000, 400))
    saving = 1 - split / all_strong
    # Measured, not aspirational: 9x Sonnet is $0.108/cycle, 8x Haiku plus a
    # Sonnet chair is $0.060 - a 44% saving for the same chairman quality.
    assert 0.40 < saving < 0.50, f"expected ~44% saving, got {saving:.0%}"
