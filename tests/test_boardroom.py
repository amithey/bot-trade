"""Offline tests for the analyst boardroom — fake LLM, no network."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from decision_engine.boardroom import (
    _PANEL, AnalystBoardroom, AnalystOpinion, AnalystVote,
    _contrarian_packet, _flow_packet, _fundamental_packet, _macro_packet,
    _news_packet, _risk_packet, _technical_packet,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeStructured:
    def __init__(self, factory):
        self._factory = factory

    def invoke(self, prompt):
        return self._factory(prompt)


class FakeLLM:
    """Stands in for ChatAnthropic. Routes by output schema."""

    def __init__(self, analyst_vote="BUY", chair_raises=False):
        self.analyst_vote = analyst_vote
        self.chair_raises = chair_raises

    def with_structured_output(self, schema):
        if schema is AnalystVote:
            return _FakeStructured(lambda p: AnalystVote(
                vote=self.analyst_vote, conviction=0.8,
                opinion="From my desk this setup looks clean and "
                        "well-supported by the data in my packet."))

        def chair_factory(prompt):
            if self.chair_raises:
                raise RuntimeError("chair offline")
            from decision_engine.ai_engine import TradingDecision
            return TradingDecision(
                action=self.analyst_vote,
                confidence_score=0.75,
                reasoning="Weighing Maya, David, Noa and Leo — the panel "
                          "agrees, ruling accordingly.",
                risk_level="MEDIUM",
            )
        return _FakeStructured(chair_factory)


class _FailingNews:
    def fetch(self, ticker):
        raise RuntimeError("no network in tests")


def make_snapshot(n=120, ticker="TEST"):
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + rng.normal(0.002, 0.01, n))
    df = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": np.full(n, 5_000.0),
        "SMA_20": close, "RSI_14": np.full(n, 55.0),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))
    return SimpleNamespace(ticker=ticker, data=df, latest=df.iloc[-1],
                           fundamentals=None)


@pytest.fixture
def boardroom(monkeypatch):
    b = AnalystBoardroom.__new__(AnalystBoardroom)
    b._llm = FakeLLM()
    b._vote_llm = b._llm.with_structured_output(AnalystVote)
    b._news = _FailingNews()
    return b


# ---------------------------------------------------------------------------
# Packets
# ---------------------------------------------------------------------------

def test_panel_is_a_full_desk():
    assert len(_PANEL) == 8
    kinds = {kind for _, _, _, kind, _ in _PANEL}
    assert kinds == {"technical", "fundamental", "news", "quant",
                     "macro", "risk", "flow", "contrarian"}


def test_packets_are_isolated():
    snap = make_snapshot()
    tech = _technical_packet(snap)
    fund = _fundamental_packet(snap)
    news = _news_packet("TEST", None)
    macro = _macro_packet("TEST", None)
    risk = _risk_packet(snap, in_position=True, entry_price=90.0,
                        daily_pnl_pct=-1.2)
    flow = _flow_packet(snap)
    contra = _contrarian_packet(snap, None, None)
    assert "RSI_14" in tech
    assert "No fundamental data" in fund      # crypto/index path
    assert "unavailable" in news              # graceful news failure
    assert "macro" in macro.lower()
    assert "OPEN POSITION" in risk and "-1.2" in risk
    assert "OBV" in flow
    assert "OTHER side" in contra
    # The fundamental packet must not leak indicator values
    assert "RSI" not in fund


# ---------------------------------------------------------------------------
# Majority fallback
# ---------------------------------------------------------------------------

def _op(vote, conv=0.8, name="X"):
    return AnalystOpinion(name=name, emoji="·", role="r", vote=vote,
                          conviction=conv, opinion="o" * 30)


def test_majority_ruling_buy():
    ops = [_op("BUY"), _op("BUY"), _op("BUY", 0.6), _op("SELL", 0.9)]
    d = AnalystBoardroom._majority_ruling("T", ops)
    assert d.action == "BUY"
    assert 0.5 <= d.confidence_score <= 1.0


def test_majority_ruling_split_is_hold():
    ops = [_op("BUY", 0.6), _op("SELL", 0.6),
           _op("BUY", 0.4), _op("SELL", 0.4)]
    d = AnalystBoardroom._majority_ruling("T", ops)
    assert d.action == "HOLD"


def test_majority_ruling_abstains_ignored():
    ops = [_op("ABSTAIN", 0.0), _op("ABSTAIN", 0.0),
           _op("SELL", 0.9), _op("SELL", 0.8)]
    d = AnalystBoardroom._majority_ruling("T", ops)
    assert d.action == "SELL"


# ---------------------------------------------------------------------------
# Full meeting with fake LLM
# ---------------------------------------------------------------------------

def test_convene_full_meeting(boardroom):
    ruling = boardroom.convene(make_snapshot(), verdict=None,
                               in_position=False)
    assert len(ruling.opinions) == len(_PANEL)
    assert all(o.ok for o in ruling.opinions)
    assert ruling.decision.action == "BUY"
    assert not ruling.chair_is_fallback
    s = ruling.summary_dict()
    assert s["tally"]["BUY"] == len(_PANEL)
    assert s["chair"]["name"] == ruling.chair_name


def test_convene_chair_fallback(boardroom):
    boardroom._llm = FakeLLM(analyst_vote="SELL", chair_raises=True)
    boardroom._vote_llm = boardroom._llm.with_structured_output(AnalystVote)
    ruling = boardroom.convene(make_snapshot(), verdict=None,
                               in_position=True, entry_price=90.0)
    assert ruling.chair_is_fallback
    assert ruling.decision.action == "SELL"   # unanimous panel
