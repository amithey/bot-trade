from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from config.user_profile import RISK_ENVELOPES
from strategy.committee import IndicatorCommittee
from strategy.committee_policy import assess_committee_entry, committee_envelope
from strategy.committee_replay import replay_committee
from strategy.research import entry_features


def evidence(side="LONG"):
    return {"entry_allowed": True, "signal_side": side,
            "regime": "UPTREND" if side == "LONG" else "DOWNTREND",
            "setup": "TREND_PULLBACK" if side == "LONG" else "SHORT_RALLY_REJECTION",
            "timeframe_bias": side, "metrics": {"close": 100., "atr": .2}}


def assess(report=None, **kwargs):
    args = dict(score=.5, quorum=True, categories={k: .5 for k in ("Trend", "Momentum", "Volume")},
                fee_rate=.001, slippage_bps=5., stop_loss_pct=.8, take_profit_pct=2.)
    args.update(kwargs)
    return assess_committee_entry(evidence() if report is None else report, **args)


def test_admission_uses_costs_and_caps_estimated_risk():
    result = assess()
    assert result.allowed
    assert result.size_pct / 100 * (.8 + result.round_trip_cost_pct) == pytest.approx(.25)
    assert not assess(fee_rate=.01).allowed
    assert not assess(take_profit_pct=.5).allowed
    assert not assess(minutes_since_exit=14.99).allowed
    assert assess(minutes_since_exit=15.).allowed


@pytest.mark.parametrize("field,value", [("timeframe_bias", "UNKNOWN"), ("timeframe_bias", "SHORT"),
                                         ("regime", "RANGE"), ("setup", "RANGE_REVERSAL"),
                                         ("entry_allowed", False)])
def test_majority_cannot_bypass_missing_thesis(field, value):
    report = evidence()
    report[field] = value
    assert not assess(report).allowed


def test_short_rules_are_symmetric_and_require_independent_groups():
    negative = {k: -.5 for k in ("Trend", "Momentum", "Volume")}
    assert assess(evidence("SHORT"), score=-.5, categories=negative).allowed
    assert not assess(evidence("SHORT"), score=.5, categories=negative).allowed
    assert not assess(categories={"Trend": 1., "Momentum": 1., "Volume": 0.}).allowed
    assert not assess(categories={}).allowed
    assert not assess(quorum=False).allowed


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1., 0.])
def test_missing_or_invalid_volatility_fails_closed(value):
    report = evidence()
    report["metrics"]["atr"] = value
    assert not assess(report).allowed


def test_committee_envelope_does_not_mutate_other_modes():
    original = deepcopy(RISK_ENVELOPES["Balanced"])
    adjusted = committee_envelope(original)
    assert adjusted.stop_loss_pct == original.stop_loss_pct
    assert adjusted.take_profit_pct == 2.
    assert RISK_ENVELOPES["Balanced"] == original


def replay_data():
    df = pd.DataFrame({"Open": 100., "High": 100.1, "Low": 99.9,
                       "Close": 100., "Volume": 1000.},
                      index=pd.date_range("2026-01-01", periods=800, freq="5min", tz="UTC"))
    features = entry_features(df)
    features["eligible"] = False
    features.loc[df.index[720], ["eligible", "signal_side", "regime", "setup",
                                "higher_timeframe_bias", "required_committee_score", "signal_confidence"]] = [
        True, "LONG", "UPTREND", "TREND_PULLBACK", "LONG", .1, .7]
    features["exit_recommended"] = False
    votes = pd.DataFrame(1, index=df.index, columns=[a.name for a in IndicatorCommittee().agents])
    return df, features, votes


def test_replay_next_open_accounting_and_cost_stress():
    df, features, votes = replay_data()
    kwargs = dict(enhanced=True, evaluation_start=720, features=features, votes=votes)
    base = replay_committee(df, **kwargs)
    stress = replay_committee(df, slippage_multiplier=2., **kwargs)
    assert base["trades"] == stress["trades"] == 1
    assert base["trade_log"][0]["entry_time"] == str(df.index[721])
    assert base["net_pnl"] == pytest.approx(base["return_pct"] * 100)
    assert stress["return_pct"] < base["return_pct"] < 0


def test_replay_gap_stop_cannot_fill_at_stale_stop_level():
    df, features, votes = replay_data()
    df.loc[df.index[722], ["Open", "High", "Low", "Close"]] = [98., 98.1, 97.9, 98.]
    result = replay_committee(df, enhanced=True, evaluation_start=720, features=features, votes=votes)
    trade = result["trade_log"][0]
    assert trade["reason"] == "hard stop"
    assert trade["exit"] < 98.


def test_replay_requires_a_chronological_window():
    df, _, _ = replay_data()
    with pytest.raises(ValueError):
        replay_committee(df.iloc[::-1], enhanced=True, evaluation_start=720)
