from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strategy.committee_evidence import chart_features, chart_admission, fundamental_assessment, decision_evidence
from strategy.committee_swing import completed_hourly_bars, swing_inputs
from strategy.committee_replay import replay_committee
from strategy.trend_controls import trend_control_features
from tests.test_committee_policy import replay_data


def test_chart_features_and_confirmed_pivots_never_see_future():
    df, _, _ = replay_data()
    rng = np.random.default_rng(19)
    df.Close = 100 + rng.normal(0, .1, len(df)).cumsum()
    df.High, df.Low = df.Close + .1, df.Close - .1
    df.Open = df.Close.shift().fillna(df.Close.iloc[0])
    df.High = df[["High", "Open"]].max(axis=1)
    df.Low = df[["Low", "Open"]].min(axis=1)
    full = chart_features(df)
    for cutoff in (211, 499, 722):
        pd.testing.assert_frame_equal(full.iloc[:cutoff], chart_features(df.iloc[:cutoff]))
    assert full.efficiency.dropna().between(0, 1.00000001).all()


def test_direction_structure_and_rejections_are_symmetric():
    row = {"di_spread": 10., "efficiency": .5, "structure": "RISING"}
    assert chart_admission(row, "LONG")["allowed"]
    assert not chart_admission(row, "SHORT")["allowed"]
    assert not chart_admission({**row, "failed_breakout": True}, "LONG")["allowed"]
    assert chart_admission({**row, "di_spread": -10., "structure": "FALLING"}, "SHORT")["allowed"]
    assert not chart_admission({**row, "efficiency": np.nan}, "LONG")["allowed"]


def test_company_quality_uses_fresh_evidence_and_never_approves_missing_data():
    now = datetime(2026, 9, 19, tzinfo=timezone.utc)
    bad = SimpleNamespace(profit_margin=-.1, free_cash_flow=-20., revenue_growth=-.1,
                          debt_to_equity=250., fetched_at=now)
    assert fundamental_assessment("META", "LONG", fundamentals=bad, now=now)["veto"]
    assert not fundamental_assessment("META", "SHORT", fundamentals=bad, now=now)["veto"]
    bad.fetched_at = now - timedelta(hours=3)
    stale = fundamental_assessment("META", "LONG", fundamentals=bad, now=now)
    assert stale["status"] == "STALE" and not stale["veto"]
    missing = fundamental_assessment("SPY", "LONG", now=now)
    assert missing["status"] == "UNAVAILABLE"
    assert missing["directional_edge"] == "NOT_ESTABLISHED"


def test_crypto_is_not_company_valuation_and_future_snapshots_are_rejected():
    now = datetime(2026, 9, 19, tzinfo=timezone.utc)
    crypto = {"status": "AVAILABLE", "observed_at": now.isoformat(),
              "metrics": {"price": 100000, "market_cap": 2e12, "volume_24h": 2e10}}
    result = fundamental_assessment("BTC-USD", "LONG", crypto=crypto, now=now)
    assert result["status"] == "CONTEXT_ONLY" and not result["veto"]
    crypto["observed_at"] = (now + timedelta(minutes=1)).isoformat()
    assert fundamental_assessment("BTC-USD", "LONG", crypto=crypto, now=now)["status"] == "UNAVAILABLE"


def test_hourly_adapter_rejects_partial_hours_and_has_prefix_invariance():
    df, _, _ = replay_data()
    all_hours = completed_hourly_bars(df)
    for cutoff in (647, 648, 649):
        partial = completed_hourly_bars(df.iloc[:cutoff])
        pd.testing.assert_frame_equal(partial, all_hours.loc[partial.index])
        assert (partial.index + pd.Timedelta(hours=1) <= df.index[cutoff - 1] + pd.Timedelta(minutes=5)).all()
    missing = df.drop(df.index[10])
    assert df.index[0] not in completed_hourly_bars(missing).index


def test_swing_signals_keep_all_execution_bars_and_no_future_hour():
    df, _, _ = replay_data()
    full = swing_inputs(df)
    prefix = swing_inputs(df.iloc[:647])
    for whole, partial in zip(full, prefix):
        assert whole.index.equals(df.index)
        # Last prefix row can be ineligible because its execution bar has
        # not arrived yet. Earlier decisions must be prefix-invariant.
        pd.testing.assert_frame_equal(whole.iloc[:646], partial.iloc[:646])
    assert not full[0].eligible.loc[df.index.minute != 55].any()


def test_wider_swing_stop_reduces_size_and_stays_fixed_after_entry():
    df, features, votes = replay_data()
    df, features, votes = df.iloc[:730], features.iloc[:730].copy(), votes.iloc[:730]
    features["atr"] = 1.
    controls = trend_control_features(df)
    controls["chandelier_long"] = 90.
    result = replay_committee(df, enhanced=True, evaluation_start=720, features=features,
        votes=votes, controls=controls, chandelier=True, trend_holding=True, volatility_stop=True, finalize=False)
    pos = result["open_position"]
    assert pos["stop_loss_pct"] == pytest.approx(2.5)
    assert pos["notional"] < 1000.
    assert pos["notional"] * (pos["stop_loss_pct"] + .3) / 100 <= 25.
    features["atr"] = 3.
    blocked = replay_committee(df, enhanced=True, evaluation_start=720, features=features,
        votes=votes, controls=controls, chandelier=True, trend_holding=True, volatility_stop=True)
    assert blocked["trades"] == 0


def test_evidence_is_serializable_and_explicitly_shadow():
    import json
    df, _, _ = replay_data()
    packet = decision_evidence(df, ticker="BTC-USD", side="LONG")
    json.dumps(packet, allow_nan=False)
    assert packet["mode"] == "SHADOW"
    assert packet["fundamentals"]["status"] == "UNAVAILABLE"


def test_observation_deduplicates_bars_and_excludes_private_fields(tmp_path):
    import json
    import sqlite3
    from strategy.committee_observations import record_observation
    path = tmp_path / "audit.sqlite3"
    report = {"ticker": "BTC-USD", "bar_closed_at": "2026-09-19T10:00:00Z", "committee_policy_version": "v5",
              "entry_allowed": False, "credentials": "must-not-persist", "balance": 12345}
    assert not record_observation(None, report, path=path)
    record_observation("account@example.test", report, path=path)
    record_observation("account@example.test", report, path=path)
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT account,packet FROM observations").fetchall()
    assert len(rows) == 1 and len(rows[0][0]) == 64
    assert "credentials" not in json.loads(rows[0][1])
    assert "balance" not in json.loads(rows[0][1])
    assert "account@example.test" not in str(rows)
