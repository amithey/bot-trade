import numpy as np
import pandas as pd
import pytest

from strategy.trend_controls import trend_control_features, ratchet_chandelier
from strategy.committee_replay import replay_committee
from tests.test_committee_policy import replay_data


def test_chandelier_ratcheting_is_position_specific_and_symmetric():
    assert ratchet_chandelier("LONG", 95., 94.) == 95.
    assert ratchet_chandelier("LONG", 95., 97.) == 97.
    assert ratchet_chandelier("SHORT", 105., 106.) == 105.
    assert ratchet_chandelier("SHORT", 105., 103.) == 103.
    assert ratchet_chandelier("LONG", None, 80.) == 80.  # new position resets
    assert ratchet_chandelier("LONG", 95., float("nan")) == 95.


def test_trend_controls_never_see_future_candles():
    df, _, _ = replay_data()
    rng = np.random.default_rng(91)
    df.Close = 100 + rng.normal(0, .3, len(df)).cumsum()
    df.Open = df.Close.shift().fillna(df.Close.iloc[0])
    df.High = df[["Open", "Close"]].max(axis=1) + .2
    df.Low = df[["Open", "Close"]].min(axis=1) - .2
    full = trend_control_features(df)
    pd.testing.assert_frame_equal(full.iloc[:650], trend_control_features(df.iloc[:650]))


def test_adx_is_an_entry_veto_even_with_unanimous_votes():
    df, features, votes = replay_data()
    controls = trend_control_features(df)
    controls["adx"] = 19.
    kwargs = dict(enhanced=True, evaluation_start=720, features=features, votes=votes, controls=controls)
    assert replay_committee(df, **kwargs)["trades"] == 1
    assert replay_committee(df, adx_min=25., **kwargs)["trades"] == 0
    controls["adx"] = 25.
    assert replay_committee(df, adx_min=25., **kwargs)["trades"] == 1


def test_chandelier_uses_prior_bar_level_and_adverse_gap_fill():
    df, features, votes = replay_data()
    controls = trend_control_features(df)
    controls["chandelier_long"] = 99.5
    df.loc[df.index[722], ["Open", "High", "Low", "Close"]] = [99., 99.1, 98.9, 99.]
    # A future/current bar level must not tighten today's stop.
    controls.loc[df.index[722], "chandelier_long"] = 200.
    result = replay_committee(df, enhanced=True, evaluation_start=720, features=features,
                              votes=votes, controls=controls, chandelier=True)
    trade = result["trade_log"][0]
    assert trade["reason"] == "chandelier stop"
    assert trade["exit"] < 99.


def test_chandelier_does_not_loosen_hard_stop():
    df, features, votes = replay_data()
    controls = trend_control_features(df)
    controls["chandelier_long"] = 80.
    df.loc[df.index[722], ["Open", "High", "Low", "Close"]] = [98., 98.1, 97.9, 98.]
    result = replay_committee(df, enhanced=True, evaluation_start=720, features=features,
                              votes=votes, controls=controls, chandelier=True)
    assert result["trade_log"][0]["reason"] == "hard stop"


def test_default_replay_remains_unchanged_when_controls_disabled():
    df, features, votes = replay_data()
    kwargs = dict(enhanced=True, evaluation_start=720, features=features, votes=votes)
    assert replay_committee(df, **kwargs) == replay_committee(df, adx_min=None, chandelier=False, **kwargs)


@pytest.mark.parametrize("threshold", [-1, 101, float("nan")])
def test_invalid_adx_threshold_rejected(threshold):
    df, _, _ = replay_data()
    with pytest.raises(ValueError):
        replay_committee(df, enhanced=True, evaluation_start=720, adx_min=threshold)


@pytest.mark.parametrize("timeframe", ["30min", "1h"])
def test_higher_timeframe_never_uses_incomplete_candles(timeframe):
    df, _, _ = replay_data()
    df.High = np.linspace(100.1, 110., len(df))
    full = trend_control_features(df, chandelier_timeframe=timeframe)
    for end in (647, 648, 649, 653):
        pd.testing.assert_frame_equal(full.iloc[:end], trend_control_features(df.iloc[:end], chandelier_timeframe=timeframe))
    # Alter a later component of the same hourly candle: earlier values stay fixed.
    changed = df.copy()
    changed.loc[changed.index[659], "High"] = 1000.
    after = trend_control_features(changed, chandelier_timeframe=timeframe)
    pd.testing.assert_frame_equal(full.iloc[:659], after.iloc[:659])


def test_forward_mark_keeps_open_trade_and_accounts_for_entry_fee():
    df, features, votes = replay_data()
    # End before maximum hold; no exit event should be invented at snapshot time.
    df, features, votes = df.iloc[:730], features.iloc[:730], votes.iloc[:730]
    kwargs = dict(enhanced=True, evaluation_start=720, features=features, votes=votes)
    live = replay_committee(df, finalize=False, **kwargs)
    closed = replay_committee(df, **kwargs)
    assert live["trades"] == 0 and live["open_position"] is not None
    assert closed["trades"] == 1 and closed["open_position"] is None
    assert closed["equity"] < live["equity"] < 10000
    assert live["return_pct"] == pytest.approx((live["equity"] / 10000 - 1) * 100)


def test_session_gap_does_not_change_candle_close_timestamp():
    df, features, votes = replay_data()
    # New entry on the first bar after a weekend; force only end-of-window exit.
    df, features, votes = df.iloc[:723].copy(), features.iloc[:723].copy(), votes.iloc[:723].copy()
    new_index = df.index[:-1].append(pd.DatetimeIndex([df.index[-1] + pd.Timedelta(days=2)]))
    df.index = features.index = votes.index = new_index
    features.loc[:, "eligible"] = False
    features.loc[new_index[-2]] = features.loc[new_index[720]]
    features.loc[new_index[-2], "eligible"] = True
    result = replay_committee(df, enhanced=True, evaluation_start=720, features=features, votes=votes)
    trade = result["trade_log"][0]
    assert trade["reason"] == "end of evaluation"
    assert pd.Timestamp(trade["exit_time"]) == new_index[-1] + pd.Timedelta(minutes=5)
