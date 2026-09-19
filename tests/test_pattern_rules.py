import numpy as np
import pandas as pd
import pytest

from strategy.pattern_rules import VARIANTS, FAMILY
from tools.evaluate_patterns_daily import daily_returns, newey_west_t


def bars(n=700, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(.0004, .015, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, .003, n))
    spread = close * rng.uniform(.002, .02, n)
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + spread,
                         "Low": np.minimum(open_, close) - spread, "Close": close},
                        index=pd.date_range("2020-01-01", periods=n, freq="D"))


@pytest.mark.parametrize("name", list(VARIANTS))
def test_rule_never_uses_future_bars(name):
    df = bars()
    full = VARIANTS[name](df)
    for cut in (320, 450, 610):
        assert full.iloc[cut] == VARIANTS[name](df.iloc[:cut + 1]).iloc[-1], f"{name} changed when later bars were removed"


@pytest.mark.parametrize("name", list(VARIANTS))
def test_rule_output_is_binary_exposure(name):
    signal = VARIANTS[name](bars())
    assert set(signal.dropna().unique()) <= {0., 1.}


def test_every_variant_belongs_to_a_family():
    assert set(FAMILY) == set(VARIANTS) and len(VARIANTS) == 16


def test_signal_is_traded_at_next_open_and_costs_hit_that_day():
    df = pd.DataFrame({"Open": [100., 100., 110., 121., 121.], "High": 130., "Low": 90., "Close": 100.},
                      index=pd.date_range("2024-01-01", periods=5))
    signal = pd.Series([0., 1., 1., 0., 0.], index=df.index)  # decided at each close
    returns, held = daily_returns(df, signal, .01)
    # Decided at the second bar's close, so first held from the third bar's open: earns 110->121 = +10%, minus entry cost.
    assert held.tolist() == [0., 0., 1., 1.]
    assert returns.iloc[2] == pytest.approx(.10 - .01)
    assert returns.iloc[1] == 0.  # the signal day itself earns nothing
    assert returns.iloc[3] == 0.  # already held, flat open-to-open, no trade cost


def test_newey_west_needs_enough_data():
    assert np.isnan(newey_west_t(np.ones(10)))
    assert newey_west_t(np.random.default_rng(1).normal(.01, .01, 500)) > 3
