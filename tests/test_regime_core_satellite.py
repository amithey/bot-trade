import numpy as np
import pandas as pd
import pytest

from portfolio.virtual_account import LivePortfolio
from strategy.regime_core_satellite import (
    CoreSatelliteConfig,
    detect_regime,
    plan_core_satellite,
    calculate_core_rebalance_action,
)


def sample_daily_data(above_sma=True):
    n = 250
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    base = 100.0
    # Create trend up or down
    if above_sma:
        prices = np.linspace(80, 150, n)
    else:
        prices = np.linspace(150, 80, n)
    return pd.DataFrame({"Close": prices}, index=dates)


def test_detect_regime_bullish_and_bearish():
    df_bull = sample_daily_data(above_sma=True)
    reg_bull = detect_regime(df_bull, benchmark="SPY")
    assert reg_bull.is_bullish
    assert reg_bull.regime == "BULLISH"
    assert reg_bull.pct_from_sma > 0

    df_bear = sample_daily_data(above_sma=False)
    reg_bear = detect_regime(df_bear, benchmark="SPY")
    assert not reg_bear.is_bullish
    assert reg_bear.regime == "BEARISH"
    assert reg_bear.pct_from_sma < 0


def test_plan_core_satellite():
    df_bull = sample_daily_data(above_sma=True)
    plan_bull = plan_core_satellite(10000.0, df_bull, CoreSatelliteConfig(core_target_pct=45.0))
    assert plan_bull.target_core_pct == 45.0
    assert plan_bull.target_core_value == pytest.approx(4500.0)
    assert plan_bull.target_satellite_pct == 55.0
    assert plan_bull.target_satellite_value == pytest.approx(5500.0)

    df_bear = sample_daily_data(above_sma=False)
    plan_bear = plan_core_satellite(10000.0, df_bear, CoreSatelliteConfig(core_target_pct=45.0))
    # In bear regime, core rotates to 0% cash protection
    assert plan_bear.target_core_pct == 0.0
    assert plan_bear.target_core_value == 0.0


def test_core_rebalance_action():
    port = LivePortfolio(initial_capital=10000.0)
    df_bull = sample_daily_data(above_sma=True)
    
    # Empty portfolio -> needs BUY rebalance into Core
    action = calculate_core_rebalance_action(port, df_bull)
    assert action["action"] == "BUY"
    assert action["rebalance_cash"] > 4000.0

    # Simulate buying core
    port.buy("SPY", 150.0, cash_amount=4500.0)
    # Now within tolerance -> HOLD
    action_hold = calculate_core_rebalance_action(port, df_bull)
    assert action_hold["action"] == "HOLD"

    # Shift regime to bear -> needs SELL
    df_bear = sample_daily_data(above_sma=False)
    action_sell = calculate_core_rebalance_action(port, df_bear)
    assert action_sell["action"] == "SELL"
