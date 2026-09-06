"""Regression coverage for account integrity and aggregate exposure."""
import math

import pytest

from portfolio.virtual_account import LivePortfolio
from risk.sizing import allocate_buy


def allocation(**overrides):
    args = dict(equity=10_000., cash=10_000., existing_value=0.,
                user_cap_pct=20., profile_cap_pct=30., suggested_pct=None,
                fee_rate=.001)
    args.update(overrides)
    return allocate_buy(**args)


def test_pyramiding_cannot_exceed_total_symbol_cap():
    order = allocation(existing_value=1900.)
    assert 0 < order.cash_amount < 100
    post_equity = 10_000 - order.cash_amount * .001
    assert (1900 + order.cash_amount) / post_equity <= .2 + 1e-12
    assert allocation(existing_value=2100.).cash_amount == 0


def test_smaller_analyst_suggestion_is_respected():
    assert allocation(suggested_pct=1.).cash_amount == 100


def test_available_cash_includes_fees():
    order = allocation(cash=50.)
    assert order.cash_amount * 1.001 == pytest.approx(50)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., 0., 101.])
def test_invalid_suggestion_does_not_become_a_buy(value):
    assert allocation(suggested_pct=value).cash_amount == 0


@pytest.mark.parametrize("field", ["equity", "cash", "existing_value", "user_cap_pct", "profile_cap_pct", "fee_rate"])
def test_nonfinite_sizing_fails_closed(field):
    assert allocation(**{field: math.nan}).cash_amount == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -1., 0.])
@pytest.mark.parametrize("operation", ["buy_price", "buy_quantity", "buy_cash", "sell_price", "sell_quantity", "mark"])
def test_bad_order_cannot_corrupt_account(value, operation):
    port = LivePortfolio(initial_capital=10_000.)
    port.buy("AAPL", 100., quantity=10.)
    before = (port.cash, port.get_total_value(), len(port.trade_log))
    with pytest.raises(ValueError):
        if operation == "buy_price":
            port.buy("AAPL", value, quantity=1.)
        elif operation == "buy_quantity":
            port.buy("AAPL", 100., quantity=value)
        elif operation == "buy_cash":
            port.buy("AAPL", 100., cash_amount=value)
        elif operation == "sell_price":
            port.sell("AAPL", value)
        elif operation == "sell_quantity":
            port.sell("AAPL", 100., quantity=value)
        else:
            port.update_price("AAPL", value)
    assert (port.cash, port.get_total_value(), len(port.trade_log)) == before
