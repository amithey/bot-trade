import json
import math
import threading
from types import SimpleNamespace as NS

import pytest

from market_data.protection_quotes import ProtectionQuote, latest_protection_quote
from portfolio.virtual_account import LivePortfolio
from risk.profit_protection import ProfitProtectionConfig, assess_profit_protection


def assess(**overrides):
    args = {"side": "LONG", "entry_price": 100., "quantity": 10., "entry_fees": 1.,
            "price": 100., "fee_rate": .001, "exit_slippage_bps": 5.}
    args.update(overrides)
    return assess_profit_protection(**args)


@pytest.mark.parametrize("side,price", [("LONG", 100.2), ("SHORT", 99.8)])
def test_gross_profit_that_cannot_pay_costs_does_not_arm(side, price):
    result = assess(side=side, price=price)
    assert result.estimated_net_pct < 0
    assert not result.armed and result.stop_price is None


@pytest.mark.parametrize("side,peak,higher,retrace", [
    ("LONG", 101., 101.2, 100.5), ("SHORT", 99., 98.8, 99.5),
])
def test_stop_ratchets_and_exits_with_positive_net_result(side, peak, higher, retrace):
    first = assess(side=side, price=peak)
    assert first.armed and not first.should_exit
    better = assess(side=side, price=higher, best_price=first.best_price,
                    stop_price=first.stop_price, armed=True)
    assert (better.stop_price > first.stop_price if side == "LONG"
            else better.stop_price < first.stop_price)
    down = assess(side=side, price=retrace, best_price=better.best_price,
                  stop_price=better.stop_price, armed=True)
    assert down.should_exit
    assert down.stop_price == better.stop_price
    assert down.best_price == higher
    assert down.estimated_net_pct > 0


@pytest.mark.parametrize("side,peak,gap", [("LONG", 101., 98.), ("SHORT", 99., 102.)])
def test_gap_uses_observed_price_and_can_realize_a_loss(side, peak, gap):
    first = assess(side=side, price=peak)
    result = assess(side=side, price=gap, best_price=first.best_price,
                    stop_price=first.stop_price, armed=True)
    assert result.should_exit and result.estimated_net_pct < 0
    assert result.estimated_fill == pytest.approx(gap * (1.0005 if side == "SHORT" else .9995))
    assert result.estimated_fill != result.stop_price


@pytest.mark.parametrize("side,peak", [("LONG", 101.), ("SHORT", 99.)])
def test_trigger_price_accounts_for_both_commissions_and_slippage(side, peak):
    result = assess(side=side, price=peak)
    at_stop = assess(side=side, price=result.stop_price, best_price=result.best_price,
                     stop_price=result.stop_price, armed=True)
    assert at_stop.should_exit
    assert at_stop.estimated_net_pct == pytest.approx(result.peak_net_pct * .5)


@pytest.mark.parametrize("field", ["price", "entry_price", "quantity", "entry_fees", "fee_rate", "exit_slippage_bps", "best_price", "stop_price"])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_invalid_numbers_never_change_protection(field, value):
    with pytest.raises(ValueError):
        assess(**{field: value})


@pytest.mark.parametrize("overrides", [
    {"retain_fraction": 0}, {"retain_fraction": 1},
    {"activation_net_pct": 0}, {"minimum_net_pct": .2},
])
def test_invalid_policy_is_rejected(overrides):
    with pytest.raises(ValueError):
        ProfitProtectionConfig(**overrides)


def portfolio(side="LONG"):
    port = LivePortfolio(10_000)
    if side == "LONG":
        port.buy("TEST", 100., quantity=10.)
    else:
        port.open_short("TEST", 100., cash_amount=1000.)
    return port


def protect(port, price, at=1., **kwargs):
    return port.protect_profit("TEST", price, observed_at=at, exit_slippage_bps=5., **kwargs)


@pytest.mark.parametrize("side,peak,retrace", [("LONG", 101., 100.5), ("SHORT", 99., 99.5)])
def test_restart_and_price_updates_preserve_fees_peak_and_stop(tmp_path, side, peak, retrace):
    port = portfolio(side)
    state, trade = protect(port, peak)
    assert not trade
    path = tmp_path / "paper.json"
    port.save(path)
    restored = LivePortfolio.load(path)
    restored.update_price("TEST", peak)
    pos = restored.positions["TEST"]
    assert pos.entry_fees == 1.
    assert pos.best_price == peak and pos.profit_stop_price == state.stop_price
    _, trade = protect(restored, retrace, at=2.)
    assert trade.realized_pnl > 0
    assert restored.cash - restored.initial_capital == pytest.approx(trade.realized_pnl)
    assert not restored.positions


def test_partial_sale_keeps_stop_and_allocated_costs():
    port = portfolio()
    state, _ = protect(port, 101.)
    port.sell("TEST", 101., quantity=4.)
    pos = port.positions["TEST"]
    assert pos.profit_stop_price == state.stop_price
    assert pos.entry_fees == pytest.approx(.6)
    _, trade = protect(port, 100.5, at=2.)
    assert trade.quantity == 6.
    assert port.cash - port.initial_capital == pytest.approx(port.get_realized_pnl())


def test_armed_position_cannot_be_diluted_by_pyramiding():
    port = portfolio()
    protect(port, 101.)
    before = (port.cash, port.positions, port.trade_log)
    with pytest.raises(ValueError, match="armed profit protection"):
        port.buy("TEST", 101., quantity=10.)
    assert (port.cash, port.positions, port.trade_log) == before


def test_new_position_does_not_inherit_an_old_peak():
    port = portfolio()
    protect(port, 101.)
    protect(port, 100.5, at=2.)
    port.buy("TEST", 105., quantity=10.)
    assert not port.positions["TEST"].profit_protection_armed
    assert port.positions["TEST"].best_price is None
    assert not protect(port, 105., at=3.)[0].armed


def test_old_inflight_quote_cannot_close_a_replacement_position():
    port = portfolio()
    revision = len(port.trade_log)
    port.sell("TEST", 100.)
    port.buy("TEST", 98., quantity=10.)
    assert protect(port, 101., expected_revision=revision) == (None, None)
    assert port.positions["TEST"].best_price is None


def test_out_of_order_quote_is_ignored():
    port = portfolio()
    protect(port, 101., at=2.)
    assert protect(port, 98., at=1.) == (None, None)
    assert len(port.trade_log) == 1


def test_failed_exit_is_retried_even_if_price_recovers(monkeypatch):
    port = portfolio()
    protect(port, 101.)
    original = port.sell
    monkeypatch.setattr(port, "sell", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fill failed")))
    with pytest.raises(RuntimeError):
        protect(port, 100.5, at=2.)
    assert port.positions["TEST"].profit_exit_pending
    monkeypatch.setattr(port, "sell", original)
    assert protect(port, 101.2, at=3.)[1] is not None


def test_schema_three_position_starts_without_an_invented_peak(tmp_path):
    port = portfolio()
    path = tmp_path / "old.json"
    port.save(path)
    raw = json.loads(path.read_text())
    raw["schema_version"] = 3
    raw["positions"]["TEST"] = {k: v for k, v in raw["positions"]["TEST"].items()
                                 if not k.startswith("profit_") and k != "best_price"}
    path.write_text(json.dumps(raw))
    restored = LivePortfolio.load(path)
    assert restored.positions["TEST"].entry_fees == 1
    assert restored.positions["TEST"].best_price is None


def test_concurrent_saves_produce_a_complete_loadable_file(tmp_path):
    port = portfolio()
    path = tmp_path / "paper.json"
    failures = []
    def save():
        try:
            for _ in range(8):
                port.save(path)
        except Exception as exc:  # noqa: BLE001 - surface worker failures in the test thread
            failures.append(exc)
    threads = [threading.Thread(target=save) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert not failures and not any(t.is_alive() for t in threads)
    assert LivePortfolio.load(path).positions["TEST"].entry_fees == 1


@pytest.mark.parametrize("delta", [-10, 121])
def test_protection_quote_rejects_future_and_stale_candles(delta):
    with pytest.raises(ValueError):
        ProtectionQuote(100., 1000. - delta, 1000.).validate(now=1000.)


def test_price_only_feed_uses_bounded_request_and_no_fundamentals(monkeypatch):
    import time
    from contextlib import nullcontext

    import requests
    calls = []
    payload = {"chart": {"error": None, "result": [{"timestamp": [time.time()],
               "indicators": {"quote": [{"close": [100.]}]}}]}}
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return nullcontext(NS(raise_for_status=lambda: None, json=lambda: payload))
    monkeypatch.setattr(requests, "get", get)
    quote = latest_protection_quote("TEST/A")
    assert quote.price == 100.
    assert len(calls) == 1
    assert calls[0][0].endswith("/chart/TEST%2FA")
    assert calls[0][1]["timeout"] == (2, 5)
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["params"] == {"range": "1d", "interval": "1m", "includePrePost": "false"}


def test_expired_request_is_rejected_even_with_recent_candle():
    with pytest.raises(ValueError):
        ProtectionQuote(100., 1000., 960.).validate(now=1000.)


@pytest.mark.parametrize("payload", [
    {"chart": {"error": {"code": "Not Found"}, "result": None}},
    {"chart": {"error": None, "result": []}},
    {"chart": {"result": [{"timestamp": [], "indicators": {"quote": [{"close": []}]}}]}},
    {"chart": {"result": [{"timestamp": [1.], "indicators": {"quote": [{"close": [None]}]}}]}},
    {"chart": {"result": [{"timestamp": [1., 2.], "indicators": {"quote": [{"close": [100.]}]}}]}},
])
def test_price_feed_rejects_missing_or_misaligned_data(monkeypatch, payload):
    from contextlib import nullcontext

    import requests

    response = NS(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(requests, "get", lambda *a, **k: nullcontext(response))
    with pytest.raises(ValueError):
        latest_protection_quote("TEST")


def test_reduced_cost_estimate_cannot_loosen_an_armed_stop():
    first = assess(price=101.)
    next_quote = assess(price=101., best_price=first.best_price, armed=True,
                        stop_price=first.stop_price, fee_rate=0., exit_slippage_bps=0.)
    assert next_quote.stop_price >= first.stop_price
