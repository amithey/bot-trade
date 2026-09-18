"""Bounded public crypto fundamentals; no account keys or order endpoints.

Market cap/turnover describe liquidity and supply, not intrinsic value. Company
financial statements are supplied separately by the existing equity fetcher.
"""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite
import threading
import time

COINS = {"BTC-USD": "btc-bitcoin", "ETH-USD": "eth-ethereum",
         "SOL-USD": "sol-solana", "DOGE-USD": "doge-dogecoin",
         "ADA-USD": "ada-cardano", "XRP-USD": "xrp-xrp"}
_lock = threading.Lock()
_cache = OrderedDict()
_busy = set()


def number(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (ValueError, TypeError):
        return None


def normalize_crypto(payload, ticker, *, now=None):
    now = now or datetime.now(timezone.utc)
    source = f"https://api.coinpaprika.com/v1/tickers/{COINS[ticker]}"
    result = {"status": "UNAVAILABLE", "asset_type": "CRYPTO", "source": source,
              "fetched_at": now.isoformat(), "metrics": {},
              "limitations": "Market/supply context, not intrinsic valuation. No on-chain or funding feed. Supply fields may be unavailable on the free API."}
    if payload.get("id") != COINS[ticker] or payload.get("symbol") != ticker.split("-")[0]:
        result["reason"] = "Provider asset identity mismatch"
        return result
    try:
        stamp = datetime.fromisoformat(payload["last_updated"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Timestamp needs timezone")
        age = (now - stamp).total_seconds()
    except (KeyError, ValueError, TypeError):
        result["reason"] = "Missing provider timestamp"
        return result
    quote = payload.get("quotes", {}).get("USD", {})
    metrics = {key: number(quote.get(key)) for key in ("price", "market_cap", "volume_24h")}
    metrics.update({key: number(payload.get(key)) for key in ("circulating_supply", "max_supply")})
    result.update(observed_at=stamp.isoformat(), metrics=metrics)
    if not 0 <= age <= 900:
        result.update(status="STALE", reason="Crypto snapshot is stale or future-dated")
    elif any(metrics[key] is None or metrics[key] <= 0 for key in ("price", "market_cap", "volume_24h")):
        result["reason"] = "Missing positive price, market cap or traded volume"
    else:
        result.update(status="AVAILABLE", reason="Timestamped market capitalization and liquidity available")
    return result


def _refresh(ticker):
    result = {"status": "UNAVAILABLE", "reason": "Public fundamentals provider unavailable", "metrics": {}}
    try:
        import requests
        response = requests.get(f"https://api.coinpaprika.com/v1/tickers/{COINS[ticker]}",
                                timeout=(2, 4), allow_redirects=False)
        response.raise_for_status()
        result = normalize_crypto(response.json(), ticker)
    except Exception:
        pass
    finally:
        with _lock:
            _cache[ticker] = (time.monotonic(), result)
            _cache.move_to_end(ticker)
            while len(_cache) > 32:
                _cache.popitem(last=False)
            _busy.discard(ticker)


def crypto_context(ticker):
    """Never block the trading/risk thread or turn missing data into approval."""
    if ticker not in COINS:
        return {"status": "UNAVAILABLE", "reason": "No verified crypto asset mapping", "metrics": {}}
    with _lock:
        cached = _cache.get(ticker)
        ttl = 300 if cached and cached[1]["status"] == "AVAILABLE" else 60
        if cached and time.monotonic() - cached[0] < ttl:
            result = deepcopy(cached[1])
            if result["status"] == "AVAILABLE":
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(result["observed_at"])).total_seconds()
                if not 0 <= age <= 900:
                    result.update(status="STALE", reason="Cached provider snapshot expired")
            return result
        if ticker not in _busy and len(_busy) < 4:
            _busy.add(ticker)
            threading.Thread(target=_refresh, args=(ticker,), daemon=True,
                             name="committee_fundamentals").start()
    return {"status": "LOADING", "reason": "Waiting for timestamped crypto fundamentals", "metrics": {}}


def fundamental_gate(ticker, *, side, price, order_value, fundamentals=None, crypto=None, now=None):
    """Risk vetoes, not a directional price forecast. Missing evidence blocks entry only."""
    now = now or datetime.now(timezone.utc)
    checks = []
    if ticker.endswith(("-USD", "-USDT")):
        context = deepcopy(crypto) if crypto is not None else crypto_context(ticker)
        metrics = context.get("metrics", {})
        checks.append((context.get("status") == "AVAILABLE", context.get("reason", "Crypto fundamentals unavailable")))
        reference, volume = number(metrics.get("price")), number(metrics.get("volume_24h"))
        checks.append((reference is not None and reference > 0 and abs(price / reference - 1) <= .02,
                       "Execution quote must agree with independent USD market price within 2%"))
        checks.append((volume is not None and volume > 0 and 0 < order_value <= volume * .0001,
                       "Order must be <= 0.01% of reported daily USD volume; volume is not order-book depth"))
    else:
        keys = ("profit_margin", "free_cash_flow", "revenue_growth", "debt_to_equity")
        metrics = {key: number(getattr(fundamentals, key, None)) for key in keys}
        fetched = getattr(fundamentals, "fetched_at", None)
        if fetched is not None and fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        fresh = fetched is not None and 0 <= (now - fetched).total_seconds() <= 7200
        context = {"status": "AVAILABLE" if fresh else "UNAVAILABLE", "asset_type": "EQUITY",
                   "source": "Yahoo Finance company snapshot", "metrics": metrics,
                   "fetched_at": fetched.isoformat() if fetched else None,
                   "limitations": "Provider financial snapshot, not verified filings or point-in-time historical fundamentals; not supported for funds without company statements."}
        checks.append((fresh and all(metrics[k] is not None for k in keys),
                       "Need current provider snapshot of profitability, free cash flow, growth and leverage"))
        distressed = (metrics["profit_margin"] is not None and metrics["profit_margin"] < 0
                      and metrics["free_cash_flow"] is not None and metrics["free_cash_flow"] < 0)
        checks.append((side != "LONG" or not distressed,
                       "Long entry veto when both profit margin and free cash flow are negative"))
        fragile = (metrics["revenue_growth"] is not None and metrics["revenue_growth"] < 0
                   and metrics["debt_to_equity"] is not None and metrics["debt_to_equity"] > 200)
        checks.append((side != "LONG" or not fragile,
                       "Long entry veto for shrinking revenue with debt/equity above 200%"))
    failed = [detail for passed, detail in checks if not passed]
    return {"allowed": not failed, "reason": "; ".join(failed) if failed else "Fundamental risk checks passed",
            "checks": [{"name": "Fundamental risk", "passed": bool(ok), "detail": detail} for ok, detail in checks],
            "context": context}
