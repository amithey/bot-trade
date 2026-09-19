"""Causal chart structure and asset-specific fundamental decision evidence.

Research rules are hypotheses, not fitted probabilities or profit forecasts.
No network calls; callers supply completed candles and timestamped snapshots.
"""
from datetime import datetime, timezone
from math import isfinite

import numpy as np
import pandas as pd

from strategy.committee import _adx

VERSION = "committee-evidence-v1"


def chart_features(df):
    adx, plus, minus = _adx(df, 14)
    close, high, low = df.Close, df.High, df.Low
    travel = close.diff().abs().rolling(24, min_periods=24).sum()
    efficiency = (close - close.shift(24)).abs() / travel.replace(0, np.nan)
    efficiency = efficiency.mask(travel == 0, 0.)
    # A swing at t-2 is confirmed only at t. No centered rolling window
    # is exposed before its two right-hand candles actually close.
    pivot_high = high.shift(2).where(
        (high.shift(2) > high) & (high.shift(2) > high.shift(1)) &
        (high.shift(2) > high.shift(3)) & (high.shift(2) > high.shift(4)))
    pivot_low = low.shift(2).where(
        (low.shift(2) < low) & (low.shift(2) < low.shift(1)) &
        (low.shift(2) < low.shift(3)) & (low.shift(2) < low.shift(4)))
    prior_high = pivot_high.ffill().shift().where(pivot_high.notna()).ffill()
    prior_low = pivot_low.ffill().shift().where(pivot_low.notna()).ffill()
    last_high, last_low = pivot_high.ffill(), pivot_low.ffill()
    rising = (last_high > prior_high) & (last_low > prior_low)
    falling = (last_high < prior_high) & (last_low < prior_low)
    structure = pd.Series("MIXED", index=df.index)
    structure.loc[prior_high.isna() | prior_low.isna()] = "UNKNOWN"
    structure.loc[rising] = "RISING"
    structure.loc[falling] = "FALLING"
    ceiling = high.rolling(20, min_periods=20).max().shift()
    floor = low.rolling(20, min_periods=20).min().shift()
    breakout, breakdown = close > ceiling, close < floor
    breakout_level = ceiling.where(breakout).ffill(limit=6).shift()
    breakdown_level = floor.where(breakdown).ffill(limit=6).shift()
    retest_long = (low <= breakout_level) & (close > breakout_level) & (close > df.Open)
    retest_short = (high >= breakdown_level) & (close < breakdown_level) & (close < df.Open)
    return pd.DataFrame({
        "adx": adx, "di_spread": plus - minus, "efficiency": efficiency,
        "structure": structure, "swing_high": last_high, "swing_low": last_low,
        "failed_breakout": (high > ceiling) & (close <= ceiling),
        "failed_breakdown": (low < floor) & (close >= floor),
        "breakout_retest": retest_long, "breakdown_retest": retest_short,
        "breakout": breakout, "breakdown": breakdown,
    }, index=df.index)


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (TypeError, ValueError):
        return None


def chart_admission(row, side, variant="structure"):
    if variant not in ("direction", "efficiency", "structure", "retest"):
        raise ValueError("Unknown chart hypothesis")
    sign = 1 if side == "LONG" else -1
    spread, efficiency = number(row.get("di_spread")), number(row.get("efficiency"))
    checks = [{"name": "Directional movement", "passed": side in ("LONG", "SHORT") and spread is not None and sign * spread > 0,
               "detail": f"DI direction must confirm {side}; +DI minus -DI={spread}"}]
    if variant in ("efficiency", "structure"):
        checks.append({"name": "Trend efficiency", "passed": efficiency is not None and efficiency >= .25,
                       "detail": f"24-bar net movement / traveled distance >= 0.25; observed {efficiency}"})
    if variant == "structure":
        opposing = "FALLING" if side == "LONG" else "RISING"
        # UNKNOWN is recorded, never described as positive confirmation.
        checks.append({"name": "No opposing swing structure", "passed": row.get("structure") != opposing,
                       "detail": f"Confirmed swing structure: {row.get('structure', 'UNKNOWN')}"})
    rejection = bool(row.get("failed_breakout" if side == "LONG" else "failed_breakdown", False))
    checks.append({"name": "No failed break", "passed": not rejection,
                   "detail": "Candle must not reject the 20-bar break in the intended direction"})
    if variant == "retest":
        checks.append({"name": "Breakout retest", "passed": bool(row.get("breakout_retest" if side == "LONG" else "breakdown_retest", False)),
                       "detail": "Require a recent breakout level to be retested and held"})
    failed = [c["detail"] for c in checks if not c["passed"]]
    return {"allowed": not failed, "checks": checks, "reason": "; ".join(failed) if failed else "Chart hypothesis passed"}


def fundamental_assessment(ticker, side, *, fundamentals=None, crypto=None, now=None):
    now = now or datetime.now(timezone.utc)
    result = {"status": "UNAVAILABLE", "veto": False, "observations": [], "metrics": {},
              "directional_edge": "NOT_ESTABLISHED", "historical_validation": "NOT_TESTED"}
    if ticker.endswith(("-USD", "-USDT")):
        result.update(asset_type="CRYPTO", limitations="Market/liquidity evidence only. No on-chain, exchange-flow or futures-carry feed; not intrinsic valuation.")
        context = crypto or {}
        result.update(source=context.get("source"), observed_at=context.get("observed_at"))
        try:
            stamp = datetime.fromisoformat(context["observed_at"].replace("Z", "+00:00"))
            fresh = stamp.tzinfo is not None and 0 <= (now - stamp).total_seconds() <= 900
        except (KeyError, TypeError, ValueError):
            fresh = False
        if context.get("status") != "AVAILABLE" or not fresh:
            result["observations"] = ["Timestamped crypto context unavailable; no fundamental confirmation"]
            return result
        metrics = {key: number(context.get("metrics", {}).get(key)) for key in ("price", "market_cap", "volume_24h")}
        if any(value is None or value <= 0 for value in metrics.values()):
            return result
        result.update(status="CONTEXT_ONLY", metrics=metrics, observations=["Independent market price, capitalization and daily reported liquidity available; not a buy/sell thesis"])
        return result
    result.update(asset_type="EQUITY_OR_FUND", limitations="Current provider snapshot, not filing-date historical data. Financial-company and fund metrics need separate models.")
    metrics = {key: number(getattr(fundamentals, key, None)) for key in ("profit_margin", "free_cash_flow", "revenue_growth", "debt_to_equity")}
    result["metrics"] = metrics
    fetched = getattr(fundamentals, "fetched_at", None)
    if not isinstance(fetched, datetime):
        return result
    fetched = fetched.replace(tzinfo=timezone.utc) if fetched.tzinfo is None else fetched
    result["observed_at"] = fetched.isoformat()
    if not 0 <= (now - fetched).total_seconds() <= 7200:
        result["status"] = "STALE"
        return result
    if any(value is None for value in metrics.values()):
        result["status"] = "INSUFFICIENT"
        result["observations"] = ["Incomplete company statements; do not apply company-quality rules to funds"]
        return result
    result["status"] = "AVAILABLE"
    distressed = metrics["profit_margin"] < 0 and metrics["free_cash_flow"] < 0
    fragile = metrics["revenue_growth"] < 0 and metrics["debt_to_equity"] > 200
    result["veto"] = side == "LONG" and (distressed or fragile)
    result["observations"] = ["Negative profitability and cash flow" if distressed else "No combined profitability/cash-flow red flag",
                              "Shrinking revenue and high debt/equity" if fragile else "No combined growth/leverage red flag"]
    return result


def decision_evidence(df, *, ticker, side, fundamentals=None, crypto=None, now=None):
    row = chart_features(df).iloc[-1]
    chart = {key: (str(value) if isinstance(value, str) else bool(value) if isinstance(value, (bool, np.bool_)) else number(value)) for key, value in row.items()}
    fundamental = fundamental_assessment(ticker, side, fundamentals=fundamentals, crypto=crypto, now=now)
    assessment = chart_admission(row, side)
    return {"version": VERSION, "mode": "SHADOW", "chart": chart,
            "fundamentals": fundamental, "candidate_allowed": assessment["allowed"] and not fundamental["veto"],
            "chart_assessment": assessment,
            "limitation": "Candidate decision only; no proven excess return. Missing fundamentals provide no confirmation."}
