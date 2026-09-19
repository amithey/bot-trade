"""Multi-asset daily trend scanner: one long-only decision per completed daily bar.

The same `plan_day` drives the portfolio simulation and the live runner, so what
is tested is what trades. Signals use completed days and fill at the next open.
Exits: a chandelier stop (highest close since entry minus k x ATR) checked against
the next day's low, or the entry rule's own trend-break condition at the close.
No fixed profit target and no maximum holding time: winners run until the trend
breaks. The configuration grid is declared here, before results. Hypotheses.
"""
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

ENTRY_RULES = ("donchian_55", "ma_10_50", "tsmom_120")
MIN_HISTORY = 200


@dataclass(frozen=True)
class ScannerConfig:
    entry_rule: str = "donchian_55"
    stop_atr: float = 3.
    max_positions: int = 8
    risk_pct: float = 1.           # equity lost if a fresh position hits its initial stop
    max_weight_pct: float = 25.

    def __post_init__(self):
        if self.entry_rule not in ENTRY_RULES:
            raise ValueError(f"Unknown entry rule {self.entry_rule}")
        if not (self.stop_atr > 0 and self.max_positions >= 1 and 0 < self.risk_pct <= 5
                and 0 < self.max_weight_pct <= 100):
            raise ValueError("Invalid scanner configuration")


GRID = [ScannerConfig(entry_rule=rule, stop_atr=stop, max_positions=slots)
        for rule in ENTRY_RULES for stop in (3., 5.) for slots in (5, 10)]


def _atr(df, n=20):
    prev = df.Close.shift()
    true_range = pd.concat([df.High - df.Low, (df.High - prev).abs(), (df.Low - prev).abs()], axis=1).max(axis=1)
    return true_range.rolling(n).mean()


def scanner_features(df, rule):
    """Per-day evidence from that day's completed bar and earlier ones only."""
    close = df.Close
    new_high_20 = close > df.High.rolling(20).max().shift()
    if rule == "donchian_55":
        enter = close > df.High.rolling(55).max().shift()
        valid = close >= df.Low.rolling(20).min().shift()
    elif rule == "ma_10_50":
        fast, slow = close.rolling(10).mean(), close.rolling(50).mean()
        valid = (fast > slow) & (close > slow)
        enter = valid & new_high_20
    elif rule == "tsmom_120":
        valid = close > close.shift(120)
        enter = valid & new_high_20
    else:
        raise ValueError(f"Unknown entry rule {rule}")
    volatility = close.pct_change().rolling(60).std() * np.sqrt(252)
    strength = (close / close.shift(126) - 1) / volatility
    history = pd.Series(np.arange(1, len(df) + 1), index=df.index) >= MIN_HISTORY
    return pd.DataFrame({"close": close, "atr": _atr(df), "strength": strength,
                         "enter": enter & history, "valid": valid & history}, index=df.index)


def _finite(*values):
    return all(isinstance(v, (int, float, np.floating)) and isfinite(v) for v in values)


def plan_day(today, positions, equity, config):
    """Decide from completed-day evidence.

    today: {ticker: features row as dict} for assets whose day just completed.
    positions: {ticker: {"peak_close", "stop"}} currently held (all assets).
    Returns (exits, entries, stops): tickers to sell at the next open, entries as
    {ticker, weight_pct, stop, strength}, and each held ticker's stop for its next bar.
    """
    exits, stops = [], {}
    for ticker, position in positions.items():
        row = today.get(ticker)
        if row is None:          # no completed bar today (e.g. a stock on Saturday)
            stops[ticker] = position["stop"]
            continue
        peak = max(position["peak_close"], row["close"])
        stop = position["stop"]
        if _finite(row["atr"]):
            stop = max(stop, peak - config.stop_atr * row["atr"])
        position.update(peak_close=peak, stop=stop)
        stops[ticker] = stop
        if not row["valid"]:
            exits.append(ticker)
    free = config.max_positions - (len(positions) - len(exits))
    candidates = sorted(
        (t for t, r in today.items() if t not in positions and bool(r["enter"]) and bool(r["valid"])
         and _finite(r["atr"], r["strength"], r["close"]) and r["atr"] > 0 and r["strength"] > 0),
        key=lambda t: today[t]["strength"], reverse=True)
    entries = []
    for ticker in candidates[:max(0, free)]:
        row = today[ticker]
        stop_distance = config.stop_atr * row["atr"] / row["close"]
        weight = min(config.max_weight_pct, config.risk_pct / stop_distance)
        entries.append({"ticker": ticker, "weight_pct": weight, "strength": row["strength"],
                        "stop": row["close"] - config.stop_atr * row["atr"]})
    return exits, entries, stops


def simulate_portfolio(frames, config, start, end=None, *, fee_rate=.001, slippage_bps=5., capital=10000.):
    """Daily multi-asset replay. frames: {ticker: OHLC DataFrame on a naive daily index}."""
    frames = {t: df for t, df in frames.items() if len(df)}
    features = {t: scanner_features(df, config.entry_rule) for t, df in frames.items()}
    bars = {t: df[["Open", "High", "Low", "Close"]].to_dict("index") for t, df in frames.items()}
    evidence = {t: f.to_dict("index") for t, f in features.items()}
    start, end = pd.Timestamp(start), pd.Timestamp(end) if end is not None else None
    dates = sorted({d for df in frames.values() for d in df.index if d >= start and (end is None or d <= end)})
    slip = slippage_bps / 10000
    cash, positions, pending_exit, pending_entry = capital, {}, set(), {}
    last_close, curve, trades = {}, [], []

    def equity():
        return cash + sum(p["quantity"] * last_close[t] for t, p in positions.items())

    def close_position(ticker, price, day, reason):
        nonlocal cash
        position = positions.pop(ticker)
        fill = price * (1 - slip)
        proceeds = position["quantity"] * fill
        fee = proceeds * fee_rate
        cash += proceeds - fee
        trades.append({"ticker": ticker, "entry_date": str(position["opened"].date()), "exit_date": str(day.date()),
                       "entry": position["entry"], "exit": fill, "reason": reason,
                       "pnl": proceeds - fee - position["cost"],
                       "return_pct": (proceeds - fee) / position["cost"] * 100 - 100})

    for day in dates:
        today = {t: bars[t][day] for t in frames if day in bars[t]}
        for ticker, bar in today.items():
            if ticker in pending_exit and ticker in positions:
                close_position(ticker, bar["Open"], day, "trend break")
            pending_exit.discard(ticker)
            order = pending_entry.pop(ticker, None)   # filled now or superseded by today's decision
            if order is not None and ticker not in positions and len(positions) < config.max_positions:
                fill = bar["Open"] * (1 + slip)
                spend = min(equity() * order["weight_pct"] / 100, cash / (1 + fee_rate))
                if spend > 1 and order["stop"] < fill:
                    cash -= spend * (1 + fee_rate)
                    positions[ticker] = {"quantity": spend / fill, "entry": fill, "cost": spend * (1 + fee_rate),
                                         "opened": day, "peak_close": fill, "stop": order["stop"]}
            if ticker in positions and bar["Low"] <= positions[ticker]["stop"]:
                close_position(ticker, min(bar["Open"], positions[ticker]["stop"]), day, "chandelier stop")
            last_close[ticker] = bar["Close"]
        # Decide at today's close for tomorrow's open.
        completed = {t: evidence[t][day] for t in today}
        exits, entries, _ = plan_day(completed, positions, equity(), config)
        pending_exit.update(exits)
        pending_entry.update({e["ticker"]: e for e in entries})
        curve.append((day, equity(), sum(p["quantity"] * last_close[t] for t, p in positions.items())))
    frame = pd.DataFrame(curve, columns=["date", "equity", "invested"]).set_index("date")
    return frame, trades
