"""Paper runner for the multi-asset daily trend scanner.

Each `step()` fetches daily bars for the whole universe and:
1. fills orders decided on an earlier completed day once that asset's next
   session has started (the simulator fills these at the next open);
2. treats each chandelier stop as a resting stop order: if the session low
   touches it, the exit fills at the worse of the stop and the session open;
3. decides with `plan_day` on every newly completed daily bar.
Decisions never use a still-forming bar. State lives in two JSON files so a
restart resumes exactly. Paper only: no broker connection exists here.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio.virtual_account import LivePortfolio
from strategy.trend_scanner import ScannerConfig, plan_day, scanner_features

UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EEM", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA",
            "JPM", "XOM", "JNJ", "WMT", "KO", "INTC", "PFE", "BA", "DIS", "VZ", "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"]
# Chosen by training-window Sharpe in tools/evaluate_trend_scanner.py (2016-2021).
CONFIG = ScannerConfig(entry_rule="ma_10_50", stop_atr=5., max_positions=10)
SLIPPAGE_BPS = 5.
STOCK_CLOSE_HOUR_NY = 16


def fetch_daily(ticker):
    import yfinance as yf
    df = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True, actions=False, timeout=30)
    return df[["Open", "High", "Low", "Close"]].dropna()


def session_frame(df, ticker, now):
    """Naive session-date index plus whether the last bar's session has closed."""
    crypto = ticker.endswith("-USD")
    zone = "UTC" if crypto else "America/New_York"
    index = pd.DatetimeIndex(df.index)
    index = index.tz_localize(zone) if index.tz is None else index.tz_convert(zone)
    frame = df.copy()
    frame.index = index.tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")]
    local_now = pd.Timestamp(now).tz_convert(zone)
    today = local_now.tz_localize(None).normalize()
    if crypto:
        last_closed = frame.index[-1] < today
    else:
        last_closed = frame.index[-1] < today or local_now.hour >= STOCK_CLOSE_HOUR_NY
    return frame, bool(len(frame)) and last_closed


class TrendScannerRunner:
    def __init__(self, state_dir, *, universe=UNIVERSE, config=CONFIG, fetch=fetch_daily, capital=10000.):
        self.state_dir = Path(state_dir)
        self.universe, self.config, self.fetch = list(universe), config, fetch
        self.portfolio_path = self.state_dir / "portfolio.json"
        self.state_path = self.state_dir / "scanner_state.json"
        self.portfolio = (LivePortfolio.load(self.portfolio_path) if self.portfolio_path.exists()
                          else LivePortfolio(initial_capital=capital, name="Trend scanner (paper)"))
        self.state = (json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists()
                      else {"positions": {}, "pending_exits": {}, "pending_entries": {}, "decided": {}, "events": []})

    def _event(self, now, kind, ticker, detail):
        self.state["events"] = (self.state["events"] + [{"at": now.isoformat(), "kind": kind, "ticker": ticker, "detail": detail}])[-500:]

    def _sell(self, ticker, price, now, reason):
        fill = price * (1 - SLIPPAGE_BPS / 10000)
        trade = self.portfolio.sell(ticker, fill, reasoning=reason)
        self.state["positions"].pop(ticker, None)
        self._event(now, "SELL", ticker, f"{reason} @ {fill:.4f}; P&L {trade.realized_pnl:+.2f}")

    def step(self, now=None):
        now = now or datetime.now(timezone.utc)
        frames, closed, errors = {}, {}, {}
        for ticker in self.universe:
            try:
                frames[ticker], closed[ticker] = session_frame(self.fetch(ticker), ticker, now)
            except Exception as exc:                      # one bad feed must not stop the rest
                errors[ticker] = str(exc)
        # Positions the portfolio no longer holds (manual close) lose their scanner state.
        for ticker in list(self.state["positions"]):
            if ticker not in self.portfolio.positions:
                self.state["positions"].pop(ticker)
        for ticker, df in frames.items():
            bar, stamp = df.iloc[-1], df.index[-1]
            self.portfolio.update_price(ticker, float(bar.Close))
            decided_on = pd.Timestamp(self.state["decided"].get(ticker, "1900-01-01"))
            session_started = stamp > decided_on
            if session_started and ticker in self.state["pending_exits"]:
                self.state["pending_exits"].pop(ticker)
                if ticker in self.portfolio.positions:
                    self._sell(ticker, float(bar.Open), now, "Trend break at completed daily close")
            order = self.state["pending_entries"].get(ticker) if session_started else None
            if order is not None:
                self.state["pending_entries"].pop(ticker)
                fill = float(bar.Open) * (1 + SLIPPAGE_BPS / 10000)
                equity = self.portfolio.get_total_value()
                spend = min(equity * order["weight_pct"] / 100, self.portfolio.cash / (1 + self.portfolio.fee_rate))
                if (ticker not in self.portfolio.positions and len(self.portfolio.positions) < self.config.max_positions
                        and spend > 1 and order["stop"] < fill):
                    self.portfolio.buy(ticker, fill, cash_amount=spend,
                                       reasoning=f"Trend scanner entry; strength {order['strength']:.2f}; stop {order['stop']:.4f}")
                    self.state["positions"][ticker] = {"peak_close": fill, "stop": order["stop"]}
                    self._event(now, "BUY", ticker, f"{spend:.2f} @ {fill:.4f}; stop {order['stop']:.4f}")
            held = self.state["positions"].get(ticker)
            if held is not None and stamp > pd.Timestamp(held.get("stop_set_on", "1900-01-01")) and float(bar.Low) <= held["stop"]:
                self._sell(ticker, min(float(bar.Open), held["stop"]), now, "Chandelier stop")
        # Decide only on bars that are complete and not decided before.
        today = {}
        for ticker, df in frames.items():
            complete = df if closed[ticker] else df.iloc[:-1]
            if len(complete) == 0:
                continue
            stamp = complete.index[-1]
            if stamp > pd.Timestamp(self.state["decided"].get(ticker, "1900-01-01")):
                row = scanner_features(complete, self.config.entry_rule).iloc[-1]
                today[ticker] = {k: (bool(v) if k in ("enter", "valid") else float(v)) for k, v in row.items()}
                self.state["decided"][ticker] = str(stamp.date())
        if today:
            exits, entries, _ = plan_day(today, self.state["positions"], self.portfolio.get_total_value(), self.config)
            for ticker in exits:
                self.state["pending_exits"][ticker] = self.state["decided"][ticker]
                self._event(now, "EXIT_PLANNED", ticker, "Trend rule no longer valid at the close")
            for entry in entries:
                self.state["pending_entries"][entry["ticker"]] = entry
                self._event(now, "ENTRY_PLANNED", entry["ticker"],
                            f"weight {entry['weight_pct']:.1f}%, stop {entry['stop']:.4f}, strength {entry['strength']:.2f}")
            for ticker in today:
                if ticker in self.state["positions"]:
                    self.state["positions"][ticker]["stop_set_on"] = self.state["decided"][ticker]
        self.state["last_step"] = {"at": now.isoformat(), "errors": errors, "decided": sorted(today)}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio.save(self.portfolio_path)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        return self.state["last_step"]
