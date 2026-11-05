"""Paper runner for the multi-asset daily / 4H trend scanner with Core/Satellite regime allocation.

Each `step()` fetches bars for the universe and:
1. maintains Core regime allocation (when enabled) to prevent runaway underperformance;
2. fills orders decided on earlier completed bars;
3. manages trailing chandelier ATR stops;
4. scans and ranks momentum breakouts across the multi-asset universe.
Paper only: virtual portfolio, no live money brokerage.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio.virtual_account import LivePortfolio
from strategy.trend_scanner import ScannerConfig, plan_day, scanner_features

UNIVERSE = [
    # Broad Index & Sector Leaders
    "SPY", "QQQ", "IWM", "SMH", "TLT", "GLD", "EEM",
    # Megacap Tech & Semiconductor Momentum Leaders
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    # Quality Growth & Defensive Compounders
    "LLY", "COST", "NFLX", "JPM", "XOM", "JNJ", "WMT", "KO", "DIS", "BA",
    # Top Liquid Crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"
]

# Default configuration declared before results
CONFIG = ScannerConfig(entry_rule="ma_10_50", stop_atr=5., max_positions=10)
SLIPPAGE_BPS = 5.
STOCK_CLOSE_HOUR_NY = 16


def fetch_bars(ticker, interval="1d"):
    """Fetch market bars for a given ticker and interval."""
    import yfinance as yf
    if interval == "4h":
        df = yf.Ticker(ticker).history(period="60d", interval="1h", auto_adjust=True, actions=False, timeout=30)
        if len(df) == 0:
            return pd.DataFrame()
        df_4h = df.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last"
        }).dropna()
        return df_4h[["Open", "High", "Low", "Close"]]
    df = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True, actions=False, timeout=30)
    return df[["Open", "High", "Low", "Close"]].dropna()


def fetch_daily(ticker):
    return fetch_bars(ticker, interval="1d")


def session_frame(df, ticker, now):
    """Naive session-date index plus whether the last bar's session has closed."""
    if df is None or len(df) == 0:
        return pd.DataFrame(), False
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
    def __init__(self, state_dir, *, universe=UNIVERSE, config=CONFIG, fetch=None, interval="1d",
                 core_satellite=False, capital=10000.):
        self.state_dir = Path(state_dir)
        self.universe = list(universe)
        self.config = config
        self.interval = interval
        self.core_satellite = core_satellite
        if fetch is None:
            self.fetch = lambda t: fetch_bars(t, interval=self.interval)
        else:
            self.fetch = fetch
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

        # Core/Satellite Regime rebalance check (if enabled)
        if self.core_satellite and "SPY" in frames and len(frames["SPY"]) > 0:
            try:
                from strategy.regime_core_satellite import calculate_core_rebalance_action
                core_action = calculate_core_rebalance_action(self.portfolio, frames["SPY"])
                if core_action["action"] == "BUY" and core_action["rebalance_cash"] > 10:
                    core_fill = float(frames["SPY"].iloc[-1]["Close"]) * (1 + SLIPPAGE_BPS / 10000)
                    self.portfolio.buy("SPY", core_fill, cash_amount=core_action["rebalance_cash"],
                                       reasoning=f"Core/Satellite Regime: {core_action['reason']}")
                    self._event(now, "CORE_REBALANCE", "SPY", f"BUY ${core_action['rebalance_cash']:,.2f} @ {core_fill:.2f}")
                elif core_action["action"] == "SELL" and "SPY" in self.portfolio.positions:
                    core_fill = float(frames["SPY"].iloc[-1]["Close"]) * (1 - SLIPPAGE_BPS / 10000)
                    self.portfolio.sell("SPY", core_fill, reasoning=f"Core/Satellite Regime: {core_action['reason']}")
                    self._event(now, "CORE_REBALANCE", "SPY", f"SELL @ {core_fill:.2f}")
            except Exception as exc:
                errors["core_satellite"] = str(exc)

        for ticker, df in frames.items():
            if len(df) == 0:
                continue
            bar, stamp = df.iloc[-1], df.index[-1]
            self.portfolio.update_price(ticker, float(bar.Close))
            decided_on = pd.Timestamp(self.state["decided"].get(ticker, "1900-01-01"))
            session_started = stamp > decided_on
            if session_started and ticker in self.state["pending_exits"]:
                self.state["pending_exits"].pop(ticker)
                if ticker in self.portfolio.positions:
                    self._sell(ticker, float(bar.Open), now, "Trend break at completed close")
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
            if len(df) == 0:
                continue
            complete = df if closed.get(ticker, False) else df.iloc[:-1]
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

        self.state["last_step"] = {
            "at": now.isoformat(),
            "errors": errors,
            "decided": sorted(today),
            "interval": self.interval,
            "core_satellite": self.core_satellite,
            "universe_count": len(self.universe),
        }
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio.save(self.portfolio_path)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        return self.state["last_step"]
