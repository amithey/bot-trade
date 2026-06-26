"""Performance metrics: Sharpe / Sortino / Calmar, max drawdown, win rate,
profit factor, expectancy, plus equity / drawdown / benchmark curves.

All functions accept a ``trade_log`` (list of ``TradeRecord``) and a list
of ``DailySnapshot`` objects, both lifted directly from
``LivePortfolio``. The math is intentionally vanilla — annualizes on a
**252-trading-day** basis from daily portfolio-value snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

_TRADING_DAYS = 252
_RF_DAILY = 0.0   # risk-free rate; keep at 0 for simplicity


# ─────────────────────────────────────────────────────────────────────────────
# Core dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PerformanceMetrics:
    n_trades: int
    n_round_trips: int
    n_wins: int
    n_losses: int
    win_rate: float                 # 0-1
    avg_win: float                  # USD
    avg_loss: float                 # USD (negative)
    profit_factor: float            # gross_win / |gross_loss|
    expectancy: float               # avg P&L per round trip (USD)
    total_return_pct: float         # since inception (%)
    cagr_pct: float                 # annualised (%)
    sharpe: float                   # daily-returns based, annualised
    sortino: float                  # downside-deviation, annualised
    calmar: float                   # CAGR / |max_dd|
    max_drawdown_pct: float         # most negative peak-to-trough (%)
    max_drawdown_dollars: float     # peak-to-trough dollars
    longest_drawdown_days: int      # longest underwater stretch
    volatility_pct: float           # annualised stdev of daily returns (%)
    best_day_pct: float
    worst_day_pct: float
    avg_holding_period_days: float

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_records_df(trade_log: Iterable) -> pd.DataFrame:
    """Coerce TradeRecord (or dicts) to a DataFrame with strict types."""
    rows: list[dict] = []
    for r in trade_log:
        if hasattr(r, "to_dict"):
            d = r.to_dict()
        elif isinstance(r, dict):
            d = dict(r)
        else:
            continue
        try:
            d["executed_at"] = (
                datetime.fromisoformat(d["executed_at"])
                if isinstance(d["executed_at"], str)
                else d["executed_at"]
            )
        except Exception:
            continue
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=[
            "executed_at", "action", "ticker", "quantity", "price",
            "realized_pnl", "portfolio_value",
        ])
    return pd.DataFrame(rows).sort_values("executed_at").reset_index(drop=True)


def _to_snapshots_df(snaps: Iterable) -> pd.DataFrame:
    """Coerce DailySnapshot (or dicts) to a DataFrame indexed by date."""
    rows: list[dict] = []
    for s in snaps:
        if hasattr(s, "to_dict"):
            d = s.to_dict()
        elif isinstance(s, dict):
            d = dict(s)
        else:
            continue
        try:
            sd = d["snapshot_date"]
            if isinstance(sd, str):
                sd = date.fromisoformat(sd)
            d["snapshot_date"] = sd
            d["portfolio_value"] = float(d["portfolio_value"])
        except Exception:
            continue
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["snapshot_date", "portfolio_value"])
    df = pd.DataFrame(rows).sort_values("snapshot_date").drop_duplicates(
        subset=["snapshot_date"], keep="last"
    ).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Curves
# ─────────────────────────────────────────────────────────────────────────────


def equity_curve(
    trade_log: Iterable,
    daily_snapshots: Iterable,
    initial_capital: float,
    current_value: Optional[float] = None,
) -> pd.DataFrame:
    """Return a daily equity curve as ``DataFrame(index=date, columns=['equity'])``.

    Strategy:
      1. Use daily_snapshots when available — they are sampled at start-of-day.
      2. If trades exist outside the snapshot range, fall back to the
         ``portfolio_value`` field on each TradeRecord, sampled to the trade's
         calendar date (last value of the day wins).
      3. Forward-fill missing calendar days within the observed range.
    """
    snaps = _to_snapshots_df(daily_snapshots)
    trades = _to_records_df(trade_log)

    series: dict[date, float] = {}
    for _, row in snaps.iterrows():
        series[row["snapshot_date"]] = float(row["portfolio_value"])
    for _, row in trades.iterrows():
        ts = row["executed_at"]
        if not isinstance(ts, datetime):
            continue
        d = ts.date()
        # Last trade of the day wins (most accurate end-of-day value)
        series[d] = float(row.get("portfolio_value") or series.get(d, initial_capital))

    if current_value is not None:
        series[date.today()] = float(current_value)

    if not series:
        idx = pd.DatetimeIndex([pd.Timestamp.today().normalize()])
        return pd.DataFrame({"equity": [initial_capital]}, index=idx)

    s = pd.Series(series).sort_index()
    s.index = pd.DatetimeIndex(s.index)

    # Reindex to a continuous business-day range and forward-fill.
    # If the trades happen on a weekend, bdate_range can be empty — fall
    # back to the raw observed index in that case.
    full_idx = pd.bdate_range(s.index.min(), s.index.max())
    if len(full_idx) == 0:
        full_idx = s.index
    # Union ensures any weekend trade dates are kept too.
    full_idx = full_idx.union(s.index)
    s = s.reindex(full_idx).ffill()
    if len(s) and pd.isna(s.iloc[0]):
        s.iloc[0] = initial_capital
        s = s.ffill()
    return pd.DataFrame({"equity": s})


def drawdown_series(equity: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns ``equity, peak, drawdown_pct``."""
    if isinstance(equity, pd.DataFrame):
        eq = equity["equity"].astype(float)
    else:
        eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    return pd.DataFrame({"equity": eq, "peak": peak, "drawdown_pct": dd})


def benchmark_equity(
    benchmark_ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_value: float,
) -> Optional[pd.DataFrame]:
    """Fetch a benchmark price series and rescale so it starts at *initial_value*.

    Returns ``DataFrame(index=date, columns=['equity'])`` or None on failure.
    """
    try:
        import yfinance as yf
        from pathlib import Path
        cache = Path("data/yfinance_cache")
        cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache))

        df = yf.Ticker(benchmark_ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
    except Exception as exc:
        logger.debug(f"benchmark fetch failed for {benchmark_ticker}: {exc}")
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    # Make tz-naive for clean joins with the equity curve
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    closes = df["Close"].astype(float).dropna()
    if closes.empty:
        return None
    scaled = closes / float(closes.iloc[0]) * float(initial_value)
    return pd.DataFrame({"equity": scaled})


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def _longest_underwater_run(dd_pct: pd.Series) -> int:
    """Length (in business days) of the longest stretch where dd_pct < 0."""
    mask = (dd_pct < -1e-9).astype(int).to_numpy()
    if mask.size == 0:
        return 0
    # Run-length encode the 1s
    longest = cur = 0
    for v in mask:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return int(longest)


def _annualised_return(eq: pd.Series) -> float:
    """Geometric annualised return from a daily equity series."""
    if len(eq) < 2:
        return 0.0
    n_days = (eq.index[-1] - eq.index[0]).days
    if n_days <= 0:
        return 0.0
    total_ret = float(eq.iloc[-1] / eq.iloc[0])
    if total_ret <= 0:
        return -100.0
    years = n_days / 365.25
    return (total_ret ** (1.0 / max(years, 1e-9)) - 1.0) * 100.0


def compute_metrics(
    trade_log: Iterable,
    daily_snapshots: Iterable,
    initial_capital: float,
    current_value: Optional[float] = None,
) -> PerformanceMetrics:
    """Compute the full :class:`PerformanceMetrics` block."""
    trades = _to_records_df(trade_log)
    eq = equity_curve(trade_log, daily_snapshots, initial_capital, current_value)
    eq_s = eq["equity"].astype(float)

    # ── Trade-based stats ───────────────────────────────────────────────
    sells = trades[trades["action"].isin(["SELL", "FORCE_CLOSE"])]
    pnls = sells["realized_pnl"].astype(float)
    n_round_trips = int(len(pnls))
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / n_round_trips) if n_round_trips else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 1e-9 else float("inf") \
        if gross_win > 0 else 0.0
    expectancy = float(pnls.mean()) if n_round_trips else 0.0

    # ── Holding period (avg seconds between paired BUY → SELL) ──────────
    holding_secs: list[float] = []
    open_buys: dict[str, datetime] = {}
    for _, r in trades.iterrows():
        if r["action"] == "BUY":
            open_buys[r["ticker"]] = r["executed_at"]
        elif r["action"] in ("SELL", "FORCE_CLOSE") and r["ticker"] in open_buys:
            entry = open_buys.pop(r["ticker"])
            try:
                holding_secs.append((r["executed_at"] - entry).total_seconds())
            except Exception:
                pass
    avg_hold_days = (sum(holding_secs) / len(holding_secs) / 86400.0) \
        if holding_secs else 0.0

    # ── Equity-based stats ──────────────────────────────────────────────
    rets = eq_s.pct_change().dropna()
    if len(rets) >= 2:
        mu = float(rets.mean())
        sigma = float(rets.std(ddof=1))
        sharpe = ((mu - _RF_DAILY) / sigma * np.sqrt(_TRADING_DAYS)) if sigma > 0 else 0.0
        downside = rets[rets < 0]
        d_sigma = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = ((mu - _RF_DAILY) / d_sigma * np.sqrt(_TRADING_DAYS)) if d_sigma > 0 else 0.0
        vol_pct = sigma * np.sqrt(_TRADING_DAYS) * 100.0
        best_day_pct = float(rets.max() * 100.0)
        worst_day_pct = float(rets.min() * 100.0)
    else:
        sharpe = sortino = vol_pct = best_day_pct = worst_day_pct = 0.0

    dd = drawdown_series(eq_s)
    max_dd_pct = float(dd["drawdown_pct"].min()) if len(dd) else 0.0
    if len(dd):
        peak_at_min = float(dd.loc[dd["drawdown_pct"].idxmin(), "peak"])
        eq_at_min = float(dd.loc[dd["drawdown_pct"].idxmin(), "equity"])
        max_dd_dollars = float(eq_at_min - peak_at_min)
    else:
        max_dd_dollars = 0.0
    longest_dd = _longest_underwater_run(dd["drawdown_pct"]) if len(dd) else 0

    cagr_pct = _annualised_return(eq_s)
    calmar = (cagr_pct / abs(max_dd_pct)) if abs(max_dd_pct) > 1e-9 else 0.0

    final_val = float(eq_s.iloc[-1]) if len(eq_s) else float(initial_capital)
    total_ret_pct = (final_val - initial_capital) / initial_capital * 100.0

    return PerformanceMetrics(
        n_trades=int(len(trades)),
        n_round_trips=n_round_trips,
        n_wins=int(len(wins)),
        n_losses=int(len(losses)),
        win_rate=round(win_rate, 4),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
        expectancy=round(expectancy, 2),
        total_return_pct=round(total_ret_pct, 3),
        cagr_pct=round(cagr_pct, 3),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        calmar=round(calmar, 3),
        max_drawdown_pct=round(max_dd_pct, 3),
        max_drawdown_dollars=round(max_dd_dollars, 2),
        longest_drawdown_days=int(longest_dd),
        volatility_pct=round(vol_pct, 3),
        best_day_pct=round(best_day_pct, 3),
        worst_day_pct=round(worst_day_pct, 3),
        avg_holding_period_days=round(avg_hold_days, 3),
    )
