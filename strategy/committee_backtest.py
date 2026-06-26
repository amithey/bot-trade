"""
Committee Backtester — fast, offline, zero API calls.
=====================================================

Replays the 38-indicator committee over historical bars for any ticker
(crypto OR stocks) and produces an equity curve vs buy-and-hold,
exactly the comparison shown in the "green line vs gray line" style:

    green — the committee bot (long / cash)
    gray  — buy & hold, no touch

Execution model (point-in-time correct):
    * votes are computed on bar *t* close
    * fills happen at bar *t+1* OPEN, with taker fees applied both ways
    * long-only: SELL means exit to cash, never short

Because every agent is vectorized, a 5-year daily backtest runs in
well under a second — no Claude calls, no rate limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from strategy.committee import CommitteeConfig, IndicatorCommittee


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CommitteeTrade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    pnl_pct: Optional[float]          # net of fees, None while open


@dataclass
class CommitteeBacktestResult:
    ticker: str
    interval: str
    bars: int
    start: pd.Timestamp
    end: pd.Timestamp

    equity: pd.Series                 # strategy equity curve (normalized 1.0)
    buy_hold: pd.Series               # buy & hold curve (normalized 1.0)
    score: pd.Series                  # committee net score per bar
    position: pd.Series               # 1 = long, 0 = cash

    total_return_pct: float
    buy_hold_return_pct: float
    alpha_pct: float
    max_drawdown_pct: float
    bh_max_drawdown_pct: float
    drawdown_edge_pp: float           # how much shallower our worst DD is
    downside_captured_pct: float      # % of B&H's losing-bar losses we ate
    upside_captured_pct: float        # % of B&H's winning-bar gains we kept
    time_in_market_pct: float
    win_rate_pct: float
    total_trades: int
    fees_paid_pct: float
    trades: list[CommitteeTrade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------------

def backtest_committee(
    df: pd.DataFrame,
    ticker: str = "",
    interval: str = "1d",
    config: Optional[CommitteeConfig] = None,
    fee_pct: float = 0.1,             # taker fee per side, in percent
    committee: Optional[IndicatorCommittee] = None,
    votes: Optional[pd.DataFrame] = None,   # precomputed vote matrix
) -> CommitteeBacktestResult:
    """
    Run the committee over an OHLCV DataFrame and compute performance.

    Args:
        df:        OHLCV DataFrame (DatetimeIndex). Raw bars are fine —
                   the committee computes its own indicators.
        ticker:    Display label only.
        interval:  Display label only ("1d", "1h", …).
        config:    Committee thresholds (defaults to CommitteeConfig()).
        fee_pct:   Taker fee per side in percent (0.1 = 0.1 %).
        committee: Reuse an existing IndicatorCommittee (optional).
    """
    if df is None or len(df) < 120:
        raise ValueError(f"Need at least 120 bars to backtest, got "
                         f"{0 if df is None else len(df)}.")

    com = committee or IndicatorCommittee(config)
    cfg = com.config
    fee = fee_pct / 100.0

    if votes is None:
        votes = com.vote_matrix(df)
    bulls = (votes == 1).sum(axis=1)
    bears = (votes == -1).sum(axis=1)
    total = len(com.agents)
    score = (bulls - bears) / total
    quorum = (bulls + bears) >= cfg.min_quorum

    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    score_np = score.to_numpy()
    quorum_np = quorum.to_numpy()
    n = len(df)

    # Skip the indicator warm-up: first bar where a decent share of the
    # committee is actually voting (SMA200 etc. are the binding window).
    active = (bulls + bears).to_numpy()
    warm_candidates = np.nonzero(active >= max(cfg.min_quorum, total // 3))[0]
    warm = int(warm_candidates[0]) if len(warm_candidates) else n

    position = np.zeros(n, dtype=int)
    equity = np.ones(n, dtype=float)
    cash, units = 1.0, 0.0
    in_pos = False
    fees_paid = 0.0
    trades: list[CommitteeTrade] = []

    for i in range(n):
        # Execute the signal decided on the PREVIOUS bar at this bar's open
        if i > warm:
            prev_score, prev_q = score_np[i - 1], quorum_np[i - 1]
            if not in_pos and prev_q and prev_score >= cfg.enter_score:
                fill = open_[i] * (1 + fee)
                units = cash / fill
                fees_paid += cash * fee
                cash, in_pos = 0.0, True
                trades.append(CommitteeTrade(
                    entry_time=df.index[i], exit_time=None,
                    entry_price=fill, exit_price=None, pnl_pct=None))
            elif in_pos and prev_q and prev_score <= cfg.exit_score:
                fill = open_[i] * (1 - fee)
                cash = units * fill
                fees_paid += units * open_[i] * fee
                units, in_pos = 0.0, False
                t = trades[-1]
                t.exit_time = df.index[i]
                t.exit_price = fill
                t.pnl_pct = (fill / t.entry_price - 1) * 100

        position[i] = 1 if in_pos else 0
        equity[i] = cash + units * close[i]

    # Mark any open trade to market at the last close
    if trades and trades[-1].exit_price is None:
        t = trades[-1]
        t.pnl_pct = (close[-1] * (1 - fee) / t.entry_price - 1) * 100

    eq = pd.Series(equity, index=df.index)
    bh = pd.Series(close / close[0], index=df.index)

    def max_dd(series: pd.Series) -> float:
        peak = series.cummax()
        return float(((series / peak) - 1).min() * 100)

    strat_dd = max_dd(eq)
    bh_dd = max_dd(bh)

    # Upside / downside capture vs buy & hold (bar-level)
    bh_ret = bh.pct_change(fill_method=None).fillna(0)
    eq_ret = eq.pct_change(fill_method=None).fillna(0)
    up_mask, dn_mask = bh_ret > 0, bh_ret < 0
    up_capture = (float(eq_ret[up_mask].sum() / bh_ret[up_mask].sum()) * 100
                  if up_mask.any() and bh_ret[up_mask].sum() != 0 else 0.0)
    dn_capture = (float(eq_ret[dn_mask].sum() / bh_ret[dn_mask].sum()) * 100
                  if dn_mask.any() and bh_ret[dn_mask].sum() != 0 else 0.0)

    closed = [t for t in trades if t.exit_price is not None]
    wins = [t for t in closed if (t.pnl_pct or 0) > 0]

    return CommitteeBacktestResult(
        ticker=ticker, interval=interval, bars=n,
        start=df.index[0], end=df.index[-1],
        equity=eq, buy_hold=bh, score=score,
        position=pd.Series(position, index=df.index),
        total_return_pct=round(float(eq.iloc[-1] - 1) * 100, 2),
        buy_hold_return_pct=round(float(bh.iloc[-1] - 1) * 100, 2),
        alpha_pct=round(float(eq.iloc[-1] - bh.iloc[-1]) * 100, 2),
        max_drawdown_pct=round(strat_dd, 2),
        bh_max_drawdown_pct=round(bh_dd, 2),
        drawdown_edge_pp=round(strat_dd - bh_dd, 2),
        downside_captured_pct=round(dn_capture, 1),
        upside_captured_pct=round(up_capture, 1),
        time_in_market_pct=round(float(position[warm:].mean() * 100)
                                 if n > warm else 0.0, 1),
        win_rate_pct=round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
        total_trades=len(trades),
        fees_paid_pct=round(fees_paid * 100, 2),
        trades=trades,
    )


# ---------------------------------------------------------------------------
# Auto-optimizer — grid search over entry/exit vote margins
# ---------------------------------------------------------------------------

@dataclass
class OptimizationCell:
    enter_votes: int
    exit_votes: int
    total_return_pct: float
    max_drawdown_pct: float
    total_trades: int
    win_rate_pct: float
    fitness: float                    # risk-adjusted ranking score


def optimize_committee(
    df: pd.DataFrame,
    ticker: str = "",
    interval: str = "1d",
    fee_pct: float = 0.1,
    margins: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14),
) -> tuple[list[OptimizationCell], CommitteeBacktestResult]:
    """
    Grid-search entry/exit vote margins and rank by a risk-adjusted score:

        fitness = total_return − 0.5 · |max_drawdown|

    (rewarding profit but penalising deep pain — the committee's whole
    point is a smoother ride, not just raw return).

    The 38-agent vote matrix is computed ONCE and reused for every cell,
    so a full 7×7 grid takes about as long as a single backtest.

    Returns ``(all_cells_sorted_best_first, best_full_result)``.
    """
    com = IndicatorCommittee()
    votes = com.vote_matrix(df)

    cells: list[OptimizationCell] = []
    n_agents = len(com.agents)
    for ev in margins:
        for xv in margins:
            cfg = CommitteeConfig(enter_score=ev / n_agents,
                                  exit_score=-xv / n_agents)
            res = backtest_committee(
                df, ticker=ticker, interval=interval, config=cfg,
                fee_pct=fee_pct, committee=IndicatorCommittee(cfg),
                votes=votes,
            )
            cells.append(OptimizationCell(
                enter_votes=ev, exit_votes=xv,
                total_return_pct=res.total_return_pct,
                max_drawdown_pct=res.max_drawdown_pct,
                total_trades=res.total_trades,
                win_rate_pct=res.win_rate_pct,
                fitness=round(res.total_return_pct
                              - 0.5 * abs(res.max_drawdown_pct), 2),
            ))

    cells.sort(key=lambda c: c.fitness, reverse=True)
    best = cells[0]
    best_cfg = CommitteeConfig(enter_score=best.enter_votes / n_agents,
                               exit_score=-best.exit_votes / n_agents)
    best_res = backtest_committee(
        df, ticker=ticker, interval=interval, config=best_cfg,
        fee_pct=fee_pct, committee=IndicatorCommittee(best_cfg), votes=votes,
    )
    return cells, best_res


# ---------------------------------------------------------------------------
# Data helper — fetch history within yfinance interval limits
# ---------------------------------------------------------------------------

# yfinance hard limits per interval (calendar days of history available)
_INTERVAL_MAX_DAYS = {
    "5m": 59, "15m": 59, "30m": 59,
    "1h": 729,
    "1d": 365 * 30, "1wk": 365 * 30,
}


def fetch_history(ticker: str, days: int, interval: str = "1d") -> pd.DataFrame:
    """
    Download raw OHLCV bars for the committee backtest.

    Clamps *days* to what Yahoo Finance actually serves for the interval
    (e.g. hourly bars only go back ~2 years; for 5 years use daily bars).
    Works for crypto (24/7) and stocks (exchange hours) alike.
    """
    import yfinance as yf
    days = min(int(days), _INTERVAL_MAX_DAYS.get(interval, 365 * 30))
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=days)
    df = yf.Ticker(ticker.upper().strip()).history(
        start=str(start), end=str(end), interval=interval,
        auto_adjust=True, actions=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No {interval} data returned for '{ticker}'.")
    if df.index.tzinfo is not None:
        df.index = df.index.tz_localize(None)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"]
            if c in df.columns]
    df = df[cols].astype(float).sort_index()
    # Yahoo occasionally returns rows with NaN prices (halts, partial bars)
    # which would poison the equity curve — drop them outright.
    df = df.dropna(subset=[c for c in ("Open", "Close") if c in df.columns])
    return df


# ---------------------------------------------------------------------------
# __main__ — smoke test (no API key needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    df = fetch_history(tk, days=365 * 3, interval="1d")
    res = backtest_committee(df, ticker=tk, interval="1d")
    print(f"\n{tk} — {res.bars} bars  {res.start:%Y-%m-%d} → {res.end:%Y-%m-%d}")
    print(f"  Committee return : {res.total_return_pct:+.1f}%")
    print(f"  Buy & hold       : {res.buy_hold_return_pct:+.1f}%")
    print(f"  Alpha            : {res.alpha_pct:+.1f}pp")
    print(f"  Max DD (bot/B&H) : {res.max_drawdown_pct:.1f}% / "
          f"{res.bh_max_drawdown_pct:.1f}%")
    print(f"  Upside captured  : {res.upside_captured_pct:.0f}%   "
          f"Downside captured: {res.downside_captured_pct:.0f}%")
    print(f"  Trades {res.total_trades} · win {res.win_rate_pct:.0f}% · "
          f"in-market {res.time_in_market_pct:.0f}% · "
          f"fees {res.fees_paid_pct:.2f}%")
