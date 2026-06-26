"""
Backtesting Engine — Phase 5
==============================

Simulates the full BotTrade pipeline (fetch → RAG retrieve → AI decide)
over a historical date range and evaluates financial performance using
``vectorbt``.

Architecture
------------
**Single-download, rolling-slice strategy**

The full enriched DataFrame is downloaded once at startup, extended
backwards by a configurable warm-up window so that SMA_200 is fully
computed before the first simulation day.  On each simulation step the
engine receives a *slice* of that DataFrame (``full_df.loc[:sim_date]``),
which is wrapped in a ``MarketSnapshot`` and fed to the existing Phase 2–4
pipeline without any additional network calls.

Why this is point-in-time correct: SMA, RSI, and MACD values at day ``t``
depend only on Close prices ≤ ``t``.  Computing them over the full dataset
and then slicing produces bit-identical values to computing them online,
so there is zero look-ahead bias.

Position model
--------------
Long-only, all-in / all-out (MVP):

- **BUY**  (when not in position, confidence ≥ ``min_confidence``):
  Allocate 100 % of current cash at next-bar's open price.
- **SELL** (when in position):
  Liquidate entire position at next-bar's open price.
- **HOLD** or low-confidence: No action.

Execution uses next-bar's ``Open`` price to prevent look-ahead bias
(we decide at Close, execute at next Open).

vectorbt layer
--------------
After the loop the recorded ``entries`` / ``exits`` boolean Series are
fed into ``vbt.Portfolio.from_signals()`` so the library handles all the
P&L, drawdown, and Sharpe arithmetic with its optimised NumPy engine.

Typical usage
-------------
    from market_data.fetcher    import MarketDataFetcher
    from rag.retriever          import StrategyRetriever
    from decision_engine.ai_engine import AITradingEngine
    from backtesting.backtest_runner import BacktestRunner

    runner = BacktestRunner(
        fetcher   = MarketDataFetcher(),
        retriever = StrategyRetriever(),
        engine    = AITradingEngine(),
    )
    result = runner.run("QQQ", start_date="2023-01-01", end_date="2023-12-31")
    result.print_summary()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings
from decision_engine.ai_engine import AITradingEngine, TradingDecision
from market_data.fetcher import MACDParams, MarketDataFetcher, MarketSnapshot
from rag.retriever import RetrievalResult, StrategyRetriever
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Calendar days prepended before start_date so that SMA_200 (199 bars) and
# a comfortable margin are fully warmed up before the first signal day.
_DEFAULT_WARMUP_DAYS: int = 550   # ≈ 390 trading days → ~190 usable after dropna

# Minimum confidence threshold below which BUY/SELL decisions are treated as HOLD
_DEFAULT_MIN_CONFIDENCE: float = 0.55

# Seconds between consecutive API calls — respects Anthropic rate limits
_DEFAULT_INTER_CALL_DELAY: float = 1.0

# Broker fees per side and slippage (both as fractions, e.g. 0.001 = 0.1%)
_DEFAULT_FEES: float = 0.001
_DEFAULT_SLIPPAGE: float = 0.001


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """
    Immutable record of a single AI decision made during the simulation.

    Every day produces a ``TradeRecord`` regardless of action — HOLD records
    are kept so the reasoning log is complete and can be reviewed later.

    Attributes:
        sim_date:          The simulation date (the bar whose data fed the AI).
        action:            "BUY", "SELL", or "HOLD".
        exec_price:        Next-bar open price at which the order was filled.
                           ``None`` for HOLD records.
        confidence:        AI confidence score [0.0, 1.0].
        risk_level:        "LOW" / "MEDIUM" / "HIGH".
        rag_quality:       "STRONG" / "MODERATE" / "WEAK" / "NONE".
        is_fallback:       True if the AI engine returned an error fallback.
        reasoning_snippet: First 200 characters of the AI's reasoning text.
        portfolio_value:   Portfolio value *after* this action is applied.
    """

    sim_date: date
    action: str
    exec_price: Optional[float]
    confidence: float
    risk_level: str
    rag_quality: str
    is_fallback: bool
    reasoning_snippet: str
    portfolio_value: float


@dataclass
class _PositionState:
    """Mutable state tracking the current open position."""

    in_position: bool = False
    entry_price: float = 0.0
    entry_date: Optional[date] = None
    shares: float = 0.0
    cash: float = 0.0
    trade_count: int = 0           # number of completed round-trips
    winning_trades: int = 0        # completed trades with positive P&L

    def portfolio_value(self, current_price: float) -> float:
        """Mark-to-market value given the current close price."""
        return self.cash + self.shares * current_price

    def open_trade(self, price: float, dt: date) -> None:
        """Allocate all available cash into shares at *price*."""
        self.shares = self.cash / price
        self.entry_price = price
        self.entry_date = dt
        self.cash = 0.0
        self.in_position = True

    def close_trade(self, price: float) -> float:
        """
        Liquidate all shares at *price*.

        Returns:
            Realised P&L (positive = win, negative = loss).
        """
        proceeds = self.shares * price
        pnl = proceeds - (self.shares * self.entry_price)
        self.cash = proceeds
        self.shares = 0.0
        self.in_position = False
        self.trade_count += 1
        if pnl > 0:
            self.winning_trades += 1
        return pnl


@dataclass
class BacktestResult:
    """
    Complete result of a :meth:`BacktestRunner.run` call.

    Holds both the raw trade log and the fully-computed ``vectorbt``
    ``Portfolio`` object.  The ``print_summary()`` method renders a
    formatted Rich console report.

    Attributes:
        ticker:                Asset that was tested.
        start_date:            First simulation day (inclusive).
        end_date:              Last simulation day (inclusive).
        initial_capital:       Starting cash in USD.
        final_portfolio_value: Ending mark-to-market portfolio value.
        total_return_pct:      Strategy total return in percent.
        buy_hold_return_pct:   Passive buy-and-hold return for the same period.
        alpha_pct:             total_return_pct − buy_hold_return_pct.
        sharpe_ratio:          Annualised Sharpe ratio (from vectorbt).
        max_drawdown_pct:      Maximum peak-to-trough drawdown in percent.
        win_rate_pct:          Percentage of closed trades that were profitable.
        total_trades:          Number of completed round-trip trades.
        total_days_simulated:  Calendar days in the simulation window.
        ai_calls_made:         Total calls to the Anthropic API.
        fallback_count:        Number of days where the AI returned a fallback HOLD.
        trade_log:             Ordered list of :class:`TradeRecord` for every bar.
        portfolio:             The ``vectorbt.Portfolio`` object (may be None if vbt
                               is unavailable or no trades were executed).
    """

    ticker: str
    start_date: date
    end_date: date
    initial_capital: float
    final_portfolio_value: float
    total_return_pct: float
    buy_hold_return_pct: float
    alpha_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    total_days_simulated: int
    ai_calls_made: int
    fallback_count: int
    trade_log: list[TradeRecord] = field(default_factory=list)
    portfolio: Optional[object] = field(default=None, repr=False)  # vbt.Portfolio

    def __repr__(self) -> str:
        return (
            f"BacktestResult(ticker={self.ticker!r}, "
            f"return={self.total_return_pct:+.2f}%, "
            f"alpha={self.alpha_pct:+.2f}%, "
            f"sharpe={self.sharpe_ratio:.3f}, "
            f"max_dd={self.max_drawdown_pct:.2f}%, "
            f"trades={self.total_trades})"
        )

    def print_summary(self) -> None:
        """Render a complete Rich console report of the backtest results."""
        _render_backtest_report(self)


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------


class BacktestRunner:
    """
    Orchestrates the full Phase 1–4 pipeline over a historical date range
    to simulate and evaluate trading performance.

    Parameters
    ----------
    fetcher:
        Initialised :class:`~market_data.fetcher.MarketDataFetcher`.
        Provides enriched OHLCV + indicator data.
    retriever:
        Initialised :class:`~rag.retriever.StrategyRetriever`.
        Provides RAG context for each simulation day.
    engine:
        Initialised :class:`~decision_engine.ai_engine.AITradingEngine`.
        Provides the AI trading decision.
    rag_top_k:
        Number of strategy chunks to retrieve per day.
    min_confidence:
        Minimum AI confidence required to execute a BUY or SELL.
        Decisions below this threshold are treated as HOLD.
    inter_call_delay:
        Seconds to sleep between consecutive Anthropic API calls.
        Prevents hitting rate limits on long backtests.
    fees:
        Broker commission as a fraction of trade value (e.g. 0.001 = 0.1 %).
    slippage:
        Estimated execution slippage as a fraction of trade value.
    """

    def __init__(
        self,
        fetcher: MarketDataFetcher,
        retriever: StrategyRetriever,
        engine: AITradingEngine,
        rag_top_k: int = 3,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        inter_call_delay: float = _DEFAULT_INTER_CALL_DELAY,
        fees: float = _DEFAULT_FEES,
        slippage: float = _DEFAULT_SLIPPAGE,
    ) -> None:
        self._fetcher = fetcher
        self._retriever = retriever
        self._engine = engine
        self._rag_top_k = rag_top_k
        self._min_confidence = min_confidence
        self._inter_call_delay = inter_call_delay
        self._fees = fees
        self._slippage = slippage

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        ticker: str,
        start_date: str | date,
        end_date: str | date,
        initial_capital: float = 10_000.0,
        warmup_days: int = _DEFAULT_WARMUP_DAYS,
    ) -> BacktestResult:
        """
        Run the full backtesting simulation.

        Steps:
        1. Fetch enriched OHLCV + indicators for [start − warmup, end].
        2. Isolate simulation days within [start, end].
        3. For each simulation day, slice the DataFrame, build a
           ``MarketSnapshot``, call the RAG retriever and AI engine.
        4. Execute BUY / SELL logic against a ``_PositionState``.
        5. Build a ``vectorbt`` Portfolio from the entry/exit signal Series.
        6. Compute and return a ``BacktestResult``.

        Args:
            ticker:          Yahoo Finance ticker symbol (e.g. "QQQ").
            start_date:      First simulation day.  Accepts "YYYY-MM-DD" or
                             :class:`datetime.date`.
            end_date:        Last simulation day (inclusive).
            initial_capital: Starting cash in USD.
            warmup_days:     Extra calendar days fetched before ``start_date``
                             to warm up the SMA_200 indicator.

        Returns:
            A fully populated :class:`BacktestResult`.

        Raises:
            ValueError: If no simulation days remain after warm-up or if
                        the date range is invalid.
            RuntimeError: If the market data fetch fails.
        """
        sim_start, sim_end = self._validate_dates(start_date, end_date)
        fetch_start = sim_start - timedelta(days=warmup_days)

        logger.info(
            f"[bold]Backtest:[/bold] {ticker} | "
            f"sim window: {sim_start} → {sim_end} | "
            f"fetch window: {fetch_start} → {sim_end} | "
            f"capital: ${initial_capital:,.2f}"
        )

        # ---------------------------------------------------------------- #
        # Step 1: Download the full enriched DataFrame once
        # ---------------------------------------------------------------- #
        full_snapshot = self._fetcher.fetch(
            ticker=ticker,
            start=fetch_start,
            end=sim_end,
        )
        full_df = full_snapshot.data

        logger.debug(
            f"Full dataset: {len(full_df)} rows after warm-up dropna "
            f"({full_df.index[0].date()} → {full_df.index[-1].date()})"
        )

        # ---------------------------------------------------------------- #
        # Step 2: Isolate simulation days
        # ---------------------------------------------------------------- #
        sim_mask = (full_df.index >= pd.Timestamp(sim_start)) & (
            full_df.index <= pd.Timestamp(sim_end)
        )
        sim_df = full_df.loc[sim_mask]

        if sim_df.empty:
            raise ValueError(
                f"No trading days found in [{sim_start}, {sim_end}] after the "
                f"SMA warm-up window was removed.  "
                f"Extend the date range or reduce SMA periods."
            )

        sim_dates = list(sim_df.index)
        n_days = len(sim_dates)
        logger.info(f"Simulation window contains {n_days} trading days.")

        # ---------------------------------------------------------------- #
        # Step 3–4: Main simulation loop
        # ---------------------------------------------------------------- #
        position = _PositionState(cash=initial_capital)
        trade_log: list[TradeRecord] = []
        entries_list: list[bool] = []
        exits_list: list[bool] = []
        ai_calls = 0
        fallbacks = 0

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        from rich.console import Console

        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[action]:<6}"),
            TextColumn("[green]${task.fields[value]:>9,.2f}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                description=f"[bold cyan]Backtesting {ticker}[/bold cyan]",
                total=n_days,
                action="START",
                value=initial_capital,
            )

            for bar_idx, sim_ts in enumerate(sim_dates):
                sim_date_obj: date = sim_ts.date()  # type: ignore[union-attr]

                # Determine next bar's open price for execution
                # (decide at Close of sim_ts, execute at Open of next bar)
                exec_price: Optional[float] = None
                if bar_idx + 1 < len(sim_dates):
                    next_ts = sim_dates[bar_idx + 1]
                    exec_price = float(full_df.loc[next_ts, "Open"])
                else:
                    # Last bar: use same bar's close (no next bar)
                    exec_price = float(sim_df.loc[sim_ts, "Close"])

                current_close = float(sim_df.loc[sim_ts, "Close"])
                current_value = position.portfolio_value(current_close)

                # ---- Build a point-in-time MarketSnapshot ---- #
                data_slice = full_df.loc[:sim_ts]
                snap = MarketSnapshot(
                    ticker=ticker,
                    data=data_slice,
                    sma_periods=full_snapshot.sma_periods,
                    rsi_period=full_snapshot.rsi_period,
                    macd_params=full_snapshot.macd_params,
                )

                # ---- RAG retrieval ---- #
                try:
                    retrieval: RetrievalResult = self._retriever.get_relevant_strategies(
                        snap, top_k=self._rag_top_k
                    )
                except Exception as exc:
                    logger.warning(f"{sim_date_obj}: RAG retrieval failed: {exc}")
                    retrieval = self._empty_retrieval(ticker, snap)

                # ---- AI decision ---- #
                try:
                    decision: TradingDecision = self._engine.evaluate_market(snap, retrieval)
                    ai_calls += 1
                    if decision.is_fallback:
                        fallbacks += 1
                except Exception as exc:
                    logger.warning(f"{sim_date_obj}: AI engine raised unexpectedly: {exc}")
                    decision = AITradingEngine._make_fallback(ticker, str(exc))
                    ai_calls += 1
                    fallbacks += 1

                # ---- Translate decision to position change ---- #
                entry_signal = False
                exit_signal = False
                effective_action = decision.action
                executed_at: Optional[float] = None

                is_confident = decision.confidence_score >= self._min_confidence
                is_last_bar = (bar_idx + 1 >= n_days)

                if (
                    decision.action == "BUY"
                    and not position.in_position
                    and is_confident
                    and not is_last_bar
                    and not decision.is_fallback
                ):
                    # Apply fees + slippage to execution price
                    fill_price = exec_price * (1 + self._fees + self._slippage)
                    position.open_trade(fill_price, sim_date_obj)
                    entry_signal = True
                    executed_at = fill_price
                    logger.debug(
                        f"{sim_date_obj}: BUY @ ${fill_price:.2f} "
                        f"(conf={decision.confidence_score:.2f})"
                    )

                elif (
                    decision.action == "SELL"
                    and position.in_position
                    and (is_confident or is_last_bar)
                ):
                    # Apply slippage/fees on exit
                    fill_price = exec_price * (1 - self._fees - self._slippage)
                    pnl = position.close_trade(fill_price)
                    exit_signal = True
                    executed_at = fill_price
                    logger.debug(
                        f"{sim_date_obj}: SELL @ ${fill_price:.2f} "
                        f"(P&L: ${pnl:+.2f})"
                    )

                else:
                    effective_action = "HOLD"

                # Force-close open position on last bar
                if is_last_bar and position.in_position and not exit_signal:
                    fill_price = current_close * (1 - self._fees)
                    pnl = position.close_trade(fill_price)
                    exit_signal = True
                    effective_action = "SELL*"  # asterisk = forced close
                    executed_at = fill_price
                    logger.debug(f"{sim_date_obj}: FORCED CLOSE @ ${fill_price:.2f}")

                entries_list.append(entry_signal)
                exits_list.append(exit_signal)

                post_value = position.portfolio_value(current_close)

                # ---- Record the bar ---- #
                trade_log.append(
                    TradeRecord(
                        sim_date=sim_date_obj,
                        action=effective_action,
                        exec_price=executed_at,
                        confidence=decision.confidence_score,
                        risk_level=decision.risk_level,
                        rag_quality=decision.rag_context_quality,
                        is_fallback=decision.is_fallback,
                        reasoning_snippet=decision.reasoning[:200],
                        portfolio_value=post_value,
                    )
                )

                # ---- Progress bar update ---- #
                progress.update(
                    task,
                    advance=1,
                    action=effective_action[:6],
                    value=post_value,
                )

                # ---- Rate-limiting sleep (skip on last bar) ---- #
                if not is_last_bar and self._inter_call_delay > 0:
                    time.sleep(self._inter_call_delay)

        # ---------------------------------------------------------------- #
        # Step 5: Build vectorbt Portfolio
        # ---------------------------------------------------------------- #
        price_series = sim_df["Close"]
        entries_series = pd.Series(entries_list, index=sim_df.index, dtype=bool)
        exits_series = pd.Series(exits_list, index=sim_df.index, dtype=bool)

        vbt_portfolio, vbt_stats = self._build_vbt_portfolio(
            price=price_series,
            entries=entries_series,
            exits=exits_series,
            initial_capital=initial_capital,
        )

        # ---------------------------------------------------------------- #
        # Step 6: Compute summary metrics
        # ---------------------------------------------------------------- #
        final_value = position.portfolio_value(float(price_series.iloc[-1]))
        total_return = (final_value / initial_capital - 1.0) * 100.0

        first_price = float(price_series.iloc[0])
        last_price = float(price_series.iloc[-1])
        bh_return = (last_price / first_price - 1.0) * 100.0

        win_rate = (
            position.winning_trades / position.trade_count * 100.0
            if position.trade_count > 0
            else 0.0
        )

        return BacktestResult(
            ticker=ticker,
            start_date=sim_start,
            end_date=sim_end,
            initial_capital=initial_capital,
            final_portfolio_value=final_value,
            total_return_pct=round(total_return, 4),
            buy_hold_return_pct=round(bh_return, 4),
            alpha_pct=round(total_return - bh_return, 4),
            sharpe_ratio=round(vbt_stats.get("sharpe_ratio", 0.0), 4),
            max_drawdown_pct=round(vbt_stats.get("max_drawdown_pct", 0.0), 4),
            win_rate_pct=round(win_rate, 2),
            total_trades=position.trade_count,
            total_days_simulated=n_days,
            ai_calls_made=ai_calls,
            fallback_count=fallbacks,
            trade_log=trade_log,
            portfolio=vbt_portfolio,
        )

    # ------------------------------------------------------------------ #
    # vectorbt helper
    # ------------------------------------------------------------------ #

    def _build_vbt_portfolio(
        self,
        price: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        initial_capital: float,
    ) -> tuple[Optional[object], dict]:
        """
        Build a ``vectorbt.Portfolio`` from signal Series and extract
        key performance metrics into a plain dict.

        Falls back gracefully if vectorbt is not installed or if no trades
        were executed (an empty signal set causes vectorbt to raise).

        Args:
            price:           Close price Series with DatetimeIndex.
            entries:         Boolean Series; True on BUY execution days.
            exits:           Boolean Series; True on SELL execution days.
            initial_capital: Starting cash in USD.

        Returns:
            Tuple of ``(vbt.Portfolio or None, metrics_dict)``.
        """
        default_stats: dict = {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0,
        }

        if not entries.any():
            logger.warning(
                "No BUY signals were generated during the simulation.  "
                "vectorbt portfolio is unavailable — all metrics set to 0."
            )
            return None, default_stats

        try:
            import vectorbt as vbt
        except ImportError:
            logger.warning("vectorbt not installed.  Install it: pip install vectorbt")
            return None, default_stats

        try:
            portfolio = vbt.Portfolio.from_signals(
                close=price,
                entries=entries,
                exits=exits,
                init_cash=initial_capital,
                fees=self._fees,
                slippage=self._slippage,
                freq="1D",
            )

            stats = portfolio.stats()
            # Map to our standard key names, handling version differences
            metrics: dict = {
                "sharpe_ratio": float(
                    stats.get("Sharpe Ratio") or stats.get("sharpe_ratio") or 0.0
                ),
                "max_drawdown_pct": abs(float(
                    stats.get("Max Drawdown [%]") or
                    stats.get("max_drawdown_pct") or 0.0
                )),
                "total_return_pct": float(
                    stats.get("Total Return [%]") or
                    stats.get("total_return_pct") or 0.0
                ),
                "win_rate_pct": float(
                    stats.get("Win Rate [%]") or
                    stats.get("win_rate_pct") or 0.0
                ),
            }
            logger.debug(f"vectorbt stats: {metrics}")
            return portfolio, metrics

        except Exception as exc:
            logger.error(f"vectorbt portfolio construction failed: {exc}")
            return None, default_stats

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_dates(
        start: str | date,
        end: str | date,
    ) -> tuple[date, date]:
        """Parse and validate start/end dates."""
        from market_data.fetcher import _to_date
        s = _to_date(start) if isinstance(start, str) else start
        e = _to_date(end) if isinstance(end, str) else end
        if s >= e:
            raise ValueError(f"start_date ({s}) must be before end_date ({e}).")
        if (e - s).days < 20:
            raise ValueError(
                f"Simulation window is only {(e - s).days} days — "
                "need at least 20 trading days for meaningful results."
            )
        return s, e

    @staticmethod
    def _empty_retrieval(ticker: str, snapshot: "MarketSnapshot") -> "RetrievalResult":
        """Create a no-op RetrievalResult for use when the retriever fails."""
        from rag.retriever import RetrievalResult
        return RetrievalResult(
            query="retrieval unavailable",
            ticker=ticker,
            chunks=[],
            market_regime={},
            collection_size=0,
        )


# ---------------------------------------------------------------------------
# Rich console renderer
# ---------------------------------------------------------------------------


def _render_backtest_report(result: "BacktestResult") -> None:
    """
    Render a fully formatted backtesting report to the console using Rich.

    Displays three panels:
    1. Performance summary table (returns, risk, trade stats).
    2. Trade activity log (last 20 executed trades).
    3. AI engine quality metrics (fallback rate, RAG quality distribution).
    """
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    console = Console()

    console.print()
    console.print(Rule(f"[bold cyan]BACKTEST REPORT — {result.ticker}[/bold cyan]"))

    # ------------------------------------------------------------------ #
    # Panel 1: Performance Summary
    # ------------------------------------------------------------------ #
    def _color_pct(val: float, invert: bool = False) -> str:
        good = val > 0 if not invert else val < 0
        bad  = val < 0 if not invert else val > 0
        if good:
            return f"[bold green]{val:+.2f}%[/bold green]"
        if bad:
            return f"[bold red]{val:+.2f}%[/bold red]"
        return f"[dim]{val:+.2f}%[/dim]"

    perf_table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    perf_table.add_column("Metric", style="dim", no_wrap=True)
    perf_table.add_column("Value", justify="right")

    perf_table.add_row("Ticker",              f"[bold]{result.ticker}[/bold]")
    perf_table.add_row("Period",
        f"{result.start_date}  →  {result.end_date}")
    perf_table.add_row("Trading Days",        str(result.total_days_simulated))
    perf_table.add_row("Initial Capital",     f"${result.initial_capital:>12,.2f}")
    perf_table.add_row("Final Value",         f"${result.final_portfolio_value:>12,.2f}")
    perf_table.add_row("─" * 28,             "─" * 12)
    perf_table.add_row("Strategy Return",     _color_pct(result.total_return_pct))
    perf_table.add_row("Buy & Hold Return",   _color_pct(result.buy_hold_return_pct))
    perf_table.add_row("Alpha (vs B&H)",
        _color_pct(result.alpha_pct))
    perf_table.add_row("─" * 28,             "─" * 12)
    sharpe_color = (
        "green" if result.sharpe_ratio > 1.0 else
        "yellow" if result.sharpe_ratio > 0 else "red"
    )
    perf_table.add_row("Sharpe Ratio",
        f"[{sharpe_color}]{result.sharpe_ratio:.3f}[/{sharpe_color}]")
    perf_table.add_row("Max Drawdown",
        _color_pct(-result.max_drawdown_pct, invert=False)
        .replace("+", ""))
    perf_table.add_row("Win Rate",
        f"[{'green' if result.win_rate_pct >= 50 else 'red'}]"
        f"{result.win_rate_pct:.1f}%"
        f"[/{'green' if result.win_rate_pct >= 50 else 'red'}]")
    perf_table.add_row("Total Trades",        str(result.total_trades))
    perf_table.add_row("─" * 28,             "─" * 12)
    perf_table.add_row("AI API Calls",        str(result.ai_calls_made))
    fallback_color = "red" if result.fallback_count > result.ai_calls_made * 0.1 else "green"
    perf_table.add_row("Fallback HOLDs",
        f"[{fallback_color}]{result.fallback_count}[/{fallback_color}] "
        f"({result.fallback_count / max(result.ai_calls_made, 1) * 100:.1f}%)")

    console.print(Panel(
        perf_table,
        title="[bold]Performance Summary[/bold]",
        expand=False,
    ))

    # ------------------------------------------------------------------ #
    # Panel 2: Executed Trade Log
    # ------------------------------------------------------------------ #
    executed = [r for r in result.trade_log if r.action not in ("HOLD",)]
    if executed:
        trade_table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        trade_table.add_column("Date",       no_wrap=True)
        trade_table.add_column("Action",     width=7)
        trade_table.add_column("Price",      justify="right")
        trade_table.add_column("Confidence", justify="right")
        trade_table.add_column("Risk",       width=8)
        trade_table.add_column("RAG",        width=9)
        trade_table.add_column("Portfolio",  justify="right")
        trade_table.add_column("Reasoning (snippet)")

        for rec in executed[-20:]:  # show last 20 to keep table manageable
            act_color = (
                "green"  if "BUY"  in rec.action else
                "red"    if "SELL" in rec.action else
                "yellow"
            )
            conf_color = (
                "green"  if rec.confidence >= 0.70 else
                "yellow" if rec.confidence >= 0.45 else
                "red"
            )
            risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(
                rec.risk_level, "white"
            )
            rag_color = {"STRONG": "green", "MODERATE": "cyan",
                         "WEAK": "yellow", "NONE": "red"}.get(rec.rag_quality, "white")

            trade_table.add_row(
                str(rec.sim_date),
                f"[bold {act_color}]{rec.action}[/bold {act_color}]",
                f"${rec.exec_price:.2f}" if rec.exec_price else "—",
                f"[{conf_color}]{rec.confidence:.2f}[/{conf_color}]",
                f"[{risk_color}]{rec.risk_level}[/{risk_color}]",
                f"[{rag_color}]{rec.rag_quality}[/{rag_color}]",
                f"${rec.portfolio_value:,.2f}",
                rec.reasoning_snippet[:80] + "…" if len(rec.reasoning_snippet) > 80
                else rec.reasoning_snippet,
            )

        console.print(Panel(
            trade_table,
            title=f"[bold]Executed Trades[/bold] (showing last {min(len(executed), 20)} of {len(executed)})",
        ))
    else:
        console.print(Panel(
            "[yellow]No BUY or SELL orders were executed during the simulation.[/yellow]\n"
            "Possible causes:\n"
            "• All AI decisions were HOLD or below the confidence threshold.\n"
            "• Knowledge base is empty (run knowledge ingestion first).\n"
            "• API key is invalid (all decisions fell back to HOLD).",
            title="[yellow]No Trades Executed[/yellow]",
            expand=False,
        ))

    # ------------------------------------------------------------------ #
    # Panel 3: AI Quality Distribution
    # ------------------------------------------------------------------ #
    action_counts: dict[str, int] = {}
    rag_counts: dict[str, int] = {}
    for rec in result.trade_log:
        action_counts[rec.action] = action_counts.get(rec.action, 0) + 1
        rag_counts[rec.rag_quality] = rag_counts.get(rec.rag_quality, 0) + 1

    quality_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    quality_table.add_column("Signal Distribution")
    quality_table.add_column("Count", justify="right")
    quality_table.add_column("Share", justify="right")
    total = max(result.total_days_simulated, 1)
    for action, cnt in sorted(action_counts.items()):
        color = "green" if "BUY" in action else "red" if "SELL" in action else "dim"
        quality_table.add_row(
            f"[{color}]{action}[/{color}]",
            str(cnt),
            f"{cnt / total * 100:.1f}%",
        )

    rag_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    rag_table.add_column("RAG Context Quality")
    rag_table.add_column("Count", justify="right")
    for quality, cnt in sorted(rag_counts.items()):
        color = {"STRONG": "green", "MODERATE": "cyan",
                 "WEAK": "yellow", "NONE": "red"}.get(quality, "white")
        rag_table.add_row(f"[{color}]{quality}[/{color}]", str(cnt))

    from rich.columns import Columns
    console.print(Panel(
        Columns([quality_table, rag_table], equal=True, expand=True),
        title="[bold]AI Signal & RAG Quality Distribution[/bold]",
    ))

    console.print()


# ---------------------------------------------------------------------------
# __main__ — integration smoke-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import os
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    console = Console()
    console.print(Rule("[bold yellow]Phase 5 — Backtesting Engine Test[/bold yellow]"))

    # ------------------------------------------------------------------ #
    # Check for API key before initialising the engine
    # ------------------------------------------------------------------ #
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    has_api_key = bool(api_key) and api_key.startswith("sk-ant-")

    if not has_api_key:
        console.print(
            Panel(
                "[yellow]ANTHROPIC_API_KEY not found in environment.[/yellow]\n\n"
                "The BacktestRunner requires a live Anthropic API key because it\n"
                "calls the AI Decision Engine on every simulation bar.\n\n"
                "Set your key and re-run:\n"
                "  [bold cyan]export ANTHROPIC_API_KEY=sk-ant-...[/bold cyan]\n"
                "  [bold cyan]python -m backtesting.backtest_runner[/bold cyan]\n\n"
                "[dim]Note: A 30-day backtest makes ~30 API calls.\n"
                "With prompt caching the cost is minimal.[/dim]",
                title="[yellow]API Key Required[/yellow]",
                expand=False,
            )
        )
        import sys
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Initialise the three pipeline dependencies
    # ------------------------------------------------------------------ #
    console.print("[dim]Initialising pipeline components…[/dim]")

    fetcher   = MarketDataFetcher()
    retriever = StrategyRetriever()
    engine    = AITradingEngine()

    runner = BacktestRunner(
        fetcher=fetcher,
        retriever=retriever,
        engine=engine,
        rag_top_k=3,
        min_confidence=0.55,
        inter_call_delay=1.0,   # 1 s between API calls
        fees=0.001,
        slippage=0.001,
    )

    # ------------------------------------------------------------------ #
    # Run a short 30-trading-day backtest on QQQ
    # ------------------------------------------------------------------ #
    from datetime import date, timedelta

    end   = date.today() - timedelta(days=1)          # yesterday (markets may be open today)
    start = end - timedelta(days=60)                  # ~30 trading days in ~60 calendar days

    console.print(
        f"Running backtest: [bold cyan]QQQ[/bold cyan] | "
        f"{start} → {end} | "
        f"Initial capital: $10,000"
    )

    try:
        result = runner.run(
            ticker="QQQ",
            start_date=start,
            end_date=end,
            initial_capital=10_000.0,
        )
        result.print_summary()

    except (ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Backtest failed:[/bold red] {exc}")
