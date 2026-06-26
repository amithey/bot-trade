"""
Market Data Fetcher — Phase 2
==============================

Fetches historical OHLCV price data from Yahoo Finance and augments it
with the technical indicators most commonly used in systematic trading:

  - Simple Moving Averages  (SMA_20, SMA_50, SMA_200  — configurable)
  - Relative Strength Index (RSI-14  — Wilder's smoothing)
  - MACD                    (12 / 26 / 9  — line, signal, histogram)

All indicator arithmetic is implemented in pure pandas / NumPy — no
external TA library is required, which keeps the dependency surface small
and makes the math fully transparent.

Typical usage
-------------
    from market_data.fetcher import MarketDataFetcher

    fetcher = MarketDataFetcher()

    # By explicit date range
    snap = fetcher.fetch("AAPL", start="2023-01-01", end="2024-01-01")

    # By look-back window (most common for live trading)
    snap = fetcher.fetch_latest("QQQ", lookback_days=400)

    print(snap.latest)          # most recent row as pd.Series
    print(snap.data.tail(5))    # last 5 rows of full DataFrame
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from utils.logger import get_logger

# Silence yfinance's own noisy progress/debug output
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_SMA_PERIODS: tuple[int, ...] = (20, 50, 200)
DEFAULT_RSI_PERIOD: int = 14
DEFAULT_MACD_FAST: int = 12
DEFAULT_MACD_SLOW: int = 26
DEFAULT_MACD_SIGNAL: int = 9

# Minimum rows needed before dropna (dominated by SMA_200 warm-up window)
_MIN_ROWS_WARN: int = 250

# Fundamentals change slowly — cache yfinance.Ticker.info for this long
FUNDAMENTALS_TTL_SEC: int = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Parameter / result data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FundamentalsData:
    """
    Point-in-time fundamental snapshot fetched from yfinance ``Ticker.info``.

    All fields are ``Optional`` — fundamentals are frequently missing for
    ETFs, index tickers (e.g. ``^TA35``), and tickers outside the main
    US exchanges.  Callers must always guard against ``None``.

    Attributes
    ----------
    pe_trailing:    Trailing twelve-month P/E ratio.
    pe_forward:     Analyst-consensus forward P/E.
    market_cap:     Total market capitalisation (USD).
    week_52_high:   Highest closing price over the trailing 52 weeks.
    week_52_low:    Lowest closing price over the trailing 52 weeks.
    dividend_yield: Annual dividend yield as a decimal (0.02 = 2 %).
    beta:           Beta vs the S&P 500.
    sector:         GICS sector string (e.g. ``"Technology"``).
    industry:       GICS industry string (e.g. ``"Semiconductors"``).
    long_name:      Full company / fund name.
    currency:       Reporting currency (e.g. ``"USD"``, ``"ILS"``).
    fetched_at:     UTC timestamp of the info call.
    """

    pe_trailing:    Optional[float]
    pe_forward:     Optional[float]
    market_cap:     Optional[int]
    week_52_high:   Optional[float]
    week_52_low:    Optional[float]
    dividend_yield: Optional[float]
    beta:           Optional[float]
    sector:         Optional[str]
    industry:       Optional[str]
    long_name:      Optional[str]
    currency:       Optional[str]
    fetched_at:     datetime = field(default_factory=datetime.utcnow)

    @property
    def pe_discount_to_52w_high(self) -> Optional[float]:
        """Current price discount from 52-week high (requires week_52_high)."""
        return None  # placeholder — current price is in MarketSnapshot

    def summary_dict(self) -> dict[str, Any]:
        """Return a display-friendly dict (None values shown as ``'N/A'``)."""
        def _fmt(v: Any) -> Any:
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return round(v, 4)
            return v

        return {
            "P/E (Trailing)":  _fmt(self.pe_trailing),
            "P/E (Forward)":   _fmt(self.pe_forward),
            "Market Cap":      f"${self.market_cap:,}" if self.market_cap else "N/A",
            "52W High":        _fmt(self.week_52_high),
            "52W Low":         _fmt(self.week_52_low),
            "Dividend Yield":  f"{self.dividend_yield * 100:.2f}%" if self.dividend_yield else "N/A",
            "Beta":            _fmt(self.beta),
            "Sector":          self.sector or "N/A",
            "Industry":        self.industry or "N/A",
            "Name":            self.long_name or "N/A",
            "Currency":        self.currency or "N/A",
        }


@dataclass(frozen=True)
class MACDParams:
    """Encapsulates the three MACD period parameters."""

    fast: int = DEFAULT_MACD_FAST
    slow: int = DEFAULT_MACD_SLOW
    signal: int = DEFAULT_MACD_SIGNAL

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError(
                f"MACD fast period ({self.fast}) must be less than slow ({self.slow})."
            )
        if self.signal < 1:
            raise ValueError(f"MACD signal period must be >= 1, got {self.signal}.")


@dataclass
class MarketSnapshot:
    """
    Returned by every fetch call.  Wraps the enriched DataFrame together
    with the parameters that were used to build it.

    Attributes:
        ticker:      Ticker symbol, upper-cased.
        data:        Clean OHLCV + indicator DataFrame, index = DatetimeIndex.
        sma_periods: SMA periods that were computed.
        rsi_period:  RSI look-back period.
        macd_params: MACD configuration that was applied.
        fetched_at:  UTC timestamp of the fetch call.
    """

    ticker: str
    data: pd.DataFrame
    sma_periods: tuple[int, ...]
    rsi_period: int
    macd_params: MACDParams
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    # Optional fundamental data — populated only by fetch_with_fundamentals()
    fundamentals: Optional[FundamentalsData] = field(default=None, compare=False)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    @property
    def latest(self) -> pd.Series:
        """Most recent row as a labeled Series."""
        return self.data.iloc[-1]

    @property
    def indicator_columns(self) -> list[str]:
        """All column names beyond the base OHLCV set."""
        base = {"Open", "High", "Low", "Close", "Volume"}
        return [c for c in self.data.columns if c not in base]

    @property
    def start_date(self) -> date:
        return self.data.index[0].date()  # type: ignore[union-attr]

    @property
    def end_date(self) -> date:
        return self.data.index[-1].date()  # type: ignore[union-attr]

    @property
    def row_count(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"MarketSnapshot(ticker={self.ticker!r}, rows={self.row_count}, "
            f"range={self.start_date}→{self.end_date}, "
            f"indicators={self.indicator_columns})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MarketDataFetcher:
    """
    Fetches historical market data and computes technical indicators.

    Design notes
    ~~~~~~~~~~~~
    * Uses ``yf.Ticker.history()`` rather than ``yf.download()`` to avoid
      the MultiIndex column issue present in yfinance ≥ 0.2.x for single
      tickers.
    * All indicator math is pure pandas / NumPy — no TA library required.
    * RSI uses **Wilder's smoothed moving average** (EWM with ``com = period - 1``),
      which is the industry-standard formulation and matches TradingView.
    * After all indicators are appended the DataFrame is passed through
      ``dropna()`` — the SMA_200 warm-up window (199 rows) is the binding
      constraint, so callers should request at least ``max(sma_periods) + 50``
      trading days of raw data to end up with a meaningful result.

    Parameters
    ----------
    sma_periods:
        Tuple of look-back windows for Simple Moving Averages.
    rsi_period:
        RSI look-back period (Wilder's method).
    macd_params:
        Fast / slow / signal periods for MACD.  Pass a :class:`MACDParams`
        instance to override defaults.
    """

    def __init__(
        self,
        sma_periods: tuple[int, ...] = DEFAULT_SMA_PERIODS,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        macd_params: Optional[MACDParams] = None,
    ) -> None:
        if not sma_periods:
            raise ValueError("`sma_periods` must contain at least one value.")
        if rsi_period < 2:
            raise ValueError(f"`rsi_period` must be >= 2, got {rsi_period}.")

        self.sma_periods: tuple[int, ...] = tuple(sorted(sma_periods))
        self.rsi_period: int = rsi_period
        self.macd_params: MACDParams = macd_params or MACDParams()
        # TTL cache for fundamentals — ticker.info is slow (2-10s) and stable
        # intraday, so we only refresh every FUNDAMENTALS_TTL_SEC.
        self._fund_cache: dict[str, tuple[float, Optional["FundamentalsData"]]] = {}

    # ------------------------------------------------------------------ #
    # Public fetch API
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        ticker: str,
        start: Optional[str | date | datetime] = None,
        end: Optional[str | date | datetime] = None,
        period: Optional[str] = None,
        interval: str = "1d",
    ) -> MarketSnapshot:
        """
        Fetch OHLCV data for *ticker* and compute technical indicators.
        Supports either (start, end) dates OR a yfinance 'period' string.

        Args:
            ticker:   Yahoo Finance ticker symbol.
            start:    Start date (inclusive).
            end:      End date (inclusive). Defaults to today if start is provided.
            period:   yfinance period string (e.g., "1d", "5d", "1mo").
                      If provided, start and end are ignored.
            interval: yfinance interval string: "1d", "1h", "5m", etc.

        Returns:
            A :class:`MarketSnapshot` with clean indicator data.
        """
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker symbol must not be empty.")

        if period is None:
            if start is None:
                raise ValueError("Must provide either 'period' or 'start' date.")
            start_dt = _to_date(start)
            end_dt = _to_date(end) if end is not None else date.today()
            if start_dt >= end_dt:
                raise ValueError(f"start ({start_dt}) must be before end ({end_dt}).")
            raw = self._download(ticker, start=start_dt, end=end_dt, interval=interval)
        else:
            raw = self._download(ticker, period=period, interval=interval)

        enriched = self._add_indicators(raw)

        return MarketSnapshot(
            ticker=ticker,
            data=enriched,
            sma_periods=self.sma_periods,
            rsi_period=self.rsi_period,
            macd_params=self.macd_params,
        )

    def fetch_latest(
        self,
        ticker: str,
        lookback_days: int = 400,
        interval: str = "1d",
    ) -> MarketSnapshot:
        """
        Convenience method: fetch the *lookback_days* most recent calendar
        days of data ending today.

        The default of 400 calendar days (≈ 280 trading days) comfortably
        exceeds the SMA_200 warm-up window and leaves ~80 clean rows after
        ``dropna()``.

        Args:
            ticker:        Yahoo Finance ticker symbol.
            lookback_days: Calendar days to look back from today.
            interval:      yfinance interval string.

        Returns:
            :class:`MarketSnapshot` — same as :meth:`fetch`.
        """
        if lookback_days < 1:
            raise ValueError(f"lookback_days must be >= 1, got {lookback_days}.")

        end = date.today()
        start = end - timedelta(days=lookback_days)
        return self.fetch(ticker, start=start, end=end, interval=interval)

    def fetch_multiple(
        self,
        tickers: list[str],
        start: str | date | datetime,
        end: Optional[str | date | datetime] = None,
        interval: str = "1d",
    ) -> dict[str, MarketSnapshot]:
        """
        Fetch and enrich data for multiple tickers.

        Failed tickers are logged and excluded from the result dict rather
        than raising, so one bad ticker does not abort the batch.

        Args:
            tickers:  List of ticker symbols.
            start:    Start date (same semantics as :meth:`fetch`).
            end:      End date (same semantics as :meth:`fetch`).
            interval: yfinance interval string.

        Returns:
            ``{ticker: MarketSnapshot}`` for each successfully fetched ticker.
        """
        results: dict[str, MarketSnapshot] = {}
        for ticker in tickers:
            try:
                results[ticker.upper()] = self.fetch(
                    ticker, start=start, end=end, interval=interval
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning(f"Skipping {ticker!r}: {exc}")
        return results

    # ------------------------------------------------------------------ #
    # Fundamental data
    # ------------------------------------------------------------------ #

    def fetch_fundamentals(self, ticker: str) -> Optional[FundamentalsData]:
        """
        Fetch fundamental data for *ticker* from ``yfinance.Ticker.info``.

        Design goals
        ~~~~~~~~~~~~
        * **Never raises** — all failures (network, missing keys, unsupported
          tickers) are caught and logged; ``None`` is returned so callers
          can decide how to handle absence.
        * **Graceful on ETFs / indices** — ETFs (e.g. ``QQQ``, ``HACK``,
          ``URA``) have no P/E; Israeli TA-35 tickers (e.g. ``TEVA.TA``,
          ``^TA35``) may return sparse info.  Every field individually
          defaults to ``None`` if absent.
        * **Read-only** — no caching; callers should cache at the
          application layer if frequent calls are a concern.

        Supported ticker formats
        ~~~~~~~~~~~~~~~~~~~~~~~~
        * US equities:            ``AAPL``, ``NVDA``, ``MSFT``
        * US ETFs (sector/theme): ``QQQ``, ``HACK``, ``BOTZ``, ``URA``
        * Israeli stocks:         ``TEVA.TA``, ``BEZQ.TA``
        * Israeli index:          ``^TA35`` (limited info)
        * Indices:                ``^GSPC``, ``^NDX``

        Parameters
        ----------
        ticker:
            Yahoo Finance ticker symbol (case-insensitive; normalised to
            uppercase internally).

        Returns
        -------
        FundamentalsData or None
            ``None`` if the fetch failed entirely; a ``FundamentalsData``
            with ``None`` fields for any missing values otherwise.
        """
        import time as _time
        ticker = ticker.strip().upper()

        # TTL cache check
        cached = self._fund_cache.get(ticker)
        if cached is not None:
            ts, payload = cached
            if (_time.time() - ts) < FUNDAMENTALS_TTL_SEC:
                logger.debug("Fundamentals cache HIT for %s", ticker)
                return payload

        logger.debug("Fetching fundamentals for %s …", ticker)

        try:
            info: dict[str, Any] = yf.Ticker(ticker).info
        except Exception as exc:
            logger.warning("yfinance.Ticker(%r).info raised: %s", ticker, exc)
            # Cache the miss too so we don't hammer yfinance when it's down
            self._fund_cache[ticker] = (_time.time(), None)
            return None

        if not info or not isinstance(info, dict):
            logger.warning("yfinance returned empty info dict for %r.", ticker)
            self._fund_cache[ticker] = (_time.time(), None)
            return None

        def _float(key: str) -> Optional[float]:
            val = info.get(key)
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        def _int(key: str) -> Optional[int]:
            val = info.get(key)
            try:
                return int(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        def _str(key: str) -> Optional[str]:
            val = info.get(key)
            return str(val).strip() if val else None

        fundamentals = FundamentalsData(
            pe_trailing=    _float("trailingPE"),
            pe_forward=     _float("forwardPE"),
            market_cap=     _int("marketCap"),
            week_52_high=   _float("fiftyTwoWeekHigh"),
            week_52_low=    _float("fiftyTwoWeekLow"),
            dividend_yield= _float("dividendYield"),
            beta=           _float("beta"),
            sector=         _str("sector"),
            industry=       _str("industry"),
            long_name=      _str("longName") or _str("shortName"),
            currency=       _str("currency"),
        )

        present = sum(
            1 for v in (
                fundamentals.pe_trailing, fundamentals.pe_forward,
                fundamentals.market_cap,  fundamentals.week_52_high,
            )
            if v is not None
        )
        logger.debug(
            "Fundamentals for %s: pe_trail=%s, pe_fwd=%s, mktcap=%s, "
            "52wH=%s, 52wL=%s (%d/4 key fields populated)",
            ticker,
            fundamentals.pe_trailing, fundamentals.pe_forward,
            fundamentals.market_cap,
            fundamentals.week_52_high, fundamentals.week_52_low,
            present,
        )
        self._fund_cache[ticker] = (_time.time(), fundamentals)
        return fundamentals

    def fetch_with_fundamentals(
        self,
        ticker: str,
        start: Optional[str | date | datetime] = None,
        end: Optional[str | date | datetime] = None,
        period: Optional[str] = None,
        interval: str = "1d",
    ) -> MarketSnapshot:
        """
        Combined fetch: OHLCV + technical indicators **and** fundamentals.

        Equivalent to calling :meth:`fetch` followed by
        :meth:`fetch_fundamentals` and attaching the result to
        ``snapshot.fundamentals``.  If the fundamentals fetch fails,
        the snapshot is returned with ``fundamentals=None`` rather than
        raising.

        Parameters
        ----------
        ticker, start, end, interval:
            Same semantics as :meth:`fetch`.

        Returns
        -------
        MarketSnapshot
            ``snapshot.fundamentals`` is a :class:`FundamentalsData` instance
            when available, or ``None`` if the fundamentals call failed.
        """
        snapshot = self.fetch(ticker=ticker, start=start, end=end,
                              period=period, interval=interval)
        fundamentals = self.fetch_fundamentals(ticker)

        # Inject fundamentals into the snapshot via dataclass replacement
        import dataclasses
        snapshot = dataclasses.replace(snapshot, fundamentals=fundamentals)

        if fundamentals is not None:
            logger.info(
                "[bold green]%s[/bold green] fundamentals attached — "
                "P/E=%.2f, MktCap=%s, 52wH=%.2f",
                ticker,
                fundamentals.pe_trailing or 0.0,
                f"${fundamentals.market_cap / 1e9:.1f}B" if fundamentals.market_cap else "N/A",
                fundamentals.week_52_high or 0.0,
            )
        return snapshot

    # ------------------------------------------------------------------ #
    # Private: data download
    # ------------------------------------------------------------------ #

    def _download(
        self,
        ticker: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        period: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download raw OHLCV data from Yahoo Finance.
        """
        msg = f"Downloading {ticker} [{interval}]"
        if period:
            msg += f" (period={period})"
        else:
            msg += f" ({start} to {end})"
        logger.debug(msg)

        try:
            yticker = yf.Ticker(ticker)
            if period:
                df: pd.DataFrame = yticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    actions=False,
                )
            else:
                df: pd.DataFrame = yticker.history(
                    start=str(start),
                    end=str(end + timedelta(days=1)) if end else None,
                    interval=interval,
                    auto_adjust=True,
                    actions=False,
                )
        except Exception as exc:
            raise RuntimeError(f"yfinance error for '{ticker}': {exc}")

        if df is None or df.empty:
            raise RuntimeError(f"No data returned for ticker '{ticker}'.")

        # yfinance returns a timezone-aware index; strip tz to make it naive UTC.
        # Plotly handles naive DatetimeIndex cleanly; rangebreaks work correctly.
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)

        # Keep only the canonical OHLCV columns, drop anything yfinance appends
        ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[ohlcv_cols].copy()

        # Ensure float dtype for price columns (Volume stays int-like)
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df.sort_index(inplace=True)

        raw_rows = len(df)
        if raw_rows < _MIN_ROWS_WARN:
            logger.warning(
                f"{ticker}: only {raw_rows} raw rows fetched. "
                f"After SMA_{max(self.sma_periods)} warm-up ({max(self.sma_periods) - 1} rows), "
                f"the clean DataFrame may have very few usable rows. "
                f"Consider requesting at least {_MIN_ROWS_WARN + 50} calendar days of data."
            )

        return df

    # ------------------------------------------------------------------ #
    # Private: indicator pipeline
    # ------------------------------------------------------------------ #

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full indicator pipeline to a raw OHLCV DataFrame and
        return a clean (no-NaN) copy.

        Pipeline order matters:
        1. SMAs        — depend only on Close
        2. RSI         — depends only on Close
        3. MACD        — depends only on Close
        4. dropna()    — removes warm-up rows where any column is NaN

        Args:
            df: Raw OHLCV DataFrame (DatetimeIndex, float columns).

        Returns:
            Enriched DataFrame with all indicator columns appended and all
            NaN rows dropped.
        """
        df = df.copy()

        for period in self.sma_periods:
            df = self._add_sma(df, period)

        df = self._add_rsi(df, self.rsi_period)
        df = self._add_macd(df, self.macd_params)
        df = self._add_bollinger(df, period=20, stdev=2.0)
        df = self._add_atr(df, period=14)
        df = self._add_vwap(df, period=20)
        df = self._add_support_resistance(df, lookback=20)

        rows_before = len(df)
        df.dropna(inplace=True)
        rows_after = len(df)

        dropped = rows_before - rows_after
        if dropped > 0:
            logger.debug(
                f"dropna() removed {dropped} warm-up rows "
                f"({rows_before} → {rows_after})."
            )

        if rows_after == 0:
            raise RuntimeError(
                "All rows were removed by dropna() after indicator calculation. "
                "The fetched date range is too short for the configured SMA periods. "
                f"Longest SMA = {max(self.sma_periods)} periods; "
                "request more historical data."
            )

        return df

    @staticmethod
    def _add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Append a Simple Moving Average column ``SMA_{period}`` to *df*.

        SMA_n(t) = mean(Close[t-n+1 : t+1])

        The first ``period - 1`` rows will be NaN (insufficient history).
        These are handled by the final ``dropna()`` in :meth:`_add_indicators`.

        Args:
            df:     OHLCV DataFrame (must contain a ``Close`` column).
            period: Rolling window size in bars.

        Returns:
            The same DataFrame with ``SMA_{period}`` appended in-place.
        """
        col_name = f"SMA_{period}"
        df[col_name] = df["Close"].rolling(window=period, min_periods=period).mean()
        return df

    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Append RSI and its intermediate series to *df*.

        Implementation uses **Wilder's Smoothed Moving Average** via
        ``pandas.Series.ewm(com=period-1)``, which is algebraically
        identical to the recursive Wilder formula and matches the output
        of TradingView and Bloomberg.

        Columns added
        ~~~~~~~~~~~~~
        ``RSI_{period}``  — Relative Strength Index [0, 100]

        Maths
        ~~~~~
        ::

            delta   = Close.diff()
            gain    = delta.clip(lower=0)
            loss    = (-delta).clip(lower=0)

            avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period-1, min_periods=period).mean()

            RS  = avg_gain / avg_loss
            RSI = 100 − (100 / (1 + RS))

        Edge cases
        ~~~~~~~~~~
        When ``avg_loss == 0`` (all gains, no losses) RS is infinite and
        RSI should be 100.  We handle this via ``np.where``.

        Args:
            df:     OHLCV (+ SMA) DataFrame.
            period: Look-back window (default 14).

        Returns:
            DataFrame with ``RSI_{period}`` appended.
        """
        col_name = f"RSI_{period}"
        close = df["Close"]

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing: alpha = 1/period  →  com = period - 1
        avg_gain: pd.Series = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss: pd.Series = loss.ewm(com=period - 1, min_periods=period).mean()

        # Prevent division-by-zero: where avg_loss == 0, RSI is 100
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = np.where(np.isinf(rs), 100.0, 100.0 - (100.0 / (1.0 + rs)))

        df[col_name] = rsi
        # Restore NaN for warm-up rows so dropna() cleans them correctly
        df.loc[close.diff().isna() | avg_gain.isna(), col_name] = np.nan

        return df

    @staticmethod
    def _add_macd(df: pd.DataFrame, params: MACDParams) -> pd.DataFrame:
        """
        Append MACD line, signal line, and histogram columns to *df*.

        The standard MACD is based on **exponential** moving averages, not
        simple ones.  ``adjust=False`` mimics the recursive EMA formula
        used by most trading platforms.

        Columns added
        ~~~~~~~~~~~~~
        ``MACD``           — Fast EMA − Slow EMA
        ``MACD_Signal``    — *signal*-period EMA of the MACD line
        ``MACD_Histogram`` — MACD − Signal  (positive = bullish momentum)

        Maths
        ~~~~~
        ::

            EMA_fast = Close.ewm(span=fast, adjust=False).mean()
            EMA_slow = Close.ewm(span=slow, adjust=False).mean()
            MACD     = EMA_fast − EMA_slow
            Signal   = MACD.ewm(span=signal, adjust=False).mean()
            Hist     = MACD − Signal

        Args:
            df:     OHLCV (+ SMA + RSI) DataFrame.
            params: :class:`MACDParams` with fast / slow / signal periods.

        Returns:
            DataFrame with ``MACD``, ``MACD_Signal``, ``MACD_Histogram`` appended.
        """
        close = df["Close"]

        ema_fast: pd.Series = close.ewm(span=params.fast, adjust=False).mean()
        ema_slow: pd.Series = close.ewm(span=params.slow, adjust=False).mean()

        macd_line: pd.Series = ema_fast - ema_slow
        signal_line: pd.Series = macd_line.ewm(span=params.signal, adjust=False).mean()
        histogram: pd.Series = macd_line - signal_line

        df["MACD"] = macd_line
        df["MACD_Signal"] = signal_line
        df["MACD_Histogram"] = histogram

        return df

    @staticmethod
    def _add_bollinger(df: pd.DataFrame, period: int = 20, stdev: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands: SMA ± N·σ over `period` bars."""
        close = df["Close"]
        mid   = close.rolling(window=period, min_periods=period).mean()
        sd    = close.rolling(window=period, min_periods=period).std()
        df[f"BB_Mid_{period}"]   = mid
        df[f"BB_Upper_{period}"] = mid + stdev * sd
        df[f"BB_Lower_{period}"] = mid - stdev * sd
        # Band width as % of mid — useful for squeeze detection
        df[f"BB_Width_{period}"] = (df[f"BB_Upper_{period}"] - df[f"BB_Lower_{period}"]) / mid * 100.0
        return df

    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range (Wilder). Volatility in absolute price units."""
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        # Wilder's smoothing
        df[f"ATR_{period}"] = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        return df

    @staticmethod
    def _add_vwap(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Rolling VWAP over the last `period` bars — intraday-friendly."""
        typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vol = df["Volume"].astype(float)
        pv  = typical * vol
        num = pv.rolling(window=period, min_periods=period).sum()
        den = vol.rolling(window=period, min_periods=period).sum()
        df[f"VWAP_{period}"] = num / den.replace(0, np.nan)
        return df

    @staticmethod
    def _add_support_resistance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Local support/resistance — rolling min(Low) / max(High) over `lookback`."""
        df[f"Support_{lookback}"]    = df["Low"].rolling(window=lookback, min_periods=lookback).min()
        df[f"Resistance_{lookback}"] = df["High"].rolling(window=lookback, min_periods=lookback).max()
        return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(value: str | date | datetime) -> date:
    """Coerce a string, date, or datetime to :class:`datetime.date`."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # string — try ISO format first, then common US format
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date string {value!r}. "
        "Expected ISO format: YYYY-MM-DD."
    )


# ---------------------------------------------------------------------------
# __main__ — quick smoke-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from datetime import date, timedelta

    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint

    console = Console()
    fetcher = MarketDataFetcher()

    TICKERS = ["QQQ", "AAPL"]
    LOOKBACK = 420  # calendar days — gives ~80 clean rows after SMA_200 warm-up

    for symbol in TICKERS:
        console.rule(f"[bold cyan]{symbol}[/bold cyan]")

        try:
            snap = fetcher.fetch_latest(symbol, lookback_days=LOOKBACK)
        except RuntimeError as e:
            rprint(f"[bold red]ERROR:[/bold red] {e}")
            continue

        # ---------------------------------------------------------------- #
        # Build a Rich table for the last 5 rows
        # ---------------------------------------------------------------- #
        tail = snap.data.tail(5).copy()

        # Round for display
        price_cols = ["Open", "High", "Low", "Close"]
        sma_cols = [f"SMA_{p}" for p in snap.sma_periods]
        rsi_col = f"RSI_{snap.rsi_period}"
        float_cols = price_cols + sma_cols + [rsi_col, "MACD", "MACD_Signal", "MACD_Histogram"]

        table = Table(
            title=f"{symbol}  —  Last 5 trading days  "
                  f"(total clean rows: {snap.row_count})",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Date", style="dim", no_wrap=True)
        table.add_column("Close", justify="right")
        for c in sma_cols:
            table.add_column(c, justify="right")
        table.add_column(rsi_col, justify="right")
        table.add_column("MACD", justify="right")
        table.add_column("MACD_Signal", justify="right")
        table.add_column("MACD_Hist", justify="right")

        for idx, row in tail.iterrows():
            date_str = str(idx.date())  # type: ignore[union-attr]

            # Colour RSI: green if oversold (<30), red if overbought (>70)
            rsi_val = row[rsi_col]
            if rsi_val < 30:
                rsi_str = f"[bold green]{rsi_val:.2f}[/bold green]"
            elif rsi_val > 70:
                rsi_str = f"[bold red]{rsi_val:.2f}[/bold red]"
            else:
                rsi_str = f"{rsi_val:.2f}"

            # Colour MACD histogram: green = positive momentum, red = negative
            hist = row["MACD_Histogram"]
            hist_str = (
                f"[green]{hist:.4f}[/green]"
                if hist >= 0
                else f"[red]{hist:.4f}[/red]"
            )

            table.add_row(
                date_str,
                f"{row['Close']:.2f}",
                *[f"{row[c]:.2f}" for c in sma_cols],
                rsi_str,
                f"{row['MACD']:.4f}",
                f"{row['MACD_Signal']:.4f}",
                hist_str,
            )

        console.print(table)

        # ---------------------------------------------------------------- #
        # Print a quick signal summary based on the latest bar
        # ---------------------------------------------------------------- #
        latest = snap.latest
        close = latest["Close"]
        sma20 = latest["SMA_20"]
        sma50 = latest["SMA_50"]
        rsi = latest[rsi_col]
        macd = latest["MACD"]
        signal = latest["MACD_Signal"]

        trend = "UPTREND" if close > sma20 > sma50 else (
            "DOWNTREND" if close < sma20 < sma50 else "MIXED"
        )
        rsi_zone = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL")
        macd_cross = "BULLISH CROSS" if macd > signal else "BEARISH CROSS"

        summary = (
            f"  Trend   : [bold]{'[green]' + trend + '[/green]' if trend == 'UPTREND' else '[red]' + trend + '[/red]' if trend == 'DOWNTREND' else '[yellow]' + trend + '[/yellow]'}[/bold]\n"
            f"  RSI     : {rsi:.2f}  →  [bold]{rsi_zone}[/bold]\n"
            f"  MACD    : {macd:.4f}  vs  Signal {signal:.4f}  →  [bold]{macd_cross}[/bold]"
        )
        console.print(Panel(summary, title=f"[bold]{symbol} Signal Snapshot[/bold]", expand=False))
        console.print()
