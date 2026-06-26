"""
Virtual Portfolio Manager — Phase 7
=====================================

Simulates a real brokerage account without connecting to any live exchange.
All positions, cash, and trade history are stored in-memory and persisted to
a JSON file so the portfolio survives application restarts.

Architecture
------------
``LivePortfolio``
    The central stateful object.  One instance per virtual account.
    Thread-safe via ``threading.Lock`` (safe for Streamlit reruns).

``Position``
    Tracks a single open long position: quantity, average cost basis,
    current mark-to-market price, and derived P&L metrics.

``TradeRecord``
    Immutable record of every executed BUY, SELL, or FORCE_CLOSE.
    Serialised to JSON for the audit log.

``DailySnapshot``
    Portfolio value captured at the start of each calendar day, used to
    compute intra-day P&L %.

Typical usage
-------------
    from portfolio.virtual_account import LivePortfolio

    port = LivePortfolio(initial_capital=25_000.0)

    # Execute a buy (all-in on QQQ)
    port.buy("QQQ", price=415.20, reasoning="RSI oversold + MACD bullish")

    # Update prices after market close
    port.update_price("QQQ", 418.50)

    # Inspect
    summary = port.get_summary()
    print(summary["total_value"])       # current portfolio value
    print(summary["daily_pnl_pct"])     # intraday gain/loss %
    print(summary["unrealized_pnl"])    # open position gain/loss $

    # Close position
    port.sell("QQQ", price=418.50, reasoning="Trailing stop hit")

    # Persist
    port.save("data/virtual_portfolio.json")
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger, log_trade_execution

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Position:
    """
    A single open long position.

    Attributes
    ----------
    ticker:           Asset symbol.
    quantity:         Shares held (supports fractional shares).
    avg_entry_price:  Weighted-average cost basis per share.
    current_price:    Most recent mark-to-market price.
    opened_at:        UTC datetime when the position was first opened.
    """

    ticker: str
    quantity: float
    avg_entry_price: float
    current_price: float
    opened_at: datetime = field(default_factory=datetime.utcnow)

    # ── Derived metrics ───────────────────────────────────────────────────────

    @property
    def market_value(self) -> float:
        """Current mark-to-market value of the entire position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total amount originally invested (ignoring fees for simplicity)."""
        return self.quantity * self.avg_entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Absolute unrealised gain / loss in USD."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealised gain / loss as a percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["opened_at"] = self.opened_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        d["opened_at"] = datetime.fromisoformat(d["opened_at"])
        return cls(**d)


@dataclass(frozen=True)
class TradeRecord:
    """
    Immutable record of a single executed order.

    Attributes
    ----------
    executed_at:     UTC datetime of execution.
    action:          ``"BUY"``, ``"SELL"``, or ``"FORCE_CLOSE"``.
    ticker:          Asset symbol.
    quantity:        Shares transacted.
    price:           Execution price per share.
    gross_value:     ``quantity × price`` before fees.
    fee:             Commission charged (USD).
    net_value:       ``gross_value ± fee`` (positive for BUY cost, negative for SELL proceeds).
    realized_pnl:    Profit / loss realised on this trade (0.0 for BUY).
    cash_after:      Cash balance immediately after execution.
    portfolio_value: Total portfolio value (cash + open positions) after execution.
    reasoning:       AI reasoning snippet that triggered this order.
    """

    executed_at: datetime
    action: str
    ticker: str
    quantity: float
    price: float
    gross_value: float
    fee: float
    net_value: float
    realized_pnl: float
    cash_after: float
    portfolio_value: float
    reasoning: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["executed_at"] = self.executed_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        d = dict(d)
        d["executed_at"] = datetime.fromisoformat(d["executed_at"])
        return cls(**d)


@dataclass
class DailySnapshot:
    """Portfolio value recorded at the open of each calendar day."""

    snapshot_date: date
    portfolio_value: float

    def to_dict(self) -> dict:
        return {
            "snapshot_date": self.snapshot_date.isoformat(),
            "portfolio_value": self.portfolio_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DailySnapshot":
        return cls(
            snapshot_date=date.fromisoformat(d["snapshot_date"]),
            portfolio_value=float(d["portfolio_value"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Live Portfolio
# ─────────────────────────────────────────────────────────────────────────────


class LivePortfolio:
    """
    Simulated brokerage account for virtual live trading.

    All monetary values are in USD.  The portfolio supports a single
    long-only position per ticker (buying into an existing position
    increases the average cost basis via a weighted average).

    Parameters
    ----------
    initial_capital:
        Starting cash balance in USD.
    fee_rate:
        Commission charged per trade as a fraction of gross value
        (default 0.001 = 0.1 %).
    name:
        Human-readable label for the portfolio (shown in reports).
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,
        name: str = "BotTrade Virtual Account",
    ) -> None:
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
        if not (0.0 <= fee_rate < 0.10):
            raise ValueError(f"fee_rate must be in [0, 0.10), got {fee_rate}")

        self.name: str = name
        self.initial_capital: float = initial_capital
        self.fee_rate: float = fee_rate

        self._cash: float = initial_capital
        self._positions: dict[str, Position] = {}
        self._trade_log: list[TradeRecord] = []
        self._daily_snapshots: list[DailySnapshot] = []
        self._realized_pnl: float = 0.0

        self._lock = threading.Lock()

        # Record the opening-day snapshot
        self._record_daily_snapshot_if_new()

        logger.info(
            "LivePortfolio '%s' initialised — capital=$%.2f, fee_rate=%.3f%%",
            self.name, initial_capital, fee_rate * 100,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Core read properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        """Current uninvested cash balance."""
        with self._lock:
            return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        """Shallow copy of the open positions dict (safe to iterate)."""
        with self._lock:
            return dict(self._positions)

    @property
    def trade_log(self) -> list[TradeRecord]:
        """Ordered list of all executed trades (oldest first)."""
        with self._lock:
            return list(self._trade_log)

    # ─────────────────────────────────────────────────────────────────────────
    # Computed metrics
    # ─────────────────────────────────────────────────────────────────────────

    def get_total_value(self) -> float:
        """Cash balance + sum of all open position market values."""
        with self._lock:
            return self._cash + sum(p.market_value for p in self._positions.values())

    def get_unrealized_pnl(self) -> float:
        """Sum of unrealised P&L across all open positions (USD)."""
        with self._lock:
            return sum(p.unrealized_pnl for p in self._positions.values())

    def get_realized_pnl(self) -> float:
        """Cumulative realised profit / loss on all closed trades (USD)."""
        with self._lock:
            return self._realized_pnl

    def get_total_return_pct(self) -> float:
        """(current_value − initial_capital) / initial_capital × 100."""
        total = self.get_total_value()
        return (total - self.initial_capital) / self.initial_capital * 100.0

    def get_daily_pnl(self) -> float:
        """
        Intraday absolute P&L in USD:
        ``current_value − today's_opening_value``.

        Returns ``0.0`` if no snapshot is available for today.
        """
        today_value = self.get_total_value()
        today = date.today()
        with self._lock:
            snaps_today = [
                s for s in self._daily_snapshots
                if s.snapshot_date == today
            ]
        if not snaps_today:
            return 0.0
        return today_value - snaps_today[0].portfolio_value

    def get_daily_pnl_pct(self) -> float:
        """
        Intraday P&L as a percentage of today's opening portfolio value.

        Returns ``0.0`` if no snapshot is available for today.
        """
        today = date.today()
        with self._lock:
            snaps_today = [
                s for s in self._daily_snapshots
                if s.snapshot_date == today
            ]
        if not snaps_today:
            return 0.0
        start_val = snaps_today[0].portfolio_value
        if start_val == 0:
            return 0.0
        return self.get_daily_pnl() / start_val * 100.0

    # ─────────────────────────────────────────────────────────────────────────
    # Trade execution
    # ─────────────────────────────────────────────────────────────────────────

    def buy(
        self,
        ticker: str,
        price: float,
        quantity: Optional[float] = None,
        cash_amount: Optional[float] = None,
        reasoning: str = "",
    ) -> TradeRecord:
        """
        Execute a BUY order.

        Exactly one of *quantity* or *cash_amount* must be provided.
        If neither is given the entire available cash balance is used
        (all-in position, matching the backtest model).

        Parameters
        ----------
        ticker:       Asset symbol (e.g. ``"QQQ"``).
        price:        Execution price per share.
        quantity:     Number of shares to buy.  Fractional shares allowed.
        cash_amount:  USD amount to spend (converted to shares at *price*).
                      Overrides *quantity* if both are given.
        reasoning:    AI reasoning text snippet for the audit log.

        Returns
        -------
        TradeRecord
            The immutable record of this execution.

        Raises
        ------
        ValueError
            If *price* ≤ 0, insufficient cash, or ticker is empty.
        """
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker must not be empty.")
        if price <= 0:
            raise ValueError(f"Execution price must be > 0, got {price}.")

        with self._lock:
            self._record_daily_snapshot_if_new()

            available_cash = self._cash

            # Determine quantity
            if cash_amount is not None:
                spend = min(float(cash_amount), available_cash)
            elif quantity is not None:
                spend = float(quantity) * price
            else:
                spend = available_cash  # all-in

            if spend <= 0 or available_cash < spend * 0.001:
                raise ValueError(
                    f"Insufficient cash to BUY {ticker}: "
                    f"need ${spend:.2f}, have ${available_cash:.2f}."
                )

            fee = spend * self.fee_rate
            total_cost = spend + fee
            if total_cost > available_cash:
                # Reduce position size so fees fit within available cash
                spend = available_cash / (1.0 + self.fee_rate)
                fee = spend * self.fee_rate
                total_cost = spend + fee

            qty = spend / price

            # Update or create position (weighted-average cost basis)
            if ticker in self._positions:
                existing = self._positions[ticker]
                new_qty = existing.quantity + qty
                new_avg = (
                    (existing.avg_entry_price * existing.quantity + price * qty)
                    / new_qty
                )
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=new_qty,
                    avg_entry_price=new_avg,
                    current_price=price,
                    opened_at=existing.opened_at,
                )
            else:
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=qty,
                    avg_entry_price=price,
                    current_price=price,
                )

            self._cash -= total_cost
            portfolio_val = self._cash + sum(
                p.market_value for p in self._positions.values()
            )

            record = TradeRecord(
                executed_at=datetime.utcnow(),
                action="BUY",
                ticker=ticker,
                quantity=qty,
                price=price,
                gross_value=spend,
                fee=fee,
                net_value=total_cost,
                realized_pnl=0.0,
                cash_after=self._cash,
                portfolio_value=portfolio_val,
                reasoning=reasoning[:300],
            )
            self._trade_log.append(record)

        log_trade_execution(
            logger,
            action="BUY",
            ticker=ticker,
            price=price,
            quantity=qty,
            cash_after=self._cash,
            portfolio_value=portfolio_val,
            reasoning_snippet=reasoning,
        )
        return record

    def sell(
        self,
        ticker: str,
        price: float,
        quantity: Optional[float] = None,
        action_label: str = "SELL",
        reasoning: str = "",
    ) -> TradeRecord:
        """
        Execute a SELL order.

        Parameters
        ----------
        ticker:       Asset symbol.
        price:        Execution price per share.
        quantity:     Shares to sell.  ``None`` → liquidate the entire position.
        action_label: Label stored in the trade record (``"SELL"`` or
                      ``"FORCE_CLOSE"``).
        reasoning:    AI reasoning snippet.

        Returns
        -------
        TradeRecord

        Raises
        ------
        ValueError
            If the ticker has no open position, or *quantity* exceeds holdings.
        """
        ticker = ticker.strip().upper()
        if price <= 0:
            raise ValueError(f"Execution price must be > 0, got {price}.")

        with self._lock:
            if ticker not in self._positions:
                raise ValueError(
                    f"Cannot SELL {ticker}: no open position exists."
                )

            self._record_daily_snapshot_if_new()

            position = self._positions[ticker]
            qty = float(quantity) if quantity is not None else position.quantity

            if qty > position.quantity + 1e-9:
                raise ValueError(
                    f"Cannot sell {qty:.4f} shares of {ticker}: "
                    f"only {position.quantity:.4f} held."
                )

            gross_proceeds = qty * price
            fee = gross_proceeds * self.fee_rate
            net_proceeds = gross_proceeds - fee

            # Realised P&L = (sell_price - avg_cost) × qty − fees
            realised = (price - position.avg_entry_price) * qty - fee
            self._realized_pnl += realised

            self._cash += net_proceeds

            # Remove or reduce position
            remaining_qty = position.quantity - qty
            if remaining_qty < 1e-9:
                del self._positions[ticker]
            else:
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=remaining_qty,
                    avg_entry_price=position.avg_entry_price,
                    current_price=price,
                    opened_at=position.opened_at,
                )

            portfolio_val = self._cash + sum(
                p.market_value for p in self._positions.values()
            )

            record = TradeRecord(
                executed_at=datetime.utcnow(),
                action=action_label,
                ticker=ticker,
                quantity=qty,
                price=price,
                gross_value=gross_proceeds,
                fee=fee,
                net_value=net_proceeds,
                realized_pnl=realised,
                cash_after=self._cash,
                portfolio_value=portfolio_val,
                reasoning=reasoning[:300],
            )
            self._trade_log.append(record)

        log_trade_execution(
            logger,
            action=action_label,
            ticker=ticker,
            price=price,
            quantity=qty,
            cash_after=self._cash,
            portfolio_value=portfolio_val,
            reasoning_snippet=reasoning,
        )
        return record

    def force_close(self, ticker: str, price: float, reasoning: str = "") -> TradeRecord:
        """Liquidate an open position unconditionally (e.g. end-of-day)."""
        return self.sell(
            ticker=ticker,
            price=price,
            quantity=None,
            action_label="FORCE_CLOSE",
            reasoning=reasoning,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Price updates
    # ─────────────────────────────────────────────────────────────────────────

    def update_price(self, ticker: str, price: float) -> None:
        """
        Update the mark-to-market price for a single ticker.

        No-op if the ticker has no open position (avoids KeyError).
        """
        ticker = ticker.strip().upper()
        with self._lock:
            if ticker in self._positions:
                pos = self._positions[ticker]
                self._positions[ticker] = Position(
                    ticker=pos.ticker,
                    quantity=pos.quantity,
                    avg_entry_price=pos.avg_entry_price,
                    current_price=price,
                    opened_at=pos.opened_at,
                )

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Batch-update mark-to-market prices.

        Parameters
        ----------
        prices:
            ``{ticker: latest_close_price}`` mapping.  Unknown tickers
            (no open position) are silently ignored.
        """
        for ticker, price in prices.items():
            self.update_price(ticker, price)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary / reporting
    # ─────────────────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """
        Return a serialisable snapshot of the entire portfolio state.

        Keys
        ----
        name, initial_capital, cash, total_value,
        unrealized_pnl, realized_pnl, total_return_pct,
        daily_pnl, daily_pnl_pct,
        open_positions  (list of position dicts),
        trade_count, last_trade_at
        """
        with self._lock:
            positions_snap = list(self._positions.values())
            trade_count = len(self._trade_log)
            last_trade = self._trade_log[-1].executed_at if self._trade_log else None

        total_val = self.get_total_value()
        unreal_pnl = self.get_unrealized_pnl()

        return {
            "name":              self.name,
            "initial_capital":   self.initial_capital,
            "cash":              round(self._cash, 4),
            "total_value":       round(total_val, 4),
            "unrealized_pnl":    round(unreal_pnl, 4),
            "realized_pnl":      round(self._realized_pnl, 4),
            "total_return_pct":  round(self.get_total_return_pct(), 4),
            "daily_pnl":         round(self.get_daily_pnl(), 4),
            "daily_pnl_pct":     round(self.get_daily_pnl_pct(), 4),
            "open_positions": [
                {
                    "ticker":              p.ticker,
                    "quantity":            round(p.quantity, 6),
                    "avg_entry_price":     round(p.avg_entry_price, 4),
                    "current_price":       round(p.current_price, 4),
                    "market_value":        round(p.market_value, 2),
                    "unrealized_pnl":      round(p.unrealized_pnl, 2),
                    "unrealized_pnl_pct":  round(p.unrealized_pnl_pct, 2),
                    "opened_at":           p.opened_at.isoformat(),
                }
                for p in positions_snap
            ],
            "trade_count":  trade_count,
            "last_trade_at": last_trade.isoformat() if last_trade else None,
        }

    def print_summary(self) -> None:
        """Print a Rich-formatted portfolio summary to the console."""
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        summary = self.get_summary()

        # KPI panel
        ret_color = "green" if summary["total_return_pct"] >= 0 else "red"
        dpnl_color = "green" if summary["daily_pnl_pct"] >= 0 else "red"

        kpi_lines = [
            f"[bold]Portfolio:[/bold]  {self.name}",
            f"[bold]Cash:[/bold]       ${summary['cash']:>12,.2f}",
            f"[bold]Total Value:[/bold] ${summary['total_value']:>11,.2f}",
            f"[bold]Total Return:[/bold] [{ret_color}]{summary['total_return_pct']:+.2f}%[/{ret_color}]",
            f"[bold]Daily P&L:[/bold]  [{dpnl_color}]{summary['daily_pnl_pct']:+.2f}%[/{dpnl_color}]  (${summary['daily_pnl']:+,.2f})",
            f"[bold]Unrealised:[/bold] ${summary['unrealized_pnl']:>10,.2f}",
            f"[bold]Realised:[/bold]   ${summary['realized_pnl']:>10,.2f}",
        ]
        console.print(Panel("\n".join(kpi_lines), title="[cyan]Virtual Portfolio[/cyan]", expand=False))

        # Open positions table
        if summary["open_positions"]:
            t = Table(title="Open Positions", box=box.ROUNDED, header_style="bold magenta")
            t.add_column("Ticker")
            t.add_column("Qty", justify="right")
            t.add_column("Avg Entry", justify="right")
            t.add_column("Current", justify="right")
            t.add_column("Market Value", justify="right")
            t.add_column("Unrealised P&L", justify="right")
            for p in summary["open_positions"]:
                color = "green" if p["unrealized_pnl"] >= 0 else "red"
                t.add_row(
                    p["ticker"],
                    f"{p['quantity']:.4f}",
                    f"${p['avg_entry_price']:.2f}",
                    f"${p['current_price']:.2f}",
                    f"${p['market_value']:,.2f}",
                    f"[{color}]${p['unrealized_pnl']:+,.2f} ({p['unrealized_pnl_pct']:+.2f}%)[/{color}]",
                )
            console.print(t)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence (JSON)
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Serialise the entire portfolio to a JSON file.

        The file is written atomically (temp → rename) so a crash mid-write
        cannot corrupt the saved state.

        Parameters
        ----------
        path:
            File path.  Parent directories are created if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            payload = {
                "schema_version":   2,
                "name":             self.name,
                "initial_capital":  self.initial_capital,
                "fee_rate":         self.fee_rate,
                "cash":             self._cash,
                "realized_pnl":     self._realized_pnl,
                "positions":        {k: v.to_dict() for k, v in self._positions.items()},
                "trade_log":        [r.to_dict() for r in self._trade_log],
                "daily_snapshots":  [s.to_dict() for s in self._daily_snapshots],
                "saved_at":         datetime.utcnow().isoformat(),
            }

        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

        logger.debug("Portfolio saved to '%s' (%d trades)", path, len(self._trade_log))

    @classmethod
    def load(cls, path: str | Path) -> "LivePortfolio":
        """
        Restore a ``LivePortfolio`` from a previously saved JSON file.

        Parameters
        ----------
        path:
            Path to the JSON file produced by :meth:`save`.

        Returns
        -------
        LivePortfolio
            A fully reconstructed portfolio instance.

        Raises
        ------
        FileNotFoundError:
            If *path* does not exist.
        ValueError:
            If the JSON schema version is unrecognised.
        """
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        version = raw.get("schema_version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unrecognised portfolio schema version: {version}")

        port = cls.__new__(cls)
        port.name             = raw["name"]
        port.initial_capital  = float(raw["initial_capital"])
        port.fee_rate         = float(raw.get("fee_rate", 0.001))
        port._cash            = float(raw["cash"])
        port._realized_pnl    = float(raw.get("realized_pnl", 0.0))
        port._positions       = {
            k: Position.from_dict(v) for k, v in raw.get("positions", {}).items()
        }
        port._trade_log       = [TradeRecord.from_dict(r) for r in raw.get("trade_log", [])]
        port._daily_snapshots = [DailySnapshot.from_dict(s) for s in raw.get("daily_snapshots", [])]
        port._lock            = threading.Lock()

        logger.info(
            "Portfolio '%s' loaded from '%s' — %d trades, cash=$%.2f",
            port.name, path, len(port._trade_log), port._cash,
        )
        return port

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _record_daily_snapshot_if_new(self) -> None:
        """
        Store today's opening portfolio value if we haven't done so yet.
        Called internally at the start of every buy/sell to ensure the
        daily-P&L baseline is always available.

        Must be called *within* ``self._lock``.
        """
        today = date.today()
        already_recorded = any(
            s.snapshot_date == today for s in self._daily_snapshots
        )
        if not already_recorded:
            value = self._cash + sum(p.market_value for p in self._positions.values())
            self._daily_snapshots.append(
                DailySnapshot(snapshot_date=today, portfolio_value=value)
            )
            logger.debug(
                "Daily snapshot recorded: date=%s, value=$%.2f", today, value
            )
