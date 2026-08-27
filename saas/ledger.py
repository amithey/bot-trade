"""
UsageLedger — durable, per-account record of every Claude call and its cost.

Two jobs:

* **Enforcement.**  Platform-funded (trial) usage draws against a monthly
  budget; :meth:`month_spend` is what :func:`saas.plans.resolve` reads to
  decide whether a free user still has funding.
* **Transparency.**  A BYOK subscriber is spending their own money through
  your platform, so they are entitled to see exactly what it cost, broken
  down by model, strategy mode and symbol.  A bot that quietly burns someone
  else's key without showing the bill does not survive contact with users.

Storage is SQLite at ``data/usage.db`` — no server to run, survives restarts,
and handles the write rate of a trading loop comfortably.  WAL mode plus a
process-level lock keeps the background engine threads and the Streamlit
request threads from stepping on each other.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from saas.pricing import cost_usd

_DEFAULT_DB = Path("./data/usage.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id   TEXT PRIMARY KEY,
    plan_id      TEXT NOT NULL DEFAULT 'FREE',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    note         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS usage_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,
    day                TEXT    NOT NULL,
    month              TEXT    NOT NULL,
    account_id         TEXT    NOT NULL,
    funding            TEXT    NOT NULL,
    model              TEXT    NOT NULL DEFAULT '',
    mode               TEXT    NOT NULL DEFAULT '',
    ticker             TEXT    NOT NULL DEFAULT '',
    input_tokens       INTEGER NOT NULL DEFAULT 0,
    output_tokens      INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd           REAL    NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS ix_usage_account_month
    ON usage_events (account_id, month);
CREATE INDEX IF NOT EXISTS ix_usage_account_ts
    ON usage_events (account_id, ts);

-- Calls the shared decision cache avoided, so the operator can prove the
-- saving rather than assert it.
CREATE TABLE IF NOT EXISTS savings_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    month       TEXT NOT NULL,
    account_id  TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT '',
    ticker      TEXT NOT NULL DEFAULT '',
    calls_saved INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ix_savings_account_month
    ON savings_events (account_id, month);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _month_key(d: Optional[date] = None) -> str:
    d = d or _now().date()
    return f"{d.year:04d}-{d.month:02d}"


@dataclass
class UsageRow:
    ts: str
    model: str
    mode: str
    ticker: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    funding: str

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_write_tokens)


class UsageLedger:
    """Thread-safe SQLite ledger of API spend."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._path = Path(db_path or _DEFAULT_DB)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, timeout=15.0,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- accounts ----------------------------------------------------------
    def ensure_account(self, account_id: str, plan_id: str = "FREE") -> str:
        now = _now().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO accounts "
                "(account_id, plan_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (account_id, plan_id, now, now),
            )
            self._conn.commit()
        return account_id

    def get_plan_id(self, account_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT plan_id FROM accounts WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        return row["plan_id"] if row else "FREE"

    def set_plan_id(self, account_id: str, plan_id: str) -> None:
        self.ensure_account(account_id, plan_id)
        with self._lock:
            self._conn.execute(
                "UPDATE accounts SET plan_id = ?, updated_at = ? "
                "WHERE account_id = ?",
                (plan_id, _now().isoformat(), account_id),
            )
            self._conn.commit()

    def list_accounts(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM accounts ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- recording ---------------------------------------------------------
    def record(
        self,
        account_id: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        funding: str = "PLATFORM",
        mode: str = "",
        ticker: str = "",
    ) -> float:
        """Persist one API call and return its cost in USD."""
        cost = cost_usd(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_events (ts, day, month, account_id, funding,"
                " model, mode, ticker, input_tokens, output_tokens,"
                " cache_read_tokens, cache_write_tokens, cost_usd)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    now.isoformat(), now.date().isoformat(), _month_key(now.date()),
                    account_id, funding, model, mode, ticker,
                    int(input_tokens), int(output_tokens),
                    int(cache_read_tokens), int(cache_write_tokens), float(cost),
                ),
            )
            self._conn.commit()
        return cost

    def record_saving(
        self,
        account_id: str,
        mode: str = "",
        ticker: str = "",
        calls_saved: int = 1,
    ) -> None:
        """Note that the shared cache served a decision instead of the API."""
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO savings_events (ts, month, account_id, mode,"
                " ticker, calls_saved) VALUES (?,?,?,?,?,?)",
                (now.isoformat(), _month_key(now.date()), account_id,
                 mode, ticker, int(calls_saved)),
            )
            self._conn.commit()

    # -- queries -----------------------------------------------------------
    def _scalar(self, sql: str, args: tuple, default: Any = 0.0) -> Any:
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
        value = row[0] if row else None
        return default if value is None else value

    def month_spend(
        self,
        account_id: str,
        funding: Optional[str] = None,
        month: Optional[str] = None,
    ) -> float:
        """Total spend this calendar month, optionally filtered by funding.

        Pass ``funding="PLATFORM"`` for budget enforcement — a user's own
        BYOK spend must never count against the operator's trial budget.
        """
        m = month or _month_key()
        if funding:
            return float(self._scalar(
                "SELECT SUM(cost_usd) FROM usage_events"
                " WHERE account_id = ? AND month = ? AND funding = ?",
                (account_id, m, funding),
            ))
        return float(self._scalar(
            "SELECT SUM(cost_usd) FROM usage_events"
            " WHERE account_id = ? AND month = ?",
            (account_id, m),
        ))

    def day_spend(self, account_id: str, day: Optional[str] = None) -> float:
        d = day or _now().date().isoformat()
        return float(self._scalar(
            "SELECT SUM(cost_usd) FROM usage_events"
            " WHERE account_id = ? AND day = ?",
            (account_id, d),
        ))

    def calls_saved(self, account_id: str, month: Optional[str] = None) -> int:
        m = month or _month_key()
        return int(self._scalar(
            "SELECT SUM(calls_saved) FROM savings_events"
            " WHERE account_id = ? AND month = ?",
            (account_id, m), default=0,
        ))

    def recent(self, account_id: str, limit: int = 50) -> list[UsageRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, model, mode, ticker, input_tokens, output_tokens,"
                " cache_read_tokens, cache_write_tokens, cost_usd, funding"
                " FROM usage_events WHERE account_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (account_id, int(limit)),
            ).fetchall()
        return [UsageRow(**dict(r)) for r in rows]

    def breakdown(
        self,
        account_id: str,
        by: str = "mode",
        month: Optional[str] = None,
    ) -> list[dict]:
        """Spend grouped by ``mode``, ``model`` or ``ticker`` for one month."""
        if by not in ("mode", "model", "ticker", "funding"):
            raise ValueError(f"cannot group usage by {by!r}")
        m = month or _month_key()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {by} AS label, COUNT(*) AS calls,"
                f" SUM(input_tokens + cache_read_tokens + cache_write_tokens)"
                f"   AS input_tokens,"
                f" SUM(output_tokens) AS output_tokens,"
                f" SUM(cost_usd) AS cost_usd"
                f" FROM usage_events WHERE account_id = ? AND month = ?"
                f" GROUP BY {by} ORDER BY cost_usd DESC",
                (account_id, m),
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self, account_id: str) -> dict:
        """Everything the usage page needs, in one call."""
        month = _month_key()
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS calls,"
                " SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok,"
                " SUM(cache_read_tokens) AS cread,"
                " SUM(cache_write_tokens) AS cwrite,"
                " SUM(cost_usd) AS cost"
                " FROM usage_events WHERE account_id = ? AND month = ?",
                (account_id, month),
            ).fetchone()
        d = dict(row) if row else {}
        return {
            "month":            month,
            "calls":            int(d.get("calls") or 0),
            "input_tokens":     int(d.get("in_tok") or 0),
            "output_tokens":    int(d.get("out_tok") or 0),
            "cache_read":       int(d.get("cread") or 0),
            "cache_write":      int(d.get("cwrite") or 0),
            "cost_usd":         float(d.get("cost") or 0.0),
            "today_usd":        self.day_spend(account_id),
            "platform_usd":     self.month_spend(account_id, funding="PLATFORM"),
            "byok_usd":         self.month_spend(account_id, funding="BYOK"),
            "calls_saved":      self.calls_saved(account_id),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_LEDGER: Optional[UsageLedger] = None
_LEDGER_GUARD = threading.Lock()


def get_ledger() -> UsageLedger:
    global _LEDGER
    if _LEDGER is None:
        with _LEDGER_GUARD:
            if _LEDGER is None:
                _LEDGER = UsageLedger()
    return _LEDGER
