"""Read-only, offline diagnosis: python -m tools.audit_portfolio PORTFOLIO.json.

Outputs JSON to stdout. Exit fills are explicitly distinguished from complete
round trips. No market requests, account writes or order execution occur.
"""
import argparse
import json
from math import isfinite
from pathlib import Path

from portfolio.accounting import migrate_entry_fees


def audit_portfolio(raw: dict) -> dict:
    version = raw.get("schema_version", 1)
    if version not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported portfolio schema: {version}")
    data = migrate_entry_fees(raw) if version < 3 else raw
    trades = data.get("trade_log", [])
    positions = data.get("positions", {})
    exits = [t for t in trades if t["action"] in ("SELL", "COVER", "FORCE_CLOSE")]
    entry_fees = sum(float(t["fee"]) for t in trades if t["action"] in ("BUY", "SHORT"))
    remaining_fees = sum(float(p.get("entry_fees", 0)) for p in positions.values())
    exit_fees = sum(float(t["fee"]) for t in exits)
    closed_fees = entry_fees - remaining_fees + exit_fees
    net_realized = sum(float(t["realized_pnl"]) for t in exits)
    open_gross, marks = 0., 0.
    for p in positions.values():
        qty, entry, price = (float(p[k]) for k in ("quantity", "avg_entry_price", "current_price"))
        short = p.get("side", "LONG") == "SHORT"
        pnl = qty * (entry - price if short else price - entry)
        open_gross += pnl
        marks += qty * entry + pnl if short else qty * price
    initial, cash = float(data["initial_capital"]), float(data["cash"])
    equity = cash + marks
    net_open = open_gross - remaining_fees
    by_symbol = {}
    for tr in exits:
        item = by_symbol.setdefault(tr["ticker"], {"exit_fills": 0, "net_realized_pnl": 0.})
        item["exit_fills"] += 1
        item["net_realized_pnl"] += float(tr["realized_pnl"])
    reconciliation_error = (equity - initial) - (net_realized + net_open)
    realized_error = float(data.get("realized_pnl", 0)) - net_realized
    if not all(isfinite(n) for n in (equity, net_realized, net_open, closed_fees, realized_error)):
        raise ValueError("Non-finite portfolio accounting input")
    notes = [
        "Exit fills can include partial sales; counts and win rate are not complete round trips.",
        "Open positions use saved marks; these are not verified current prices.",
        "Slippage is embedded in fills and cannot be separated without reference quotes. API costs are excluded.",
        "This journal alone cannot establish predictive edge or diagnose rejected signals.",
    ]
    if not trades:
        notes.insert(0, "No recorded trades: the supplied file cannot explain a trading loss.")
    if abs(reconciliation_error) > .01 or abs(realized_error) > .01:
        notes.insert(0, "ACCOUNTING MISMATCH: do not trust performance totals until reconciled.")
    return {
        "saved_at": data.get("saved_at"), "source_schema": version,
        "orders": len(trades), "exit_fills": len(exits),
        "first_execution": min((t["executed_at"] for t in trades), default=None),
        "last_execution": max((t["executed_at"] for t in trades), default=None),
        "initial_capital": initial, "saved_equity": equity,
        "total_pnl": equity - initial, "net_realized_pnl": net_realized,
        "net_open_pnl": net_open, "closed_trade_commissions": closed_fees,
        "gross_realized_before_commissions": net_realized + closed_fees,
        "all_paid_commissions": entry_fees + exit_fees,
        "net_winning_exit_fills": sum(float(t["realized_pnl"]) > 0 for t in exits),
        "net_losing_exit_fills": sum(float(t["realized_pnl"]) < 0 for t in exits),
        "mean_net_pnl_per_exit_fill": net_realized / len(exits) if exits else None,
        "equity_reconciliation_error": reconciliation_error,
        "realized_reconciliation_error": realized_error,
        "by_symbol": by_symbol, "limitations": notes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", type=Path)
    args = parser.parse_args()
    try:
        raw = json.loads(args.portfolio.read_text(encoding="utf-8"))
        report = audit_portfolio(raw)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
