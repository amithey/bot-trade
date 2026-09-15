"""Legacy entry-fee allocation. Replays quantities, never executes orders."""
from copy import deepcopy
from math import isclose, isfinite


def migrate_entry_fees(payload: dict) -> dict:
    """Convert schema 1/2 to net realised P&L without changing cash or marks.

    Refuse incomplete journals: guessing an entry commission would silently
    corrupt both performance labels and the loss circuit breaker.
    """
    result = deepcopy(payload)
    inventory: dict[str, dict] = {}
    allocated_total = 0.0
    for trade in result.get("trade_log", []):
        action, ticker = trade["action"], trade["ticker"]
        qty, fee = float(trade["quantity"]), float(trade["fee"])
        if not (isfinite(qty) and qty > 0 and isfinite(fee) and fee >= 0):
            raise ValueError("Cannot migrate entry fees: invalid journal quantity or fee")
        if action in ("BUY", "SHORT"):
            pos = inventory.setdefault(ticker, {"quantity": 0., "fees": 0., "side": action})
            if pos["side"] != action or (action == "SHORT" and pos["quantity"]):
                raise ValueError("Cannot migrate entry fees: incompatible entry history")
            pos["quantity"] += qty
            pos["fees"] += fee
        elif action in ("SELL", "COVER", "FORCE_CLOSE"):
            pos = inventory.get(ticker)
            if pos is None or qty > pos["quantity"] + 1e-9:
                raise ValueError("Cannot migrate entry fees: incomplete entry history")
            if ((action == "SELL" and pos["side"] != "BUY")
                    or (action == "COVER" and pos["side"] != "SHORT")):
                raise ValueError("Cannot migrate entry fees: incompatible exit history")
            allocated = pos["fees"] * min(qty / pos["quantity"], 1.)
            trade["realized_pnl"] = float(trade["realized_pnl"]) - allocated
            allocated_total += allocated
            pos["quantity"] -= qty
            pos["fees"] -= allocated
            if pos["quantity"] < 1e-9:
                del inventory[ticker]
        else:
            raise ValueError(f"Cannot migrate entry fees: unsupported action {action}")
    positions = result.get("positions", {})
    if set(inventory) != set(positions):
        raise ValueError("Cannot migrate entry fees: journal and positions disagree")
    for ticker, pos in positions.items():
        replay = inventory[ticker]
        expected_side = "SHORT" if replay["side"] == "SHORT" else "LONG"
        if (not isclose(replay["quantity"], float(pos["quantity"]), rel_tol=1e-9, abs_tol=1e-9)
                or pos.get("side", "LONG") != expected_side):
            raise ValueError("Cannot migrate entry fees: journal and position quantities/sides disagree")
        pos["entry_fees"] = replay["fees"]
    result["realized_pnl"] = float(result.get("realized_pnl", 0.)) - allocated_total
    result["schema_version"] = 3
    return result
