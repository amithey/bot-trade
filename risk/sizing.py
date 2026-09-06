"""Pure order-sizing policy shared by all live strategy modes.

The ceiling applies to total symbol exposure, including existing holdings.
An analyst's smaller suggestion is never rounded up to a profile minimum.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class BuyAllocation:
    cash_amount: float
    equity_pct: float
    reason: str


def allocate_buy(*, equity: float, cash: float, existing_value: float,
                 user_cap_pct: float, profile_cap_pct: float,
                 suggested_pct: float | None, fee_rate: float) -> BuyAllocation:
    values = (equity, cash, existing_value, user_cap_pct, profile_cap_pct, fee_rate)
    if not all(isfinite(v) for v in values):
        return BuyAllocation(0, 0, "Non-finite sizing input")
    if (equity <= 0 or cash <= 0 or existing_value < 0 or
            not 0 <= fee_rate < .1 or not 0 < user_cap_pct <= 100 or
            not 0 < profile_cap_pct <= 100):
        return BuyAllocation(0, 0, "Invalid sizing input or no available cash")
    ceiling = min(user_cap_pct, profile_cap_pct)
    requested = ceiling if suggested_pct is None else suggested_pct
    if not isfinite(requested) or requested <= 0 or requested > 100:
        return BuyAllocation(0, 0, "Invalid analyst position size")
    # Leave room for entry fees, which reduce post-trade equity.
    headroom = max(0., (equity * ceiling / 100 - existing_value)
                   / (1 + fee_rate * ceiling / 100))
    amount = min(equity * requested / 100, headroom, cash / (1 + fee_rate))
    if amount < .01:
        return BuyAllocation(0, 0, "Symbol exposure limit reached or cash exhausted")
    return BuyAllocation(amount, amount / equity * 100, "Within exposure and cash limits")
