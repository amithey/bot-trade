"""
Market Hours Logic
==================
Handles US/Eastern equity session checks and 24/7 crypto detection.

Crypto tickers (those containing "-USD", e.g. BTC-USD, ETH-USD, SOL-USD)
trade continuously on decentralised exchanges — they have no market hours.
Pass the ticker to ``get_market_status`` and it will short-circuit equity
hours logic and return OPEN immediately.
"""

from datetime import datetime, time
import pytz

_ET = pytz.timezone("US/Eastern")

# Any ticker whose symbol contains this suffix is treated as crypto.
_CRYPTO_SUFFIX = "-USD"


def _is_crypto(ticker: str) -> bool:
    return _CRYPTO_SUFFIX in ticker.upper()


def get_market_status(ticker: str = "") -> tuple[bool, str]:
    """
    Return ``(is_open, status_message)`` for the given ticker.

    Crypto assets (ticker contains "-USD") are always open — the function
    returns immediately without consulting calendar or clock.

    For equities, checks the standard US regular session:
    Mon–Fri, 09:30–16:00 US/Eastern.

    Args:
        ticker: Asset symbol.  If empty, assumes an equity.

    Returns:
        (is_open, status_message)
    """
    if _is_crypto(ticker):
        return True, "OPEN (24/7 Crypto)"

    now_utc = datetime.now(pytz.utc)
    now_et  = now_utc.astimezone(_ET)

    # Day of week: 0=Mon, 6=Sun
    weekday      = now_et.weekday()
    current_time = now_et.time()

    if weekday >= 5:
        day_name = now_et.strftime("%A")
        return False, f"CLOSED (Weekend — {day_name})"

    market_open  = time(9, 30)
    market_close = time(16, 0)

    if current_time < market_open:
        return False, "CLOSED (Pre-Market — opens 09:30 ET)"
    elif current_time >= market_close:
        return False, "CLOSED (After-Hours — closed 16:00 ET)"

    # Calculate time remaining in session
    closing_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    delta      = closing_dt - now_et
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes, _       = divmod(remainder, 60)

    return True, f"OPEN ({hours}h {minutes}m remaining)"


def is_market_open(ticker: str = "") -> bool:
    """Simple boolean check — passes ticker through for crypto detection."""
    is_open, _ = get_market_status(ticker)
    return is_open
