# Local profit protection

This change implements deterministic trailing exits in the paper portfolio.
It has not been deployed and its thresholds have not been calibrated on the
user's trades. Synthetic demonstrations test execution mechanics only.

## Policy

- Track the highest observed quote for a long and lowest for a short.
- Arm after estimated net profit reaches **0.10% of entry notional**.
- Set the exit trigger to retain **50% of peak net profit**, with a **0.02%**
  minimum net-profit target. Ratchet toward profit; never loosen a saved trigger.
- Use actual remaining entry commissions, the entry fill, exit commission and
  configured exit slippage. Entry slippage is already included in the entry fill.
- Close the paper position at the currently observed quote with adverse modeled
  slippage. A price gap can produce a loss even after protection was armed.

These are initial test defaults in `risk/profit_protection.py`. The engine accepts
an explicit `ProfitProtectionConfig`; the UI currently uses the defaults.

## Execution and persistence

`LiveTradingEngine.start()` starts a separate profit monitor. It checks every
open ticker using price-only one-minute Yahoo Finance data, then waits ten
seconds between rounds. Direct chart requests are sequential, with two-second
connection and five-second read timeouts; this is polling, not a tick stream or
an exchange-native stop. No timezone/cookie database or fundamental lookup is
required by the protection feed. Network/DNS and response delays can extend a round.
Quotes older than two minutes or requests older than thirty seconds are rejected.
Unavailable quotes appear in the activity log and protection status.

The monitor does not wait for AI, RAG, the next strategy candle or permission to
open new trades. A decision already in flight is discarded if another trade
changed the portfolio; execution checks the revision again under its lock.
Existing fixed stops, daily guards and strategy exits continue alongside it.

Schema **4** saves the observed peak, ratcheted trigger, activation state, quote
time and pending exit. Concurrent saves are serialized. Partial long sales keep
the trigger and allocate remaining entry fees. Adding to an armed long is
rejected. A failed protective close is retried on the next valid quote even if
the price recovers. Reopened positions start fresh.

Schemas 1–3 can be loaded. No historical peak is invented for old positions:
tracking starts with quotes seen by the updated version. Keep a backup before
upgrading; older application versions cannot load schema 4.

**Stopping the bot or exiting the application stops monitoring.** Saved triggers
resume when the engine starts again. Polling can miss intrabar peaks or crossings;
gaps, stale data and outages prevent a guaranteed profit floor.

## Local checks

```powershell
python -m tools.simulate_profit_protection
python -m pytest -q tests/test_profit_protection.py tests/test_profit_monitor.py tests/test_profit_protection_ui.py
python -X utf8 -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8502
```

The simulation uses an isolated in-memory account and synthetic long, short and
gap paths; it does not touch saved customer accounts. The dashboard's **Profit
protection** table shows the observed peak, current trigger and activation state.

The existing committee backtest does not yet model this new monitor. Neither its
results nor this synthetic demo establish the new policy's historical or future
profitability. The supplied trade export lacks the price path needed to replay
actual peak-to-exit behavior.

## Validation on 2026-09-15

- Full local suite: **1,057 passed, 2 skipped**. One existing Windows CPU-count
  warning in the ML tests; joblib used logical cores instead.
- All 58 profit-protection tests passed, including a real background monitor
  closing while a fake AI call was blocked and rejecting its late BUY decision.
- UI component rendered successfully; local dashboard health returned HTTP 200.
- The direct feed returned a fresh public BTC-USD quote in a network-enabled check.
- Synthetic $1,000 positions closed at net +$7.49 (long), +$7.51 (short), and
  -$22.47 (gap through the stop). These intentionally chosen paths demonstrate
  mechanics and do not estimate expected returns.
- No deployment was performed. The local dashboard is available on port 8502;
  opening it does not automatically start the user's trading account.
