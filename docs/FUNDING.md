# Paper-account funding

The editable capital preference did not update an existing portfolio. **Manage
funds** replaces it on Live, Portfolio and Settings with three explicit actions:

- **Deposit:** add cash and increase net capital by the amount entered.
- **Withdraw:** remove available cash and reduce net capital. Positions cannot
  fund a withdrawal automatically; the remaining net capital must stay positive.
- **Reset account:** archive the old account and start again with the chosen
  balance. Requires explicit confirmation, no open positions, and a fully stopped
  agent. It does not silently liquidate positions or restart trading.

Deposits and withdrawals preserve trades, positions, protective stops and
realized P&L. For example, $200,000 of capital with $1,009.93 of trading losses
has $198,990.07 cash when no positions are open. Depositing $50,000 produces
$250,000 net capital and $248,990.07 cash; the loss remains $1,009.93.

Cash transfers are journaled separately. Total return is trading P&L divided by
current net capital. Daily risk limits use the day's original opening equity,
with external cash flows excluded from daily P&L. Historical analytics and the
live equity chart are rebased to current net capital so funding changes do not
appear as trading gains or losses. These rebased percentages are not a
time-weighted or money-weighted investment return.

Schema **5** adds a funding journal and daily cash-flow metadata. Schemas 1–4
remain loadable. Older app versions cannot load schema 5. Transfers are locked
against trades and concurrent browser sessions, saved immediately, and rolled
back in memory if saving fails. In-flight strategy orders are invalidated after
a cash transfer. The active engine and every page share the same wallet object.

Reset archives are stored under `data/portfolios/archive/` on the existing
persistent volume. No live funds are moved: this remains a paper account.

The old unsynchronized capital preference is not automatically treated as a
deposit or reset. Users explicitly choose an operation and amount in Manage funds.
