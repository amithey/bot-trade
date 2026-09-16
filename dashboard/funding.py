"""Explicit paper-account cash transfers and an archived reset."""
import math
from contextlib import nullcontext
from uuid import uuid4

import streamlit as st


def change_funds(operation, amount, *, confirmed=False):
    from dashboard import _shared as shared

    # Serialize cross-tab wallet operations even before an engine exists.
    _, guard = shared.portfolio_store()
    with guard:
        return _change_funds(operation, amount, confirmed=confirmed)


def _change_funds(operation, amount, *, confirmed=False):
    from dashboard import _shared as shared
    from portfolio.virtual_account import LivePortfolio

    if operation not in ("Deposit", "Withdraw", "Reset account"):
        raise ValueError("Unknown wallet operation")
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Enter a finite positive amount")
    shared.ensure_portfolio_in_session()
    engine = shared.current_engine()
    account = shared.account_id()
    # This lock also guards engine.start(), so a reset cannot race a restart.
    with engine._lock if engine is not None else nullcontext():
        port = engine.portfolio if engine is not None else st.session_state["portfolio"]
        persist = lambda p: shared.save_portfolio(p, account=account)
        if operation == "Reset account":
            if not confirmed:
                raise ValueError("Confirm that you want to archive this account and start again")
            if engine is not None and (engine.is_running() or any(
                t is not None and t.is_alive()
                for t in (engine._thread, engine._protection_thread)
            )):
                raise ValueError("Pause the agent and wait for its current cycle to finish before resetting")
            if port.positions:
                raise ValueError("Close open positions before resetting the account")
            path = shared.portfolio_path(account)
            archive = path.parent / "archive" / f"{path.stem}-{uuid4().hex}.json"
            port.save(archive)
            replacement = LivePortfolio(amount, fee_rate=port.fee_rate, name=port.name)
            persist(replacement)
            if engine is not None:
                engine.portfolio = replacement
                engine._equity_history.clear()
                engine._last_signal_bars.clear()
                engine._protection_status.clear()
                engine._last_decision = engine._last_research = None
                engine._last_ai_context.clear()
                engine._halt_reason = ""
                from risk import SafetyController
                engine._safety = SafetyController(engine._safety.config)
            port = replacement
        else:
            port.transfer_cash(amount if operation == "Deposit" else -amount, persist=persist)
        st.session_state["portfolio"] = port
        st.session_state["starting_capital"] = port.initial_capital
        wallets, _ = shared.portfolio_store()
        wallets[str(shared.portfolio_path(account))] = port
    # The portfolio is authoritative even if a secondary profile save fails.
    try:
        shared.save_profile()
    except OSError:
        st.warning("Wallet saved. Profile preferences could not be saved; the wallet balance is authoritative.")
    return port


def render_funding_controls(key):
    from dashboard import _shared as shared

    shared.ensure_portfolio_in_session()
    port = st.session_state["portfolio"]
    scoped = f"{key}_{shared.account_id()}"
    flash = st.session_state.pop(f"{scoped}_success", None)
    if flash:
        st.success(flash)
    st.caption(f"Net capital ${port.initial_capital:,.2f} · Available cash ${port.cash:,.2f}")
    operation = st.radio("Wallet action", ["Deposit", "Withdraw", "Reset account"],
                         horizontal=True, key=f"{scoped}_action")
    with st.form(f"{scoped}_{operation}_form"):
        amount = st.number_input("New starting balance ($)" if operation == "Reset account" else "Amount ($)",
                                 min_value=1., max_value=10_000_000., value=1000., step=1000.,
                                 key=f"{scoped}_{operation}_amount")
        confirmed = False
        if operation == "Reset account":
            st.warning("Starts a new paper account with no trades or P&L. The old account is archived. Pause the agent and close positions first.")
            confirmed = st.checkbox("Archive the current account and start again", key=f"{scoped}_confirm")
        else:
            st.caption("Keeps trades, positions and trading P&L. Cash transfers are not trading returns. Return % uses net capital.")
        submitted = st.form_submit_button(operation, type="primary")
    if submitted:
        try:
            updated = change_funds(operation, amount, confirmed=confirmed)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
        else:
            st.session_state[f"{scoped}_success"] = f"{operation} saved. Available cash ${updated.cash:,.2f}."
            st.rerun()
    if port.capital_changes:
        st.caption("Recent cash transfers (separate from trades)")
        st.dataframe([
            {"Time (UTC)": c["changed_at"], "Amount ($)": c["amount"],
             "Net capital ($)": c["capital_after"]}
            for c in reversed(port.capital_changes[-20:])
        ], hide_index=True, width="stretch")
