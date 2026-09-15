"""Read-only position protection status, shared by both account views."""
import streamlit as st


def render_profit_protection(portfolio, engine=None):
    st.markdown("##### Profit protection")
    positions = portfolio.positions
    if not positions:
        st.caption("No open positions to protect.")
        return
    running = engine is not None and engine.is_running()
    if not running:
        st.warning("Bot stopped: saved exit triggers are not being monitored.")
    states = engine.profit_protection_status() if engine is not None else {}
    rows = []
    for ticker, pos in positions.items():
        status = "Exit pending" if pos.profit_exit_pending else (
            "Armed" if pos.profit_protection_armed else "Waiting for net profit"
        )
        if states.get(ticker, {}).get("status") == "UNAVAILABLE":
            status = "Quote unavailable"
        rows.append({
            "Ticker": ticker, "Side": pos.side, "Protection": status,
            "Best observed price": pos.best_price,
            "Exit trigger": pos.profit_stop_price,
            "Peak net return (%)": pos.profit_peak_net_pct,
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "The exit trigger follows favorable prices and never moves backward. "
        "Net estimates include entry/exit fees and modeled exit slippage. "
        "Polling can miss price moves; gaps may exit below the intended profit."
    )
