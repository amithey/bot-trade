"""Account-level paper portfolio controls, intentionally outside the main desk."""
import streamlit as st


def render_reset():
    from dashboard._shared import account_id as _account_id, save_portfolio
    if st.button("Reset paper portfolio", width="stretch", help="Reset portfolio"):
        from portfolio.virtual_account import LivePortfolio
        from trading.registry import get_registry
        # Stop and drop the engine first: it holds a reference to the old
        # portfolio and would otherwise keep checkpointing it back over the
        # fresh one. get_live_engine() rebuilds on the next run.
        get_registry().stop(_account_id())
        fresh = LivePortfolio(
            initial_capital=float(st.session_state["starting_capital"]))
        st.session_state["portfolio"] = fresh
        save_portfolio(fresh)
        st.rerun()

