"""Reusable workspace components. These render supplied data without fetching it."""
from html import escape

import streamlit as st

from dashboard.theme import C_BUY, C_SELL, TEXT_DIM


def page_header(title: str, description: str, *, section: str) -> None:
    st.markdown(
        f'<div class="workspace-header"><div>'
        f'<div class="workspace-eyebrow">{escape(section)}</div>'
        f'<h1 class="page-title">{escape(title)}</h1>'
        f'<p class="page-sub">{escape(description)}</p></div>'
        '<span class="badge badge-blue">BotTrade workspace</span></div>',
        unsafe_allow_html=True,
    )


def empty_workspace(title: str, description: str) -> None:
    st.markdown(
        '<div class="empty-workspace"><div class="empty-symbol">▥</div>'
        f'<h3>{escape(title)}</h3><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def _select_symbol(symbol: str) -> None:
    # Callbacks run before the next script pass, before this widget is built.
    from dashboard._shared import DEFAULT_TICKERS, CUSTOM_LABEL
    known = list(st.session_state.get("watchlist") or []) + DEFAULT_TICKERS
    st.session_state["ticker_sel_widget"] = symbol if symbol in known else CUSTOM_LABEL
    if symbol not in known:
        st.session_state["ticker_custom"] = symbol
    st.session_state["ticker_sel"] = symbol


def render_desk_sidebar(state, portfolio, watchlist, *, risk: str, cap_pct: float) -> None:
    """Trading desk context using known prices only, never fabricated quotes."""
    with st.container(border=True):
        st.markdown('<div class="bt-section-title">WATCHLIST</div>', unsafe_allow_html=True)
        positions = portfolio.positions
        symbols = list(dict.fromkeys([state["ticker"], *watchlist]))
        for symbol in symbols[:8]:
            position = positions.get(symbol)
            price = (state.get("last_price") if symbol == state["ticker"]
                     else position.current_price if position else None)
            left, right = st.columns([1.2, 1], gap="small")
            left.button(symbol, key=f"desk_symbol_{symbol}", width="stretch",
                        type="primary" if symbol == state["ticker"] else "secondary",
                        on_click=_select_symbol, args=(symbol,))
            right.markdown(
                f'<div style="text-align:right;padding:.45rem 0;font-size:.78rem">'
                f'{f"{price:,.2f}" if price else "—"}</div>', unsafe_allow_html=True,
            )
        st.caption("Last known prices · — means no quote loaded")
        st.page_link("pages/6_Watchlist_Scanner.py", label="Open scanner", icon=":material/filter_alt:")

    with st.container(border=True):
        st.markdown('<div class="bt-section-title">STRATEGY STATUS</div>', unsafe_allow_html=True)
        decision = state.get("last_decision")
        if decision is None:
            st.caption("No signal yet. Analysis appears after the first cycle.")
        else:
            color = {"BUY": C_BUY, "SELL": C_SELL}.get(decision.action, TEXT_DIM)
            st.markdown(
                f'<div style="font-size:1.4rem;font-weight:600;color:{color}">'
                f'{escape(decision.action)}</div>', unsafe_allow_html=True,
            )
            st.caption(f"Signal confidence {decision.confidence_score:.0%}")
            st.caption(decision.reasoning[:220])
        st.markdown(
            '<div class="desk-facts">'
            f'<span>Strategy</span><b>{escape(state.get("strategy_mode", "—"))}</b>'
            f'<span>Risk profile</span><b>{escape(risk)}</b>'
            f'<span>Exposure cap</span><b>{cap_pct:g}%</b>'
            f'<span>Cycle</span><b>{state["interval_sec"]}s</b>'
            '<span>Execution</span><b>Paper</b></div>', unsafe_allow_html=True,
        )
