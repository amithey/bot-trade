"""Reusable workspace components. These render supplied data without fetching it."""
from html import escape
from urllib.parse import urlparse

import streamlit as st

from dashboard.theme import C_BUY, C_SELL, TEXT_DIM


def page_header(title: str, description: str, *, section: str) -> None:
    st.markdown(
        f'<div class="workspace-header"><div>'
        f'<div class="workspace-eyebrow">{escape(section)}</div>'
        f'<h1 class="page-title">{escape(title)}</h1>'
        f'<p class="page-sub">{escape(description)}</p></div>'
        '<div class="workspace-context"><i></i> BotTrade <span> / </span> Workspace</div></div>',
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
            label = "Vote strength" if state.get("strategy_mode") == "COMMITTEE" else "Model confidence"
            st.caption(f"{label} {decision.confidence_score:.0%} · not a win probability")
            st.caption(decision.reasoning[:220])
        research = state.get("last_research") or {}
        if research.get("regime"):
            st.caption(f"{research['regime']} · {research['setup']}")
            st.caption(research.get("explanation", ""))
        st.markdown(
            '<div class="desk-facts">'
            f'<span>Strategy</span><b>{escape(state.get("strategy_mode", "—"))}</b>'
            f'<span>Risk profile</span><b>{escape(risk)}</b>'
            f'<span>Exposure cap</span><b>{cap_pct:g}%</b>'
            f'<span>Cycle</span><b>{state["interval_sec"]}s</b>'
            '<span>Execution</span><b>Paper</b></div>', unsafe_allow_html=True,
        )


def render_entry_research(report) -> None:
    if not report:
        st.info("No research snapshot yet. The next completed candle will produce an entry assessment.")
        return
    if report.get("status") == "DATA_BLOCKED":
        st.warning(report["reason"])
        return
    st.caption(f"{report['ticker']} · {report['interval']} · candle closed {report['bar_closed_at']}")
    cols = st.columns(3)
    cols[0].metric("Market regime", report["regime"])
    cols[1].metric("Setup", report["setup"])
    cols[2].metric("New entry gate", "Eligible" if report["entry_allowed"] else "Blocked")
    st.write(report["explanation"])
    if report.get("raw_action"):
        st.caption(f"Strategy candidate: {report['raw_action']} → filtered signal: {report.get('filtered_action', '—')}. "
                   "An eligible signal still needs the account's confidence, cash and safety checks.")
    evidence, context = st.columns([1.35, 1], gap="large")
    with evidence:
        st.dataframe(report["checks"], hide_index=True, width="stretch",
                     column_config={"name": "Check", "passed": "Passed", "detail": "Requirement"})
        with st.expander("Technical evidence", expanded=True):
            st.dataframe([{"Metric": k, "Value": v} for k, v in report["metrics"].items()],
                         hide_index=True, width="stretch")
            st.caption("Support/resistance and volume averages exclude the signal candle. Entry gates are rule-based hypotheses, not return forecasts.")
    with context:
        with st.expander("Fundamental context"):
            metrics = report.get("fundamentals") or {}
            if metrics:
                st.dataframe([{"Metric": k, "Value": str(v)} for k, v in metrics.items()], hide_index=True, width="stretch")
            else:
                st.caption("No company fundamentals available. Corporate P/E, margins and cash flow are not applicable to BTC.")
        with st.expander("News & macro sources", expanded=True):
            news = report.get("news_context") or {}
            items = news.get("items", [])
            if not items:
                st.caption("Loading public sources in the background…" if news.get("status") == "loading"
                           else "No dated headlines from the last 72 hours are available. Missing news is not neutral sentiment.")
            for item in items:
                parsed = urlparse(item["url"])
                title = escape(item["title"])
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    title = f'<a href="{escape(item["url"], quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                st.markdown(title, unsafe_allow_html=True)
                st.caption(f"{item['source']} · {item['published']}")
            st.caption("Public headlines are context, not an automatic buy/sell trigger. Committee uses technical rules; AI and Hybrid also evaluate supplied news and fundamentals.")
