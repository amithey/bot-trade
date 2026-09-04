"""Market Research — on-demand macro/geopolitical briefing from Claude.

Heavy on tokens, so the LLM call only fires when the user clicks the
'Run Research' button. Cached result is shown until refreshed.
"""
from __future__ import annotations

import html as _html
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st


def _safe_url(raw: str) -> str:
    """Return URL only if scheme is http/https, else '#'."""
    try:
        u = urlparse(raw or "")
        if u.scheme in ("http", "https") and u.netloc:
            return raw
    except Exception:
        pass
    return "#"

# ── sys.path bootstrap (so `from dashboard...` works in direct `streamlit run`)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ────────────────────────────────────────────────────────────────────────────

from dashboard._shared import (
    BORDER, C_BUY, C_HOLD, C_SELL, CYAN, DEFAULT_TICKERS,
    secure_page, ensure_profile_in_session,
)

st.set_page_config(page_title="BotTrade - Market Research", page_icon=":material/manage_search:",
                   layout="wide", initial_sidebar_state="expanded")
secure_page()
ensure_profile_in_session()

st.markdown('<div class="page-title">MARKET RESEARCH</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">On-demand macro briefing — hot sectors, tickers to watch, '
    'key risks. Each run costs API tokens, so use sparingly (once or twice a day).</div>',
    unsafe_allow_html=True,
)

# ── Control bar ─────────────────────────────────────────────────────────────
watchlist = list(st.session_state.get("watchlist") or DEFAULT_TICKERS)

col_a, col_b, col_c = st.columns([2, 2, 1])
with col_a:
    scope = st.multiselect(
        "Scan these symbols for sentiment",
        options=watchlist,
        default=watchlist[:6],
        help="News is pulled per ticker; bias is aggregated across all.",
    )
with col_b:
    extra = st.text_input(
        "Extra focus (optional)",
        placeholder="e.g. rate cuts, semiconductors, oil, election risk…",
        help="Free-text hint injected into the prompt. Leave blank for a general scan.",
    )
with col_c:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Research", type="primary", use_container_width=True)

st.markdown("---")

# ── Fire the research call ─────────────────────────────────────────────────
if run:
    if not scope:
        st.warning("Pick at least one symbol to scan.")
    else:
        try:
            from dashboard._shared import get_pipeline
            pipeline = get_pipeline()
            engine = pipeline.ai_engine if hasattr(pipeline, "ai_engine") else None
            if engine is None:
                from decision_engine.ai_engine import AITradingEngine
                engine = AITradingEngine()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load AI engine: {e}")
            engine = None

        if engine is not None:
            with st.spinner("Scanning news feeds and asking Claude…"):
                try:
                    result = engine.research_market(
                        watchlist=scope,
                        extra_focus=extra.strip() or None,
                    )
                    st.session_state["_research_result"] = result
                except Exception as e:  # noqa: BLE001
                    st.error(f"Research failed: {e}")

# ── Render cached / fresh result ───────────────────────────────────────────
result = st.session_state.get("_research_result")

if not result:
    st.info(
        "No research run yet. Click **Run Research** above — the bot will "
        "pull fresh headlines, classify sentiment, and ask Claude to "
        "identify hot sectors, tickers to watch, and key market risks.",
        icon=None,
    )
else:
    bias = result.get("bias", {})
    ts: datetime = result.get("ts") or datetime.utcnow()
    age_min = (datetime.utcnow() - ts).total_seconds() / 60.0

    # ── Bias summary strip ──────────────────────────────────────────────────
    label = bias.get("label", "neutral")
    score = bias.get("score", 0)
    color = {"bullish": C_BUY, "bearish": C_SELL, "mixed": "#ffb347"}.get(label, C_HOLD)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="bt-panel" style="text-align:center">'
            f'<div style="font-size:0.75rem;color:#8a98ad;letter-spacing:1px">AGGREGATE BIAS</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:{color}">{label.upper()}</div>'
            f'<div style="color:#c9d1da">score {score:+.2f}</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="bt-panel" style="text-align:center">'
            f'<div style="font-size:0.75rem;color:#8a98ad;letter-spacing:1px">HEADLINES</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:{CYAN}">{len(result.get("headlines", []))}</div>'
            f'<div style="color:#c9d1da">across scope</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="bt-panel" style="text-align:center">'
            f'<div style="font-size:0.75rem;color:#8a98ad;letter-spacing:1px">MODEL</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:#e6eef8;font-family:monospace">{result.get("model","?")}</div>'
            f'<div style="color:#c9d1da">free-form</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="bt-panel" style="text-align:center">'
            f'<div style="font-size:0.75rem;color:#8a98ad;letter-spacing:1px">REPORT AGE</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#e6eef8">{age_min:.0f}m</div>'
            f'<div style="color:#c9d1da">since last run</div>'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main report + headlines side-by-side ───────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="bt-section-title">AI BRIEFING</div>',
                    unsafe_allow_html=True)
        st.markdown(result.get("report", "_empty_"))

    with right:
        st.markdown('<div class="bt-section-title">HEADLINES USED</div>',
                    unsafe_allow_html=True)
        heads = result.get("headlines", [])
        if not heads:
            st.caption("No headlines pulled.")
        else:
            for h in heads[:25]:
                tag = h.sentiment
                tag_color = {"bull": C_BUY, "bear": C_SELL}.get(tag, C_HOLD)
                tag_label = {"bull": "BULL", "bear": "BEAR"}.get(tag, "NEUT")
                title_safe = _html.escape(h.title or "")
                url = _safe_url(getattr(h, "url", "") or "")
                src = _html.escape(getattr(h, "source", "?") or "?")
                st.markdown(
                    f'<div style="padding:8px 10px;margin-bottom:6px;'
                    f'background:#0c1016;border:1px solid {BORDER};border-radius:4px;">'
                    f'<span style="color:{tag_color};font-family:monospace;'
                    f'font-size:0.72rem;font-weight:700">[{tag_label}]</span> '
                    f'<span style="color:#8a98ad;font-size:0.72rem">{src}</span><br>'
                    f'<a href="{url}" target="_blank" style="color:#e6eef8;'
                    f'text-decoration:none;font-size:0.88rem">{title_safe}</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Per-ticker bias breakdown ──────────────────────────────────────────
    per_ticker = bias.get("per_ticker") or []
    if per_ticker:
        st.markdown("---")
        st.markdown('<div class="bt-section-title">PER-TICKER SENTIMENT</div>',
                    unsafe_allow_html=True)
        cols = st.columns(min(len(per_ticker), 6) or 1)
        for i, b in enumerate(per_ticker):
            lbl = b.get("label", "neutral")
            sc = float(b.get("score", 0) or 0)
            c = {"bullish": C_BUY, "bearish": C_SELL, "mixed": "#ffb347"}.get(lbl, C_HOLD)
            with cols[i % len(cols)]:
                st.markdown(
                    f'<div class="bt-panel" style="text-align:center">'
                    f'<div style="font-family:monospace;color:#8a98ad;font-size:0.75rem">'
                    f'{(scope[i] if i < len(scope) else "")}</div>'
                    f'<div style="color:{c};font-weight:700;font-size:1.1rem">{lbl.upper()}</div>'
                    f'<div style="color:#c9d1da;font-size:0.8rem">{sc:+.1f} · '
                    f'{b.get("bull_count",0)}↑ / {b.get("bear_count",0)}↓</div>'
                    f'</div>', unsafe_allow_html=True,
                )

st.markdown("---")
st.caption(
    "Generated briefing. This is not investment advice. Claude reads "
    "only the headlines shown above — it has no access to real-time prices, "
    "fundamentals, or order flow. Always cross-check before trading."
)
