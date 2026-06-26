"""ML Lab — unified view of the bot's machine-learning signals.

Shows, for any ticker:
  • Regime classification (KMeans)
  • Anomaly score (IsolationForest)
  • Detected chart patterns (rule-based with scipy peak detection)
  • Short-horizon forecast (EWMA-drift + volatility band)
  • Self-learning win probability (RandomForest on closed trades)

Also offers manual retrain buttons and a "wire into engine" toggle.
Zero LLM tokens — all models run locally.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

# ── sys.path bootstrap ─────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from dashboard._shared import (
    BORDER, C_BUY, C_HOLD, C_SELL, CYAN, DEFAULT_TICKERS, TEXT, TEXT_DIM,
    apply_theme, ensure_portfolio_in_session, ensure_profile_in_session,
)

st.set_page_config(page_title="BotTrade - ML Lab", page_icon=":material/model_training:",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()
ensure_profile_in_session()
ensure_portfolio_in_session()

st.markdown('<div class="page-title">ML LAB</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">All machine-learning signals in one place — regime, '
    'anomaly, chart patterns, forecast, and the self-learning trade-journal '
    'classifier. No LLM tokens consumed.</div>',
    unsafe_allow_html=True,
)

watchlist = list(st.session_state.get("watchlist") or DEFAULT_TICKERS)

# ─── Controls ─────────────────────────────────────────────────────────────
c_sym, c_run, c_info = st.columns([2, 1, 3])
with c_sym:
    symbol = st.selectbox("Ticker", watchlist,
                          index=0 if watchlist else None,
                          key="ml_ticker")
with c_run:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Analyze", use_container_width=True, type="primary")
with c_info:
    st.caption(f"Models persist under `data/ml_models/` · "
               f"Rendered {datetime.now().strftime('%H:%M:%S')}")

st.markdown("---")


# ─── Fetch + run all models ───────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_and_score(ticker: str) -> dict:
    from market_data.fetcher import MarketDataFetcher
    from ml.anomaly import AnomalyDetector
    from ml.forecaster import forecast
    from ml.pattern_detector import detect_patterns
    from ml.regime import RegimeClassifier
    from ml.trade_journal_ml import TradeJournalML

    fetcher = MarketDataFetcher()
    snap = fetcher.fetch_latest(ticker, lookback_days=420)
    df = snap.data

    regime = RegimeClassifier(ticker)
    if not regime.is_fitted:
        regime.fit(df)
    regime_res = regime.predict(df)

    anomaly = AnomalyDetector(ticker)
    if not anomaly.is_fitted:
        anomaly.fit(df)
    anom_res = anomaly.predict(df)

    patterns = detect_patterns(df)
    fc = forecast(df, horizon=5)

    jm = TradeJournalML()
    win = jm.predict(df)

    return {
        "df":         df,
        "close":      float(df["Close"].iloc[-1]),
        "regime":     regime_res,
        "anomaly":    anom_res,
        "patterns":   patterns,
        "forecast":   fc,
        "win":        win,
    }


if not watchlist:
    st.warning("Watchlist empty — add symbols in **Settings**.")
    st.stop()

if run:
    _fetch_and_score.clear()

try:
    result = _fetch_and_score(symbol)
except Exception as exc:
    st.error(f"Could not analyze **{symbol}**: {exc}")
    st.stop()


# ─── KPI strip ────────────────────────────────────────────────────────────
reg = result["regime"]
an = result["anomaly"]
fc = result["forecast"]
win = result["win"]

_REG_COLORS = {
    "TRENDING_UP":   C_BUY,
    "TRENDING_DOWN": C_SELL,
    "RANGING":       C_HOLD,
    "VOLATILE":      "#ffb347",
}
_SEV_COLORS = {
    "NONE":    C_HOLD,
    "MILD":    "#ffb347",
    "STRONG":  "#ff7a4f",
    "EXTREME": C_SELL,
}
_BIAS_COLORS = {"up": C_BUY, "down": C_SELL, "flat": C_HOLD}

def _kpi(col, label, value, sub, color):
    col.markdown(
        f'<div class="bt-panel" style="text-align:center">'
        f'<div style="color:{TEXT_DIM};font-size:0.7rem;letter-spacing:1.5px">'
        f'{label}</div>'
        f'<div style="color:{color};font-size:1.4rem;font-weight:700;margin-top:2px">'
        f'{value}</div>'
        f'<div style="color:{TEXT_DIM};font-size:0.75rem">{sub}</div>'
        f'</div>', unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5 = st.columns(5)
_kpi(k1, "REGIME",
     reg.label.replace("_", " "),
     f"conf {reg.confidence*100:.0f}%",
     _REG_COLORS.get(reg.label, TEXT))
_kpi(k2, "ANOMALY",
     an.severity,
     f"score {an.score:+.2f}",
     _SEV_COLORS.get(an.severity, TEXT))
_kpi(k3, "5-BAR FORECAST",
     f"{(fc.point_return*100):+.2f}%",
     f"band ±{((fc.hi_95-fc.lo_95)/2*100):.2f}%",
     _BIAS_COLORS.get(fc.bias, TEXT))
_kpi(k4, "WIN PROBABILITY",
     f"{win.prob_win*100:.0f}%",
     f"n={win.n_train} · {win.confidence_band}",
     C_BUY if win.prob_win >= 0.55 else (C_SELL if win.prob_win <= 0.45 else C_HOLD))
_kpi(k5, "CLOSE",
     f"${result['close']:.2f}",
     symbol,
     TEXT)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Patterns panel ──────────────────────────────────────────────────────
st.markdown('<div class="bt-section-title">CHART PATTERNS</div>',
            unsafe_allow_html=True)
pats = result["patterns"]
if not pats:
    st.info("No canonical patterns detected in the last 80 bars.")
else:
    for p in pats:
        dir_color = C_BUY if p.direction == "bullish" else C_SELL
        bar = int(p.confidence * 100)
        st.markdown(
            f'<div class="bt-panel" style="border-left:4px solid {dir_color}">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline">'
            f'<div><span style="font-family:monospace;font-weight:700;'
            f'color:{TEXT};font-size:1.05rem">{p.name.replace("_"," ")}</span>'
            f' <span style="color:{dir_color};font-size:0.8rem">'
            f'[{p.direction.upper()}]</span></div>'
            f'<div style="color:{dir_color};font-weight:700">'
            f'{p.confidence*100:.0f}%</div>'
            f'</div>'
            f'<div style="height:5px;background:#1a202c;border-radius:3px;'
            f'margin-top:6px;overflow:hidden">'
            f'<div style="height:100%;width:{bar}%;background:{dir_color}"></div>'
            f'</div>'
            f'<div style="color:{TEXT_DIM};font-size:0.78rem;margin-top:4px">'
            f'{p.notes}</div>'
            f'</div>', unsafe_allow_html=True,
        )

st.markdown("---")


# ─── Regime clusters + forecast chart ─────────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown('<div class="bt-section-title">REGIME CLUSTER DISTANCES</div>',
                unsafe_allow_html=True)
    dists = reg.distances
    max_d = max(dists.values()) or 1.0
    for name, d in sorted(dists.items(), key=lambda kv: kv[1]):
        c = _REG_COLORS.get(name, TEXT)
        bar_pct = max(0, min(100, int((1 - d / max_d) * 100)))
        st.markdown(
            f'<div style="margin-bottom:6px">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:0.8rem">'
            f'<span style="color:{c};font-family:monospace">'
            f'{name.replace("_"," ")}</span>'
            f'<span style="color:{TEXT_DIM};font-family:monospace">'
            f'd={d:.2f}</span></div>'
            f'<div style="height:6px;background:#1a202c;border-radius:3px;'
            f'overflow:hidden">'
            f'<div style="height:100%;width:{bar_pct}%;background:{c}"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Closest cluster → **{reg.label.replace('_',' ')}** "
               f"(confidence {reg.confidence*100:.0f}%).")

with right:
    st.markdown('<div class="bt-section-title">5-BAR FORECAST</div>',
                unsafe_allow_html=True)
    try:
        import plotly.graph_objects as go
        df = result["df"].tail(40)
        last_px = float(df["Close"].iloc[-1])
        h = fc.horizon
        # build forecast cone
        import numpy as np
        steps = list(range(1, h + 1))
        mid = [last_px * np.exp(fc.point_return * i / h) for i in steps]
        lo = [last_px * np.exp(fc.lo_95 * i / h) for i in steps]
        hi = [last_px * np.exp(fc.hi_95 * i / h) for i in steps]
        future_x = list(range(len(df), len(df) + h))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["Close"],
            mode="lines", line=dict(color=CYAN, width=2), name="Close",
        ))
        fig.add_trace(go.Scatter(
            x=future_x + future_x[::-1],
            y=hi + lo[::-1],
            fill="toself", fillcolor="rgba(0,212,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=mid, mode="lines+markers",
            line=dict(color=_BIAS_COLORS.get(fc.bias, TEXT),
                      width=2, dash="dot"),
            name="forecast",
        ))
        fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_DIM, size=11),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(gridcolor="#1a202c", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.caption(f"(chart unavailable: {exc})")

    bias_c = _BIAS_COLORS.get(fc.bias, TEXT)
    st.markdown(
        f'<div style="color:{TEXT_DIM};font-size:0.85rem">'
        f'Bias <span style="color:{bias_c};font-weight:700">'
        f'{fc.bias.upper()}</span> · μ={fc.point_return*100:+.2f}% · '
        f'σ={fc.sigma*100:.2f}%/bar · 95% CI '
        f'[{fc.lo_95*100:+.2f}%, {fc.hi_95*100:+.2f}%]'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─── Trade Journal ML training ───────────────────────────────────────────
st.markdown('<div class="bt-section-title">SELF-LEARNING JOURNAL</div>',
            unsafe_allow_html=True)
jcol1, jcol2, jcol3 = st.columns([2, 2, 1])
with jcol1:
    st.metric("Trained on", f"{win.n_train} closed trades")
with jcol2:
    st.metric("Confidence band", win.confidence_band)
with jcol3:
    retrain = st.button("Retrain now", use_container_width=True)

if retrain:
    from portfolio.virtual_account import LivePortfolio
    port: LivePortfolio = st.session_state.get("portfolio")
    if port is None:
        st.error("No portfolio loaded yet.")
    else:
        from market_data.fetcher import MarketDataFetcher
        from ml.trade_journal_ml import TradeJournalML
        fetcher = MarketDataFetcher()

        def _fetch_near(ticker: str, dt):
            # Fetch 420 days ending around dt (we use the full history
            # and rely on extract_latest using the tail).
            try:
                return fetcher.fetch_latest(
                    ticker, lookback_days=420,
                ).data
            except Exception:
                return None

        with st.spinner("Rebuilding training set from closed trades…"):
            n = TradeJournalML().fit_from_portfolio(
                list(port._trade_log), fetch_history=_fetch_near,
            )
        if n < 10:
            st.warning(f"Only {n} closed trades found. Need ≥10 to train. "
                       f"The journal will stay neutral until then.")
        else:
            st.success(f"Retrained on {n} closed trades. Reload to see updated predictions.")

st.caption(
    "After the bot closes 10+ trades, use **Retrain now**. The journal "
    "learns which of your setups actually win and feeds that probability "
    "back into every future decision."
)

st.markdown("---")
st.caption(
    "All models persist under `data/ml_models/`. Delete a file there to "
    "force retraining on next analyze. Models auto-retrain on first use."
)
