"""Watchlist Scanner — zero-token technical screener.

Scans every ticker in the user's watchlist and flags actionable setups:
  • RSI oversold / overbought
  • Golden/Death cross proximity
  • Price near 20-bar support/resistance
  • Bollinger squeeze
  • Above/below VWAP
  • MACD cross

Uses the existing MarketDataFetcher — same indicators as the live engine.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

# ── sys.path bootstrap (so `from dashboard...` works in direct `streamlit run`)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ────────────────────────────────────────────────────────────────────────────

from dashboard._shared import (
    C_BUY, C_HOLD, C_SELL, DEFAULT_TICKERS, TEXT, TEXT_DIM,
    secure_page, ensure_profile_in_session,
)

st.set_page_config(page_title="BotTrade - Scanner", page_icon=":material/filter_alt:",
                   layout="wide", initial_sidebar_state="expanded")
secure_page()
ensure_profile_in_session()

st.markdown('<div class="page-title">WATCHLIST SCANNER</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Scans every ticker in your watchlist for actionable '
    'technical setups — no LLM tokens consumed. Purely rule-based.</div>',
    unsafe_allow_html=True,
)

watchlist = list(st.session_state.get("watchlist") or DEFAULT_TICKERS)


@st.cache_data(ttl=300, show_spinner=False)
def _scan(tickers: tuple[str, ...]) -> list[dict]:
    """Fetch indicators and evaluate rule-based signals per ticker."""
    from market_data.fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher()
    rows: list[dict] = []
    for tk in tickers:
        try:
            snap = fetcher.fetch_latest(tk, lookback_days=420)
        except Exception as exc:  # noqa: BLE001
            rows.append({"ticker": tk, "error": str(exc)[:120]})
            continue
        last = snap.latest
        prev = snap.data.iloc[-2] if len(snap.data) >= 2 else last
        signals: list[tuple[str, str]] = []  # (label, polarity)

        # RSI
        rsi = float(last.get("RSI_14", 50.0))
        if rsi <= 30:
            signals.append((f"RSI {rsi:.0f} oversold", "bull"))
        elif rsi >= 70:
            signals.append((f"RSI {rsi:.0f} overbought", "bear"))

        # MACD cross
        m_now = float(last.get("MACD", 0))
        s_now = float(last.get("MACD_Signal", 0))
        m_prev = float(prev.get("MACD", 0))
        s_prev = float(prev.get("MACD_Signal", 0))
        if m_prev <= s_prev and m_now > s_now:
            signals.append(("MACD bull cross", "bull"))
        elif m_prev >= s_prev and m_now < s_now:
            signals.append(("MACD bear cross", "bear"))

        # SMA regime + cross proximity
        sma50 = float(last.get("SMA_50", 0))
        sma200 = float(last.get("SMA_200", 0))
        if sma50 and sma200:
            spread = (sma50 / sma200 - 1.0) * 100.0
            if sma50 > sma200 and abs(spread) < 0.5:
                signals.append(("Near golden cross", "bull"))
            elif sma50 < sma200 and abs(spread) < 0.5:
                signals.append(("Near death cross", "bear"))
            elif sma50 > sma200:
                signals.append(("Golden-cross regime", "bull"))
            else:
                signals.append(("Death-cross regime", "bear"))

        # Support/resistance proximity
        close = float(last["Close"])
        sup = float(last.get("Support_20", close))
        res = float(last.get("Resistance_20", close))
        if sup and (close - sup) / close * 100.0 < 1.0:
            signals.append(("Near 20-bar support", "bull"))
        if res and (res - close) / close * 100.0 < 1.0:
            signals.append(("Near 20-bar resistance", "bear"))

        # Bollinger squeeze
        bw = float(last.get("BB_Width_20", 0))
        # historical avg width
        try:
            bw_mean = float(snap.data["BB_Width_20"].tail(60).mean())
            if bw and bw_mean and bw < bw_mean * 0.6:
                signals.append(("Bollinger squeeze", "neutral"))
        except Exception:
            pass

        # VWAP position
        vwap = float(last.get("VWAP_20", close))
        if vwap:
            diff = (close / vwap - 1.0) * 100.0
            if diff > 0.3:
                signals.append((f"{diff:+.1f}% above VWAP", "bull"))
            elif diff < -0.3:
                signals.append((f"{diff:+.1f}% below VWAP", "bear"))

        # Composite score
        score = sum(1 for _, p in signals if p == "bull") - \
                sum(1 for _, p in signals if p == "bear")

        rows.append({
            "ticker":  tk,
            "price":   close,
            "rsi":     rsi,
            "score":   score,
            "signals": signals,
        })
    return rows


# ─── Controls ─────────────────────────────────────────────────────────────────
c_scan, c_filter, c_info = st.columns([1, 2, 3])
with c_scan:
    if st.button("Re-scan", use_container_width=True):
        _scan.clear()
        st.rerun()
with c_filter:
    min_score = st.slider("Min |score| to show", 0, 5, 0,
                          help="Hide tickers with weak / neutral reads.")
with c_info:
    st.caption(f"Watchlist: {len(watchlist)} tickers · "
               f"Rendered {datetime.now().strftime('%H:%M:%S')}")

st.markdown("---")

if not watchlist:
    st.warning("Watchlist is empty. Add tickers in **Settings**.")
    st.stop()

with st.spinner("Fetching indicators…"):
    results = _scan(tuple(watchlist))

# Sort: most bullish first
results.sort(key=lambda r: r.get("score", 0), reverse=True)

# ─── Render rows ─────────────────────────────────────────────────────────────
for r in results:
    if "error" in r:
        st.markdown(
            f'<div class="bt-panel" style="border-left:4px solid {C_SELL};opacity:0.7">'
            f'<span style="font-family:monospace;font-weight:700">{r["ticker"]}</span> '
            f'<span style="color:{C_SELL}">— {r["error"]}</span>'
            f'</div>', unsafe_allow_html=True,
        )
        continue

    if abs(r["score"]) < min_score:
        continue

    score = r["score"]
    if score > 0:
        color = C_BUY
        verdict = f"+{score} bullish"
    elif score < 0:
        color = C_SELL
        verdict = f"{score} bearish"
    else:
        color = C_HOLD
        verdict = "neutral"

    # Signal chips
    chips = []
    for lbl, pol in r["signals"]:
        c = {"bull": C_BUY, "bear": C_SELL}.get(pol, TEXT_DIM)
        chips.append(
            f'<span style="display:inline-block;padding:2px 8px;margin:2px 4px 2px 0;'
            f'border:1px solid {c};color:{c};border-radius:3px;'
            f'font-size:0.72rem;font-family:monospace">{lbl}</span>'
        )
    chips_html = "".join(chips) or \
        f'<span style="color:{TEXT_DIM};font-style:italic">no strong signals</span>'

    st.markdown(
        f'<div class="bt-panel" style="border-left:4px solid {color}">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:baseline;margin-bottom:6px">'
        f'<span style="font-family:monospace;font-weight:700;font-size:1.1rem;'
        f'color:{TEXT}">{r["ticker"]}</span>'
        f'<span style="color:{TEXT_DIM}">'
        f'${r["price"]:.2f} · RSI {r["rsi"]:.0f} · '
        f'<span style="color:{color};font-weight:700">{verdict}</span>'
        f'</span></div>'
        f'<div>{chips_html}</div>'
        f'</div>', unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Use the scanner to find candidates, then open the ticker in **Live Trading** "
    "for full AI evaluation. Bullish score ≥ 2 with RSI 30-55 often marks a "
    "healthy entry window."
)
