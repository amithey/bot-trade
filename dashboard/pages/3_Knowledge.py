"""Knowledge Base page — view chunk count, ingest from URL, re-seed books."""
from __future__ import annotations

import subprocess
import sys
from urllib.parse import urlparse

import streamlit as st

# ── sys.path bootstrap (so `from dashboard...` works in direct `streamlit run`)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ────────────────────────────────────────────────────────────────────────────

from dashboard._shared import ROOT, apply_theme, ensure_profile_in_session

_ALLOWED_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "youtu.be", "www.youtu.be",
    "music.youtube.com",
}


def _is_youtube_url(raw: str) -> bool:
    try:
        u = urlparse(raw or "")
        return u.scheme in ("http", "https") and u.netloc.lower() in _ALLOWED_HOSTS
    except Exception:
        return False

st.set_page_config(page_title="BotTrade - Knowledge", page_icon=":material/database:", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()
ensure_profile_in_session()

st.markdown('<div class="page-title">KNOWLEDGE BASE</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">ChromaDB status, ingestion, and written playbooks</div>',
            unsafe_allow_html=True)

# ── Collection status ────────────────────────────────────────────────────────
try:
    from rag.retriever import StrategyRetriever
    retr = StrategyRetriever()
    info = retr.collection_info()
    doc_count = int(info.get("document_count", 0))
except Exception as exc:
    info = {}
    doc_count = 0
    st.error(f"Couldn't open ChromaDB: {exc}")

k1, k2, k3 = st.columns(3)
k1.metric("Documents", f"{doc_count:,}")
k2.metric("Collection", info.get("collection_name", "—"))
k3.metric("Embedding model", (info.get("embedding_model", "—") or "—").split("/")[-1])

st.caption(f"Path: `{info.get('persist_dir', '—')}`")
st.markdown("---")

# ── Ingest from YouTube URL / Playlist ───────────────────────────────────────
st.markdown("##### Ingest from YouTube")
st.caption("Single video URL or a playlist URL — both are supported.")

col_url, col_btn = st.columns([4, 1])
url = col_url.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=... or /playlist?list=...",
    label_visibility="collapsed",
    key="yt_url_input",
)
go_btn = col_btn.button("Ingest", use_container_width=True)

if go_btn and url.strip():
    clean_url = url.strip()
    if not _is_youtube_url(clean_url):
        st.error("Only YouTube URLs are allowed "
                 "(youtube.com / youtu.be / music.youtube.com).")
    else:
        is_playlist = "list=" in clean_url or "/playlist?" in clean_url
        module = "knowledge_ingestion.playlist_scraper" if is_playlist \
                 else "knowledge_ingestion.youtube_scraper"
        args = [sys.executable, "-X", "utf8", "-m", module, "--url", clean_url]
        with st.status(f"Running {module} …", expanded=True) as s:
            try:
                # Streamlit session dir may differ from ROOT on some installs;
                # spawn from the project root so relative paths match.
                proc = subprocess.run(
                    args, cwd=str(ROOT),
                    capture_output=True, text=True,
                    timeout=20 * 60,     # 20-min hard cap (was 60m)
                )
                output = (proc.stdout or "") + (proc.stderr or "")
                st.code(output[-4000:] or "(no output)", language="text")
                if proc.returncode == 0:
                    s.update(label="Ingestion complete",
                             state="complete", expanded=False)
                else:
                    s.update(label=f"Failed (exit {proc.returncode})",
                             state="error")
            except subprocess.TimeoutExpired:
                s.update(label="Timed out after 20 min", state="error")
            except Exception as exc:
                s.update(label=f"{exc}", state="error")

st.markdown("---")

# ── Ingest from Web Article ──────────────────────────────────────────────────
st.markdown("##### Ingest from Web Article")
st.caption("Any public strategy/analysis page — Investopedia, TradingView "
           "ideas, broker blogs, Substack… Text is extracted, chunked and "
           "deduplicated automatically. One URL per line for batches.")

art_col, art_btn_col = st.columns([4, 1])
art_urls_raw = art_col.text_area(
    "Article URLs",
    placeholder="https://www.investopedia.com/terms/m/macd.asp\n"
                "https://www.tradingview.com/ideas/...",
    label_visibility="collapsed",
    height=80,
    key="article_urls_input",
)
art_go = art_btn_col.button("Ingest Articles", use_container_width=True)

if art_go and art_urls_raw.strip():
    art_urls = [u.strip() for u in art_urls_raw.splitlines() if u.strip()]
    bad = [u for u in art_urls
           if urlparse(u).scheme not in ("http", "https")]
    if bad:
        st.error(f"Not valid http(s) URLs: {', '.join(bad[:3])}")
    else:
        args = [sys.executable, "-X", "utf8", "-m",
                "knowledge_ingestion.article_scraper"]
        for u in art_urls:
            args += ["--url", u]
        with st.status(f"Ingesting {len(art_urls)} article(s)…",
                       expanded=True) as s:
            try:
                proc = subprocess.run(
                    args, cwd=str(ROOT),
                    capture_output=True, text=True,
                    timeout=10 * 60,
                )
                output = (proc.stdout or "") + (proc.stderr or "")
                st.code(output[-4000:] or "(no output)", language="text")
                if proc.returncode == 0:
                    s.update(label="Articles ingested",
                             state="complete", expanded=False)
                else:
                    s.update(label=f"Some articles failed "
                                   f"(exit {proc.returncode})",
                             state="error")
            except subprocess.TimeoutExpired:
                s.update(label="Timed out after 10 min", state="error")
            except Exception as exc:
                s.update(label=f"{exc}", state="error")

st.markdown("---")

# ── Re-run seed strategies ───────────────────────────────────────────────────
st.markdown("##### Written Playbooks (Seeded Strategies)")
st.caption("Built-in textual knowledge: RSI, MACD, candlesticks, S/R, "
           "fundamentals, day-trading rules. Idempotent — safe to re-run.")

if st.button("Re-run Seed Ingestion", use_container_width=False):
    with st.status("Re-seeding…", expanded=True) as s:
        try:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "knowledge_ingestion.seed_strategies"],
                cwd=str(ROOT), capture_output=True, text=True, timeout=300,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            st.code(out[-3000:], language="text")
            if proc.returncode == 0:
                s.update(label="Seed complete", state="complete", expanded=False)
            else:
                s.update(label=f"Failed (exit {proc.returncode})", state="error")
        except Exception as exc:
            s.update(label=f"{exc}", state="error")

with st.expander("What's inside the seeded playbooks?"):
    st.markdown("""
1. **RSI Playbook** — oversold/overbought rules, divergence, crypto adjustments.
2. **MACD Playbook** — crossovers, histogram momentum, signal-line strategy.
3. **Golden Cross & Death Cross** — phases of each regime, signal bars.
4. **Candlestick & Chart Patterns** — Hammer, Engulfing, H&S, triangles, flags.
5. **Support, Resistance & Trendlines** — level identification, bounces, breakouts.
6. **Fundamental Analysis** — P/E, PEG, FCF, earnings, macro (Fed/CPI/NFP), sector rotation.
7. **Day Trading Rules & Psychology** — 1% rule, Kelly, drawdown limits, daily workflow.
    """)
