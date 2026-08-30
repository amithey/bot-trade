"""
News & Geopolitics Feed — ticker-specific + global macro headlines.

This module gives the AI engine a read on the *narrative* surrounding a
ticker and the broader market at decision time. Technical signals tell
the bot what happened; headlines tell it why — and, crucially, whether
a sudden move was triggered by a fundamental catalyst (trade cautiously)
or by ordinary flow (trade the pattern).

Design goals
------------
* Zero paid APIs — use public RSS feeds that do not require keys.
* Pluggable providers — the `NewsProvider` abstract class lets us add
  NewsAPI, Bloomberg Terminal, Alpaca News, etc. later without changing
  call sites.
* Aggressive caching — 30 min TTL per feed; headlines are cheap context
  but expensive bandwidth if pulled every AI cycle.
* Graceful degradation — any feed failure returns an empty list and
  logs a debug message; never blocks the decision loop.

Feed selection (default)
------------------------
* Yahoo Finance per-ticker RSS  -> ticker-specific headlines.
* Reuters + BBC World top stories -> global macro / geopolitics.
* CoinDesk RSS when ticker looks like crypto (``-USD`` suffix on major
  coins) -> crypto-specific flow.

Public API
----------
```python
from market_data.news import NewsFeed

feed = NewsFeed()              # singleton-friendly (internally cached)
headlines = feed.fetch("AAPL") # list[Headline], best-recent first
summary   = feed.summary_for_prompt("AAPL", max_items=6)
```

The ``summary_for_prompt()`` helper returns a trim, LLM-ready string
block the AI engine can splice directly into its prompt.
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from typing import Iterable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Very rough crypto-ticker detector: "BTC-USD", "ETH-USD", etc.
_CRYPTO_RE = re.compile(r"^(BTC|ETH|SOL|XRP|DOGE|ADA|AVAX|DOT|MATIC|LINK|LTC)-USD$",
                        re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

_DEFAULT_TTL_SECONDS = 30 * 60  # 30 min per feed
_DEFAULT_TIMEOUT_SEC = 6        # hard cap per HTTP call
_USER_AGENT = ("Mozilla/5.0 (compatible; BotTrade/1.0; "
               "+https://github.com/bottrade)")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight sentiment lexicon
# ─────────────────────────────────────────────────────────────────────────────
# A true LLM-driven classifier would produce sharper labels but cost tokens
# on every AI cycle. A curated keyword lexicon catches the majority of
# high-signal headlines (earnings beats/misses, rate cuts/hikes, rallies,
# crashes, escalations, record highs/lows) with zero latency and zero cost.
# Each side holds both unigrams and bigrams. Bigrams are scored first so
# "rate hike" beats the generic "rate" match.

_BULL_BIGRAMS = {
    "record high", "all time high", "beats estimates", "beat estimates",
    "raises guidance", "upgraded to", "cut rates", "rate cut",
    "strong demand", "better than expected", "rally continues",
    "buyback announced", "earnings beat", "profit surges", "revenue beat",
    "breaks out", "bullish outlook", "upside surprise", "reopens trade",
    "ceasefire agreed", "deal reached", "tariffs lifted", "sanctions lifted",
    "soft landing",
}
_BEAR_BIGRAMS = {
    "record low", "all time low", "misses estimates", "misses expectations",
    "cuts guidance", "downgraded to", "rate hike", "hikes rates",
    "weak demand", "worse than expected", "selloff deepens",
    "earnings miss", "profit warning", "revenue miss",
    "breaks down", "bearish outlook", "downside surprise",
    "war escalates", "attack on", "tariffs imposed", "sanctions imposed",
    "hard landing", "hits lowest", "bank run", "default risk",
    "investigation launched", "fraud alleged", "layoffs announced",
    "bankruptcy filed", "credit downgrade",
}
_BULL_UNIGRAMS = {
    "surge", "surges", "jumps", "jumped", "rallies", "rally", "soars",
    "soared", "climbs", "climbed", "gains", "gaining", "rebound",
    "rebounds", "upgraded", "bullish", "outperform", "beat", "beats",
    "record", "breakthrough", "approved", "booming", "accelerates",
    "expands", "inflows", "buyback", "dividend",
}
_BEAR_UNIGRAMS = {
    "plunge", "plunges", "plunged", "crashes", "crashed", "slump",
    "slumps", "slumped", "tumbles", "tumbled", "sinks", "sank",
    "downgrade", "downgraded", "bearish", "underperform", "miss",
    "misses", "recession", "crisis", "collapse", "collapses",
    "layoffs", "bankruptcy", "default", "warns", "fears", "shock",
    "selloff", "sell-off", "scandal", "fraud", "investigation",
    "war", "attack", "invasion", "sanctions", "tariffs", "strike",
    "outflows", "halts", "probe",
}


def _classify_headline(title: str) -> tuple[str, int]:
    """Return ``(label, score)`` where label is ``bull``/``bear``/``neutral``
    and score ∈ {-3..+3}. Bigrams are worth 2 points, unigrams 1 point."""
    if not title:
        return "neutral", 0
    t = title.lower()
    score = 0
    for phrase in _BULL_BIGRAMS:
        if phrase in t:
            score += 2
    for phrase in _BEAR_BIGRAMS:
        if phrase in t:
            score -= 2
    words = re.findall(r"[a-z'-]+", t)
    wordset = set(words)
    score += sum(1 for w in wordset if w in _BULL_UNIGRAMS)
    score -= sum(1 for w in wordset if w in _BEAR_UNIGRAMS)
    score = max(-3, min(3, score))
    if score > 0:
        return "bull", score
    if score < 0:
        return "bear", score
    return "neutral", 0


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Headline:
    """One news item — provider-agnostic."""
    title: str
    source: str             # human label, e.g. "Yahoo Finance"
    url: str
    published: Optional[datetime]   # UTC, None if provider omitted
    summary: str = ""              # short 1-2 sentence excerpt if given

    @property
    def age_minutes(self) -> Optional[int]:
        if self.published is None:
            return None
        delta = datetime.now(timezone.utc) - self.published
        return max(0, int(delta.total_seconds() // 60))

    def short(self, max_len: int = 140) -> str:
        t = self.title.strip()
        return t if len(t) <= max_len else t[: max_len - 1] + "…"

    @property
    def sentiment(self) -> str:
        """Cached keyword-based sentiment label (bull / bear / neutral)."""
        return _classify_headline(self.title)[0]

    @property
    def sentiment_score(self) -> int:
        """Signed score ∈ [-3, +3]. Positive = bullish; negative = bearish."""
        return _classify_headline(self.title)[1]


@dataclass
class NewsBundle:
    """All headlines returned for a single ``fetch(ticker)`` call."""
    ticker: str
    ticker_specific: list[Headline] = field(default_factory=list)
    macro: list[Headline] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def all(self) -> list[Headline]:
        return list(self.ticker_specific) + list(self.macro)

    @property
    def empty(self) -> bool:
        return not self.ticker_specific and not self.macro

    def sentiment_bias(self) -> dict[str, object]:
        """Aggregate sentiment across all headlines into a single bias dict.

        Returns keys:
            score         — mean signed score ∈ [-3, +3]
            label         — 'bullish' / 'bearish' / 'neutral' / 'mixed'
            bull_count    — number of bullish headlines
            bear_count    — number of bearish headlines
            neutral_count — number of neutral headlines
            strongest     — the single most extreme Headline (or None)
        """
        headlines = self.all
        if not headlines:
            return {"score": 0.0, "label": "neutral",
                    "bull_count": 0, "bear_count": 0, "neutral_count": 0,
                    "strongest": None}
        bull = sum(1 for h in headlines if h.sentiment == "bull")
        bear = sum(1 for h in headlines if h.sentiment == "bear")
        neu  = len(headlines) - bull - bear
        mean = sum(h.sentiment_score for h in headlines) / len(headlines)
        if bull > 0 and bear > 0 and abs(bull - bear) <= max(1, len(headlines) // 5):
            label = "mixed"
        elif mean >= 0.6:
            label = "bullish"
        elif mean <= -0.6:
            label = "bearish"
        else:
            label = "neutral"
        strongest = max(headlines,
                        key=lambda h: abs(h.sentiment_score), default=None)
        return {"score": round(mean, 2), "label": label,
                "bull_count": bull, "bear_count": bear, "neutral_count": neu,
                "strongest": strongest}


# ─────────────────────────────────────────────────────────────────────────────
# Provider abstraction
# ─────────────────────────────────────────────────────────────────────────────


class NewsProvider(ABC):
    """A pluggable source of :class:`Headline`. Implementations should be
    side-effect-free and must never raise — return ``[]`` on any error."""

    name: str = "abstract"

    @abstractmethod
    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        """Return a list of headlines. ``ticker=None`` means "macro / global"."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# RSS helpers
# ─────────────────────────────────────────────────────────────────────────────


def _http_get(url: str, timeout: float = _DEFAULT_TIMEOUT_SEC) -> Optional[bytes]:
    """GET with a browser-like UA, returning bytes or ``None`` on any error."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        logger.debug(f"news http_get failed url={url} err={exc}")
        return None


def _clean_text(raw: str) -> str:
    """Strip HTML tags + decode entities + collapse whitespace."""
    if not raw:
        return ""
    txt = _HTML_TAG_RE.sub("", raw)
    txt = unescape(txt)
    return " ".join(txt.split())


def _parse_rss_time(raw: str) -> Optional[datetime]:
    """Parse common RSS date formats, returning UTC-aware datetime or None."""
    if not raw:
        return None
    raw = raw.strip()
    # RFC 822 is by far the most common RSS format.
    for fmt in ("%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _parse_rss_items(xml_bytes: bytes, source: str, limit: int = 15) -> list[Headline]:
    """Parse an RSS 2.0 feed into :class:`Headline` objects."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.debug(f"news rss parse failed source={source} err={exc}")
        return []

    items: list[Headline] = []
    # RSS 2.0: <rss><channel><item>; Atom would need different handling
    for item in root.iter("item"):
        title = _clean_text((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        pub = _parse_rss_time(item.findtext("pubDate") or "")
        desc = _clean_text(item.findtext("description") or "")
        if not title:
            continue
        items.append(Headline(title=title, source=source,
                              url=link, published=pub, summary=desc[:280]))
        if len(items) >= limit:
            break
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Concrete providers
# ─────────────────────────────────────────────────────────────────────────────


class YahooFinanceRSS(NewsProvider):
    """Per-ticker Yahoo Finance RSS feed — no API key required."""
    name = "yahoo_finance"
    _URL_TMPL = ("https://feeds.finance.yahoo.com/rss/2.0/headline"
                 "?s={ticker}&region=US&lang=en-US")

    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        if not ticker:
            return []
        symbol = ticker.upper().strip()
        url = self._URL_TMPL.format(ticker=urllib.parse.quote(symbol))
        body = _http_get(url)
        if not body:
            return []
        return _parse_rss_items(body, source="Yahoo Finance", limit=10)


class ReutersTopNewsRSS(NewsProvider):
    """Reuters top-news RSS — macro + geopolitical coverage.

    Reuters retired their public RSS in 2020; we fall back to a Google
    News 'site:reuters.com' query feed, which is still free + unauthed."""
    name = "reuters_via_gnews"
    _URL = ("https://news.google.com/rss/search"
            "?q=site:reuters.com+when:1d&hl=en-US&gl=US&ceid=US:en")

    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        body = _http_get(self._URL)
        if not body:
            return []
        return _parse_rss_items(body, source="Reuters", limit=8)


class GoogleMarketsNewsRSS(NewsProvider):
    """Google News 'markets' topic RSS — broad macro/geopolitics layer."""
    name = "gnews_markets"
    _URL_MACRO = ("https://news.google.com/rss/headlines/section/topic/BUSINESS"
                  "?hl=en-US&gl=US&ceid=US:en")

    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        body = _http_get(self._URL_MACRO)
        if not body:
            return []
        return _parse_rss_items(body, source="Google News — Business", limit=10)


class CoinDeskRSS(NewsProvider):
    """Crypto-specific headlines — only queried for crypto tickers."""
    name = "coindesk"
    _URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"

    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        body = _http_get(self._URL)
        if not body:
            return []
        return _parse_rss_items(body, source="CoinDesk", limit=10)


class MarketWatchRSS(NewsProvider):
    """MarketWatch top-stories RSS — US equity + macro coverage."""
    name = "marketwatch"
    _URL = "https://feeds.marketwatch.com/marketwatch/topstories/"

    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        body = _http_get(self._URL)
        if not body:
            return []
        return _parse_rss_items(body, source="MarketWatch", limit=8)


class CNBCMarketsRSS(NewsProvider):
    """CNBC Markets RSS — real-time market-moving headlines."""
    name = "cnbc_markets"
    _URL = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"

    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        body = _http_get(self._URL)
        if not body:
            return []
        return _parse_rss_items(body, source="CNBC Markets", limit=8)


class BBCBusinessRSS(NewsProvider):
    """BBC Business RSS — international business + geopolitical angle."""
    name = "bbc_business"
    _URL = "https://feeds.bbci.co.uk/news/business/rss.xml"

    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        body = _http_get(self._URL)
        if not body:
            return []
        return _parse_rss_items(body, source="BBC Business", limit=6)


class BloombergGNewsRSS(NewsProvider):
    """Bloomberg coverage — Bloomberg doesn't publish a free RSS, so we
    proxy via Google News 'site:bloomberg.com' search."""
    name = "bloomberg_via_gnews"
    _URL = ("https://news.google.com/rss/search"
            "?q=site:bloomberg.com+when:1d&hl=en-US&gl=US&ceid=US:en")

    def fetch(self, ticker: Optional[str]) -> list[Headline]:
        body = _http_get(self._URL)
        if not body:
            return []
        return _parse_rss_items(body, source="Bloomberg", limit=6)


# ─────────────────────────────────────────────────────────────────────────────
# NewsFeed — façade with per-provider caching
# ─────────────────────────────────────────────────────────────────────────────


class NewsFeed:
    """High-level headline aggregator with per-feed TTL cache.

    The same instance can be reused across AI cycles — caching is
    internal and thread-safe enough for Streamlit's single-writer
    access pattern.
    """

    def __init__(
        self,
        ticker_providers: Optional[Iterable[NewsProvider]] = None,
        macro_providers: Optional[Iterable[NewsProvider]] = None,
        crypto_providers: Optional[Iterable[NewsProvider]] = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        # `is None` on purpose, not truthiness: `x or default` treats an
        # explicitly empty list the same as "not provided" and silently
        # substitutes the real defaults - so passing providers=[] to turn a
        # feed off (e.g. to cut a rate-limited or costly source) would
        # silently reactivate every default provider instead.
        self.ticker_providers = list(
            [YahooFinanceRSS()] if ticker_providers is None else ticker_providers)
        self.macro_providers = list(
            [
                ReutersTopNewsRSS(),
                BloombergGNewsRSS(),
                MarketWatchRSS(),
                CNBCMarketsRSS(),
                BBCBusinessRSS(),
                GoogleMarketsNewsRSS(),
            ] if macro_providers is None else macro_providers)
        self.crypto_providers = list(
            [CoinDeskRSS()] if crypto_providers is None else crypto_providers)
        self._ttl = ttl_seconds
        # key -> (fetched_ts, list[Headline])
        self._cache: dict[str, tuple[float, list[Headline]]] = {}

    # ---- public API ---------------------------------------------------- #

    def fetch(self, ticker: str) -> NewsBundle:
        """Return ticker-specific + macro headlines for *ticker*.

        Fully cached — repeat calls within the TTL hit memory, not HTTP.
        """
        ticker = ticker.upper().strip()
        ticker_headlines = self._collect(self.ticker_providers, ticker=ticker)
        macro_headlines  = self._collect(self.macro_providers, ticker=None,
                                         cache_key_suffix="macro")
        if _CRYPTO_RE.match(ticker):
            macro_headlines = (
                self._collect(self.crypto_providers, ticker=None,
                              cache_key_suffix="crypto")
                + macro_headlines
            )

        return NewsBundle(
            ticker=ticker,
            ticker_specific=ticker_headlines,
            macro=macro_headlines,
        )

    def macro_summary_for_prompt(self, ticker: str, max_items: int = 5) -> str:
        """Macro/geopolitics headlines only — compact block for the LLM prompt.

        Headlines are tagged with their keyword-classified sentiment so the
        LLM can weigh them faster. An aggregate bias line sits at the top.
        Returns an empty string when no macro headlines are available.
        """
        bundle = self.fetch(ticker)
        if not bundle.macro:
            return ""
        bias = bundle.sentiment_bias()
        lines: list[str] = [
            f"Aggregate macro sentiment: {bias['label'].upper()} "
            f"(score {bias['score']:+.2f} | "
            f"bull {bias['bull_count']} / bear {bias['bear_count']} / "
            f"neutral {bias['neutral_count']})",
            "",
        ]
        for h in bundle.macro[:max_items]:
            age = f"{h.age_minutes}m ago" if h.age_minutes is not None else "recent"
            tag = {"bull": "BULL", "bear": "BEAR", "neutral": "NEUT"}[h.sentiment]
            lines.append(
                f"  - [{tag}] [{h.source} · {age}] {h.short(150)}"
            )
        return "\n".join(lines)

    def summary_for_prompt(self, ticker: str, max_items: int = 6) -> str:
        """Return a compact block suitable for splicing into the LLM prompt.

        Empty string if no headlines at all — callers should add their own
        section header if they want one.
        """
        bundle = self.fetch(ticker)
        if bundle.empty:
            return ""
        lines: list[str] = []

        tspec = bundle.ticker_specific[:max_items]
        if tspec:
            lines.append(f"Ticker-specific headlines ({ticker}):")
            for h in tspec:
                age = f"{h.age_minutes}m ago" if h.age_minutes is not None else "recent"
                lines.append(f"  - [{h.source} · {age}] {h.short(150)}")

        macro = bundle.macro[:max(2, max_items // 2)]
        if macro:
            lines.append("")
            lines.append("Macro / geopolitical context:")
            for h in macro:
                age = f"{h.age_minutes}m ago" if h.age_minutes is not None else "recent"
                lines.append(f"  - [{h.source} · {age}] {h.short(150)}")
        return "\n".join(lines)

    # ---- internals ----------------------------------------------------- #

    def _collect(
        self,
        providers: Iterable[NewsProvider],
        ticker: Optional[str],
        cache_key_suffix: str = "",
    ) -> list[Headline]:
        out: list[Headline] = []
        for p in providers:
            key = f"{p.name}:{ticker or cache_key_suffix or 'macro'}"
            now = time.time()
            cached = self._cache.get(key)
            if cached and (now - cached[0] < self._ttl):
                out.extend(cached[1])
                continue
            try:
                items = p.fetch(ticker)
            except Exception as exc:
                logger.debug(f"provider {p.name} failed: {exc}")
                items = []
            self._cache[key] = (now, items)
            out.extend(items)
        # De-duplicate by title (different feeds often syndicate the same story)
        seen: set[str] = set()
        unique: list[Headline] = []
        for h in out:
            k = h.title.lower()
            if k in seen:
                continue
            seen.add(k)
            unique.append(h)
        # Newest first when we have timestamps
        unique.sort(key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True)
        return unique


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    feed = NewsFeed()
    for sym in ("AAPL", "BTC-USD"):
        print("=" * 70)
        print(f"  {sym}")
        print("=" * 70)
        print(feed.summary_for_prompt(sym, max_items=5) or "(no headlines)")
        print()
