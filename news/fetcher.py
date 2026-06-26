"""
News Fetcher — Recent headlines for a ticker.

Pulls headlines from Yahoo Finance via yfinance (no extra dependency — yfinance
is already part of the stack). Normalises the nested yfinance schema into a
simple NewsItem dataclass and caches results per-ticker to avoid hammering
Yahoo on every AUTO cycle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: str
    publisher: str
    published: Optional[datetime]
    url: str

    def age_str(self, now: Optional[datetime] = None) -> str:
        if self.published is None:
            return "unknown age"
        ref = now or datetime.now(timezone.utc)
        delta = ref - self.published
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m ago"
        if hours < 48:
            return f"{int(hours)}h ago"
        return f"{int(hours / 24)}d ago"


class NewsFetcher:
    """
    Fetches recent news headlines for a given ticker. Results are cached per
    ticker for ``ttl_seconds`` (default 600 = 10 minutes) so repeated calls
    inside the AUTO loop don't re-hit Yahoo.
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, list[NewsItem]]] = {}

    def fetch(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        now = time.time()
        cached = self._cache.get(ticker)
        if cached and (now - cached[0] < self._ttl):
            return cached[1][:limit]

        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as exc:
            logger.warning(f"News fetch failed for {ticker}: {exc}")
            self._cache[ticker] = (now, [])
            return []

        items = [self._normalise(entry) for entry in raw]
        items = [i for i in items if i is not None]
        items.sort(key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        self._cache[ticker] = (now, items)
        return items[:limit]

    @staticmethod
    def _normalise(entry: dict) -> Optional[NewsItem]:
        """Flatten yfinance's nested v2 schema into a NewsItem."""
        content = entry.get("content") or entry
        title = (content.get("title") or "").strip()
        if not title:
            return None

        summary = (content.get("summary") or content.get("description") or "").strip()
        provider = ((content.get("provider") or {}).get("displayName")) or ""
        url = ((content.get("canonicalUrl") or {}).get("url")) or entry.get("link") or ""

        pub_raw = content.get("pubDate") or content.get("displayTime")
        published: Optional[datetime] = None
        if isinstance(pub_raw, str):
            try:
                # Yahoo returns ISO-8601 Zulu, e.g. "2026-04-16T19:02:44Z"
                published = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            except ValueError:
                published = None
        elif isinstance(pub_raw, (int, float)):
            published = datetime.fromtimestamp(int(pub_raw), tz=timezone.utc)

        return NewsItem(
            title=title, summary=summary, publisher=provider,
            published=published, url=url,
        )

    @staticmethod
    def format_for_prompt(items: list[NewsItem], max_summary_chars: int = 220) -> str:
        """Format news items as a compact prompt section (no HTML)."""
        if not items:
            return "(no recent headlines available)"
        lines: list[str] = []
        for i, n in enumerate(items, start=1):
            summary = n.summary[:max_summary_chars].rstrip()
            if len(n.summary) > max_summary_chars:
                summary += "…"
            head = f"{i}. [{n.age_str()}] {n.title}"
            if n.publisher:
                head += f"  — {n.publisher}"
            lines.append(head + (f"\n   {summary}" if summary else ""))
        return "\n".join(lines)
