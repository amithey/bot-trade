"""Bounded, non-blocking public news context for the dashboard's research report.

One daemon refreshes at a time. The trading thread never waits on RSS; news
is displayed with sources and dates and is not a keyword-driven order trigger.
"""
from collections import OrderedDict
from datetime import datetime, timezone
import threading
import time

_lock = threading.Lock()
_cache = OrderedDict()
_busy = False
_TTL = 900


def _refresh(ticker):
    global _busy
    result = {"status": "unavailable", "items": [], "fetched_at": datetime.now(timezone.utc).isoformat()}
    try:
        from market_data.news import NewsFeed
        bundle = NewsFeed().fetch(ticker)
        now = datetime.now(timezone.utc)
        recent = []
        for item in bundle.all:
            if item.published is None:
                continue
            published = item.published
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if not 0 <= (now - published).total_seconds() <= 72 * 3600:
                continue
            recent.append({"title": item.title, "source": item.source,
                           "url": item.url, "published": published.isoformat()})
        result.update(status="available" if recent else "unavailable", items=recent[:8])
    except Exception:
        result["status"] = "unavailable"
    finally:
        with _lock:
            _cache[ticker] = (time.monotonic(), result)
            _cache.move_to_end(ticker)
            while len(_cache) > 32:
                _cache.popitem(last=False)
            _busy = False


def news_context(ticker: str) -> dict:
    """Return recent context or a loading state; never serve stale news as fresh."""
    global _busy
    with _lock:
        cached = _cache.get(ticker)
        if cached and time.monotonic() - cached[0] < _TTL:
            return {**cached[1], "items": [dict(i) for i in cached[1]["items"]]}
        if not _busy:
            _busy = True
            threading.Thread(target=_refresh, args=(ticker,), daemon=True,
                             name="bt_research_news").start()
    return {"status": "loading", "items": [], "fetched_at": None}
