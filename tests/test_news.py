"""
Tests for the two news modules: news/fetcher.py (per-ticker Yahoo headlines
via yfinance) and market_data/news.py (multi-provider RSS aggregation, the
keyword sentiment classifier, and the macro/ticker headline bundle fed into
Claude's prompt).

Both shipped with zero tests despite being pure aggregation/classification
logic that is easy to get subtly wrong - a sentiment lexicon that double-
counts, a dedup key that doesn't actually dedup, a cache that never expires.
Nothing here touches the network: yfinance calls are mocked at `yf.Ticker`,
and RSS providers are exercised either through a fake NewsProvider or with
`_http_get` mocked so no request leaves the process.
"""
from __future__ import annotations

import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from news.fetcher import NewsFetcher, NewsItem
import market_data.news as newsmod
from market_data.news import (
    Headline, NewsBundle, NewsFeed, NewsProvider,
    _classify_headline, _clean_text, _http_get, _parse_rss_items,
    _parse_rss_time, YahooFinanceRSS,
)


# =========================================================================== #
# news/fetcher.py
# =========================================================================== #
def test_age_str_unknown_when_no_published_date():
    assert NewsItem("t", "s", "p", None, "u").age_str() == "unknown age"


def test_age_str_minutes_under_an_hour():
    now = datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc)
    item = NewsItem("t", "s", "p", now - timedelta(minutes=20), "u")
    assert item.age_str(now) == "20m ago"


def test_age_str_hours_under_two_days():
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    item = NewsItem("t", "s", "p", now - timedelta(hours=5), "u")
    assert item.age_str(now) == "5h ago"


def test_age_str_days_beyond_two_days():
    now = datetime(2024, 1, 5, 12, 0, tzinfo=timezone.utc)
    item = NewsItem("t", "s", "p", now - timedelta(days=3), "u")
    assert item.age_str(now) == "3d ago"


def _yf_entry(title="Some headline", summary="A summary.", provider="Reuters",
              url="https://example.com/a", pub="2026-01-15T10:00:00Z"):
    return {"content": {
        "title": title, "summary": summary,
        "provider": {"displayName": provider},
        "canonicalUrl": {"url": url}, "pubDate": pub,
    }}


def test_normalise_flattens_the_v2_content_schema():
    item = NewsFetcher._normalise(_yf_entry())
    assert item.title == "Some headline"
    assert item.publisher == "Reuters"
    assert item.url == "https://example.com/a"
    assert item.published == datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)


def test_normalise_falls_back_to_flat_schema_without_content_key():
    item = NewsFetcher._normalise({"title": "Flat headline"})
    assert item.title == "Flat headline"


def test_normalise_returns_none_for_a_missing_title():
    assert NewsFetcher._normalise({"content": {"title": ""}}) is None
    assert NewsFetcher._normalise({}) is None


def test_normalise_parses_a_unix_timestamp_pubdate():
    entry = _yf_entry(pub=1_700_000_000)
    item = NewsFetcher._normalise(entry)
    assert item.published == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)


def test_normalise_treats_an_unparseable_date_as_none():
    item = NewsFetcher._normalise(_yf_entry(pub="not a date"))
    assert item.published is None


def test_normalise_falls_back_to_the_flat_link_when_no_canonical_url():
    entry = {"content": {"title": "x"}, "link": "https://fallback.example"}
    assert NewsFetcher._normalise(entry).url == "https://fallback.example"


def test_fetcher_caches_within_the_ttl(monkeypatch):
    calls = {"n": 0}

    class FakeTicker:
        def __init__(self, symbol):
            calls["n"] += 1

        @property
        def news(self):
            return [_yf_entry()]

    monkeypatch.setattr("news.fetcher.yf.Ticker", FakeTicker)
    f = NewsFetcher(ttl_seconds=600)
    f.fetch("AAPL")
    f.fetch("AAPL")
    assert calls["n"] == 1


def test_fetcher_refetches_after_ttl_expires(monkeypatch):
    calls = {"n": 0}

    class FakeTicker:
        def __init__(self, symbol):
            calls["n"] += 1

        @property
        def news(self):
            return [_yf_entry()]

    monkeypatch.setattr("news.fetcher.yf.Ticker", FakeTicker)
    clock = {"t": 1000.0}
    monkeypatch.setattr("news.fetcher.time.time", lambda: clock["t"])
    f = NewsFetcher(ttl_seconds=600)
    f.fetch("AAPL")
    clock["t"] += 601
    f.fetch("AAPL")
    assert calls["n"] == 2


def test_fetcher_returns_and_caches_empty_on_api_failure(monkeypatch):
    calls = {"n": 0}

    class FakeTicker:
        def __init__(self, symbol):
            calls["n"] += 1

        @property
        def news(self):
            raise RuntimeError("yahoo is down")

    monkeypatch.setattr("news.fetcher.yf.Ticker", FakeTicker)
    f = NewsFetcher(ttl_seconds=600)
    assert f.fetch("AAPL") == []
    assert f.fetch("AAPL") == []
    assert calls["n"] == 1, "a failure should be cached, not retried every call"


def test_fetcher_sorts_newest_first_and_respects_limit(monkeypatch):
    entries = [
        _yf_entry(title="old", pub="2026-01-01T00:00:00Z"),
        _yf_entry(title="new", pub="2026-01-15T00:00:00Z"),
        _yf_entry(title="mid", pub="2026-01-10T00:00:00Z"),
    ]

    class FakeTicker:
        def __init__(self, symbol): pass
        @property
        def news(self): return entries

    monkeypatch.setattr("news.fetcher.yf.Ticker", FakeTicker)
    items = NewsFetcher().fetch("AAPL", limit=2)
    assert [i.title for i in items] == ["new", "mid"]


def test_format_for_prompt_handles_the_empty_case():
    assert NewsFetcher.format_for_prompt([]) == "(no recent headlines available)"


def test_format_for_prompt_truncates_long_summaries():
    item = NewsItem("Title", "x" * 300, "Pub", None, "u")
    out = NewsFetcher.format_for_prompt([item], max_summary_chars=50)
    assert "…" in out
    assert "x" * 51 not in out


def test_format_for_prompt_omits_publisher_when_absent():
    item = NewsItem("Title", "", "", None, "u")
    out = NewsFetcher.format_for_prompt([item])
    assert "—" not in out


# =========================================================================== #
# market_data/news.py — sentiment classifier
# =========================================================================== #
def test_classify_empty_title_is_neutral():
    assert _classify_headline("") == ("neutral", 0)


def test_classify_a_clean_bull_bigram():
    # Neither "cut" nor "rates" is also a scored unigram, so this isolates
    # the bigram weight (+2) from any unigram double-count.
    label, score = _classify_headline("Fed expected to cut rates further")
    assert (label, score) == ("bull", 2)


def test_classify_a_clean_bear_bigram():
    label, score = _classify_headline("Central bank signals a rate hike")
    assert (label, score) == ("bear", -2)


def test_classify_bull_unigram():
    label, score = _classify_headline("Shares rally after announcement")
    assert label == "bull" and score > 0


def test_classify_bear_unigram():
    label, score = _classify_headline("Company shares plunge on news")
    assert label == "bear" and score < 0


def test_classify_is_case_insensitive():
    assert _classify_headline("STOCK CRASHES AMID SELLOFF")[0] == "bear"


def test_classify_score_is_clamped_to_plus_minus_three():
    # Stack several bull signals to try to exceed the ceiling.
    label, score = _classify_headline(
        "Stock surges to a record high, rallies on beats estimates and buyback announced")
    assert label == "bull" and score == 3


def test_classify_mixed_signals_can_cancel_toward_neutral():
    label, score = _classify_headline("Stock rallies then crashes")
    assert score == 0 and label == "neutral"


# =========================================================================== #
# Headline
# =========================================================================== #
def test_headline_sentiment_properties_delegate_to_the_classifier():
    h = Headline("Shares rally on strong demand", "src", "url", None)
    assert h.sentiment == "bull"
    assert h.sentiment_score > 0


def test_headline_age_minutes_is_none_without_a_timestamp():
    assert Headline("t", "s", "u", None).age_minutes is None


def test_headline_age_minutes_is_computed_from_published():
    published = datetime.now(timezone.utc) - timedelta(minutes=10)
    age = Headline("t", "s", "u", published).age_minutes
    assert 9 <= age <= 11


def test_headline_short_truncates_long_titles():
    h = Headline("x" * 200, "s", "u", None)
    out = h.short(max_len=50)
    assert len(out) == 50 and out.endswith("…")


def test_headline_short_leaves_short_titles_alone():
    h = Headline("Short title", "s", "u", None)
    assert h.short(max_len=50) == "Short title"


# =========================================================================== #
# NewsBundle
# =========================================================================== #
def _h(title, published=None, sentiment_ok=True):
    return Headline(title, "src", "url", published)


def test_bundle_all_concatenates_ticker_and_macro():
    b = NewsBundle("AAPL", ticker_specific=[_h("a")], macro=[_h("b")])
    assert [h.title for h in b.all] == ["a", "b"]


def test_bundle_empty_flag():
    assert NewsBundle("AAPL").empty is True
    assert NewsBundle("AAPL", ticker_specific=[_h("a")]).empty is False


def test_sentiment_bias_on_an_empty_bundle():
    bias = NewsBundle("AAPL").sentiment_bias()
    assert bias == {"score": 0.0, "label": "neutral", "bull_count": 0,
                    "bear_count": 0, "neutral_count": 0, "strongest": None}


def test_sentiment_bias_all_bullish():
    heads = [_h("Shares rally on strong demand"), _h("Stock surges to a record high")]
    bias = NewsBundle("AAPL", ticker_specific=heads).sentiment_bias()
    assert bias["label"] == "bullish"
    assert bias["bull_count"] == 2 and bias["bear_count"] == 0


def test_sentiment_bias_all_bearish():
    heads = [_h("Company shares plunge on news"), _h("Stock crashes amid selloff")]
    bias = NewsBundle("AAPL", ticker_specific=heads).sentiment_bias()
    assert bias["label"] == "bearish"


def test_sentiment_bias_balanced_bull_and_bear_reads_mixed():
    heads = [_h("Shares rally on strong demand"),
            _h("Company shares plunge on news")]
    bias = NewsBundle("AAPL", ticker_specific=heads).sentiment_bias()
    assert bias["label"] == "mixed"


def test_sentiment_bias_strongest_picks_the_most_extreme_headline():
    mild = _h("Shares gain slightly")               # +1 unigram
    strong = _h("Stock surges to a record high")     # bigram + unigram, higher magnitude
    bias = NewsBundle("AAPL", ticker_specific=[mild, strong]).sentiment_bias()
    assert bias["strongest"].title == strong.title


# =========================================================================== #
# RSS plumbing
# =========================================================================== #
def test_http_get_returns_bytes_on_success(monkeypatch):
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"<rss></rss>"
    monkeypatch.setattr(newsmod.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    assert _http_get("https://x") == b"<rss></rss>"


def test_http_get_never_raises_on_a_network_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("no route to host")
    monkeypatch.setattr(newsmod.urllib.request, "urlopen", boom)
    assert _http_get("https://x") is None


def test_clean_text_strips_tags_and_decodes_entities():
    assert _clean_text("<p>Rates &amp; bonds</p>") == "Rates & bonds"


def test_clean_text_collapses_whitespace():
    assert _clean_text("line one\n\n   line two") == "line one line two"


def test_clean_text_handles_empty_input():
    assert _clean_text("") == ""


@pytest.mark.parametrize("raw,expected", [
    ("Mon, 01 Jan 2024 12:00:00 +0000", datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)),
    ("2024-01-01T12:00:00Z", datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)),
    ("2024-01-01T12:00:00+00:00", datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)),
])
def test_parse_rss_time_supported_formats(raw, expected):
    assert _parse_rss_time(raw) == expected


def test_parse_rss_time_returns_none_for_garbage():
    assert _parse_rss_time("not a date at all") is None
    assert _parse_rss_time("") is None


SAMPLE_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>First &amp; Best</title><link>https://a</link>
<pubDate>Mon, 01 Jan 2024 12:00:00 +0000</pubDate>
<description>&lt;p&gt;Some description&lt;/p&gt;</description></item>
<item><title>Second</title><link>https://b</link>
<pubDate>Tue, 02 Jan 2024 12:00:00 +0000</pubDate></item>
<item><title></title><link>https://c</link></item>
</channel></rss>"""


def test_parse_rss_items_builds_headlines_and_cleans_html():
    items = _parse_rss_items(SAMPLE_RSS, source="Test Feed")
    assert len(items) == 2   # the empty-title item is skipped
    assert items[0].title == "First & Best"
    assert items[0].summary == "Some description"
    assert items[0].source == "Test Feed"


def test_parse_rss_items_respects_the_limit():
    items = _parse_rss_items(SAMPLE_RSS, source="Test Feed", limit=1)
    assert len(items) == 1


def test_parse_rss_items_returns_empty_on_malformed_xml():
    assert _parse_rss_items(b"not xml at all <<<", source="Test Feed") == []


# =========================================================================== #
# Concrete providers — network mocked at _http_get
# =========================================================================== #
def test_yahoo_finance_rss_requires_a_ticker():
    assert YahooFinanceRSS().fetch(None) == []


def test_yahoo_finance_rss_builds_a_per_ticker_url(monkeypatch):
    seen = {}

    def fake_get(url, timeout=6):
        seen["url"] = url
        return SAMPLE_RSS

    monkeypatch.setattr(newsmod, "_http_get", fake_get)
    YahooFinanceRSS().fetch("aapl")
    assert "s=AAPL" in seen["url"]


def test_a_provider_returns_empty_when_http_get_fails(monkeypatch):
    monkeypatch.setattr(newsmod, "_http_get", lambda *a, **kw: None)
    assert YahooFinanceRSS().fetch("AAPL") == []


# =========================================================================== #
# NewsFeed — aggregation, caching, crypto routing
# =========================================================================== #
class _FakeProvider(NewsProvider):
    def __init__(self, name, headlines=None, raises=False):
        self.name = name
        self._headlines = headlines or []
        self._raises = raises
        self.calls = 0

    def fetch(self, ticker):
        self.calls += 1
        if self._raises:
            raise RuntimeError("provider exploded")
        return list(self._headlines)


def test_an_explicitly_empty_provider_list_is_honoured_not_replaced():
    """Regression guard: `providers or [default]` treats [] as falsy and
    silently substitutes the real default providers - discovered because
    several tests below that passed providers=[] to disable a feed
    unexpectedly triggered live HTTP requests to Yahoo/Reuters/Google News.
    An explicit [] must mean "no providers", full stop; omitting the
    argument (None) is the only way to get the defaults."""
    feed = NewsFeed(ticker_providers=[], macro_providers=[], crypto_providers=[])
    assert feed.ticker_providers == []
    assert feed.macro_providers == []
    assert feed.crypto_providers == []


def test_omitting_providers_still_gets_the_real_defaults():
    feed = NewsFeed()
    assert isinstance(feed.ticker_providers[0], YahooFinanceRSS)
    assert len(feed.macro_providers) == 6
    assert len(feed.crypto_providers) == 1


def test_fetch_combines_ticker_and_macro_providers():
    tp = _FakeProvider("t", [_h("ticker news")])
    mp = _FakeProvider("m", [_h("macro news")])
    feed = NewsFeed(ticker_providers=[tp], macro_providers=[mp], crypto_providers=[])
    bundle = feed.fetch("AAPL")
    assert [h.title for h in bundle.ticker_specific] == ["ticker news"]
    assert [h.title for h in bundle.macro] == ["macro news"]


def test_crypto_providers_only_fire_for_crypto_tickers():
    crypto = _FakeProvider("crypto", [_h("crypto news")])
    macro = _FakeProvider("macro", [_h("macro news")])
    feed = NewsFeed(ticker_providers=[], macro_providers=[macro],
                    crypto_providers=[crypto])

    feed.fetch("AAPL")
    assert crypto.calls == 0, "a non-crypto ticker must not query crypto providers"

    feed.fetch("BTC-USD")
    assert crypto.calls == 1


def test_crypto_headlines_are_prepended_to_macro():
    crypto = _FakeProvider("crypto", [_h("crypto news")])
    macro = _FakeProvider("macro", [_h("macro news")])
    feed = NewsFeed(ticker_providers=[], macro_providers=[macro],
                    crypto_providers=[crypto])
    bundle = feed.fetch("ETH-USD")
    assert bundle.macro[0].title == "crypto news"


def test_a_failing_provider_does_not_break_the_whole_fetch():
    ok = _FakeProvider("ok", [_h("still works")])
    broken = _FakeProvider("broken", raises=True)
    feed = NewsFeed(ticker_providers=[ok, broken], macro_providers=[], crypto_providers=[])
    bundle = feed.fetch("AAPL")
    assert [h.title for h in bundle.ticker_specific] == ["still works"]


def test_duplicate_titles_across_providers_are_deduplicated():
    a = _FakeProvider("a", [_h("Same Story")])
    b = _FakeProvider("b", [_h("same story")])   # different case, same story
    feed = NewsFeed(ticker_providers=[a, b], macro_providers=[], crypto_providers=[])
    bundle = feed.fetch("AAPL")
    assert len(bundle.ticker_specific) == 1


def test_provider_results_are_cached_within_the_ttl():
    p = _FakeProvider("p", [_h("news")])
    feed = NewsFeed(ticker_providers=[p], macro_providers=[], crypto_providers=[],
                    ttl_seconds=600)
    feed.fetch("AAPL")
    feed.fetch("AAPL")
    assert p.calls == 1


def test_provider_cache_expires_after_ttl(monkeypatch):
    p = _FakeProvider("p", [_h("news")])
    feed = NewsFeed(ticker_providers=[p], macro_providers=[], crypto_providers=[],
                    ttl_seconds=600)
    clock = {"t": 1000.0}
    monkeypatch.setattr(newsmod.time, "time", lambda: clock["t"])
    feed.fetch("AAPL")
    clock["t"] += 601
    feed.fetch("AAPL")
    assert p.calls == 2


def test_headlines_sort_newest_first_with_undated_last():
    now = datetime.now(timezone.utc)
    old = Headline("old", "s", "u", now - timedelta(days=1))
    new = Headline("new", "s", "u", now)
    undated = Headline("undated", "s", "u", None)
    p = _FakeProvider("p", [old, undated, new])
    feed = NewsFeed(ticker_providers=[p], macro_providers=[], crypto_providers=[])
    bundle = feed.fetch("AAPL")
    assert [h.title for h in bundle.ticker_specific] == ["new", "old", "undated"]


def test_summary_for_prompt_is_empty_for_an_empty_bundle():
    feed = NewsFeed(ticker_providers=[], macro_providers=[], crypto_providers=[])
    assert feed.summary_for_prompt("AAPL") == ""


def test_summary_for_prompt_includes_both_sections():
    tp = _FakeProvider("t", [_h("ticker headline")])
    mp = _FakeProvider("m", [_h("macro headline")])
    feed = NewsFeed(ticker_providers=[tp], macro_providers=[mp], crypto_providers=[])
    out = feed.summary_for_prompt("AAPL")
    assert "Ticker-specific headlines (AAPL):" in out
    assert "ticker headline" in out
    assert "Macro / geopolitical context:" in out


def test_macro_summary_for_prompt_is_empty_with_no_macro_headlines():
    feed = NewsFeed(ticker_providers=[], macro_providers=[], crypto_providers=[])
    assert feed.macro_summary_for_prompt("AAPL") == ""


def test_macro_summary_for_prompt_includes_the_aggregate_bias_and_tags():
    mp = _FakeProvider("m", [_h("Shares rally on strong demand")])
    feed = NewsFeed(ticker_providers=[], macro_providers=[mp], crypto_providers=[])
    out = feed.macro_summary_for_prompt("AAPL")
    assert "Aggregate macro sentiment: BULLISH" in out
    assert "[BULL]" in out
