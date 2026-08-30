"""
Tests for knowledge_ingestion/ — article/YouTube/playlist scrapers and the
static seed-data ingestors.

No real network, no real ChromaDB, no real YouTube/yt-dlp calls anywhere:
- HTTP (requests) is mocked with small fake response objects.
- ChromaDB is mocked with a small in-memory FakeCollection standing in for
  `_get_collection()` (article/YouTube scrapers) or for
  `chromadb.PersistentClient`/`SentenceTransformerEmbeddingFunction`
  (the seed_* modules, which build the client inline in `ingest_all()`).
- yt-dlp and youtube-transcript-api are mocked with small fakes matching
  the exact call shapes the source uses.
"""
from __future__ import annotations

import sys
import types

import pytest

import knowledge_ingestion.article_scraper as article_mod
from knowledge_ingestion.article_scraper import (
    ArticleScraper,
    ArticleIngestionResult,
    _TextExtractor,
    extract_readable_text,
)

import knowledge_ingestion.youtube_scraper as yt_mod
from knowledge_ingestion.youtube_scraper import (
    YouTubeScraper,
    IngestionResult,
    _TranscriptData,
)
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api._errors import YouTubeRequestFailed

import knowledge_ingestion.playlist_scraper as playlist_mod
from knowledge_ingestion.playlist_scraper import (
    PlaylistIngestor,
    PlaylistEntry,
    PlaylistIngestionSummary,
)


# --------------------------------------------------------------------------- #
# Shared fake ChromaDB collection
# --------------------------------------------------------------------------- #
class FakeCollection:
    def __init__(self, existing_ids: set[str] | None = None, get_raises: bool = False):
        self.existing_ids = set(existing_ids or set())
        self.get_raises = get_raises
        self.upserted: list[dict] = []

    def get(self, ids, include=None):
        if self.get_raises:
            raise RuntimeError("chroma down")
        return {"ids": [i for i in ids if i in self.existing_ids]}

    def upsert(self, ids, documents, metadatas):
        self.upserted.append({"ids": list(ids), "documents": list(documents),
                               "metadatas": list(metadatas)})
        self.existing_ids.update(ids)

    def count(self):
        return len(self.existing_ids)


class FakeResponse:
    def __init__(self, text: str, status: int = 200, content_type: str = "text/html"):
        self.text = text
        self.status_code = status
        self.headers = {"Content-Type": content_type}

    # _fetch follows redirects by hand so it can re-validate each hop, so a
    # response double has to answer these.
    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308)

    is_permanent_redirect = is_redirect

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every host to a public address so the SSRF guard permits it.

    Without this the guard does real DNS — these tests are about fetching and
    parsing, not about the address rules, which tests/test_url_guard.py covers.
    """
    monkeypatch.setattr("knowledge_ingestion.url_guard._resolve",
                        lambda host: ["93.184.216.34"])


ARTICLE_HTML = """
<html><head><title>My Article | SiteName</title></head>
<body>
<nav>Home | About</nav>
<script>var x = 1;</script>
<h1>Introduction</h1>
<p>""" + ("This is a sufficiently long paragraph about trading strategies. " * 10) + """</p>
<p>""" + ("A second paragraph with more detail about risk management. " * 10) + """</p>
<footer>Copyright 2026</footer>
</body></html>
"""


# --------------------------------------------------------------------------- #
# article_scraper: _TextExtractor / extract_readable_text
# --------------------------------------------------------------------------- #
def test_extract_readable_text_drops_nav_script_and_footer():
    title, body = extract_readable_text(ARTICLE_HTML)
    assert "Home | About" not in body
    assert "var x = 1" not in body
    assert "Copyright 2026" not in body
    assert "trading strategies" in body
    assert "risk management" in body


def test_extract_readable_text_prefers_og_title_and_strips_site_suffix():
    html = """<html><head>
    <meta property="og:title" content="Best Title Ever | Site" />
    <title>Fallback Title</title>
    </head><body><p>hello world content here that is long enough to matter maybe not</p></body></html>"""
    title, _ = extract_readable_text(html)
    assert title == "Best Title Ever"


def test_extract_readable_text_falls_back_to_title_tag_when_no_og_title():
    html = "<html><head><title>Plain Title</title></head><body><p>x</p></body></html>"
    title, _ = extract_readable_text(html)
    assert title == "Plain Title"


def test_extract_readable_text_drops_short_crumbs():
    html = "<html><body><p>Home</p><p>" + ("real content here " * 5) + "</p></body></html>"
    _, body = extract_readable_text(html)
    assert "Home" not in body


def test_extract_readable_text_never_raises_on_malformed_html():
    title, body = extract_readable_text("<html><p>unclosed <div oops")
    assert isinstance(title, str)
    assert isinstance(body, str)


# --------------------------------------------------------------------------- #
# article_scraper: ArticleScraper._fetch
# --------------------------------------------------------------------------- #
def test_fetch_rejects_non_http_urls():
    with pytest.raises(ValueError, match="only http"):
        ArticleScraper._fetch("ftp://example.com/x")


def test_fetch_rejects_non_html_content_type(monkeypatch, public_dns):
    monkeypatch.setattr(
        article_mod.requests, "get",
        lambda *a, **kw: FakeResponse("{}", content_type="application/json"),
    )
    with pytest.raises(ValueError, match="did not return an HTML page"):
        ArticleScraper._fetch("https://example.com/data.json")


def test_fetch_returns_text_on_success(monkeypatch, public_dns):
    monkeypatch.setattr(
        article_mod.requests, "get",
        lambda *a, **kw: FakeResponse("<html>hi</html>"),
    )
    assert ArticleScraper._fetch("https://example.com/a") == "<html>hi</html>"


# --------------------------------------------------------------------------- #
# article_scraper: ArticleScraper._chunk_id
# --------------------------------------------------------------------------- #
def test_chunk_id_is_deterministic_and_16_hex_chars():
    a = ArticleScraper._chunk_id("https://x.com", 0, "hello world")
    b = ArticleScraper._chunk_id("https://x.com", 0, "hello world")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_chunk_id_differs_by_index():
    a = ArticleScraper._chunk_id("https://x.com", 0, "hello")
    b = ArticleScraper._chunk_id("https://x.com", 1, "hello")
    assert a != b


# --------------------------------------------------------------------------- #
# article_scraper: ArticleScraper.ingest — full flow
# --------------------------------------------------------------------------- #
def test_ingest_fails_gracefully_when_fetch_raises(monkeypatch):
    scraper = ArticleScraper()
    monkeypatch.setattr(scraper, "_fetch", staticmethod(
        lambda url: (_ for _ in ()).throw(RuntimeError("timeout"))))
    result = scraper.ingest("https://example.com/a")
    assert isinstance(result, ArticleIngestionResult)
    assert result.success is False
    assert "fetch error" in result.error


def test_ingest_fails_when_page_too_short(monkeypatch):
    scraper = ArticleScraper()
    monkeypatch.setattr(article_mod.ArticleScraper, "_fetch",
                         staticmethod(lambda url: "<html><body><p>too short</p></body></html>"))
    result = scraper.ingest("https://example.com/thin")
    assert result.success is False
    assert "readable characters" in result.error


def test_ingest_succeeds_and_upserts_new_chunks(monkeypatch):
    scraper = ArticleScraper()
    monkeypatch.setattr(article_mod.ArticleScraper, "_fetch",
                         staticmethod(lambda url: ARTICLE_HTML))
    fake_col = FakeCollection()
    monkeypatch.setattr(scraper, "_get_collection", lambda: fake_col)

    result = scraper.ingest("https://example.com/trading-101")
    assert result.success is True
    assert result.chunks_added > 0
    assert result.chunks_skipped == 0
    assert result.domain == "example.com"
    assert len(fake_col.upserted) == 1


def test_ingest_skips_chunks_that_already_exist(monkeypatch):
    scraper = ArticleScraper()
    monkeypatch.setattr(article_mod.ArticleScraper, "_fetch",
                         staticmethod(lambda url: ARTICLE_HTML))
    # Pre-populate with the same content so re-ingest sees them as duplicates.
    fake_col = FakeCollection()
    monkeypatch.setattr(scraper, "_get_collection", lambda: fake_col)
    first = scraper.ingest("https://example.com/dup")
    second = scraper.ingest("https://example.com/dup")
    assert first.chunks_added > 0
    assert second.chunks_added == 0
    assert second.chunks_skipped == first.chunks_added


def test_ingest_uses_title_override_when_given(monkeypatch):
    scraper = ArticleScraper()
    monkeypatch.setattr(article_mod.ArticleScraper, "_fetch",
                         staticmethod(lambda url: ARTICLE_HTML))
    monkeypatch.setattr(scraper, "_get_collection", lambda: FakeCollection())
    result = scraper.ingest("https://example.com/x", title="Custom Title")
    assert result.title == "Custom Title"


def test_ingest_reports_chromadb_upsert_errors(monkeypatch):
    scraper = ArticleScraper()
    monkeypatch.setattr(article_mod.ArticleScraper, "_fetch",
                         staticmethod(lambda url: ARTICLE_HTML))

    class ExplodingCollection(FakeCollection):
        def upsert(self, *a, **kw):
            raise RuntimeError("write failed")

    monkeypatch.setattr(scraper, "_get_collection", lambda: ExplodingCollection())
    result = scraper.ingest("https://example.com/boom")
    assert result.success is False
    assert "ChromaDB upsert error" in result.error


def test_article_ingestion_result_str_includes_status_and_title():
    r = ArticleIngestionResult(
        url="u", title="A Title", domain="d.com", chunks_added=3,
        chunks_skipped=1, total_characters=500, elapsed_seconds=1.5, success=True,
    )
    s = str(r)
    assert "OK" in s and "A Title" in s and "added=3" in s


# --------------------------------------------------------------------------- #
# youtube_scraper: _extract_video_id
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=30s", "dQw4w9WgXcQ"),
])
def test_extract_video_id_recognises_common_formats(url, expected):
    assert YouTubeScraper._extract_video_id(url) == expected


@pytest.mark.parametrize("url", [
    "https://example.com/watch?v=dQw4w9WgXcQ",
    "not a url at all",
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/",
    "",
])
def test_extract_video_id_returns_none_for_unrecognised_input(url):
    assert YouTubeScraper._extract_video_id(url) is None


# --------------------------------------------------------------------------- #
# youtube_scraper: _clean_text
# --------------------------------------------------------------------------- #
def test_clean_text_strips_music_notes_brackets_timestamps_and_filler():
    scraper = YouTubeScraper()
    raw = "So um, [Music] the RSI is (applause) at 1:23 like really strong &amp; rising"
    cleaned = scraper._clean_text(raw)
    assert "[Music]" not in cleaned
    assert "(applause)" not in cleaned
    assert "1:23" not in cleaned
    assert " um" not in cleaned.lower().replace("umbrella", "")
    assert "&amp;" not in cleaned
    assert "&" in cleaned  # unescaped


def test_clean_text_collapses_repeated_whitespace_and_newlines():
    scraper = YouTubeScraper()
    cleaned = scraper._clean_text("a   b\n\n\n\nc")
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


# --------------------------------------------------------------------------- #
# youtube_scraper: _split_text / _make_chunk_id
# --------------------------------------------------------------------------- #
def test_split_text_removes_empty_chunks():
    scraper = YouTubeScraper(chunk_size=200, chunk_overlap=20)
    chunks = scraper._split_text("word " * 80)
    assert all(c.strip() for c in chunks)
    assert len(chunks) > 1


def test_constructor_honours_an_explicit_zero_chunk_overlap():
    """Regression: `chunk_overlap or settings.chunk_overlap` used to treat an
    explicitly-passed 0 as "not provided" and silently substitute the real
    default instead — see the fix in YouTubeScraper.__init__."""
    scraper = YouTubeScraper(chunk_size=50, chunk_overlap=0)
    assert scraper._chunk_overlap == 0
    assert scraper._chunk_size == 50
    # And it must actually reach the real splitter without raising
    # "chunk overlap larger than chunk size".
    chunks = scraper._split_text("word " * 40)
    assert all(c.strip() for c in chunks)


def test_make_chunk_id_is_deterministic_16_hex_chars():
    a = YouTubeScraper._make_chunk_id("vid123456AB", 0, "hello")
    b = YouTubeScraper._make_chunk_id("vid123456AB", 0, "hello")
    assert a == b
    assert len(a) == 16


# --------------------------------------------------------------------------- #
# youtube_scraper: ingest() — URL parsing and transcript-error branches
# --------------------------------------------------------------------------- #
def test_ingest_fails_when_url_unparseable():
    scraper = YouTubeScraper()
    result = scraper.ingest("https://example.com/not-youtube")
    assert result.success is False
    assert "Could not parse" in result.error


def test_ingest_uses_video_id_as_title_when_none_given(monkeypatch):
    scraper = YouTubeScraper()
    monkeypatch.setattr(scraper, "_fetch_transcript",
                         lambda vid: (_ for _ in ()).throw(TranscriptsDisabled("dQw4w9WgXcQ")))
    result = scraper.ingest("https://youtu.be/dQw4w9WgXcQ")
    assert result.title == "dQw4w9WgXcQ"
    assert result.success is False
    assert "No captions" in result.error


def test_ingest_handles_video_unavailable(monkeypatch):
    scraper = YouTubeScraper()
    monkeypatch.setattr(scraper, "_fetch_transcript",
                         lambda vid: (_ for _ in ()).throw(VideoUnavailable("dQw4w9WgXcQ")))
    result = scraper.ingest("https://youtu.be/dQw4w9WgXcQ")
    assert result.success is False
    assert "unavailable" in result.error


def test_ingest_handles_youtube_request_failed(monkeypatch):
    scraper = YouTubeScraper()

    def _raise(vid):
        raise YouTubeRequestFailed("http://x", Exception("network"))

    monkeypatch.setattr(scraper, "_fetch_transcript", _raise)
    result = scraper.ingest("https://youtu.be/dQw4w9WgXcQ")
    assert result.success is False
    assert "YouTube API request failed" in result.error


def test_ingest_succeeds_end_to_end(monkeypatch):
    scraper = YouTubeScraper()
    transcript_text = "This is a great trading strategy explained in detail. " * 20
    monkeypatch.setattr(
        scraper, "_fetch_transcript",
        lambda vid: _TranscriptData(video_id=vid, raw_text=transcript_text,
                                     clean_text=transcript_text, language="en (manual)"),
    )
    fake_col = FakeCollection()
    monkeypatch.setattr(scraper, "_get_collection", lambda: fake_col)

    result = scraper.ingest("https://youtu.be/dQw4w9WgXcQ", title="Great Video")
    assert result.success is True
    assert result.video_id == "dQw4w9WgXcQ"
    assert result.chunks_added > 0
    assert result.total_characters == len(transcript_text)
    assert len(fake_col.upserted) >= 1


def test_ingest_skips_already_existing_chunks_on_reingest(monkeypatch):
    scraper = YouTubeScraper()
    transcript_text = "Repeatable transcript content for dedup testing purposes. " * 15
    monkeypatch.setattr(
        scraper, "_fetch_transcript",
        lambda vid: _TranscriptData(video_id=vid, raw_text=transcript_text,
                                     clean_text=transcript_text, language="en"),
    )
    fake_col = FakeCollection()
    monkeypatch.setattr(scraper, "_get_collection", lambda: fake_col)

    first = scraper.ingest("https://youtu.be/dQw4w9WgXcQ")
    second = scraper.ingest("https://youtu.be/dQw4w9WgXcQ")
    assert first.chunks_added > 0
    assert second.chunks_added == 0
    assert second.chunks_skipped == first.chunks_added


# --------------------------------------------------------------------------- #
# youtube_scraper: _fetch_transcript language fallback chain
# --------------------------------------------------------------------------- #
class _FakeTranscript:
    def __init__(self, language_code, segments):
        self.language_code = language_code
        self._segments = segments

    def fetch(self):
        return self._segments


class _FakeTranscriptList:
    """Mimics the object returned by YouTubeTranscriptApi().list(video_id)."""
    def __init__(self, manual=None, generated=None, any_list=None):
        self._manual = manual or {}
        self._generated = generated or {}
        self._any = any_list or []

    def find_manually_created_transcript(self, langs):
        for l in langs:
            if l in self._manual:
                return self._manual[l]
        raise NoTranscriptFound("vid", langs, self)

    def find_generated_transcript(self, langs):
        for l in langs:
            if l in self._generated:
                return self._generated[l]
        raise NoTranscriptFound("vid", langs, self)

    def __iter__(self):
        return iter(self._any)


def test_fetch_transcript_prefers_manual_over_generated(monkeypatch):
    manual_en = _FakeTranscript("en", [{"text": "manual segment"}])
    generated_en = _FakeTranscript("en", [{"text": "generated segment"}])
    tlist = _FakeTranscriptList(manual={"en": manual_en}, generated={"en": generated_en})

    class _FakeApi:
        def list(self, video_id):
            return tlist

    monkeypatch.setattr(yt_mod, "YouTubeTranscriptApi", _FakeApi)
    scraper = YouTubeScraper()
    data = scraper._fetch_transcript("dQw4w9WgXcQ")
    assert "manual segment" in data.raw_text
    assert "manual" in data.language


def test_fetch_transcript_falls_back_to_generated_when_no_manual(monkeypatch):
    generated_en = _FakeTranscript("en", [{"text": "auto segment"}])
    tlist = _FakeTranscriptList(generated={"en": generated_en})

    class _FakeApi:
        def list(self, video_id):
            return tlist

    monkeypatch.setattr(yt_mod, "YouTubeTranscriptApi", _FakeApi)
    scraper = YouTubeScraper()
    data = scraper._fetch_transcript("dQw4w9WgXcQ")
    assert "auto segment" in data.raw_text
    assert "auto-generated" in data.language


def test_fetch_transcript_falls_back_to_any_available_transcript(monkeypatch):
    any_transcript = _FakeTranscript("fr", [{"text": "french segment"}])
    tlist = _FakeTranscriptList(any_list=[any_transcript])

    class _FakeApi:
        def list(self, video_id):
            return tlist

    monkeypatch.setattr(yt_mod, "YouTubeTranscriptApi", _FakeApi)
    scraper = YouTubeScraper()
    data = scraper._fetch_transcript("dQw4w9WgXcQ")
    assert "french segment" in data.raw_text
    assert "fallback" in data.language


def test_fetch_transcript_raises_no_transcript_found_when_nothing_available(monkeypatch):
    tlist = _FakeTranscriptList()  # empty everywhere

    class _FakeApi:
        def list(self, video_id):
            return tlist

    monkeypatch.setattr(yt_mod, "YouTubeTranscriptApi", _FakeApi)
    scraper = YouTubeScraper()
    with pytest.raises(NoTranscriptFound):
        scraper._fetch_transcript("dQw4w9WgXcQ")


def test_fetch_transcript_handles_object_style_segments(monkeypatch):
    class _Segment:
        def __init__(self, text):
            self.text = text

    manual_en = _FakeTranscript("en", [_Segment("object style segment")])
    tlist = _FakeTranscriptList(manual={"en": manual_en})

    class _FakeApi:
        def list(self, video_id):
            return tlist

    monkeypatch.setattr(yt_mod, "YouTubeTranscriptApi", _FakeApi)
    scraper = YouTubeScraper()
    data = scraper._fetch_transcript("dQw4w9WgXcQ")
    assert "object style segment" in data.raw_text


# --------------------------------------------------------------------------- #
# playlist_scraper: PlaylistEntry
# --------------------------------------------------------------------------- #
def test_playlist_entry_duration_str_unknown_when_none():
    e = PlaylistEntry(video_id="v", title="t", url="u", position=1, duration_seconds=None)
    assert e.duration_str == "unknown"


def test_playlist_entry_duration_str_formats_minutes_seconds():
    e = PlaylistEntry(video_id="v", title="t", url="u", position=1, duration_seconds=125)
    assert e.duration_str == "2:05"


def test_playlist_entry_duration_str_formats_hours():
    e = PlaylistEntry(video_id="v", title="t", url="u", position=1, duration_seconds=3725)
    assert e.duration_str == "1:02:05"


def test_playlist_entry_repr_truncates_title():
    e = PlaylistEntry(video_id="v", title="x" * 60, url="u", position=1)
    assert len(repr(e)) < 200


# --------------------------------------------------------------------------- #
# playlist_scraper: PlaylistIngestionSummary
# --------------------------------------------------------------------------- #
def test_summary_success_rate_avoids_division_by_zero():
    s = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=0)
    assert s.success_rate == 0.0


def test_summary_success_rate_and_actionable_videos():
    s = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=10,
                                  successful=6, failed=2, skipped_no_captions=2)
    assert s.success_rate == 60.0
    assert s.actionable_videos == 8


def test_summary_str_contains_key_numbers():
    s = PlaylistIngestionSummary(playlist_url="u", playlist_title="My List", total_videos=5,
                                  successful=4, failed=1, total_chunks_added=20)
    text = str(s)
    assert "My List" in text and "4 OK" in text and "20 added" in text


# --------------------------------------------------------------------------- #
# playlist_scraper: _extract_entries — yt-dlp mocked
# --------------------------------------------------------------------------- #
class _FakeYoutubeDL:
    def __init__(self, info: dict):
        self._info = info

    def __call__(self, opts):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        return self._info


def test_extract_entries_parses_valid_entries_and_skips_unavailable(monkeypatch):
    info = {
        "title": "My Playlist",
        "entries": [
            {"id": "aaaaaaaaaaa", "title": "Video 1", "duration": 60},
            None,  # unavailable entry
            {"id": "short", "title": "Bad ID"},  # not 11 chars -> skipped
            {"id": "bbbbbbbbbbb", "title": "Video 2", "url": "https://youtu.be/bbbbbbbbbbb"},
        ],
    }
    monkeypatch.setattr(playlist_mod.yt_dlp, "YoutubeDL", _FakeYoutubeDL(info))
    title, entries = PlaylistIngestor._extract_entries("https://youtube.com/playlist?list=x")
    assert title == "My Playlist"
    assert [e.video_id for e in entries] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    assert entries[0].position == 1
    # An explicit "url" already starting with "http" is kept as-is, even
    # if it's a short youtu.be link rather than a full watch URL.
    assert entries[1].url == "https://youtu.be/bbbbbbbbbbb"


def test_extract_entries_wraps_a_single_video_with_no_entries_key(monkeypatch):
    info = {"title": "Solo Video", "id": "ccccccccccc"}
    monkeypatch.setattr(playlist_mod.yt_dlp, "YoutubeDL", _FakeYoutubeDL(info))
    title, entries = PlaylistIngestor._extract_entries("https://youtu.be/ccccccccccc")
    assert len(entries) == 1
    assert entries[0].video_id == "ccccccccccc"


def test_extract_entries_raises_runtime_error_when_ytdlp_returns_none(monkeypatch):
    monkeypatch.setattr(playlist_mod.yt_dlp, "YoutubeDL", _FakeYoutubeDL(None))
    with pytest.raises(RuntimeError, match="returned None"):
        PlaylistIngestor._extract_entries("https://youtube.com/playlist?list=x")


def test_extract_entries_wraps_download_error(monkeypatch):
    class _ExplodingYDL(_FakeYoutubeDL):
        def extract_info(self, url, download=False):
            raise playlist_mod.yt_dlp.utils.DownloadError("bad url")

    monkeypatch.setattr(playlist_mod.yt_dlp, "YoutubeDL", _ExplodingYDL(None))
    with pytest.raises(RuntimeError, match="yt-dlp failed"):
        PlaylistIngestor._extract_entries("https://youtube.com/playlist?list=bad")


# --------------------------------------------------------------------------- #
# playlist_scraper: _ingest_one error categorisation
# --------------------------------------------------------------------------- #
class _FakeScraper:
    def __init__(self, outcome):
        self._outcome = outcome  # either an IngestionResult or an exception instance
        self._collection_name = "trading_strategies"

    def ingest(self, url, title):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _entry(pos=1) -> PlaylistEntry:
    return PlaylistEntry(video_id="v" * 11, title="Vid", url="https://youtu.be/vvvvvvvvvvv", position=pos)


def test_ingest_one_counts_success():
    result = IngestionResult(video_id="v", title="Vid", url="u", chunks_added=5,
                              chunks_skipped=0, total_characters=100, elapsed_seconds=1.0,
                              collection_name="c", success=True)
    ingestor = PlaylistIngestor(scraper=_FakeScraper(result), delay_between_videos=0)
    summary = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=1)
    out = ingestor._ingest_one(_entry(), summary)
    assert out is result
    assert summary.successful == 1
    assert summary.total_chunks_added == 5


def test_ingest_one_counts_no_captions_exception_as_skip():
    ingestor = PlaylistIngestor(scraper=_FakeScraper(TranscriptsDisabled("v")), delay_between_videos=0)
    summary = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=1)
    out = ingestor._ingest_one(_entry(), summary)
    assert out is None
    assert summary.skipped_no_captions == 1
    assert summary.failed == 0


def test_ingest_one_counts_video_unavailable_as_skip():
    ingestor = PlaylistIngestor(scraper=_FakeScraper(VideoUnavailable("v")), delay_between_videos=0)
    summary = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=1)
    ingestor._ingest_one(_entry(), summary)
    assert summary.skipped_no_captions == 1


def test_ingest_one_counts_request_failure_as_failed():
    exc = YouTubeRequestFailed("http://x", Exception("net"))
    ingestor = PlaylistIngestor(scraper=_FakeScraper(exc), delay_between_videos=0)
    summary = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=1)
    ingestor._ingest_one(_entry(), summary)
    assert summary.failed == 1
    assert summary.skipped_no_captions == 0


def test_ingest_one_counts_unexpected_exception_as_failed():
    ingestor = PlaylistIngestor(scraper=_FakeScraper(RuntimeError("boom")), delay_between_videos=0)
    summary = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=1)
    ingestor._ingest_one(_entry(), summary)
    assert summary.failed == 1


def test_ingest_one_categorises_scraper_reported_failure_by_message():
    caption_failure = IngestionResult(video_id="v", title="Vid", url="u", chunks_added=0,
                                       chunks_skipped=0, total_characters=0, elapsed_seconds=0.1,
                                       collection_name="c", success=False,
                                       error="No captions available for this video")
    ingestor = PlaylistIngestor(scraper=_FakeScraper(caption_failure), delay_between_videos=0)
    summary = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=1)
    ingestor._ingest_one(_entry(), summary)
    assert summary.skipped_no_captions == 1
    assert summary.failed == 0


def test_ingest_one_categorises_other_scraper_failure_as_failed():
    other_failure = IngestionResult(video_id="v", title="Vid", url="u", chunks_added=0,
                                     chunks_skipped=0, total_characters=0, elapsed_seconds=0.1,
                                     collection_name="c", success=False,
                                     error="ChromaDB upsert error: disk full")
    ingestor = PlaylistIngestor(scraper=_FakeScraper(other_failure), delay_between_videos=0)
    summary = PlaylistIngestionSummary(playlist_url="u", playlist_title="t", total_videos=1)
    ingestor._ingest_one(_entry(), summary)
    assert summary.failed == 1
    assert summary.skipped_no_captions == 0


# --------------------------------------------------------------------------- #
# playlist_scraper: ingest_playlist — end-to-end orchestration
# --------------------------------------------------------------------------- #
def test_ingest_playlist_raises_when_no_entries_found(monkeypatch):
    monkeypatch.setattr(PlaylistIngestor, "_extract_entries",
                         staticmethod(lambda url: ("Empty", [])))
    ingestor = PlaylistIngestor(scraper=_FakeScraper(RuntimeError("n/a")), delay_between_videos=0)
    with pytest.raises(ValueError, match="No video entries"):
        ingestor.ingest_playlist("https://youtube.com/playlist?list=empty")


def test_ingest_playlist_applies_start_from_and_max_videos(monkeypatch):
    entries = [_entry(pos=i) for i in range(1, 11)]
    monkeypatch.setattr(PlaylistIngestor, "_extract_entries",
                         staticmethod(lambda url: ("List", entries)))
    ok_result = IngestionResult(video_id="v", title="Vid", url="u", chunks_added=1,
                                 chunks_skipped=0, total_characters=10, elapsed_seconds=0.01,
                                 collection_name="c", success=True)
    ingestor = PlaylistIngestor(scraper=_FakeScraper(ok_result), delay_between_videos=0)
    summary = ingestor.ingest_playlist("u", max_videos=3, start_from=5)
    # positions 5..10 remain (6), capped to 3
    assert summary.total_videos == 3
    assert summary.successful == 3


# --------------------------------------------------------------------------- #
# seed_strategies.py / seed_chart_analysis.py — static-data ingestors
# --------------------------------------------------------------------------- #
import knowledge_ingestion.seed_strategies as seed_strategies_mod
import knowledge_ingestion.seed_chart_analysis as seed_charts_mod


@pytest.mark.parametrize("mod", [seed_strategies_mod, seed_charts_mod])
def test_seed_module_make_chunk_id_is_deterministic(mod):
    a = mod._make_chunk_id("src", 0, "hello world")
    b = mod._make_chunk_id("src", 0, "hello world")
    assert a == b
    assert len(a) == 16


def test_strategy_library_entries_are_well_formed():
    for source_id, title, text in seed_strategies_mod.STRATEGY_LIBRARY:
        assert source_id and isinstance(source_id, str)
        assert title and isinstance(title, str)
        assert len(text.strip()) > 100


def test_chart_analysis_library_entries_are_well_formed():
    for source_id, title, text in seed_charts_mod.CHART_ANALYSIS_LIBRARY:
        assert source_id and isinstance(source_id, str)
        assert title and isinstance(title, str)
        assert len(text.strip()) > 100


def test_strategy_library_source_ids_are_unique():
    ids = [s for s, _, _ in seed_strategies_mod.STRATEGY_LIBRARY]
    assert len(ids) == len(set(ids))


def test_chart_analysis_library_source_ids_are_unique():
    ids = [s for s, _, _ in seed_charts_mod.CHART_ANALYSIS_LIBRARY]
    assert len(ids) == len(set(ids))


class _FakeClient:
    def __init__(self, collection):
        self._collection = collection

    def get_or_create_collection(self, name, embedding_function, metadata):
        return self._collection


def _patch_seed_chromadb(monkeypatch, mod, collection):
    monkeypatch.setattr(mod.chromadb, "PersistentClient", lambda path: _FakeClient(collection))
    monkeypatch.setattr(mod, "SentenceTransformerEmbeddingFunction",
                         lambda **kw: object())


def test_seed_strategies_ingest_all_upserts_every_book_once(monkeypatch):
    fake_col = FakeCollection()
    _patch_seed_chromadb(monkeypatch, seed_strategies_mod, fake_col)
    seed_strategies_mod.ingest_all()
    assert len(fake_col.upserted) == len(seed_strategies_mod.STRATEGY_LIBRARY)
    assert fake_col.count() > 0


def test_seed_strategies_ingest_all_is_idempotent_on_rerun(monkeypatch):
    fake_col = FakeCollection()
    _patch_seed_chromadb(monkeypatch, seed_strategies_mod, fake_col)
    seed_strategies_mod.ingest_all()
    first_count = fake_col.count()
    seed_strategies_mod.ingest_all()
    # Second run finds every id already present -> no new upserts.
    assert fake_col.count() == first_count


def test_seed_chart_analysis_ingest_all_upserts_every_book_once(monkeypatch):
    fake_col = FakeCollection()
    _patch_seed_chromadb(monkeypatch, seed_charts_mod, fake_col)
    seed_charts_mod.ingest_all()
    assert len(fake_col.upserted) == len(seed_charts_mod.CHART_ANALYSIS_LIBRARY)
