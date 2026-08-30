"""
Tests for rag/ownership.py and the retrieval scoping built on it.

The property under test is a tenancy boundary: content one account ingested
must not reach another account's retrieval, and therefore must not reach
another account's Claude prompt. Operator-curated seed playbooks are the
deliberate exception — they are shared because of what they are.
"""
from __future__ import annotations

import pytest

from rag.ownership import (
    SHARED_OWNER,
    TRUSTED_SHARED_SOURCES,
    is_visible_to,
    normalise_owner,
    owner_filter,
)


# --------------------------------------------------------------------------- #
# normalise_owner
# --------------------------------------------------------------------------- #
def test_normalise_owner_keeps_a_real_account():
    assert normalise_owner("user_alice-abc123") == "user_alice-abc123"


def test_normalise_owner_trims_whitespace():
    assert normalise_owner("  acct  ") == "acct"


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_normalise_owner_treats_missing_as_shared(blank):
    assert normalise_owner(blank) == SHARED_OWNER


# --------------------------------------------------------------------------- #
# owner_filter
# --------------------------------------------------------------------------- #
def test_no_owner_means_no_filter():
    """Single-user deployments must keep seeing everything they ingested."""
    assert owner_filter(None) is None
    assert owner_filter("") is None


def test_shared_owner_means_no_filter():
    assert owner_filter(SHARED_OWNER) is None


def test_filter_admits_shared_and_own_content():
    where = owner_filter("alice")
    assert where is not None
    owner_clause = where["$or"][0]["owner"]["$in"]
    assert SHARED_OWNER in owner_clause
    assert "alice" in owner_clause


def test_filter_admits_legacy_seed_chunks_by_source():
    """Seed chunks predating ownership have no `owner` field at all.

    A filter on `owner` alone matches nothing for them, which would make the
    entire seeded playbook vanish on upgrade — so the source arm exists.
    """
    where = owner_filter("alice")
    source_clause = where["$or"][1]["source"]["$in"]
    for src in TRUSTED_SHARED_SOURCES:
        assert src in source_clause


def test_every_seeder_in_the_repo_is_listed_as_trusted():
    """Guard against a seeder being added without updating the trust list.

    `seed_script` (seed_knowledge.py, at the repo root rather than inside
    knowledge_ingestion/) was missed when this list was first written and only
    surfaced by running the backfill tool against a real collection. A missing
    entry means that seeder's playbooks silently vanish from retrieval for
    every account, so it is worth catching mechanically.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    candidates = list((root / "knowledge_ingestion").glob("seed_*.py"))
    candidates += [root / "seed_knowledge.py"]

    declared = set()
    for path in candidates:
        if not path.exists():
            continue
        for match in re.finditer(r'"source":\s*"([a-z_]+)"', path.read_text(encoding="utf-8")):
            declared.add(match.group(1))

    missing = declared - set(TRUSTED_SHARED_SOURCES)
    assert not missing, (
        f"seeder source(s) {sorted(missing)} are not in TRUSTED_SHARED_SOURCES "
        f"— their chunks would drop out of every account's retrieval"
    )


def test_filter_never_names_another_account():
    where = owner_filter("alice")
    assert "bob" not in str(where)


# --------------------------------------------------------------------------- #
# is_visible_to — the rule stated directly
# --------------------------------------------------------------------------- #
def test_my_own_chunk_is_visible():
    assert is_visible_to({"owner": "alice", "source": "web_article"}, "alice")


def test_another_accounts_chunk_is_not_visible():
    assert not is_visible_to({"owner": "bob", "source": "web_article"}, "alice")


def test_another_accounts_lesson_is_not_visible():
    """Lessons are derived from one account's own trades — sharing them would
    leak trading history through the retrieval path."""
    assert not is_visible_to({"owner": "bob", "source": "lesson"}, "alice")


def test_shared_chunk_is_visible_to_everyone():
    assert is_visible_to({"owner": SHARED_OWNER, "source": "seed"}, "alice")
    assert is_visible_to({"owner": SHARED_OWNER, "source": "seed"}, "bob")


@pytest.mark.parametrize("src", TRUSTED_SHARED_SOURCES)
def test_legacy_seed_chunk_without_owner_stays_visible(src):
    assert is_visible_to({"source": src}, "alice")


def test_legacy_user_chunk_without_owner_is_hidden():
    """Fail closed: unattributed user content of unknown provenance must not
    be readable by an arbitrary account."""
    assert not is_visible_to({"source": "web_article"}, "alice")
    assert not is_visible_to({"source": "youtube"}, "alice")


def test_single_user_mode_sees_everything():
    assert is_visible_to({"source": "web_article"}, None)
    assert is_visible_to({"owner": "bob"}, None)


def test_missing_metadata_is_handled():
    assert is_visible_to(None, None)
    assert not is_visible_to(None, "alice")


# --------------------------------------------------------------------------- #
# The retriever actually applies it
# --------------------------------------------------------------------------- #
class _SpyCollection:
    """Captures the kwargs the retriever passes to ChromaDB."""

    def __init__(self):
        self.last_kwargs = None

    def count(self):
        return 25

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        return {"documents": [[]], "metadatas": [[]], "distances": [[]],
                "ids": [[]]}

    def add(self, ids, documents, metadatas):
        self.last_kwargs = {"ids": ids, "documents": documents,
                            "metadatas": metadatas}


def _snapshot():
    import numpy as np
    import pandas as pd
    from market_data.fetcher import MACDParams, MarketSnapshot

    idx = pd.bdate_range("2026-01-01", periods=60)
    close = pd.Series(np.linspace(100, 120, 60), index=idx)
    df = pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1, "Close": close,
        "Volume": 1_000_000.0, "SMA_20": close, "SMA_50": close,
        "SMA_200": close, "RSI_14": 55.0, "MACD": 0.5,
        "MACD_Signal": 0.4, "MACD_Histogram": 0.1,
    }, index=idx)
    return MarketSnapshot(ticker="AAPL", data=df, sma_periods=(20, 50, 200),
                          rsi_period=14, macd_params=MACDParams())


def test_retriever_sends_a_where_clause_for_an_owner(monkeypatch):
    from rag.retriever import StrategyRetriever

    spy = _SpyCollection()
    r = StrategyRetriever()
    monkeypatch.setattr(r, "_get_collection", lambda: spy)
    r.get_relevant_strategies(_snapshot(), top_k=3, owner="alice")
    assert "where" in spy.last_kwargs
    assert "alice" in str(spy.last_kwargs["where"])


def test_retriever_sends_no_where_clause_without_an_owner(monkeypatch):
    from rag.retriever import StrategyRetriever

    spy = _SpyCollection()
    r = StrategyRetriever()
    monkeypatch.setattr(r, "_get_collection", lambda: spy)
    r.get_relevant_strategies(_snapshot(), top_k=3)
    assert "where" not in spy.last_kwargs


def test_per_call_owner_beats_the_instance_default(monkeypatch):
    """The dashboard caches one retriever per process, so ownership has to
    travel with the call rather than the object."""
    from rag.retriever import StrategyRetriever

    spy = _SpyCollection()
    r = StrategyRetriever(owner="instance_default")
    monkeypatch.setattr(r, "_get_collection", lambda: spy)
    r.get_relevant_strategies(_snapshot(), top_k=3, owner="caller")
    where = str(spy.last_kwargs["where"])
    assert "caller" in where
    assert "instance_default" not in where


def test_add_lesson_tags_the_owner(monkeypatch):
    from rag.retriever import StrategyRetriever

    spy = _SpyCollection()
    r = StrategyRetriever()
    monkeypatch.setattr(r, "_get_collection", lambda: spy)
    r.add_lesson("RSI divergence worked here.", metadata={"ticker": "AAPL"},
                 owner="alice")
    assert spy.last_kwargs["metadatas"][0]["owner"] == "alice"


def test_add_lesson_without_owner_is_marked_shared(monkeypatch):
    from rag.retriever import StrategyRetriever

    spy = _SpyCollection()
    r = StrategyRetriever()
    monkeypatch.setattr(r, "_get_collection", lambda: spy)
    r.add_lesson("A lesson.", metadata={})
    assert spy.last_kwargs["metadatas"][0]["owner"] == SHARED_OWNER


# --------------------------------------------------------------------------- #
# Ingesters attribute what they write
# --------------------------------------------------------------------------- #
def test_article_scraper_stamps_the_owner(monkeypatch):
    import knowledge_ingestion.article_scraper as art_mod
    from knowledge_ingestion.article_scraper import ArticleScraper

    html = ("<html><head><title>T</title></head><body><p>"
            + ("Trading strategy content that is long enough to pass. " * 12)
            + "</p></body></html>")
    monkeypatch.setattr(art_mod.ArticleScraper, "_fetch",
                        staticmethod(lambda url: html))

    captured = {}

    class _Col:
        def get(self, ids, include=None):
            return {"ids": []}

        def upsert(self, ids, documents, metadatas):
            captured["metas"] = metadatas

    scraper = ArticleScraper()
    monkeypatch.setattr(scraper, "_get_collection", lambda: _Col())
    scraper.ingest("https://example.com/x", owner="alice")
    assert all(m["owner"] == "alice" for m in captured["metas"])


def test_article_scraper_without_owner_marks_shared(monkeypatch):
    import knowledge_ingestion.article_scraper as art_mod
    from knowledge_ingestion.article_scraper import ArticleScraper

    html = ("<html><head><title>T</title></head><body><p>"
            + ("Trading strategy content that is long enough to pass. " * 12)
            + "</p></body></html>")
    monkeypatch.setattr(art_mod.ArticleScraper, "_fetch",
                        staticmethod(lambda url: html))

    captured = {}

    class _Col:
        def get(self, ids, include=None):
            return {"ids": []}

        def upsert(self, ids, documents, metadatas):
            captured["metas"] = metadatas

    scraper = ArticleScraper()
    monkeypatch.setattr(scraper, "_get_collection", lambda: _Col())
    scraper.ingest("https://example.com/x")
    assert all(m["owner"] == SHARED_OWNER for m in captured["metas"])


def test_youtube_scraper_stamps_the_owner(monkeypatch):
    from knowledge_ingestion.youtube_scraper import YouTubeScraper, _TranscriptData

    text = "A useful transcript about RSI and MACD strategy. " * 20
    scraper = YouTubeScraper(owner="alice")
    monkeypatch.setattr(
        scraper, "_fetch_transcript",
        lambda vid: _TranscriptData(video_id=vid, raw_text=text,
                                    clean_text=text, language="en"),
    )

    captured = {}

    class _Col:
        def get(self, ids, include=None):
            return {"ids": []}

        def upsert(self, ids, documents, metadatas):
            captured.setdefault("metas", []).extend(metadatas)

    monkeypatch.setattr(scraper, "_get_collection", lambda: _Col())
    scraper.ingest("https://youtu.be/dQw4w9WgXcQ")
    assert captured["metas"]
    assert all(m["owner"] == "alice" for m in captured["metas"])


# --------------------------------------------------------------------------- #
# Tenant wiring
# --------------------------------------------------------------------------- #
def test_tenant_knowledge_owner_uses_the_supplied_slug(tmp_path):
    from saas.ledger import UsageLedger
    from saas.tenant import Tenant

    led = UsageLedger(db_path=tmp_path / "u.db")
    t = Tenant(account_id="user:a@b.com", knowledge_owner="user_a_b.com-abc123",
               ledger=led)
    assert t.knowledge_owner == "user_a_b.com-abc123"


def test_tenant_knowledge_owner_follows_the_person_not_the_key(tmp_path):
    """Rotating an Anthropic key must not hide the user's own documents."""
    from saas.ledger import UsageLedger
    from saas.tenant import Tenant

    led = UsageLedger(db_path=tmp_path / "u.db")
    t = Tenant(account_id="user:a@b.com", ledger=led)
    before = t.knowledge_owner
    t.set_key("sk-ant-" + "x" * 40)
    assert t.knowledge_owner == before
