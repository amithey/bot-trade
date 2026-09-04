"""
Tests for rag/retriever.py — market-regime classification, natural-language
query generation, and the ChromaDB retrieval pipeline that feeds Claude's
prompt.

rag/ shipped with zero tests despite being the piece that decides which
strategy text Claude actually sees for a given market state. Most of it is
pure, deterministic logic — regime classification, sentence building, result
parsing/filtering — and is tested directly against a synthetic
MarketSnapshot, no ChromaDB involved. The one real ChromaDB integration
point (`_get_collection`) is mocked so no test loads the sentence-transformer
model or touches the real `data/chroma_db` — matching the "no network, fast"
discipline used everywhere else in this suite.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from market_data.fetcher import MACDParams, MarketSnapshot
from rag.retriever import RetrievalResult, RetrievedChunk, StrategyRetriever


# --------------------------------------------------------------------------- #
# Synthetic snapshot builder
# --------------------------------------------------------------------------- #
def make_snapshot(
    close, rsi=50.0, macd=0.0, macd_signal=0.0, macd_hist=None,
    sma20=None, sma50=None, sma200=None, high=None, low=None,
    prev_hist=None, ticker="TEST",
) -> MarketSnapshot:
    """A minimal two-row MarketSnapshot (or one row, if prev_hist is None)
    with exactly the columns _compute_market_regime/_snapshot_to_query read.
    """
    macd_hist = macd - macd_signal if macd_hist is None else macd_hist
    rows = []
    if prev_hist is not None:
        rows.append({"Close": close, "High": high or close, "Low": low or close,
                    "RSI_14": rsi, "MACD": macd, "MACD_Signal": macd_signal,
                    "MACD_Histogram": prev_hist})
    rows.append({"Close": close, "High": high or close, "Low": low or close,
                "RSI_14": rsi, "MACD": macd, "MACD_Signal": macd_signal,
                "MACD_Histogram": macd_hist})
    for sma, val in (("SMA_20", sma20), ("SMA_50", sma50), ("SMA_200", sma200)):
        if val is not None:
            for r in rows:
                r[sma] = val
    df = pd.DataFrame(rows, index=pd.date_range("2024-01-01", periods=len(rows)))
    return MarketSnapshot(ticker=ticker, data=df, sma_periods=(20, 50, 200),
                          rsi_period=14, macd_params=MACDParams())


@pytest.fixture()
def retriever() -> StrategyRetriever:
    return StrategyRetriever()


# --------------------------------------------------------------------------- #
# RetrievedChunk
# --------------------------------------------------------------------------- #
def test_similarity_score_endpoints():
    assert RetrievedChunk("d", {}, distance=0.0, chunk_id="1").similarity_score == 1.0
    assert RetrievedChunk("d", {}, distance=2.0, chunk_id="1").similarity_score == 0.0
    assert RetrievedChunk("d", {}, distance=0.7, chunk_id="1").similarity_score == 0.65


def test_source_title_prefers_title_over_video_id():
    c = RetrievedChunk("d", {"title": "RSI Basics", "video_id": "abc"}, 0.1, "1")
    assert c.source_title == "RSI Basics"


def test_source_title_falls_back_to_video_id_then_unknown():
    assert RetrievedChunk("d", {"video_id": "abc"}, 0.1, "1").source_title == "abc"
    assert RetrievedChunk("d", {}, 0.1, "1").source_title == "Unknown source"


def test_chunk_repr_is_informative():
    r = repr(RetrievedChunk("some long document text here", {"title": "X"}, 0.2, "1"))
    assert "score=" in r and "X" in r


# --------------------------------------------------------------------------- #
# RetrievalResult
# --------------------------------------------------------------------------- #
def test_documents_property_preserves_order():
    chunks = [RetrievedChunk("first", {}, 0.1, "1"), RetrievedChunk("second", {}, 0.3, "2")]
    result = RetrievalResult("q", "TEST", chunks, {}, collection_size=10)
    assert result.documents == ["first", "second"]


def test_found_and_best_score_on_empty_result():
    result = RetrievalResult("q", "TEST", [], {}, collection_size=0)
    assert result.found is False
    assert result.best_score is None


def test_found_and_best_score_on_nonempty_result():
    chunks = [RetrievedChunk("d", {}, 0.5, "1")]
    result = RetrievalResult("q", "TEST", chunks, {}, collection_size=5)
    assert result.found is True
    assert result.best_score == chunks[0].similarity_score


# --------------------------------------------------------------------------- #
# _compute_market_regime — RSI zone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rsi,expected", [
    (75, "Overbought"), (70, "Overbought"),
    (65, "Bullish"), (60, "Bullish"),
    (50, "Neutral"),
    (35, "Bearish"), (40, "Bearish"),
    (20, "Oversold"), (30, "Oversold"),
])
def test_rsi_zone_boundaries(retriever, rsi, expected):
    snap = make_snapshot(close=100, rsi=rsi)
    assert retriever._compute_market_regime(snap)["rsi_zone"] == expected


# --------------------------------------------------------------------------- #
# _compute_market_regime — MACD
# --------------------------------------------------------------------------- #
def test_macd_cross_bullish_when_macd_above_signal(retriever):
    snap = make_snapshot(close=100, macd=1.0, macd_signal=0.5)
    assert retriever._compute_market_regime(snap)["macd_cross"] == "Bullish Crossover"


def test_macd_cross_bearish_when_macd_below_signal(retriever):
    snap = make_snapshot(close=100, macd=-1.0, macd_signal=0.5)
    assert retriever._compute_market_regime(snap)["macd_cross"] == "Bearish Crossover"


def test_macd_momentum_expanding_bullish_when_histogram_grows(retriever):
    snap = make_snapshot(close=100, macd=1.0, macd_signal=0.5,
                         macd_hist=0.6, prev_hist=0.3)
    assert retriever._compute_market_regime(snap)["macd_momentum"] == "Expanding Bullish"


def test_macd_momentum_fading_bullish_when_histogram_shrinks(retriever):
    snap = make_snapshot(close=100, macd=1.0, macd_signal=0.5,
                         macd_hist=0.2, prev_hist=0.5)
    assert retriever._compute_market_regime(snap)["macd_momentum"] == "Fading Bullish"


def test_macd_momentum_expanding_bearish_when_histogram_falls_further(retriever):
    snap = make_snapshot(close=100, macd=-1.0, macd_signal=0.0,
                         macd_hist=-0.6, prev_hist=-0.3)
    assert retriever._compute_market_regime(snap)["macd_momentum"] == "Expanding Bearish"


def test_macd_momentum_single_bar_fallback_always_reads_expanding(retriever):
    """Documents current behavior, not a fix: with no previous bar to diff
    against, the code's fallback ternary is tautological (it checks `hist>0`
    a second time inside the branch that already established `hist>0`), so
    it always reports "Expanding X" and can never say "Fading X" here. Not
    corrected because no real caller can trigger it - MarketDataFetcher
    always returns dozens of rows minimum after its warm-up dropna(), so
    the "refine using the previous bar" branch immediately below it always
    overwrites this fallback in practice. Only a hand-built single-row
    snapshot (as in this test) ever sees the raw fallback value.
    """
    bearish = make_snapshot(close=100, macd=-1.0, macd_signal=0.0, macd_hist=-0.2)
    assert retriever._compute_market_regime(bearish)["macd_momentum"] == "Expanding Bearish"
    bullish = make_snapshot(close=100, macd=1.0, macd_signal=0.5, macd_hist=0.2)
    assert retriever._compute_market_regime(bullish)["macd_momentum"] == "Expanding Bullish"


def test_macd_momentum_fading_bearish_when_histogram_recovers(retriever):
    snap = make_snapshot(close=100, macd=-1.0, macd_signal=0.0,
                         macd_hist=-0.2, prev_hist=-0.5)
    assert retriever._compute_market_regime(snap)["macd_momentum"] == "Fading Bearish"


# --------------------------------------------------------------------------- #
# _compute_market_regime — price trend / SMA structure
# --------------------------------------------------------------------------- #
def test_strong_uptrend_requires_price_above_all_three_smas(retriever):
    snap = make_snapshot(close=110, sma20=105, sma50=100, sma200=95)
    assert retriever._compute_market_regime(snap)["price_trend"] == "Strong Uptrend"


def test_uptrend_without_sma200_confirmation(retriever):
    snap = make_snapshot(close=110, sma20=105, sma50=100, sma200=120)
    assert retriever._compute_market_regime(snap)["price_trend"] == "Uptrend"


def test_strong_downtrend_requires_price_below_all_three_smas(retriever):
    snap = make_snapshot(close=90, sma20=95, sma50=100, sma200=105)
    assert retriever._compute_market_regime(snap)["price_trend"] == "Strong Downtrend"


def test_consolidating_when_price_is_within_one_percent_of_sma20(retriever):
    # sma50 sits *below* sma20 so neither the close>sma20>sma50 nor the
    # close<sma20<sma50 branch matches first - only then does the |close-sma20|
    # check get reached.
    snap = make_snapshot(close=100, sma20=100.5, sma50=99)
    assert retriever._compute_market_regime(snap)["price_trend"] == "Consolidating"


def test_sideways_when_price_and_smas_disagree(retriever):
    snap = make_snapshot(close=100, sma20=95, sma50=105)
    assert retriever._compute_market_regime(snap)["price_trend"] == "Sideways"


def test_trend_unknown_with_no_sma_data(retriever):
    snap = make_snapshot(close=100)
    assert retriever._compute_market_regime(snap)["price_trend"] == "Unknown"


def test_golden_cross_when_sma50_above_sma200(retriever):
    snap = make_snapshot(close=100, sma50=100, sma200=90)
    assert retriever._compute_market_regime(snap)["sma_structure"] == "Golden Cross"


def test_death_cross_when_sma50_below_sma200(retriever):
    snap = make_snapshot(close=100, sma50=90, sma200=100)
    assert retriever._compute_market_regime(snap)["sma_structure"] == "Death Cross"


def test_sma_structure_mixed_with_no_sma_data(retriever):
    snap = make_snapshot(close=100)
    assert retriever._compute_market_regime(snap)["sma_structure"] == "Mixed"


# --------------------------------------------------------------------------- #
# _compute_market_regime — volatility
# --------------------------------------------------------------------------- #
def test_high_volatility_above_2_5_percent_range(retriever):
    snap = make_snapshot(close=100, high=103, low=99.5)   # 3.5% range
    assert retriever._compute_market_regime(snap)["volatility"] == "High Volatility"


def test_low_volatility_below_0_8_percent_range(retriever):
    snap = make_snapshot(close=100, high=100.3, low=100.0)
    assert retriever._compute_market_regime(snap)["volatility"] == "Low Volatility"


def test_normal_volatility_in_between(retriever):
    snap = make_snapshot(close=100, high=101, low=100)   # 1% range
    assert retriever._compute_market_regime(snap)["volatility"] == "Normal Volatility"


# --------------------------------------------------------------------------- #
# _snapshot_to_query + sentence builders
# --------------------------------------------------------------------------- #
def test_query_includes_ticker_and_price(retriever):
    snap = make_snapshot(close=142.50, ticker="AAPL")
    regime = retriever._compute_market_regime(snap)
    query = retriever._snapshot_to_query(snap, regime)
    assert "AAPL" in query and "$142.50" in query


def test_rsi_sentence_mentions_the_exact_reading_per_zone():
    assert "overbought" in StrategyRetriever._rsi_sentence(75.0, "Overbought").lower()
    assert "oversold" in StrategyRetriever._rsi_sentence(20.0, "Oversold").lower()
    assert "75.0" in StrategyRetriever._rsi_sentence(75.0, "Overbought")


def test_macd_sentence_direction_matches_the_actual_values():
    above = StrategyRetriever._macd_sentence(1.0, 0.5, 0.5, "Bullish Crossover", "Expanding Bullish")
    assert "above" in above and "building" in above and "bullish" in above
    below = StrategyRetriever._macd_sentence(-1.0, 0.0, -1.0, "Bearish Crossover", "Fading Bearish")
    assert "below" in below and "weakening" in below and "bearish" in below


def test_trend_sentence_bias_follows_trend_or_golden_cross():
    up = StrategyRetriever._trend_sentence("Uptrend", "Mixed", 110.0, "SMA 20: 100")
    assert "bullish" in up
    down = StrategyRetriever._trend_sentence("Downtrend", "Death Cross", 90.0, "SMA 20: 100")
    assert "bearish" in down
    golden_but_flat = StrategyRetriever._trend_sentence("Sideways", "Golden Cross", 100.0, "x")
    assert "bullish" in golden_but_flat, "Golden Cross alone should read bullish even if trend is flat"


def test_action_request_echoes_the_classified_regime():
    text = StrategyRetriever._action_request("Overbought", "Bullish Crossover", "Uptrend")
    assert "overbought" in text and "bullish crossover" in text and "uptrend" in text


# --------------------------------------------------------------------------- #
# _parse_results
# --------------------------------------------------------------------------- #
def test_parse_results_builds_chunks_in_returned_order():
    raw = {
        "documents":  [["first", "second"]],
        "metadatas":  [[{"title": "A"}, {"title": "B"}]],
        "distances":  [[0.1, 0.4]],
        "ids":        [["id1", "id2"]],
    }
    chunks = StrategyRetriever._parse_results(raw)
    assert [c.document for c in chunks] == ["first", "second"]
    assert chunks[0].chunk_id == "id1"


def test_parse_results_skips_empty_document_slots():
    raw = {"documents": [["", "real"]], "metadatas": [[{}, {}]],
          "distances": [[0.1, 0.2]], "ids": [["a", "b"]]}
    chunks = StrategyRetriever._parse_results(raw)
    assert len(chunks) == 1 and chunks[0].document == "real"


def test_parse_results_tolerates_missing_keys():
    assert StrategyRetriever._parse_results({}) == []


def test_parse_results_defaults_none_metadata_to_empty_dict():
    raw = {"documents": [["doc"]], "metadatas": [[None]],
          "distances": [[0.1]], "ids": [["a"]]}
    chunks = StrategyRetriever._parse_results(raw)
    assert chunks[0].metadata == {}


# --------------------------------------------------------------------------- #
# _filter_by_threshold
# --------------------------------------------------------------------------- #
def test_filter_drops_chunks_over_the_threshold(retriever):
    chunks = [RetrievedChunk("a", {}, 0.5, "1"), RetrievedChunk("b", {}, 0.9, "2")]
    filtered = retriever._filter_by_threshold(chunks)
    assert [c.chunk_id for c in filtered] == ["1"]


def test_filter_keeps_a_chunk_exactly_at_the_threshold(retriever):
    chunks = [RetrievedChunk("a", {}, retriever._distance_threshold, "1")]
    assert retriever._filter_by_threshold(chunks) == chunks


# --------------------------------------------------------------------------- #
# StrategyRetriever construction
# --------------------------------------------------------------------------- #
def test_construction_defaults_come_from_settings():
    from config.settings import settings
    r = StrategyRetriever()
    assert r._embedding_model == settings.embedding_model
    assert r._collection_name == settings.chroma_collection_name


def test_construction_overrides_are_honoured():
    r = StrategyRetriever(collection_name="custom", distance_threshold=0.3)
    assert r._collection_name == "custom"
    assert r._distance_threshold == 0.3


# --------------------------------------------------------------------------- #
# get_relevant_strategies — ChromaDB mocked
# --------------------------------------------------------------------------- #
def _fake_collection(count: int, query_result: dict | None = None, query_raises=False):
    col = MagicMock()
    col.count.return_value = count
    if query_raises:
        col.query.side_effect = RuntimeError("chroma down")
    else:
        col.query.return_value = query_result or {
            "documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]],
        }
    return col


def test_top_k_below_one_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.get_relevant_strategies(make_snapshot(close=100), top_k=0)


def test_empty_collection_returns_an_empty_but_valid_result(retriever, monkeypatch):
    monkeypatch.setattr(retriever, "_get_collection", lambda: _fake_collection(0))
    result = retriever.get_relevant_strategies(make_snapshot(close=100))
    assert result.found is False
    assert result.collection_size == 0
    assert result.query   # the query is still generated even with nothing to search


def test_collection_open_failure_returns_empty_result_not_an_exception(retriever, monkeypatch):
    def _explode():
        raise RuntimeError("cannot open chroma")
    monkeypatch.setattr(retriever, "_get_collection", _explode)
    result = retriever.get_relevant_strategies(make_snapshot(close=100))
    assert result.found is False
    assert result.collection_size == 0


def test_a_failed_query_returns_empty_result_not_an_exception(retriever, monkeypatch):
    col = _fake_collection(5, query_raises=True)
    monkeypatch.setattr(retriever, "_get_collection", lambda: col)
    result = retriever.get_relevant_strategies(make_snapshot(close=100))
    assert result.found is False
    assert result.collection_size == 5   # the count was known before the query broke


def test_top_k_is_capped_to_the_collection_size(retriever, monkeypatch):
    col = _fake_collection(2)
    monkeypatch.setattr(retriever, "_get_collection", lambda: col)
    retriever.get_relevant_strategies(make_snapshot(close=100), top_k=10)
    assert col.query.call_args.kwargs["n_results"] == 2


def test_successful_retrieval_builds_a_populated_result(retriever, monkeypatch):
    raw = {
        "documents": [["strategy text"]],
        "metadatas": [[{"title": "RSI Playbook"}]],
        "distances": [[0.2]],
        "ids": [["doc1"]],
    }
    col = _fake_collection(1, query_result=raw)
    monkeypatch.setattr(retriever, "_get_collection", lambda: col)
    snap = make_snapshot(close=100, ticker="QQQ")
    result = retriever.get_relevant_strategies(snap, top_k=1)
    assert result.found is True
    assert result.ticker == "QQQ"
    assert result.documents == ["strategy text"]
    assert result.market_regime  # populated


def test_chunks_beyond_the_threshold_are_excluded_from_the_result(retriever, monkeypatch):
    raw = {
        "documents": [["close", "far"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.1, 1.5]],
        "ids": [["a", "b"]],
    }
    col = _fake_collection(2, query_result=raw)
    monkeypatch.setattr(retriever, "_get_collection", lambda: col)
    result = retriever.get_relevant_strategies(make_snapshot(close=100), top_k=2)
    assert result.documents == ["close"]


# --------------------------------------------------------------------------- #
# add_lesson
# --------------------------------------------------------------------------- #
def test_add_lesson_generates_a_prefixed_id_and_calls_add(retriever, monkeypatch):
    col = MagicMock()
    monkeypatch.setattr(retriever, "_get_collection", lambda: col)
    doc_id = retriever.add_lesson("  Sold too early on a fakeout.  ",
                                  metadata={"ticker": "BTC-USD", "pnl": -12.5})
    assert doc_id.startswith("lesson-")
    args, kwargs = col.add.call_args
    assert kwargs["ids"] == [doc_id]
    assert kwargs["documents"] == ["Sold too early on a fakeout."]   # stripped
    assert kwargs["metadatas"][0]["ticker"] == "BTC-USD"
    assert kwargs["metadatas"][0]["type"] == "lesson"


def test_add_lesson_defaults_type_but_does_not_override_a_caller_supplied_one(retriever, monkeypatch):
    col = MagicMock()
    monkeypatch.setattr(retriever, "_get_collection", lambda: col)
    retriever.add_lesson("text", metadata={"type": "meta_lesson"})
    assert col.add.call_args.kwargs["metadatas"][0]["type"] == "meta_lesson"


def test_add_lesson_always_stamps_its_own_created_at(retriever, monkeypatch):
    """created_at is a direct assignment, not setdefault - a caller-supplied
    value must not be trusted over the server's own clock."""
    col = MagicMock()
    monkeypatch.setattr(retriever, "_get_collection", lambda: col)
    retriever.add_lesson("text", metadata={"created_at": "2000-01-01T00:00:00"})
    stamped = col.add.call_args.kwargs["metadatas"][0]["created_at"]
    assert stamped != "2000-01-01T00:00:00"


def test_add_lesson_coerces_none_metadata_values_for_chromadb(retriever, monkeypatch):
    """ChromaDB rejects None-valued metadata outright."""
    col = MagicMock()
    monkeypatch.setattr(retriever, "_get_collection", lambda: col)
    retriever.add_lesson("text", metadata={"risk_profile": None})
    assert col.add.call_args.kwargs["metadatas"][0]["risk_profile"] == ""


# --------------------------------------------------------------------------- #
# collection_info
# --------------------------------------------------------------------------- #
def test_collection_info_reports_ok_on_success(retriever, monkeypatch):
    monkeypatch.setattr(retriever, "_get_collection", lambda: _fake_collection(42))
    info = retriever.collection_info()
    assert info["status"] == "ok"
    assert info["document_count"] == 42


def test_collection_info_reports_the_error_instead_of_raising(retriever, monkeypatch):
    def _explode():
        raise RuntimeError("disk full")
    monkeypatch.setattr(retriever, "_get_collection", _explode)
    info = retriever.collection_info()
    assert info["document_count"] == 0
    assert "disk full" in info["status"]
