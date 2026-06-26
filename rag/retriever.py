"""
RAG Strategy Retriever — Phase 3
==================================

Bridges the live market state (Phase 2 ``MarketSnapshot``) and the ChromaDB
knowledge base (Phase 1) to surface the most contextually relevant trading
strategy chunks for the current market conditions.

Pipeline
--------
1. ``_compute_market_regime()``   — classify each indicator into human-readable
                                    condition labels (e.g. RSI → "Overbought").
2. ``_snapshot_to_query()``       — translate those labels + raw values into a
                                    rich natural-language semantic search query.
3. ``get_relevant_strategies()``  — embed the query, cosine-search ChromaDB,
                                    return ranked :class:`RetrievedChunk` objects.

The output ``RetrievalResult`` is the *exact input contract* for the
AI Decision Engine (Phase 4): ``.documents`` returns a plain list of
strategy text strings ready to be injected into the LLM context window.

Typical usage
-------------
    from rag.retriever import StrategyRetriever
    from market_data.fetcher import MarketDataFetcher

    fetcher = MarketDataFetcher()
    snap    = fetcher.fetch_latest("QQQ", lookback_days=420)

    retriever = StrategyRetriever()
    result    = retriever.get_relevant_strategies(snap, top_k=4)

    print(result.query)          # the generated NL query
    print(result.documents)      # list[str] ready for LLM context
    print(result.market_regime)  # dict of classified conditions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from utils.hf_quiet import configure_quiet_hf, quiet_model_load

configure_quiet_hf()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config.settings import settings
from market_data.fetcher import MACDParams, MarketSnapshot
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — centralised so they can be tuned without hunting through prose
# ---------------------------------------------------------------------------

# RSI zone boundaries
_RSI_OVERBOUGHT: float = 70.0
_RSI_STRONG_BULL: float = 60.0
_RSI_STRONG_BEAR: float = 40.0
_RSI_OVERSOLD: float = 30.0

# Cosine distance ceiling — results above this are too dissimilar to be useful.
# ChromaDB uses cosine distance ∈ [0, 2]:  0 = identical, 2 = opposite.
# 0.7 corresponds to a cosine *similarity* of about 0.65, a reasonable floor
# for semantic relevance.
_DEFAULT_DISTANCE_THRESHOLD: float = 0.7


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """
    A single document chunk returned from ChromaDB.

    Attributes:
        document:    Raw text of the chunk.
        metadata:    Metadata dict stored alongside the chunk at ingestion time
                     (source, video_id, title, chunk_index, …).
        distance:    Cosine distance from the query embedding ∈ [0, 2].
                     Lower = more semantically similar.
        chunk_id:    ChromaDB document ID.
    """

    document: str
    metadata: dict[str, Any]
    distance: float
    chunk_id: str

    @property
    def similarity_score(self) -> float:
        """
        Cosine *similarity* normalised to [0, 1].

        Derived from distance via:  similarity = 1 − (distance / 2)
        """
        return round(1.0 - (self.distance / 2.0), 4)

    @property
    def source_title(self) -> str:
        """Human-readable source label from metadata, or a fallback."""
        return self.metadata.get("title") or self.metadata.get("video_id") or "Unknown source"

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(score={self.similarity_score:.3f}, "
            f"source={self.source_title!r}, "
            f"preview={self.document[:60]!r}…)"
        )


@dataclass
class RetrievalResult:
    """
    Full output of a single retrieval call.

    This is the *input contract* for the AI Decision Engine (Phase 4):
    pass ``result.documents`` as the RAG context strings when building
    the LLM prompt.

    Attributes:
        query:             The natural-language query that was embedded and searched.
        ticker:            Asset ticker the query was generated for.
        chunks:            Ranked list of :class:`RetrievedChunk` (best first).
        market_regime:     Dict of classified market conditions used to build the query.
        collection_size:   Total number of documents in the collection at query time.
        retrieved_at:      UTC timestamp of the retrieval call.
    """

    query: str
    ticker: str
    chunks: list[RetrievedChunk]
    market_regime: dict[str, str]
    collection_size: int
    retrieved_at: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------ #
    # Convenience accessors for Phase 4
    # ------------------------------------------------------------------ #

    @property
    def documents(self) -> list[str]:
        """Plain list of chunk text strings, ordered best-match first.

        Pass this directly into the LLM context window.
        """
        return [c.document for c in self.chunks]

    @property
    def found(self) -> bool:
        """``True`` if at least one relevant chunk was retrieved."""
        return len(self.chunks) > 0

    @property
    def best_score(self) -> Optional[float]:
        """Similarity score of the top result, or ``None`` if empty."""
        return self.chunks[0].similarity_score if self.chunks else None

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(ticker={self.ticker!r}, "
            f"chunks={len(self.chunks)}, "
            f"best_score={self.best_score}, "
            f"collection_size={self.collection_size})"
        )


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------


class StrategyRetriever:
    """
    Semantic retriever that translates a live ``MarketSnapshot`` into a
    natural-language query and returns the most relevant trading strategy
    chunks from ChromaDB.

    The class is intentionally stateless beyond its configuration —
    the ChromaDB client is initialised lazily on the first query call so
    that importing the module never blocks or fails if the DB is absent.

    Parameters
    ----------
    persist_dir:
        Override the ChromaDB persist directory (defaults to ``settings``).
    collection_name:
        Override the collection name (defaults to ``settings``).
    embedding_model:
        Must match the model used during ingestion.  Defaults to
        ``settings.embedding_model`` (``all-MiniLM-L6-v2``).
    distance_threshold:
        Cosine distance ceiling for returned results.  Chunks whose distance
        exceeds this value are discarded as insufficiently relevant.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        distance_threshold: float = _DEFAULT_DISTANCE_THRESHOLD,
    ) -> None:
        self._persist_dir: str = persist_dir or str(settings.chroma_persist_dir)
        self._collection_name: str = collection_name or settings.chroma_collection_name
        self._embedding_model: str = embedding_model or settings.embedding_model
        self._distance_threshold: float = distance_threshold

        # Lazily initialised
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_relevant_strategies(
        self,
        snapshot: MarketSnapshot,
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        Translate *snapshot* into a semantic query and retrieve the most
        relevant strategy chunks from ChromaDB.

        Args:
            snapshot: A :class:`~market_data.fetcher.MarketSnapshot` produced
                      by ``MarketDataFetcher.fetch()`` or ``fetch_latest()``.
            top_k:    Maximum number of chunks to return.  The actual number
                      may be lower if the collection is small or if some results
                      exceed the distance threshold.

        Returns:
            A :class:`RetrievalResult` containing the query, retrieved chunks,
            and classified market regime.  If the collection is empty or
            retrieval fails, ``result.found`` is ``False`` and ``result.chunks``
            is an empty list — no exception is raised.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        regime = self._compute_market_regime(snapshot)
        query = self._snapshot_to_query(snapshot, regime)

        logger.info(
            f"[bold cyan]{snapshot.ticker}[/bold cyan] — "
            f"regime: {', '.join(f'{k}={v}' for k, v in regime.items())}"
        )
        logger.debug(f"Generated query:\n{query}")

        try:
            collection = self._get_collection()
        except Exception as exc:
            logger.error(f"Failed to open ChromaDB collection: {exc}")
            return self._empty_result(query, snapshot.ticker, regime, collection_size=0)

        collection_size = collection.count()

        if collection_size == 0:
            logger.warning(
                f"ChromaDB collection '{self._collection_name}' is empty. "
                "Ingest at least one YouTube video or PDF before querying. "
                "Run: python -m knowledge_ingestion.youtube_scraper --url <URL>"
            )
            return self._empty_result(query, snapshot.ticker, regime, collection_size=0)

        # Cap top_k to avoid ChromaDB raising when n_results > collection size
        effective_k = min(top_k, collection_size)

        try:
            raw = collection.query(
                query_texts=[query],
                n_results=effective_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error(f"ChromaDB query failed: {exc}")
            return self._empty_result(query, snapshot.ticker, regime, collection_size)

        chunks = self._parse_results(raw)
        chunks = self._filter_by_threshold(chunks)

        if not chunks:
            logger.warning(
                f"No chunks passed the distance threshold "
                f"({self._distance_threshold}).  "
                f"Try ingesting more content or raising the threshold."
            )

        logger.info(
            f"Retrieved {len(chunks)}/{effective_k} chunks "
            f"(threshold={self._distance_threshold}, "
            f"best score={chunks[0].similarity_score if chunks else 'n/a'})"
        )

        return RetrievalResult(
            query=query,
            ticker=snapshot.ticker,
            chunks=chunks,
            market_regime=regime,
            collection_size=collection_size,
        )

    def add_lesson(self, text: str, metadata: Optional[dict[str, Any]] = None) -> str:
        """
        Persist a short post-trade lesson to the ChromaDB collection so that
        future ``get_relevant_strategies()`` queries can surface it alongside
        the seeded playbook chunks.

        Parameters
        ----------
        text:
            Natural-language lesson (2-3 sentences), e.g. Claude's reflection
            on a closed trade.
        metadata:
            Extra metadata to attach.  A few fields are always overwritten:
            ``type='lesson'``, ``created_at`` (ISO timestamp) — callers should
            supply ``ticker``, ``action``, ``pnl``, ``risk_profile``.

        Returns
        -------
        str
            The document id assigned to the new lesson (useful for tests).
        """
        import uuid
        from datetime import datetime as _dt

        doc_id = f"lesson-{uuid.uuid4().hex[:12]}"
        meta = dict(metadata or {})
        meta.setdefault("type", "lesson")
        meta["created_at"] = _dt.utcnow().isoformat(timespec="seconds")
        # ChromaDB rejects None-valued metadata — coerce
        meta = {k: ("" if v is None else v) for k, v in meta.items()}

        col = self._get_collection()
        col.add(ids=[doc_id], documents=[text.strip()], metadatas=[meta])
        logger.info("Lesson persisted: id=%s ticker=%s pnl=%s",
                    doc_id, meta.get("ticker"), meta.get("pnl"))
        return doc_id

    def collection_info(self) -> dict[str, Any]:
        """
        Return a summary dict about the current state of the ChromaDB
        collection (document count, name, persist path).

        Useful for diagnostics and health-check endpoints.
        """
        try:
            col = self._get_collection()
            return {
                "collection_name": self._collection_name,
                "persist_dir": self._persist_dir,
                "document_count": col.count(),
                "embedding_model": self._embedding_model,
                "status": "ok",
            }
        except Exception as exc:
            return {
                "collection_name": self._collection_name,
                "persist_dir": self._persist_dir,
                "document_count": 0,
                "embedding_model": self._embedding_model,
                "status": f"error: {exc}",
            }

    # ------------------------------------------------------------------ #
    # Market regime classification
    # ------------------------------------------------------------------ #

    def _compute_market_regime(self, snapshot: MarketSnapshot) -> dict[str, str]:
        """
        Classify each technical indicator in *snapshot* into a discrete,
        human-readable condition label.

        This intermediate step decouples signal classification from prose
        generation, making each part independently testable.

        Returns a dict with the following keys:

        ==================  =============================================
        Key                 Possible values
        ==================  =============================================
        ``rsi_zone``        Overbought / Bullish / Neutral / Bearish /
                            Oversold
        ``macd_cross``      Bullish Crossover / Bearish Crossover
        ``macd_momentum``   Expanding Bullish / Expanding Bearish /
                            Fading Bullish / Fading Bearish
        ``price_trend``     Strong Uptrend / Uptrend / Strong Downtrend /
                            Downtrend / Sideways / Consolidating
        ``sma_structure``   Golden Cross / Death Cross / Mixed
        ``volatility``      High Volatility / Normal Volatility /
                            Low Volatility  (uses daily range / close)
        ==================  =============================================

        Args:
            snapshot: The market snapshot to classify.

        Returns:
            Dict mapping condition keys to label strings.
        """
        latest = snapshot.latest
        close: float = float(latest["Close"])
        rsi_col = f"RSI_{snapshot.rsi_period}"
        rsi: float = float(latest[rsi_col])
        macd: float = float(latest["MACD"])
        signal: float = float(latest["MACD_Signal"])
        hist: float = float(latest["MACD_Histogram"])

        sma: dict[int, float] = {}
        for p in snapshot.sma_periods:
            col = f"SMA_{p}"
            if col in latest.index:
                sma[p] = float(latest[col])

        regime: dict[str, str] = {}

        # --- RSI zone -------------------------------------------------- #
        if rsi >= _RSI_OVERBOUGHT:
            regime["rsi_zone"] = "Overbought"
        elif rsi >= _RSI_STRONG_BULL:
            regime["rsi_zone"] = "Bullish"
        elif rsi <= _RSI_OVERSOLD:
            regime["rsi_zone"] = "Oversold"
        elif rsi <= _RSI_STRONG_BEAR:
            regime["rsi_zone"] = "Bearish"
        else:
            regime["rsi_zone"] = "Neutral"

        # --- MACD crossover -------------------------------------------- #
        regime["macd_cross"] = "Bullish Crossover" if macd > signal else "Bearish Crossover"

        # --- MACD histogram momentum ------------------------------------ #
        # Positive + expanding or negative + shrinking = strengthening
        if hist > 0:
            regime["macd_momentum"] = "Expanding Bullish" if hist > 0 else "Fading Bullish"
        else:
            regime["macd_momentum"] = "Expanding Bearish" if hist < 0 else "Fading Bearish"

        # Refine: compare histogram to previous bar if history is available
        if len(snapshot.data) >= 2:
            prev_hist = float(snapshot.data.iloc[-2]["MACD_Histogram"])
            if hist > 0:
                regime["macd_momentum"] = (
                    "Expanding Bullish" if hist > prev_hist else "Fading Bullish"
                )
            else:
                regime["macd_momentum"] = (
                    "Expanding Bearish" if hist < prev_hist else "Fading Bearish"
                )

        # --- Price / SMA trend ----------------------------------------- #
        sma20 = sma.get(20)
        sma50 = sma.get(50)
        sma200 = sma.get(200)

        if sma20 and sma50:
            if close > sma20 > sma50:
                regime["price_trend"] = "Strong Uptrend" if (sma200 and close > sma200) else "Uptrend"
            elif close < sma20 < sma50:
                regime["price_trend"] = "Strong Downtrend" if (sma200 and close < sma200) else "Downtrend"
            elif abs(close - sma20) / close < 0.01:
                regime["price_trend"] = "Consolidating"
            else:
                regime["price_trend"] = "Sideways"
        elif sma20:
            regime["price_trend"] = "Uptrend" if close > sma20 else "Downtrend"
        else:
            regime["price_trend"] = "Unknown"

        # --- Golden / Death cross (50 vs 200) -------------------------- #
        if sma50 and sma200:
            regime["sma_structure"] = (
                "Golden Cross" if sma50 > sma200 else "Death Cross"
            )
        elif sma20 and sma50:
            regime["sma_structure"] = (
                "Bullish Alignment" if sma20 > sma50 else "Bearish Alignment"
            )
        else:
            regime["sma_structure"] = "Mixed"

        # --- Volatility (daily range as % of close) -------------------- #
        if "High" in latest.index and "Low" in latest.index:
            high = float(latest["High"])
            low = float(latest["Low"])
            daily_range_pct = (high - low) / close * 100
            if daily_range_pct > 2.5:
                regime["volatility"] = "High Volatility"
            elif daily_range_pct < 0.8:
                regime["volatility"] = "Low Volatility"
            else:
                regime["volatility"] = "Normal Volatility"
        else:
            regime["volatility"] = "Normal Volatility"

        return regime

    # ------------------------------------------------------------------ #
    # Natural-language query generation
    # ------------------------------------------------------------------ #

    def _snapshot_to_query(
        self,
        snapshot: MarketSnapshot,
        regime: dict[str, str],
    ) -> str:
        """
        Translate a classified market regime and raw indicator values into a
        rich natural-language semantic search query.

        The query is intentionally verbose and uses trading terminology likely
        to appear in the ingested YouTube transcripts and PDFs.  This
        maximises the chance of high-quality cosine matches.

        Args:
            snapshot: The source market snapshot (for raw values).
            regime:   The pre-classified regime dict from
                      :meth:`_compute_market_regime`.

        Returns:
            A multi-sentence natural-language query string.
        """
        latest = snapshot.latest
        ticker = snapshot.ticker
        close: float = float(latest["Close"])
        rsi: float = float(latest[f"RSI_{snapshot.rsi_period}"])
        macd: float = float(latest["MACD"])
        signal_val: float = float(latest["MACD_Signal"])
        hist: float = float(latest["MACD_Histogram"])

        sma_parts: list[str] = []
        for p in snapshot.sma_periods:
            col = f"SMA_{p}"
            if col in latest.index:
                sma_parts.append(f"SMA {p}: {float(latest[col]):.2f}")
        sma_text = ", ".join(sma_parts) if sma_parts else "moving averages unavailable"

        # --- Compose plain-English signal sentences -------------------- #
        rsi_zone = regime.get("rsi_zone", "Neutral")
        macd_cross = regime.get("macd_cross", "")
        macd_momentum = regime.get("macd_momentum", "")
        price_trend = regime.get("price_trend", "")
        sma_structure = regime.get("sma_structure", "")
        volatility = regime.get("volatility", "Normal Volatility")

        rsi_sentence = self._rsi_sentence(rsi, rsi_zone)
        macd_sentence = self._macd_sentence(macd, signal_val, hist, macd_cross, macd_momentum)
        trend_sentence = self._trend_sentence(price_trend, sma_structure, close, sma_text)
        action_request = self._action_request(rsi_zone, macd_cross, price_trend)

        query = (
            f"{ticker} is trading at ${close:.2f}. "
            f"{rsi_sentence} "
            f"{macd_sentence} "
            f"{trend_sentence} "
            f"Market volatility is {volatility.lower()}. "
            f"{action_request}"
        )

        return query.strip()

    # ------------------------------------------------------------------ #
    # Query sub-sentence builders (split out for readability + testability)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rsi_sentence(rsi: float, zone: str) -> str:
        if zone == "Overbought":
            return (
                f"RSI is {rsi:.1f}, firmly in overbought territory (above 70). "
                f"This signals potential exhaustion of the bullish move and a possible "
                f"mean-reversion, pullback, or distribution phase."
            )
        if zone == "Oversold":
            return (
                f"RSI is {rsi:.1f}, deep in oversold territory (below 30). "
                f"The asset may be capitulating or approaching a bounce zone; "
                f"watch for divergence or candlestick reversal patterns."
            )
        if zone == "Bullish":
            return (
                f"RSI is {rsi:.1f}, in bullish momentum territory (60–70). "
                f"Trend followers often stay long in this zone while waiting "
                f"for overbought readings to signal a trim."
            )
        if zone == "Bearish":
            return (
                f"RSI is {rsi:.1f}, in bearish momentum territory (30–40). "
                f"Sellers are in control; counter-trend longs carry elevated risk."
            )
        return (
            f"RSI is {rsi:.1f}, neutral (40–60). "
            f"No clear momentum signal; price action and volume are the primary guide."
        )

    @staticmethod
    def _macd_sentence(
        macd: float, signal: float, hist: float,
        cross: str, momentum: str,
    ) -> str:
        hist_abs = abs(hist)
        direction = "above" if macd > signal else "below"
        return (
            f"MACD line ({macd:.4f}) is {direction} the signal line ({signal:.4f}), "
            f"indicating a {cross.lower()}. "
            f"The histogram is {hist:.4f} and {momentum.lower()}, "
            f"suggesting {'building' if 'Expanding' in momentum else 'weakening'} "
            f"{'bullish' if hist >= 0 else 'bearish'} momentum. "
        )

    @staticmethod
    def _trend_sentence(trend: str, structure: str, close: float, sma_text: str) -> str:
        return (
            f"The price trend is classified as a {trend.lower()} with a "
            f"{structure.lower()} on the moving average structure "
            f"({sma_text}). "
            f"Price at ${close:.2f} relative to these averages confirms the "
            f"{'bullish' if 'Up' in trend or 'Golden' in structure else 'bearish'} bias."
        )

    @staticmethod
    def _action_request(rsi_zone: str, macd_cross: str, price_trend: str) -> str:
        return (
            f"Find trading strategies, entry and exit rules, position sizing, "
            f"stop-loss placement, and risk management techniques for an asset "
            f"showing {rsi_zone.lower()} RSI conditions combined with a "
            f"{macd_cross.lower()} and a {price_trend.lower()} trend. "
            f"Include relevant historical pattern analysis, profit target methods, "
            f"and guidelines for managing drawdown in this specific market environment."
        )

    # ------------------------------------------------------------------ #
    # ChromaDB operations
    # ------------------------------------------------------------------ #

    def _get_collection(self) -> chromadb.Collection:
        """
        Lazily initialise and return the ChromaDB collection.

        Uses :class:`SentenceTransformerEmbeddingFunction` with the same
        model name that was used at ingestion time.  A mismatch here would
        silently corrupt similarity scores, so the model name is validated
        against ``settings.embedding_model``.

        Raises:
            RuntimeError: If the persist directory does not exist or the
                          ChromaDB client cannot be created.
        """
        if self._collection is not None:
            return self._collection

        if self._embedding_model != settings.embedding_model:
            logger.warning(
                f"Retriever embedding model ({self._embedding_model!r}) differs "
                f"from the ingestion model ({settings.embedding_model!r}). "
                "Similarity scores will be unreliable if the models are different."
            )

        with quiet_model_load():
            embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model
            )

        try:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open ChromaDB at '{self._persist_dir}': {exc}. "
                "Ensure you have run the knowledge ingestion module at least once."
            ) from exc

        # get_or_create so querying before any ingestion returns an empty
        # collection rather than raising CollectionNotFoundError.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.debug(
            f"Connected to ChromaDB collection '{self._collection_name}' — "
            f"{self._collection.count()} documents."
        )
        return self._collection

    @staticmethod
    def _parse_results(raw: dict[str, Any]) -> list[RetrievedChunk]:
        """
        Parse the raw ChromaDB query response dict into a typed list of
        :class:`RetrievedChunk` objects.

        ChromaDB wraps batch results in an outer list (one entry per query
        text submitted).  We always submit exactly one query, so we index
        into ``[0]``.

        Args:
            raw: Dict returned by ``collection.query(...)``.

        Returns:
            List of :class:`RetrievedChunk`, ordered best-match first
            (ChromaDB already returns them in ascending distance order).
        """
        docs: list[str] = (raw.get("documents") or [[]])[0]
        metas: list[dict] = (raw.get("metadatas") or [[]])[0]
        dists: list[float] = (raw.get("distances") or [[]])[0]
        ids: list[str] = (raw.get("ids") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            if doc:  # skip any empty string slots
                chunks.append(
                    RetrievedChunk(
                        document=doc,
                        metadata=meta or {},
                        distance=float(dist),
                        chunk_id=cid,
                    )
                )
        return chunks

    def _filter_by_threshold(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Remove chunks whose cosine distance exceeds ``_distance_threshold``.

        Args:
            chunks: Ranked list of retrieved chunks.

        Returns:
            Filtered list — may be empty if no chunk passes the threshold.
        """
        filtered = [c for c in chunks if c.distance <= self._distance_threshold]
        dropped = len(chunks) - len(filtered)
        if dropped:
            logger.debug(
                f"Dropped {dropped} chunk(s) with distance > {self._distance_threshold}."
            )
        return filtered

    @staticmethod
    def _empty_result(
        query: str,
        ticker: str,
        regime: dict[str, str],
        collection_size: int,
    ) -> RetrievalResult:
        """Build an empty :class:`RetrievalResult` for error / empty-DB paths."""
        return RetrievalResult(
            query=query,
            ticker=ticker,
            chunks=[],
            market_regime=regime,
            collection_size=collection_size,
        )


# ---------------------------------------------------------------------------
# __main__ — smoke-test with a synthetic MarketSnapshot
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from datetime import date, timedelta

    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # ------------------------------------------------------------------ #
    # Build a synthetic MarketSnapshot to avoid needing a live network
    # call during a RAG unit test.
    #
    # Scenario: Deeply oversold stock (RSI = 25) with a bearish MACD but
    # potentially exhausted sellers — a classic "watch for reversal" setup.
    # ------------------------------------------------------------------ #
    console.rule("[bold yellow]Building synthetic MarketSnapshot[/bold yellow]")

    NUM_ROWS = 5
    today = date.today()
    idx = pd.date_range(end=today, periods=NUM_ROWS, freq="B")  # business days

    # Price around $142 with a slight downtrend
    closes = np.array([155.0, 151.0, 147.0, 143.0, 142.0])

    synthetic_data = pd.DataFrame(
        {
            "Open":  closes - 1.0,
            "High":  closes + 1.5,
            "Low":   closes - 2.0,
            "Close": closes,
            "Volume": np.array([38_000_000, 42_000_000, 55_000_000, 61_000_000, 70_000_000]),
            # Oversold RSI (25) — capitulation-level selling
            "RSI_14":       [45.0, 38.0, 32.0, 27.0, 25.0],
            # Price below all SMAs — confirmed downtrend
            "SMA_20":       [162.0, 161.0, 160.5, 159.0, 158.0],
            "SMA_50":       [168.0, 167.5, 167.0, 166.5, 166.0],
            "SMA_200":      [175.0, 174.8, 174.6, 174.4, 174.2],
            # Bearish MACD: line below signal, expanding negative histogram
            "MACD":         [-0.80, -1.10, -1.45, -1.70, -1.85],
            "MACD_Signal":  [-0.30, -0.55, -0.80, -1.05, -1.25],
            "MACD_Histogram": [-0.50, -0.55, -0.65, -0.65, -0.60],
        },
        index=idx,
    )

    mock_snapshot = MarketSnapshot(
        ticker="MOCK",
        data=synthetic_data,
        sma_periods=(20, 50, 200),
        rsi_period=14,
        macd_params=MACDParams(fast=12, slow=26, signal=9),
    )

    console.print(
        Panel(
            f"[bold]Ticker:[/bold] MOCK\n"
            f"[bold]Close:[/bold]  $142.00\n"
            f"[bold]RSI-14:[/bold] 25.0  ([red]Oversold — extreme selling[/red])\n"
            f"[bold]MACD:[/bold]   −1.85 vs Signal −1.25 "
            f"([yellow]Bearish crossover, histogram fading[/yellow])\n"
            f"[bold]Trend:[/bold]  Price below SMA_20 < SMA_50 < SMA_200 "
            f"([red]Death cross, strong downtrend[/red])",
            title="[cyan]Synthetic Market Conditions[/cyan]",
            expand=False,
        )
    )

    # ------------------------------------------------------------------ #
    # Instantiate retriever and show the generated query
    # ------------------------------------------------------------------ #
    retriever = StrategyRetriever()

    regime = retriever._compute_market_regime(mock_snapshot)
    query  = retriever._snapshot_to_query(mock_snapshot, regime)

    console.rule("[bold cyan]Classified Market Regime[/bold cyan]")
    regime_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    regime_table.add_column("Signal", style="dim")
    regime_table.add_column("Classification", style="bold")
    for key, val in regime.items():
        color = (
            "red"    if any(w in val for w in ("Bear", "Death", "Downtrend", "Oversold")) else
            "green"  if any(w in val for w in ("Bull", "Golden", "Uptrend"))             else
            "yellow"
        )
        regime_table.add_row(key.replace("_", " ").title(), f"[{color}]{val}[/{color}]")
    console.print(regime_table)

    console.rule("[bold cyan]Generated Natural-Language Query[/bold cyan]")
    console.print(Panel(query, title="Query → ChromaDB", expand=True))

    # ------------------------------------------------------------------ #
    # Attempt retrieval (will warn gracefully if collection is empty)
    # ------------------------------------------------------------------ #
    console.rule("[bold cyan]Retrieval Results[/bold cyan]")

    result = retriever.get_relevant_strategies(mock_snapshot, top_k=3)

    info = retriever.collection_info()
    console.print(
        f"[dim]Collection:[/dim] {info['collection_name']}  |  "
        f"[dim]Documents:[/dim] {info['document_count']}  |  "
        f"[dim]Status:[/dim] {info['status']}"
    )

    if not result.found:
        console.print(
            Panel(
                "[yellow]No strategy chunks retrieved.[/yellow]\n\n"
                "The ChromaDB collection is either empty or no results passed the "
                f"distance threshold ({retriever._distance_threshold}).\n\n"
                "To populate it, run:\n"
                "  [bold cyan]python -m knowledge_ingestion.youtube_scraper "
                "--url <YOUTUBE_URL> --title 'Strategy Name'[/bold cyan]",
                title="[yellow]Empty Knowledge Base[/yellow]",
                expand=False,
            )
        )
    else:
        chunks_table = Table(
            title=f"Top {len(result.chunks)} Retrieved Strategy Chunks",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        chunks_table.add_column("#", style="dim", width=3)
        chunks_table.add_column("Score", justify="right", width=7)
        chunks_table.add_column("Source", width=25)
        chunks_table.add_column("Excerpt (first 200 chars)")

        for i, chunk in enumerate(result.chunks, start=1):
            score_color = "green" if chunk.similarity_score > 0.7 else "yellow"
            chunks_table.add_row(
                str(i),
                f"[{score_color}]{chunk.similarity_score:.3f}[/{score_color}]",
                chunk.source_title[:25],
                chunk.document[:200].replace("\n", " ") + "…",
            )
        console.print(chunks_table)

    console.print()
    console.print(f"[bold green]RetrievalResult:[/bold green] {result!r}")
