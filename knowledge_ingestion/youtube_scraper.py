"""
YouTube Transcript Scraper — Knowledge Ingestion Module
========================================================

Workflow
--------
1. Parse & validate the YouTube URL → extract video ID.
2. Fetch the transcript via ``youtube-transcript-api`` (with language fallback).
3. Clean the raw transcript text (remove filler, normalise whitespace, etc.).
4. Split the cleaned text into overlapping chunks with LangChain's
   ``RecursiveCharacterTextSplitter``.
5. Attach rich metadata to each chunk (source URL, video ID, chunk index, …).
6. Upsert chunks into the persistent ChromaDB collection
   ``trading_strategies`` using a local sentence-transformer embedding model.

Usage (CLI)
-----------
    python -m knowledge_ingestion.youtube_scraper \\
        --url "https://www.youtube.com/watch?v=<VIDEO_ID>" \\
        --title "My Trading Strategy Video"

Usage (library)
---------------
    from knowledge_ingestion.youtube_scraper import YouTubeScraper

    scraper = YouTubeScraper()
    result = scraper.ingest(
        url="https://www.youtube.com/watch?v=<VIDEO_ID>",
        title="RSI + MACD Strategy Explained",
    )
    print(result)
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
from youtube_transcript_api._errors import YouTubeRequestFailed

from config.settings import settings
from utils.hf_quiet import configure_quiet_hf, quiet_model_load
from utils.logger import get_logger

configure_quiet_hf()

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Summary returned by :meth:`YouTubeScraper.ingest`."""

    video_id: str
    title: str
    url: str
    chunks_added: int
    chunks_skipped: int          # Already existed in the collection (dedup)
    total_characters: int
    elapsed_seconds: float
    collection_name: str
    success: bool
    error: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover
        status = "OK" if self.success else f"FAILED — {self.error}"
        return (
            f"[{status}] '{self.title}' ({self.video_id}) | "
            f"chunks added={self.chunks_added} skipped={self.chunks_skipped} | "
            f"chars={self.total_characters} | {self.elapsed_seconds:.1f}s"
        )


@dataclass
class _TranscriptData:
    """Internal holder for a fetched and cleaned transcript."""

    video_id: str
    raw_text: str
    clean_text: str
    language: str
    character_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.character_count = len(self.clean_text)


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------


class YouTubeScraper:
    """
    Ingests YouTube video transcripts into the ChromaDB knowledge base.

    Parameters
    ----------
    persist_dir:
        Override the ChromaDB persist directory (defaults to ``settings``).
    collection_name:
        Override the collection name (defaults to ``settings``).
    chunk_size:
        Target character count per chunk.
    chunk_overlap:
        Overlap in characters between consecutive chunks.
    embedding_model:
        HuggingFace model name for local embeddings.
    preferred_languages:
        Ordered list of BCP-47 language codes to attempt when fetching
        captions.  Falls back to auto-generated captions if manual ones
        are unavailable.
    """

    # Regex patterns used during text cleaning
    _MULTI_SPACE_RE = re.compile(r" {2,}")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _FILLER_RE = re.compile(
        r"\b(um+|uh+|er+|ah+|hmm+|like|you know|i mean|sort of|kind of)\b",
        re.IGNORECASE,
    )
    _MUSIC_NOTE_RE = re.compile(r"[\u266a\u266b\u2669♩♪♫♬]")
    _BRACKET_NOTE_RE = re.compile(r"\[.*?\]|\(.*?\)")  # [Music], (applause), etc.
    _TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        preferred_languages: Optional[list[str]] = None,
    ) -> None:
        self._persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self._collection_name = collection_name or settings.chroma_collection_name
        # `is None` on purpose, not truthiness: `x or default` treats an
        # explicitly-passed 0 (a legitimate "no overlap"/zero chunk size
        # value) the same as "not provided" and silently substitutes the
        # settings default instead of honouring the caller's 0.
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self._embedding_model = embedding_model or settings.embedding_model
        self._preferred_languages = preferred_languages or ["en", "en-US", "en-GB", "iw", "he"]

        # Lazily initialised — avoids loading heavy models on import
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._splitter: Optional[RecursiveCharacterTextSplitter] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, url: str, title: str = "") -> IngestionResult:
        """
        Full ingestion pipeline: fetch → clean → chunk → embed → store.

        Args:
            url:   YouTube video URL (any standard format).
            title: Human-readable title for metadata. If omitted the
                   video ID is used as a fallback.

        Returns:
            :class:`IngestionResult` with statistics about the run.

        Raises:
            Does **not** raise — all errors are captured in
            ``IngestionResult.error`` so callers can handle them
            gracefully.
        """
        t_start = time.perf_counter()

        # 1. Parse URL
        video_id = self._extract_video_id(url)
        if video_id is None:
            return IngestionResult(
                video_id="",
                title=title,
                url=url,
                chunks_added=0,
                chunks_skipped=0,
                total_characters=0,
                elapsed_seconds=time.perf_counter() - t_start,
                collection_name=self._collection_name,
                success=False,
                error=f"Could not parse a valid YouTube video ID from URL: {url!r}",
            )

        if not title:
            title = video_id

        logger.info(f"Starting ingestion — video_id=[bold cyan]{video_id}[/bold cyan], title='{title}'")

        # 2. Fetch transcript
        try:
            transcript_data = self._fetch_transcript(video_id)
        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            msg = (
                f"No captions available for video '{video_id}'. "
                f"The uploader may have disabled transcripts, or the video "
                f"has no auto-generated captions. Detail: {exc}"
            )
            logger.error(msg)
            return IngestionResult(
                video_id=video_id,
                title=title,
                url=url,
                chunks_added=0,
                chunks_skipped=0,
                total_characters=0,
                elapsed_seconds=time.perf_counter() - t_start,
                collection_name=self._collection_name,
                success=False,
                error=msg,
            )
        except VideoUnavailable as exc:
            msg = f"Video '{video_id}' is unavailable (private / deleted). Detail: {exc}"
            logger.error(msg)
            return IngestionResult(
                video_id=video_id,
                title=title,
                url=url,
                chunks_added=0,
                chunks_skipped=0,
                total_characters=0,
                elapsed_seconds=time.perf_counter() - t_start,
                collection_name=self._collection_name,
                success=False,
                error=msg,
            )
        except YouTubeRequestFailed as exc:
            msg = f"YouTube API request failed for '{video_id}'. Detail: {exc}"
            logger.error(msg)
            return IngestionResult(
                video_id=video_id,
                title=title,
                url=url,
                chunks_added=0,
                chunks_skipped=0,
                total_characters=0,
                elapsed_seconds=time.perf_counter() - t_start,
                collection_name=self._collection_name,
                success=False,
                error=msg,
            )

        logger.info(
            f"Transcript fetched — language=[bold]{transcript_data.language}[/bold], "
            f"chars={transcript_data.character_count:,}"
        )

        # 3. Chunk
        chunks = self._split_text(transcript_data.clean_text)
        logger.info(f"Text split into {len(chunks)} chunks "
                    f"(size={self._chunk_size}, overlap={self._chunk_overlap})")

        # 4 & 5. Embed + store (upsert into ChromaDB)
        added, skipped = self._upsert_chunks(
            chunks=chunks,
            video_id=video_id,
            title=title,
            url=url,
            language=transcript_data.language,
        )

        elapsed = time.perf_counter() - t_start
        result = IngestionResult(
            video_id=video_id,
            title=title,
            url=url,
            chunks_added=added,
            chunks_skipped=skipped,
            total_characters=transcript_data.character_count,
            elapsed_seconds=elapsed,
            collection_name=self._collection_name,
            success=True,
        )
        logger.info(f"[bold green]Ingestion complete[/bold green] — {result}")
        return result

    # ------------------------------------------------------------------
    # URL parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_video_id(url: str) -> Optional[str]:
        """
        Extract the YouTube video ID from any common URL format.

        Supports:
        - ``https://www.youtube.com/watch?v=VIDEO_ID``
        - ``https://youtu.be/VIDEO_ID``
        - ``https://www.youtube.com/embed/VIDEO_ID``
        - ``https://www.youtube.com/shorts/VIDEO_ID``
        - Bare video IDs (11 alphanumeric chars + ``_-``)

        Args:
            url: Raw URL string provided by the caller.

        Returns:
            11-character video ID string, or ``None`` if parsing fails.
        """
        url = url.strip()

        # Bare video ID pattern (YouTube IDs are always 11 chars)
        bare_id_re = re.compile(r"^[A-Za-z0-9_-]{11}$")
        if bare_id_re.match(url):
            return url

        try:
            parsed = urlparse(url)
        except ValueError:
            return None

        # youtu.be/<id>
        if parsed.netloc in ("youtu.be", "www.youtu.be"):
            vid = parsed.path.lstrip("/").split("/")[0]
            return vid if len(vid) == 11 else None

        # youtube.com/watch?v=<id>
        if parsed.netloc in ("www.youtube.com", "youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                qs = parse_qs(parsed.query)
                ids = qs.get("v", [])
                return ids[0] if ids and len(ids[0]) == 11 else None

            # /embed/<id>  or  /shorts/<id>  or  /v/<id>
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("embed", "shorts", "v", "e"):
                vid = parts[1]
                return vid if len(vid) == 11 else None

        return None

    # ------------------------------------------------------------------
    # Transcript fetching
    # ------------------------------------------------------------------

    def _fetch_transcript(self, video_id: str) -> _TranscriptData:
        """
        Fetch and assemble a transcript from the YouTube API.

        Strategy:
        1. Request the transcript list for the video.
        2. Try each language in ``_preferred_languages`` (manual captions first).
        3. Fall back to the first available auto-generated transcript.

        Args:
            video_id: 11-character YouTube video ID.

        Returns:
            :class:`_TranscriptData` with raw and cleaned text.

        Raises:
            :exc:`TranscriptsDisabled`: Captions are disabled by uploader.
            :exc:`NoTranscriptFound`:  No transcript in any language.
            :exc:`VideoUnavailable`:   Video is private or deleted.
            :exc:`YouTubeRequestFailed`: Network / API error.
        """
        # youtube-transcript-api 1.x — instance method `.list()` replaces classmethod `.list_transcripts()`
        transcript_list = YouTubeTranscriptApi().list(video_id)

        transcript = None
        used_language = "unknown"

        # Try manual captions in preferred languages
        for lang in self._preferred_languages:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                used_language = f"{lang} (manual)"
                break
            except NoTranscriptFound:
                continue

        # Fall back to auto-generated captions
        if transcript is None:
            for lang in self._preferred_languages:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    used_language = f"{lang} (auto-generated)"
                    break
                except NoTranscriptFound:
                    continue

        # Last resort: any available transcript
        if transcript is None:
            available = list(transcript_list)
            if not available:
                raise NoTranscriptFound(
                    video_id,
                    self._preferred_languages,
                    transcript_list,
                )
            transcript = available[0]
            used_language = f"{transcript.language_code} (fallback)"

        # Fetch the actual text segments and join them.
        # youtube-transcript-api 1.x returns FetchedTranscriptSnippet objects (attrs: .text/.start/.duration).
        # Older 0.x returned plain dicts. Handle both so callers don't need to care.
        segments = transcript.fetch()
        raw_text = " ".join(
            (s.text if hasattr(s, "text") else s["text"]) for s in segments
        )
        clean_text = self._clean_text(raw_text)

        return _TranscriptData(
            video_id=video_id,
            raw_text=raw_text,
            clean_text=clean_text,
            language=used_language,
        )

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """
        Normalise and clean raw transcript text for downstream chunking.

        Steps applied (in order):
        1. Decode HTML entities (``&amp;``, ``&#39;``, etc.).
        2. Strip music notes and bracketed stage directions.
        3. Remove embedded timestamps.
        4. Strip spoken filler words (um, uh, er, …).
        5. Collapse multiple spaces and newlines.
        6. Strip leading/trailing whitespace.

        Args:
            text: Raw concatenated transcript string.

        Returns:
            Cleaned plain-text string.
        """
        import html

        text = html.unescape(text)
        text = self._MUSIC_NOTE_RE.sub("", text)
        text = self._BRACKET_NOTE_RE.sub("", text)
        text = self._TIMESTAMP_RE.sub("", text)
        text = self._FILLER_RE.sub("", text)
        text = self._MULTI_SPACE_RE.sub(" ", text)
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)
        text = text.strip()
        return text

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """
        Split cleaned text into overlapping chunks using LangChain's
        ``RecursiveCharacterTextSplitter``.

        The splitter respects sentence and paragraph boundaries by trying
        to split on ``\\n\\n``, ``\\n``, ``.``, `` `` in that order,
        which is ideal for transcript-style prose.

        Args:
            text: Cleaned transcript text.

        Returns:
            List of non-empty text chunk strings.
        """
        if self._splitter is None:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )

        chunks = self._splitter.split_text(text)
        # Remove empty or whitespace-only chunks
        return [c.strip() for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # ChromaDB operations
    # ------------------------------------------------------------------

    def _get_collection(self) -> chromadb.Collection:
        """
        Return (and lazily initialise) the ChromaDB collection.

        The collection is created with a local :class:`SentenceTransformerEmbeddingFunction`
        so that no external embedding API is required.
        """
        if self._collection is not None:
            return self._collection

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(
            path=str(self._persist_dir)
        )

        with quiet_model_load():
            embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model
            )

        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=embedding_fn,
            metadata={
                "hnsw:space": "cosine",       # Cosine similarity for semantic search
                "description": (
                    "Trading strategies and educational content ingested "
                    "from YouTube and PDF sources."
                ),
            },
        )
        logger.debug(
            f"ChromaDB collection '{self._collection_name}' ready — "
            f"{self._collection.count()} documents already stored."
        )
        return self._collection

    @staticmethod
    def _make_chunk_id(video_id: str, chunk_index: int, chunk_text: str) -> str:
        """
        Generate a deterministic, content-addressed ID for a chunk.

        Using a hash of (video_id + index + first 64 chars of text) means:
        - Re-ingesting the same video produces identical IDs → safe upsert.
        - Slight text edits or re-chunking produce new IDs → no stale data.

        Args:
            video_id:    11-char YouTube video ID.
            chunk_index: 0-based position of the chunk within the transcript.
            chunk_text:  The chunk's text content.

        Returns:
            A 16-character hex digest string.
        """
        fingerprint = f"{video_id}:{chunk_index}:{chunk_text[:64]}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def _upsert_chunks(
        self,
        chunks: list[str],
        video_id: str,
        title: str,
        url: str,
        language: str,
    ) -> tuple[int, int]:
        """
        Upsert text chunks into ChromaDB, skipping duplicates.

        ChromaDB's ``upsert`` is idempotent on the document ID, so
        re-ingesting the same video is safe — existing chunks are
        overwritten in-place rather than duplicated.

        Args:
            chunks:   List of cleaned text chunks.
            video_id: YouTube video ID (for metadata + ID generation).
            title:    Human-readable video title.
            url:      Original YouTube URL.
            language: Transcript language string.

        Returns:
            Tuple of ``(added_count, skipped_count)``.
        """
        collection = self._get_collection()

        # Check which IDs already exist to report accurate skip counts
        candidate_ids = [
            self._make_chunk_id(video_id, i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        existing_ids: set[str] = set()
        try:
            existing = collection.get(ids=candidate_ids, include=[])
            existing_ids = set(existing["ids"])
        except Exception:
            # If the get fails (e.g. empty collection), proceed with full upsert
            pass

        # Build batch payload
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, (doc_id, chunk) in enumerate(zip(candidate_ids, chunks)):
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": "youtube",
                    "video_id": video_id,
                    "title": title,
                    "url": url,
                    "language": language,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                }
            )

        if not ids:
            return 0, 0

        # ChromaDB upsert is batched internally; stay under 5 000 per call
        _BATCH = 500
        for start in range(0, len(ids), _BATCH):
            batch_slice = slice(start, start + _BATCH)
            collection.upsert(
                ids=ids[batch_slice],
                documents=documents[batch_slice],
                metadatas=metadatas[batch_slice],
            )
            logger.debug(
                f"Upserted batch {start // _BATCH + 1} "
                f"({min(start + _BATCH, len(ids)) - start} chunks)"
            )

        added = len(ids) - len(existing_ids)
        skipped = len(existing_ids)
        return added, skipped


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="youtube_scraper",
        description=(
            "Ingest a YouTube video transcript into the BotTrade "
            "ChromaDB knowledge base."
        ),
    )
    parser.add_argument(
        "--url",
        required=True,
        metavar="URL",
        help="Full YouTube video URL, e.g. https://www.youtube.com/watch?v=xxxxx",
    )
    parser.add_argument(
        "--title",
        default="",
        metavar="TITLE",
        help="Human-readable title to tag this video in the vector store.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help=f"Override chunk size (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help=f"Override chunk overlap (default: {settings.chunk_overlap})",
    )
    return parser


def main() -> None:
    """CLI entry point for the YouTube scraper."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    scraper = YouTubeScraper(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    result = scraper.ingest(url=args.url, title=args.title)

    if not result.success:
        logger.error(f"Ingestion failed: {result.error}")
        sys.exit(1)

    logger.info(str(result))


if __name__ == "__main__":
    main()
