"""
Web Article Scraper — Knowledge Ingestion Module
=================================================

Ingests trading articles, strategy write-ups, and market analyses from
ANY web page (Investopedia, TradingView ideas, broker blogs, Substack…)
into the same ChromaDB collection the RAG retriever reads from.

Workflow
--------
1. Fetch the page over HTTPS (desktop User-Agent, 20 s timeout).
2. Extract the readable body text with a stdlib ``HTMLParser`` — keeps
   <p>/<li>/<h1-3>/<blockquote> blocks, drops nav/footer/script/ads.
   No extra dependencies needed.
3. Clean + split into overlapping chunks (same splitter settings as the
   YouTube/seed pipelines).
4. Upsert with content-addressed SHA256 ids → fully idempotent; re-running
   the same URL adds nothing twice.

Usage (CLI)
-----------
    python -m knowledge_ingestion.article_scraper \\
        --url "https://www.investopedia.com/articles/..." \\
        [--title "Optional override title"]

Usage (library)
---------------
    from knowledge_ingestion.article_scraper import ArticleScraper
    result = ArticleScraper().ingest("https://example.com/macd-strategy")
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlparse

import requests

from config.settings import settings
from utils.hf_quiet import configure_quiet_hf, quiet_model_load
from utils.logger import get_logger

configure_quiet_hf()
logger = get_logger(__name__)

_MIN_ARTICLE_CHARS = 400          # refuse pages with less readable text
_FETCH_TIMEOUT_SEC = 20
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


# ---------------------------------------------------------------------------
# Readable-text extraction (stdlib only)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Collects text from content tags while skipping boilerplate regions."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "aside", "header",
                 "form", "noscript", "svg", "button", "iframe", "figure"}
    BLOCK_TAGS = {"p", "li", "blockquote", "h1", "h2", "h3", "h4", "pre",
                  "td"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._block_depth = 0
        self._buf: list[str] = []
        self.blocks: list[str] = []
        self.title: str = ""
        self._in_title = False
        self._og_title: str = ""

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            d = dict(attrs)
            if d.get("property") in ("og:title", "twitter:title") \
                    and d.get("content"):
                self._og_title = d["content"].strip()
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self._block_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS \
                and self._block_depth > 0:
            self._block_depth -= 1
            if self._block_depth == 0 and self._buf:
                text = " ".join("".join(self._buf).split())
                self._buf.clear()
                if len(text) >= 30:          # drop crumbs / button labels
                    self.blocks.append(text)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip_depth == 0 and self._block_depth > 0:
            self._buf.append(data)

    @property
    def best_title(self) -> str:
        t = (self._og_title or self.title or "").strip()
        # Strip common " | SiteName" / " - SiteName" suffixes
        return re.split(r"\s+[|–—-]\s+", t)[0].strip() or t


def extract_readable_text(html: str) -> tuple[str, str]:
    """Return ``(title, body_text)`` extracted from raw HTML."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass  # HTMLParser is forgiving; partial output is fine
    body = "\n\n".join(parser.blocks)
    return parser.best_title, unescape(body)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ArticleIngestionResult:
    url: str
    title: str
    domain: str
    chunks_added: int
    chunks_skipped: int
    total_characters: int
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None

    def __str__(self) -> str:
        status = "OK" if self.success else f"FAILED — {self.error}"
        return (f"[{status}] '{self.title[:60]}' ({self.domain}) | "
                f"chunks added={self.chunks_added} "
                f"skipped={self.chunks_skipped} | "
                f"chars={self.total_characters} | "
                f"{self.elapsed_seconds:.1f}s")


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class ArticleScraper:
    """Fetches a web article and upserts its text into ChromaDB."""

    def __init__(self) -> None:
        self._collection = None   # lazy — model load is slow

    # -- internals -------------------------------------------------------

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        with quiet_model_load():
            embed_fn = SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model,
                trust_remote_code=True,
            )
        client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir))
        self._collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    @staticmethod
    def _fetch(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Not a valid http(s) URL: {url!r}")
        resp = requests.get(
            url, timeout=_FETCH_TIMEOUT_SEC,
            headers={"User-Agent": _UA,
                     "Accept-Language": "en-US,en;q=0.9,he;q=0.8"},
        )
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype and "<html" not in resp.text[:2000].lower():
            raise ValueError(f"URL did not return an HTML page "
                             f"(Content-Type: {ctype or 'unknown'}).")
        return resp.text

    @staticmethod
    def _chunk_id(url: str, idx: int, text: str) -> str:
        fingerprint = f"web:{url}:{idx}:{text[:64]}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    # -- public ----------------------------------------------------------

    def ingest(self, url: str,
               title: Optional[str] = None) -> ArticleIngestionResult:
        """Fetch *url*, extract text, chunk, and upsert. Never raises —
        check ``result.success``."""
        t0 = time.time()
        domain = urlparse(url).netloc.lower()

        def _fail(err: str) -> ArticleIngestionResult:
            logger.warning(f"Article ingest failed for {url}: {err}")
            return ArticleIngestionResult(
                url=url, title=title or "", domain=domain,
                chunks_added=0, chunks_skipped=0, total_characters=0,
                elapsed_seconds=time.time() - t0, success=False, error=err)

        try:
            html = self._fetch(url)
        except Exception as exc:
            return _fail(f"fetch error: {exc}")

        page_title, body = extract_readable_text(html)
        final_title = (title or page_title or domain).strip()

        if len(body) < _MIN_ARTICLE_CHARS:
            return _fail(
                f"only {len(body)} readable characters extracted — the page "
                f"is probably paywalled, JS-rendered, or not an article")

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(body)
        if not chunks:
            return _fail("text splitter produced no chunks")

        ids = [self._chunk_id(url, i, c) for i, c in enumerate(chunks)]
        metas = [{
            "source":       "web_article",
            "source_id":    domain,
            "url":          url,
            "title":        final_title[:200],
            "chunk_index":  i,
            "total_chunks": len(chunks),
            "chunk_size":   len(c),
        } for i, c in enumerate(chunks)]

        try:
            collection = self._get_collection()
            try:
                existing = set(collection.get(ids=ids, include=[])["ids"])
            except Exception:
                existing = set()
            new_ids = [i for i in ids if i not in existing]
            if new_ids:
                collection.upsert(
                    ids=new_ids,
                    documents=[chunks[ids.index(i)] for i in new_ids],
                    metadatas=[metas[ids.index(i)] for i in new_ids],
                )
        except Exception as exc:
            return _fail(f"ChromaDB upsert error: {exc}")

        result = ArticleIngestionResult(
            url=url, title=final_title, domain=domain,
            chunks_added=len(new_ids),
            chunks_skipped=len(ids) - len(new_ids),
            total_characters=len(body),
            elapsed_seconds=time.time() - t0,
            success=True,
        )
        logger.info(str(result))
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ingest one or more web articles into the knowledge base.")
    ap.add_argument("--url", action="append", required=True,
                    help="Article URL (repeat --url to ingest several).")
    ap.add_argument("--title", default=None,
                    help="Optional title override (single-URL mode only).")
    args = ap.parse_args()

    scraper = ArticleScraper()
    failures = 0
    for url in args.url:
        res = scraper.ingest(
            url, title=args.title if len(args.url) == 1 else None)
        print(res)
        if not res.success:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
