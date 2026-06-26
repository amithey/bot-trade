"""
Centralised logging for BotTrade.

Two simultaneous outputs:

Console (Rich)
    Level : INFO and above.
    Format: Rich-rendered, coloured, human-readable.
    Use   : Terminal / Streamlit server stdout — stays clean.

File (rotating)
    Level : DEBUG and above.
    Format: Plain text, machine-readable, one line per record.
    Path  : logs/bottrade.log  (rotates at midnight → bottrade.log.YYYY-MM-DD)
    Keep  : 30 daily files.
    Use   : Audit trail for API calls, RAG retrievals, AI decisions, errors.

Usage
-----
    from utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Decision: BUY QQQ @ $415.20")      # → console + file
    logger.debug("Raw Gemini response: %s", payload) # → file only
    logger.error("API call failed", exc_info=True)   # → console + file + traceback
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

# ── Resolve project-root-relative log directory ───────────────────────────────
# utils/logger.py lives at  <root>/utils/logger.py
# logs/             lives at  <root>/logs/
_LOGS_DIR: Path = Path(__file__).resolve().parent.parent / "logs"

# ── File format — ISO timestamp, padded level, full module path, message ──────
_FILE_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)-45s | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ── Module-level guard so we never attach handlers twice ─────────────────────
_configured: bool = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger wired to both the Rich console handler and the
    daily rotating file handler.

    Idempotent: the root logger handlers are attached exactly once across
    the entire process lifetime, regardless of how many modules call this.

    Parameters
    ----------
    name:
        Caller's ``__name__``.  Examples: ``"market_data.fetcher"``,
        ``"decision_engine.ai_engine"``.

    Returns
    -------
    logging.Logger
        A logger that writes INFO+ to the console and DEBUG+ to the file.
    """
    global _configured  # noqa: PLW0603

    if _configured:
        return logging.getLogger(name)

    # ── Load console log level from settings (graceful fallback) ─────────────
    try:
        from config.settings import settings
        console_level_str: str = settings.log_level.upper()
    except Exception:
        console_level_str = "INFO"

    console_level: int = getattr(logging, console_level_str, logging.INFO)

    # ── Create log directory (no-op if it already exists) ────────────────────
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Root logger: DEBUG so every handler can decide its own floor ─────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ─────────────────────────────────────────────────────────────────────────
    # Handler 1 — Rich console
    # ─────────────────────────────────────────────────────────────────────────
    rich_handler = RichHandler(
        level=console_level,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_path=True,
        markup=True,
        log_time_format="[%X]",
    )
    # Plain formatter: Rich already adds timestamp, level, path
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    # ─────────────────────────────────────────────────────────────────────────
    # Handler 2 — daily rotating file
    # ─────────────────────────────────────────────────────────────────────────
    log_file = _LOGS_DIR / "bottrade.log"

    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",       # rotate at midnight local time
        interval=1,
        backupCount=30,        # keep 30 days → 30 archived files
        encoding="utf-8",
        utc=False,
        delay=True,            # don't create the file until the first log record
    )
    # Suffix format controls the backup filename:  bottrade.log.2024-03-15
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=_FILE_FORMAT, datefmt=_FILE_DATEFMT)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Attach both handlers to the root logger
    # ─────────────────────────────────────────────────────────────────────────
    root.addHandler(rich_handler)
    root.addHandler(file_handler)

    # ── Silence noisy third-party loggers at the file level too ──────────────
    for _noisy in (
        "yfinance", "peewee", "urllib3", "httpx", "httpcore",
        "chromadb", "sentence_transformers", "transformers",
        "huggingface_hub", "huggingface_hub.file_download", "safetensors",
        "google.auth", "google.api_core",
    ):
        logging.getLogger(_noisy).setLevel(logging.ERROR)

    _configured = True
    return logging.getLogger(name)


# ── Convenience helpers used by the AI engine and portfolio manager ───────────

def log_api_call(
    logger: logging.Logger,
    model: str,
    ticker: str,
    prompt_tokens: Optional[int] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """Log a single Gemini API call to the file at DEBUG level."""
    parts = [f"API_CALL | model={model} | ticker={ticker}"]
    if prompt_tokens is not None:
        parts.append(f"tokens~{prompt_tokens}")
    if latency_ms is not None:
        parts.append(f"latency={latency_ms:.0f}ms")
    logger.debug(" | ".join(parts))


def log_rag_retrieval(
    logger: logging.Logger,
    ticker: str,
    n_chunks: int,
    best_score: Optional[float],
    query_snippet: str,
) -> None:
    """Log a RAG retrieval event to the file at DEBUG level."""
    score_str = f"{best_score:.3f}" if best_score is not None else "n/a"
    logger.debug(
        "RAG_RETRIEVAL | ticker=%s | chunks=%d | best_score=%s | query=%.80s",
        ticker, n_chunks, score_str, query_snippet,
    )


def log_trade_execution(
    logger: logging.Logger,
    action: str,
    ticker: str,
    price: float,
    quantity: float,
    cash_after: float,
    portfolio_value: float,
    reasoning_snippet: str = "",
) -> None:
    """Log a virtual trade execution to the file at INFO level."""
    logger.info(
        "TRADE | %s %s | price=$%.4f | qty=%.4f | "
        "cash_after=$%.2f | portfolio=$%.2f | reason=%.120s",
        action, ticker, price, quantity,
        cash_after, portfolio_value, reasoning_snippet,
    )
