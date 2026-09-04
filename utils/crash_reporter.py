"""Crash reporter — captures uncaught exceptions from both the main
thread and worker threads, writes them to a rotating ``logs/crashes.log``
with full traceback + a one-line summary suitable for ``st.error``.

Why a separate file?
--------------------
Uncaught exceptions are *the* signal that something is wrong. Buried in
``bottrade.log`` (which logs every API call and RAG query) they get lost.
Crashes are precious — they deserve their own append-only audit trail.

Hooks installed
---------------
1. ``sys.excepthook`` — main thread.
2. ``threading.excepthook`` — every ``Thread`` (Python 3.8+). The live
   trading engine runs on a daemon thread, so without this hook a crash
   there would die silently.
3. Optional ``post_crash_callback`` — called with the formatted message
   so the dispatcher / Telegram notifier can fan out.

Idempotent — calling ``install()`` twice replaces the previous hooks
cleanly.
"""
from __future__ import annotations

import logging
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

_LOGS_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
_CRASH_LOG: Path = _LOGS_DIR / "crashes.log"

# Singleton logger
_crash_logger: Optional[logging.Logger] = None
_post_callback: Optional[Callable[[str, str], None]] = None
_installed: bool = False


def _ensure_logger() -> logging.Logger:
    global _crash_logger
    if _crash_logger is not None:
        return _crash_logger
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("bottrade.crash")
    log.setLevel(logging.ERROR)
    log.propagate = False
    # 5 MB × 10 files = 50 MB max
    h = RotatingFileHandler(
        str(_CRASH_LOG), maxBytes=5_000_000, backupCount=10,
        encoding="utf-8", delay=True,
    )
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(h)
    _crash_logger = log
    return log


def _format_crash(
    *, source: str, exc_type, exc_value, exc_tb,
) -> tuple[str, str]:
    """Return (one_liner, full_text)."""
    one_liner = (
        f"[{source}] {exc_type.__name__}: {exc_value}"
        if exc_type else f"[{source}] (no exception info)"
    )
    full = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    return one_liner, full


def _record(source: str, exc_type, exc_value, exc_tb) -> None:
    log = _ensure_logger()
    one, full = _format_crash(
        source=source, exc_type=exc_type,
        exc_value=exc_value, exc_tb=exc_tb,
    )
    # File: header + traceback
    log.error("CRASH %s\n%s%s",
              one, full, "─" * 72)
    # Optional fan-out (Telegram, etc.)
    if _post_callback is not None:
        try:
            _post_callback(one, full)
        except Exception:  # noqa: BLE001
            pass


def install(post_callback: Optional[Callable[[str, str], None]] = None) -> None:
    """Install crash hooks. Safe to call multiple times.

    Parameters
    ----------
    post_callback : callable(one_liner, full_traceback), optional
        Called for every captured crash. Use this to notify Telegram or
        any other channel. Errors raised inside the callback are swallowed.
    """
    global _post_callback, _installed
    _post_callback = post_callback
    _ensure_logger()

    # --- Main thread ---
    def _excepthook(exc_type, exc_value, exc_tb):
        _record("main", exc_type, exc_value, exc_tb)
        # Still print to stderr so console operators see it too
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    # --- Worker threads (Python 3.8+) ---
    if hasattr(threading, "excepthook"):
        def _thread_excepthook(args: "threading.ExceptHookArgs"):
            tname = getattr(args.thread, "name", "unknown") if args.thread else "unknown"
            _record(
                f"thread:{tname}",
                args.exc_type, args.exc_value, args.exc_traceback,
            )
        threading.excepthook = _thread_excepthook

    _installed = True


def is_installed() -> bool:
    return _installed


def report_manual(
    summary: str, exc: Optional[BaseException] = None, source: str = "manual",
) -> None:
    """Record a crash that you caught yourself and want to preserve."""
    if exc is not None:
        _record(source, type(exc), exc, exc.__traceback__)
    else:
        log = _ensure_logger()
        log.error("CRASH [%s] %s\n%s%s",
                  source, summary, traceback.format_exc(), "─" * 72)


def crash_log_path() -> Path:
    return _CRASH_LOG


def recent_crashes(limit: int = 5) -> list[str]:
    """Return last ``limit`` crash one-liners (newest first). For UI."""
    if not _CRASH_LOG.exists():
        return []
    try:
        text = _CRASH_LOG.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    # Each crash starts with "<timestamp> | CRASH ..."
    lines = [ln for ln in text.splitlines() if " | CRASH " in ln]
    return list(reversed(lines))[:limit]
