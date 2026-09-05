"""Stall watchdog — proves whether the *process* froze, or only its output did.

Why this exists
---------------
Production showed a repeating failure on Fly.io: for minutes at a time the
app stopped answering ``/_stcore/health`` (Fly logged "context deadline
exceeded ... awaiting headers"), and with a single machine and no second
one to route to, Fly's proxy gave up entirely — "could not find a good
candidate within 40 attempts at load balancing", which is the hard
"connection error" a user actually sees. No crash, no OOM kill, no
restart: the machine stayed ``started`` the whole time and recovered on
its own.

The logs could not settle *why*, because during each window they showed
nothing at all — and "no logs" has several very different causes that are
indistinguishable after the fact:

1. The process was genuinely frozen (CPU starvation on a shared vCPU
   whose burst balance ran out, a GIL held by a long native call, or the
   hypervisor descheduling the VM). Nothing ran, so nothing logged.
2. The process was fine, but *stdout was blocked*. Writing to a full pipe
   blocks the writing thread while it holds the logging handler's lock,
   which stalls every other thread that logs — including the event loop
   serving the health check. Nothing could log, by definition.
3. The process and its logging were both fine, and only Fly's proxy or
   health-check path was unhappy.

A watchdog distinguishes them. It sleeps in a tight, cheap loop and
measures how late each wake-up actually was. After the fact:

* A gap with a large "late by" number means the thread did not get
  scheduled — case 1, the process really was frozen.
* On-time ticks through a window where Fly's logs went silent means the
  process was running fine and only its output was stuck — case 2.
* No lateness at all across a window where the health check failed means
  the app was healthy and the problem sits outside it — case 3.

Deliberately silent unless something is wrong: a normal tick logs
nothing. The failure being investigated is partly *caused* by log volume
in the first place, so a watchdog that chattered every second would be
part of the problem it is measuring.

Both clocks are recorded because they answer different questions.
``time.monotonic()`` measures scheduling: how long this thread was denied
the CPU. ``time.time()`` measures wall-clock: a large gap in monotonic
time with a matching wall-clock gap is a real stall, while wall-clock
jumping alone is just an NTP correction and not interesting.

Usage
-----
    from utils import stall_watchdog
    stall_watchdog.install()      # idempotent; safe on every Streamlit rerun
"""
from __future__ import annotations

import threading
import time

from utils.logger import get_logger

logger = get_logger(__name__)

#: How often the watchdog wakes up. Short enough to catch a stall with
#: useful resolution, long enough that the thread itself costs nothing.
_TICK_SEC = 1.0

#: Report a wake-up only when it was this many seconds later than asked
#: for. Ordinary scheduling jitter on a shared vCPU is milliseconds; Fly's
#: health check gives up after 5s, so anything at or past this threshold is
#: already long enough to have caused a failed check.
_LATE_THRESHOLD_SEC = 5.0

_thread: threading.Thread | None = None
_lock = threading.Lock()


def is_installed() -> bool:
    return _thread is not None and _thread.is_alive()


def _watch() -> None:
    while True:
        mono_before = time.monotonic()
        wall_before = time.time()
        time.sleep(_TICK_SEC)
        mono_late = (time.monotonic() - mono_before) - _TICK_SEC
        wall_late = (time.time() - wall_before) - _TICK_SEC

        if mono_late < _LATE_THRESHOLD_SEC:
            continue

        # Logging here is itself best-effort: if the stall was stdout
        # backpressure, this record only escapes once the pipe drains —
        # which is exactly the evidence wanted, since the reported lateness
        # still describes the window that just ended.
        logger.warning(
            "STALL | this thread was denied the CPU for %.1fs "
            "(asked to sleep %.1fs, wall-clock drift %.1fs). Nothing else in "
            "this process ran during that window either — that is long enough "
            "to fail Fly's 5s health check and for the proxy to report no "
            "healthy machine.",
            mono_late, _TICK_SEC, wall_late - mono_late,
        )


def install() -> None:
    """Start the watchdog once per process. Safe to call repeatedly.

    Streamlit re-executes the entrypoint on every rerun, so this is called
    many times per minute; all but the first call return immediately.
    """
    global _thread
    if is_installed():
        return
    with _lock:
        if is_installed():
            return
        _thread = threading.Thread(
            target=_watch, name="stall-watchdog", daemon=True,
        )
        _thread.start()
