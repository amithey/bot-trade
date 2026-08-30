"""
YouTube Playlist Ingestor — Bulk Knowledge Ingestion
=====================================================

Extracts all video entries from a YouTube playlist using ``yt-dlp``'s
Python API, then delegates each video to the existing
:class:`~knowledge_ingestion.youtube_scraper.YouTubeScraper` for
transcript cleaning, chunking, and ChromaDB storage.

Why yt-dlp for playlist extraction?
------------------------------------
``youtube-transcript-api`` operates on individual video IDs, not
playlists.  ``yt-dlp`` with ``extract_flat="in_playlist"`` is the
industry-standard tool for extracting the *metadata* of every video in
a playlist (ID, title, position, duration) without downloading any media —
the call takes only a few seconds regardless of playlist length.

Resilience model
----------------
Every video is wrapped in its own ``try/except``.  The following failures
are treated as *warnings* (logged, counted, and skipped):

- No captions available (``TranscriptsDisabled``, ``NoTranscriptFound``)
- Private / deleted / age-restricted video (``VideoUnavailable``)
- YouTube network error (``YouTubeRequestFailed``)
- Any other unexpected exception

The entire playlist run continues.  Only if ``yt-dlp`` itself fails to
extract the playlist metadata (bad URL, network down) does the method
raise, because there is nothing to iterate over.

Typical usage (library)
-----------------------
    from knowledge_ingestion.playlist_scraper import PlaylistIngestor

    ingestor = PlaylistIngestor()
    summary  = ingestor.ingest_playlist(
        "https://www.youtube.com/playlist?list=PLrNmo94geFHYKEZnHR6nDkPTtLWNG61a8"
    )
    print(summary)

Typical usage (CLI)
-------------------
    python -m knowledge_ingestion.playlist_scraper \\
        --url "https://www.youtube.com/playlist?list=PLrNmo94geFHYKEZnHR6nDkPTtLWNG61a8" \\
        --delay 2.0 \\
        --max-videos 10
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import yt_dlp

from knowledge_ingestion.youtube_scraper import IngestionResult, YouTubeScraper
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PLAYLIST_URL = (
    "https://www.youtube.com/watch?v=OvpTnLvZtNw"
    "&list=PLrNmo94geFHYKEZnHR6nDkPTtLWNG61a8"
)

# Courtesy delay between consecutive API calls to youtube-transcript-api.
# Prevents the IP from being soft-banned by YouTube.
_DEFAULT_DELAY_SECONDS: float = 2.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlaylistEntry:
    """
    Lightweight metadata record for a single video inside a playlist.

    Populated by ``yt-dlp`` during playlist extraction — no transcript
    fetching is done at this stage.

    Attributes:
        video_id:         11-character YouTube video ID.
        title:            Video title as returned by yt-dlp.
        url:              Full ``https://www.youtube.com/watch?v=<id>`` URL.
        position:         1-based position in the playlist.
        duration_seconds: Video duration in seconds (may be ``None`` if
                          yt-dlp could not determine it).
    """

    video_id: str
    title: str
    url: str
    position: int
    duration_seconds: Optional[int] = None

    @property
    def duration_str(self) -> str:
        """Human-readable duration string (``mm:ss`` or ``h:mm:ss``)."""
        if self.duration_seconds is None:
            return "unknown"
        secs = int(self.duration_seconds)
        h, remainder = divmod(secs, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def __repr__(self) -> str:
        return (
            f"PlaylistEntry(pos={self.position}, "
            f"id={self.video_id!r}, "
            f"title={self.title[:40]!r}, "
            f"duration={self.duration_str})"
        )


@dataclass
class PlaylistIngestionSummary:
    """
    Complete result of a :meth:`PlaylistIngestor.ingest_playlist` call.

    Attributes:
        playlist_url:          The URL that was processed.
        playlist_title:        Title returned by yt-dlp for the playlist.
        total_videos:          Number of valid video entries found.
        successful:            Videos ingested without error.
        failed:                Videos that raised a non-caption error.
        skipped_no_captions:   Videos skipped because captions were unavailable.
        total_chunks_added:    New chunks written to ChromaDB across all videos.
        total_chunks_skipped:  Chunks that already existed (idempotent upsert).
        total_characters:      Sum of cleaned transcript characters ingested.
        elapsed_seconds:       Wall-clock time for the entire run.
        results:               Per-video :class:`IngestionResult` list
                               (successful runs only).
        failures:              ``(title, error_message)`` pairs for failed videos.
    """

    playlist_url: str
    playlist_title: str
    total_videos: int
    successful: int = 0
    failed: int = 0
    skipped_no_captions: int = 0
    total_chunks_added: int = 0
    total_chunks_skipped: int = 0
    total_characters: int = 0
    elapsed_seconds: float = 0.0
    results: list[IngestionResult] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Percentage of videos successfully ingested."""
        return self.successful / max(self.total_videos, 1) * 100.0

    @property
    def actionable_videos(self) -> int:
        """Videos attempted (excludes no-caption skips)."""
        return self.successful + self.failed

    def __str__(self) -> str:
        return (
            f"PlaylistIngestionSummary: '{self.playlist_title}'\n"
            f"  Videos     : {self.total_videos} total | "
            f"{self.successful} OK | "
            f"{self.failed} failed | "
            f"{self.skipped_no_captions} no-captions\n"
            f"  Chunks     : {self.total_chunks_added} added | "
            f"{self.total_chunks_skipped} already existed\n"
            f"  Characters : {self.total_characters:,}\n"
            f"  Success    : {self.success_rate:.1f}%\n"
            f"  Elapsed    : {self.elapsed_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# Main ingestor class
# ---------------------------------------------------------------------------


class PlaylistIngestor:
    """
    Bulk-ingest every video in a YouTube playlist into the ChromaDB
    knowledge base using the existing :class:`YouTubeScraper` pipeline.

    Parameters
    ----------
    scraper:
        An initialised :class:`YouTubeScraper` instance.  If ``None``,
        a default instance is created.  Passing your own instance lets
        you share a single ChromaDB connection across callers.
    delay_between_videos:
        Seconds to sleep between consecutive video transcript requests.
        Prevents soft-banning by YouTube's transcript API.  Default: 2.0.
    """

    def __init__(
        self,
        scraper: Optional[YouTubeScraper] = None,
        delay_between_videos: float = _DEFAULT_DELAY_SECONDS,
        owner: Optional[str] = None,
    ) -> None:
        # `owner` only applies to a scraper we build ourselves — an injected
        # one already carries whatever attribution its caller chose.
        self._scraper = scraper or YouTubeScraper(owner=owner)
        self._delay = max(0.0, delay_between_videos)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest_playlist(
        self,
        playlist_url: str,
        max_videos: Optional[int] = None,
        start_from: int = 1,
    ) -> PlaylistIngestionSummary:
        """
        Extract all videos from *playlist_url* and ingest each one.

        Args:
            playlist_url: Any YouTube playlist URL format — ``/playlist?list=``,
                          ``/watch?v=...&list=``, or ``youtu.be`` with a list param.
            max_videos:   Cap the number of videos processed.  Useful for
                          dry-running on a large playlist.  ``None`` = no cap.
            start_from:   1-based position to start from.  Set to ``> 1`` to
                          resume an interrupted run.

        Returns:
            A :class:`PlaylistIngestionSummary` with full per-video statistics.

        Raises:
            ValueError:  If the URL cannot be parsed as a playlist.
            RuntimeError: If yt-dlp fails to extract playlist metadata.
        """
        t_start = time.perf_counter()

        # ---- Step 1: Extract playlist metadata via yt-dlp ------------- #
        logger.info(f"Extracting playlist metadata from: {playlist_url}")
        playlist_title, entries = self._extract_entries(playlist_url)

        if not entries:
            raise ValueError(
                f"No video entries found in the playlist at {playlist_url!r}. "
                "Check that the URL is a valid, public YouTube playlist."
            )

        # Apply start_from and max_videos slices
        entries = [e for e in entries if e.position >= start_from]
        if max_videos is not None:
            entries = entries[:max_videos]

        n = len(entries)
        logger.info(
            f"Playlist: '{playlist_title}' — "
            f"{n} video(s) to process"
            + (f" (capped at {max_videos})" if max_videos else "")
            + (f" (starting from position {start_from})" if start_from > 1 else "")
        )

        summary = PlaylistIngestionSummary(
            playlist_url=playlist_url,
            playlist_title=playlist_title,
            total_videos=n,
        )

        # ---- Step 2: Ingest each video with a Rich progress bar ------- #
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print(
            Panel(
                f"[bold]Playlist:[/bold] {playlist_title}\n"
                f"[bold]Videos  :[/bold] {n}"
                + (f"  (max: {max_videos})" if max_videos else "")
                + (f"  (from position {start_from})" if start_from > 1 else "")
                + f"\n[bold]Delay   :[/bold] {self._delay:.1f}s between videos\n"
                f"[bold]Target  :[/bold] ChromaDB collection "
                f"'{self._scraper._collection_name}'",
                title="[bold cyan]Playlist Ingestion Started[/bold cyan]",
                expand=False,
            )
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=36),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn(
                "[green]✓{task.fields[ok]}[/green] "
                "[red]✗{task.fields[fail]}[/red] "
                "[yellow]⊘{task.fields[skip]}[/yellow]"
            ),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                description="[cyan]Initialising…",
                total=n,
                ok=0,
                fail=0,
                skip=0,
            )

            for idx, entry in enumerate(entries):
                # Truncate title for display so the bar stays on one line
                display_title = (
                    entry.title[:52] + "…"
                    if len(entry.title) > 52
                    else entry.title
                )
                progress.update(
                    task,
                    description=(
                        f"[cyan][{entry.position}/{n + start_from - 1}][/cyan] "
                        f"{display_title}"
                    ),
                )

                result = self._ingest_one(entry, summary)

                if result is not None:
                    summary.results.append(result)

                progress.update(
                    task,
                    advance=1,
                    ok=summary.successful,
                    fail=summary.failed,
                    skip=summary.skipped_no_captions,
                )

                # Courtesy sleep — skip on the last video
                if idx < n - 1 and self._delay > 0:
                    time.sleep(self._delay)

        summary.elapsed_seconds = time.perf_counter() - t_start

        # ---- Step 3: Print final report ------------------------------- #
        self._print_final_report(summary, console)
        return summary

    # ------------------------------------------------------------------ #
    # yt-dlp playlist extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_entries(url: str) -> tuple[str, list[PlaylistEntry]]:
        """
        Use ``yt-dlp`` to silently extract all video entries from a playlist.

        Uses ``extract_flat="in_playlist"`` which fetches only metadata
        (ID, title, duration) without downloading any media or audio.
        The call is fast regardless of playlist length.

        Args:
            url: YouTube playlist or video-with-playlist URL.

        Returns:
            Tuple of ``(playlist_title, list[PlaylistEntry])``.

        Raises:
            RuntimeError: If yt-dlp raises any extraction error.
        """
        ydl_opts: dict = {
            # Silence all yt-dlp console output — we handle our own logging
            "quiet": True,
            "no_warnings": True,
            # Extract flat metadata only; do not fetch individual video pages
            "extract_flat": "in_playlist",
            "skip_download": True,
            # Continue past unavailable/private videos in the playlist
            "ignoreerrors": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info: dict = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(
                f"yt-dlp failed to extract playlist info from {url!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error during yt-dlp extraction: {exc}"
            ) from exc

        if info is None:
            raise RuntimeError(
                f"yt-dlp returned None for {url!r}.  "
                "The playlist may be private, deleted, or the URL is invalid."
            )

        playlist_title: str = info.get("title") or info.get("id") or "Unknown Playlist"

        raw_entries: list[dict] = info.get("entries") or []

        # If yt-dlp resolved a single video (watch?v=... without a playlist
        # param that it recognised), wrap it as a single-entry list.
        if not raw_entries and info.get("id"):
            raw_entries = [info]

        entries: list[PlaylistEntry] = []
        for pos, raw in enumerate(raw_entries, start=1):
            if not raw:                   # yt-dlp returns None for unavailable entries
                continue
            vid_id: str = raw.get("id") or ""
            if not vid_id or len(vid_id) != 11:
                # Not a recognisable YouTube video ID; skip silently
                continue
            title: str = (
                raw.get("title")
                or raw.get("fulltitle")
                or vid_id
            )
            # Prefer the explicit URL field; fall back to constructing it
            video_url: str = (
                raw.get("url")
                or raw.get("webpage_url")
                or f"https://www.youtube.com/watch?v={vid_id}"
            )
            # Ensure the URL is a proper watch URL (yt-dlp sometimes gives
            # bare IDs or youtu.be short links in flat mode)
            if not video_url.startswith("http"):
                video_url = f"https://www.youtube.com/watch?v={vid_id}"

            duration = raw.get("duration")
            entries.append(
                PlaylistEntry(
                    video_id=vid_id,
                    title=title,
                    url=video_url,
                    position=pos,
                    duration_seconds=int(duration) if duration else None,
                )
            )

        return playlist_title, entries

    # ------------------------------------------------------------------ #
    # Per-video ingestion with full error isolation
    # ------------------------------------------------------------------ #

    def _ingest_one(
        self,
        entry: PlaylistEntry,
        summary: PlaylistIngestionSummary,
    ) -> Optional[IngestionResult]:
        """
        Ingest a single video, updating *summary* in place.

        All errors are caught here so that a single bad video cannot abort
        the playlist run.  The failure category is determined from the
        exception type so that "no captions" and "real errors" are counted
        separately in the summary.

        Args:
            entry:   The video to process.
            summary: Mutable summary that is updated on every outcome.

        Returns:
            The :class:`IngestionResult` on success, or ``None`` on any failure.
        """
        from youtube_transcript_api import (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )
        from youtube_transcript_api._errors import YouTubeRequestFailed

        logger.debug(
            f"Ingesting [{entry.position}] '{entry.title}' "
            f"({entry.video_id}, {entry.duration_str})"
        )

        try:
            result: IngestionResult = self._scraper.ingest(
                url=entry.url,
                title=entry.title,
            )
        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            msg = f"No captions: {exc}"
            logger.warning(
                f"[yellow]⊘ Skipped[/yellow] [{entry.position}] "
                f"'{entry.title}' — {msg}"
            )
            summary.skipped_no_captions += 1
            summary.failures.append((entry.title, msg))
            return None

        except VideoUnavailable as exc:
            msg = f"Video unavailable (private/deleted/age-restricted): {exc}"
            logger.warning(
                f"[yellow]⊘ Skipped[/yellow] [{entry.position}] "
                f"'{entry.title}' — {msg}"
            )
            summary.skipped_no_captions += 1
            summary.failures.append((entry.title, msg))
            return None

        except YouTubeRequestFailed as exc:
            msg = f"YouTube API request failed: {exc}"
            logger.error(
                f"[red]✗ Failed[/red] [{entry.position}] "
                f"'{entry.title}' — {msg}"
            )
            summary.failed += 1
            summary.failures.append((entry.title, msg))
            return None

        except Exception as exc:
            # Catch-all: log and continue so we never crash the playlist run
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                f"[red]✗ Failed[/red] [{entry.position}] "
                f"'{entry.title}' — unexpected error — {msg}"
            )
            summary.failed += 1
            summary.failures.append((entry.title, msg))
            return None

        # ---- Handle result.success=False (e.g. bad URL parsed inside scraper) ---- #
        if not result.success:
            logger.warning(
                f"[yellow]⊘ Scraper reported failure[/yellow] [{entry.position}] "
                f"'{entry.title}' — {result.error}"
            )
            # Distinguish caption errors from other scraper failures
            error_lower = (result.error or "").lower()
            if any(kw in error_lower for kw in ("caption", "transcript", "disabled")):
                summary.skipped_no_captions += 1
            else:
                summary.failed += 1
            summary.failures.append((entry.title, result.error or "unknown"))
            return None

        # ---- Successful ingestion ---- #
        summary.successful += 1
        summary.total_chunks_added += result.chunks_added
        summary.total_chunks_skipped += result.chunks_skipped
        summary.total_characters += result.total_characters

        logger.debug(
            f"[green]✓[/green] [{entry.position}] '{entry.title}' — "
            f"{result.chunks_added} chunks added in {result.elapsed_seconds:.1f}s"
        )
        return result

    # ------------------------------------------------------------------ #
    # Final report renderer
    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_final_report(
        summary: PlaylistIngestionSummary,
        console: object,
    ) -> None:
        """Print a Rich-formatted final summary panel to the console."""
        from rich import box
        from rich.panel import Panel
        from rich.table import Table

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column("Metric", style="dim", no_wrap=True)
        t.add_column("Value")

        ok_color    = "green"  if summary.successful > 0    else "dim"
        fail_color  = "red"    if summary.failed > 0         else "dim"
        skip_color  = "yellow" if summary.skipped_no_captions > 0 else "dim"
        rate_color  = "green"  if summary.success_rate >= 70 else "yellow"

        t.add_row("Playlist",          summary.playlist_title)
        t.add_row("Total videos",      str(summary.total_videos))
        t.add_row(
            "Successful",
            f"[{ok_color}]{summary.successful}[/{ok_color}]",
        )
        t.add_row(
            "Failed (errors)",
            f"[{fail_color}]{summary.failed}[/{fail_color}]",
        )
        t.add_row(
            "Skipped (no captions)",
            f"[{skip_color}]{summary.skipped_no_captions}[/{skip_color}]",
        )
        t.add_row(
            "Success rate",
            f"[{rate_color}]{summary.success_rate:.1f}%[/{rate_color}]",
        )
        t.add_row("Chunks added",      f"{summary.total_chunks_added:,}")
        t.add_row("Chunks (already in DB)", f"{summary.total_chunks_skipped:,}")
        t.add_row("Characters ingested", f"{summary.total_characters:,}")
        t.add_row(
            "Total time",
            f"{summary.elapsed_seconds:.1f}s  "
            f"({summary.elapsed_seconds / max(summary.total_videos, 1):.1f}s/video avg)",
        )

        console.print(  # type: ignore[attr-defined]
            Panel(
                t,
                title="[bold green]Playlist Ingestion Complete[/bold green]",
                expand=False,
            )
        )

        # List failures if any
        if summary.failures:
            console.print()  # type: ignore[attr-defined]
            fail_table = Table(
                box=box.SIMPLE,
                title=f"[yellow]Failures & Skips ({len(summary.failures)})[/yellow]",
                show_header=True,
                header_style="bold",
            )
            fail_table.add_column("#",     style="dim", width=4)
            fail_table.add_column("Title", max_width=45)
            fail_table.add_column("Reason")
            for i, (title, reason) in enumerate(summary.failures, start=1):
                reason_color = "yellow" if "caption" in reason.lower() else "red"
                fail_table.add_row(
                    str(i),
                    title[:45],
                    f"[{reason_color}]{reason[:80]}[/{reason_color}]",
                )
            console.print(fail_table)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="playlist_scraper",
        description=(
            "Bulk-ingest all videos from a YouTube playlist "
            "into the BotTrade ChromaDB knowledge base."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=_DEFAULT_PLAYLIST_URL,
        metavar="URL",
        help="YouTube playlist URL",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N videos (useful for testing a large playlist)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="Skip the first N-1 videos; resume from position N",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=_DEFAULT_DELAY_SECONDS,
        metavar="SECONDS",
        help="Seconds to sleep between consecutive transcript requests",
    )
    parser.add_argument(
        "--owner",
        default=None,
        metavar="ACCOUNT",
        help="Account id these chunks belong to. Omit for shared content.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the playlist video list without ingesting anything",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    parser = _build_parser()
    args   = parser.parse_args()
    console = Console()

    if args.list_only:
        # Just show what's in the playlist without touching ChromaDB
        console.print(f"[dim]Extracting playlist metadata from:[/dim] {args.url}")
        try:
            title, entries = PlaylistIngestor._extract_entries(args.url)
        except (ValueError, RuntimeError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

        t = Table(
            title=f"Playlist: {title}  ({len(entries)} videos)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        t.add_column("#",        style="dim", width=5)
        t.add_column("Video ID", width=13)
        t.add_column("Title")
        t.add_column("Duration", justify="right", width=10)

        cap = args.max_videos or len(entries)
        for entry in entries[:cap]:
            t.add_row(
                str(entry.position),
                entry.video_id,
                entry.title,
                entry.duration_str,
            )
        console.print(t)
        return

    ingestor = PlaylistIngestor(delay_between_videos=args.delay,
                                owner=args.owner)

    try:
        summary = ingestor.ingest_playlist(
            playlist_url=args.url,
            max_videos=args.max_videos,
            start_from=args.start_from,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Fatal error:[/bold red] {exc}")
        sys.exit(1)

    if summary.failed > 0 and summary.successful == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
