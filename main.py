"""
BotTrade — AI Trading Bot with RAG Architecture
================================================

Unified entry point and CLI for the full 5-phase pipeline.

Usage
-----
  # Ingest a YouTube video into the knowledge base
  python main.py ingest --url "https://www.youtube.com/watch?v=XXX" --title "RSI Strategy"

  # Run a live decision on a single ticker
  python main.py decide --ticker QQQ

  # Run a historical backtest
  python main.py backtest --ticker QQQ --start 2023-01-01 --end 2023-12-31 --capital 10000

  # Show knowledge base status
  python main.py status

Or import and call each phase directly from your own scripts.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    """Phase 1: Ingest a YouTube URL into ChromaDB."""
    from knowledge_ingestion.youtube_scraper import YouTubeScraper
    scraper = YouTubeScraper()
    result = scraper.ingest(url=args.url, title=args.title or "")
    if not result.success:
        logger.error(f"Ingestion failed: {result.error}")
        sys.exit(1)
    logger.info(str(result))


def cmd_decide(args: argparse.Namespace) -> None:
    """Phases 2–4: Fetch live data and get an AI trading decision."""
    from market_data.fetcher import MarketDataFetcher
    from rag.retriever import StrategyRetriever
    from decision_engine.ai_engine import AITradingEngine
    from decision_engine.ai_engine import _render_decision
    from rich.console import Console

    fetcher   = MarketDataFetcher()
    retriever = StrategyRetriever()
    engine    = AITradingEngine()

    snap      = fetcher.fetch_latest(args.ticker, lookback_days=args.lookback)
    retrieval = retriever.get_relevant_strategies(snap, top_k=args.top_k)
    decision  = engine.evaluate_market(snap, retrieval)

    _render_decision(Console(), decision)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Phase 5: Run a historical backtest."""
    from market_data.fetcher import MarketDataFetcher
    from rag.retriever import StrategyRetriever
    from decision_engine.ai_engine import AITradingEngine
    from backtesting.backtest_runner import BacktestRunner

    runner = BacktestRunner(
        fetcher=MarketDataFetcher(),
        retriever=StrategyRetriever(),
        engine=AITradingEngine(),
        inter_call_delay=args.delay,
    )
    result = runner.run(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )
    result.print_summary()


def cmd_status(args: argparse.Namespace) -> None:
    """Show knowledge base and settings status."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rag.retriever import StrategyRetriever
    from config.settings import settings

    console = Console()
    retriever = StrategyRetriever()
    info = retriever.collection_info()

    t = Table(box=box.ROUNDED, show_header=False)
    t.add_column("Key", style="dim")
    t.add_column("Value")
    t.add_row("Collection", info["collection_name"])
    t.add_row("Documents", str(info["document_count"]))
    t.add_row("Embedding model", info["embedding_model"])
    t.add_row("ChromaDB path", info["persist_dir"])
    t.add_row("LLM model", settings.llm_model)
    t.add_row("Status", f"[green]{info['status']}[/green]")
    console.print(t)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bottrade",
        description="BotTrade — AI Trading Bot with RAG Architecture",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a YouTube video into ChromaDB")
    p_ingest.add_argument("--url", required=True)
    p_ingest.add_argument("--title", default="")

    # decide
    p_decide = sub.add_parser("decide", help="Get a live AI decision for a ticker")
    p_decide.add_argument("--ticker", required=True)
    p_decide.add_argument("--lookback", type=int, default=420)
    p_decide.add_argument("--top-k", type=int, default=3, dest="top_k")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run a historical backtest")
    p_bt.add_argument("--ticker", required=True)
    p_bt.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_bt.add_argument("--end",   default=str(date.today() - timedelta(days=1)))
    p_bt.add_argument("--capital", type=float, default=10_000.0)
    p_bt.add_argument("--delay", type=float, default=1.0,
                      help="Seconds between API calls (default: 1.0)")

    # status
    sub.add_parser("status", help="Show knowledge base and config status")

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    dispatch = {
        "ingest":   cmd_ingest,
        "decide":   cmd_decide,
        "backtest": cmd_backtest,
        "status":   cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
