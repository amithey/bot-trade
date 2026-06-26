"""
Post-trade reflection — after a SELL closes a round trip, ask Claude to
distill a 2-3 sentence lesson and persist it to the RAG store so that
future decisions can retrieve it alongside seeded playbook chunks.
"""
from __future__ import annotations

from typing import Optional, Any

from langchain_core.messages import HumanMessage, SystemMessage

from utils.logger import get_logger

logger = get_logger(__name__)


_SYSTEM_PROMPT = (
    "You are a senior trading coach reviewing a CLOSED round-trip trade. "
    "Your job: extract ONE durable lesson that a systematic trader could "
    "re-apply on future setups. Be specific about the signal pattern, the "
    "outcome, and the takeaway. "
    "Output 2-3 plain-English sentences — no headers, no bullet list, no "
    "markdown, no quotes. Reference concrete indicators when relevant "
    "(RSI, MACD, VWAP, Bollinger squeeze, S/R, ATR). If the trade was a "
    "loss, diagnose the likely failure mode and state the exact filter, "
    "sizing change, stop rule, or confirmation rule that should reduce a "
    "similar loss next time; if it was a win, frame it as what to repeat."
)


def reflect_on_trade(
    llm: Any,
    *,
    ticker: str,
    risk_profile: str,
    entry_price: float,
    exit_price: float,
    realized_pnl: float,
    entry_reasoning: str,
    exit_reasoning: str,
    pnl_pct: Optional[float] = None,
) -> Optional[str]:
    """
    Produce a concise lesson string via the supplied ``ChatAnthropic`` client.

    Returns ``None`` on any failure — reflection is best-effort and must
    never crash the trading loop.
    """
    user_block = (
        f"Ticker: {ticker}\n"
        f"Risk profile: {risk_profile}\n"
        f"Entry price: ${entry_price:,.2f}\n"
        f"Exit price:  ${exit_price:,.2f}\n"
        f"Realized P&L: ${realized_pnl:+,.2f}"
        + (f" ({pnl_pct:+.2f}%)" if pnl_pct is not None else "")
        + f"\n\nOriginal entry reasoning:\n{entry_reasoning or '(not recorded)'}"
        + f"\n\nExit reasoning:\n{exit_reasoning or '(not recorded)'}"
        + "\n\nWrite the lesson now."
    )
    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_block),
        ])
        text = getattr(response, "content", "") or ""
        # Guard against model returning list-of-blocks (anthropic format)
        if isinstance(text, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            ).strip()
        else:
            text = str(text).strip()
        if not text:
            return None
        # Cap length to keep RAG chunks lean
        return text[:800]
    except Exception as exc:
        logger.warning("reflect_on_trade failed: %s", exc)
        return None


_META_SYSTEM_PROMPT = (
    "You are a senior trading coach reviewing a BATCH of recently closed "
    "round-trip trades and the per-trade lessons already written about them. "
    "Your job: identify ONE recurring pattern across these trades — a "
    "repeated mistake to stop making, or a repeated setup that keeps "
    "working — and distill it into a single durable meta-rule. "
    "Output 2-4 plain-English sentences. Be concrete: reference indicators "
    "(RSI, MACD, VWAP, BB squeeze, S/R, ATR), market regimes, or risk "
    "behaviors (oversizing, ignoring SL, entering against trend). If no "
    "clear pattern exists across the batch, say so honestly in one sentence "
    "rather than invent one. No headers, no bullet list, no markdown."
)


def summarize_meta_lesson(
    llm: Any,
    *,
    risk_profile: str,
    round_trips: list[dict],
) -> Optional[str]:
    """
    Produce a meta-lesson across a batch of recent round-trip trades.

    Each entry in ``round_trips`` is a dict with keys: ``ticker``,
    ``pnl``, ``pnl_pct``, ``lesson`` (the per-trade reflection text).

    Returns ``None`` on failure — never raises.
    """
    if not round_trips:
        return None
    wins  = sum(1 for r in round_trips if r.get("pnl", 0) > 0)
    losses = sum(1 for r in round_trips if r.get("pnl", 0) < 0)
    total_pnl = sum(r.get("pnl", 0.0) for r in round_trips)

    lines = [
        f"Batch size: {len(round_trips)} round trips "
        f"({wins} wins, {losses} losses, total P&L ${total_pnl:+,.2f})",
        f"Risk profile: {risk_profile}",
        "",
        "Per-trade lessons:",
    ]
    for i, r in enumerate(round_trips, 1):
        pnl_pct = r.get("pnl_pct")
        pnl_str = f"${r.get('pnl', 0):+,.2f}"
        if pnl_pct is not None:
            pnl_str += f" ({pnl_pct:+.2f}%)"
        lesson = (r.get("lesson") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {r.get('ticker','?')} {pnl_str} — {lesson}")
    lines.append("")
    lines.append("Write the meta-lesson now.")
    user_block = "\n".join(lines)
    try:
        response = llm.invoke([
            SystemMessage(content=_META_SYSTEM_PROMPT),
            HumanMessage(content=user_block),
        ])
        text = getattr(response, "content", "") or ""
        if isinstance(text, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )
        text = text.strip()
        if not text:
            return None
        return text[:1000]
    except Exception as exc:
        logger.warning("summarize_meta_lesson failed: %s", exc)
        return None
