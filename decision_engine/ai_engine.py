"""
AI Decision Engine — Phase 4
==============================

Uses the Anthropic Claude API (via LangChain) to evaluate a live market
state alongside retrieved strategy knowledge and emit a structured,
fully-typed trading decision (BUY / SELL / HOLD).

Architecture
------------
Structured output is obtained via LangChain's ``with_structured_output``
interface.  The chain is built once at init time:

    self.llm            = ChatAnthropic(model="claude-3-haiku-20240307", …)
    self.structured_llm = self.llm.with_structured_output(TradingDecision)

Invoking ``self.structured_llm`` with a list of LangChain messages returns
a fully-validated ``TradingDecision`` Pydantic object directly — no manual
JSON parsing, no regex, no tool-call extraction required.  LangChain
surfaces Claude's tool-use mechanism transparently to implement structured
output.

Retry strategy
--------------
``tenacity`` retries on transient Anthropic API errors (rate-limit /
RateLimitError, timeout / APITimeoutError, server errors /
InternalServerError, connection drops / APIConnectionError) with
exponential back-off (2 s → 6 s, 2 attempts).  If all attempts are
exhausted, or a non-retryable error occurs, the engine logs the error
and returns a safe ``HOLD`` with ``confidence_score = 0.0``.

Typical usage
-------------
    from market_data.fetcher import MarketDataFetcher
    from rag.retriever       import StrategyRetriever
    from decision_engine.ai_engine import AITradingEngine

    fetcher   = MarketDataFetcher()
    retriever = StrategyRetriever()
    engine    = AITradingEngine()

    snap      = fetcher.fetch_latest("QQQ", lookback_days=420)
    retrieval = retriever.get_relevant_strategies(snap, top_k=4)
    decision  = engine.evaluate_market(snap, retrieval)

    print(decision.action)           # "BUY" | "SELL" | "HOLD"
    print(decision.confidence_score) # 0.0 – 1.0
    print(decision.reasoning)        # detailed chain-of-thought
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import anthropic as _anthropic
from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from market_data.fetcher import MarketSnapshot
from rag.retriever import RetrievalResult
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Load .env at module import time — before @st.cache_resource ever fires.
#
# find_dotenv() walks UP the directory tree from this file until it locates
# a .env file, handling Windows UNC paths and mixed separators correctly.
# override=True ensures an empty or stale env-var value is overwritten.
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv(usecwd=False), override=True)

# ---------------------------------------------------------------------------
# System prompt — injected as a SystemMessage on every invoke() call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = """You are a disciplined, unemotional quantitative trading analyst embedded in an autonomous algorithmic trading system.

## Your Mandate
Evaluate the provided technical market data and retrieved trading strategy knowledge to produce a single, structured trading decision. You have NO access to news, sentiment, or any information not explicitly included in the user message.

## Decision Framework

### Signal Interpretation Rules
1. **RSI**
   - < 30  : Oversold — potential reversal zone; look for confirmation from MACD / price action.
   - 30–40 : Bearish momentum — sellers in control; avoid longs unless strong strategy support.
   - 40–60 : Neutral — defer to trend and MACD for bias.
   - 60–70 : Bullish momentum — trend-following longs viable.
   - > 70  : Overbought — potential exhaustion; tighten stops on longs, watch for reversal.

2. **MACD**
   - Line above Signal + Histogram positive + expanding → bullish momentum accelerating.
   - Line below Signal + Histogram negative + expanding → bearish momentum accelerating.
   - Histogram shrinking (regardless of sign) → current momentum is fading; caution.

3. **Moving Averages**
   - Price > SMA_20 > SMA_50 > SMA_200 : textbook uptrend; bias long.
   - Price < SMA_20 < SMA_50 < SMA_200 : textbook downtrend; bias short or cash.
   - Golden Cross (SMA_50 > SMA_200)   : long-term bullish structure.
   - Death Cross  (SMA_50 < SMA_200)   : long-term bearish structure.

### Confluence Scoring
- **Strong BUY signal** (confidence ≥ 0.75) : RSI oversold OR bullish + MACD bullish cross + price above key MAs + strategy text confirms long entry rules.
- **Strong SELL signal** (confidence ≥ 0.75) : RSI overbought OR bearish + MACD bearish cross + price below key MAs + strategy text confirms exit/short rules.
- **HOLD** : Any of — mixed indicator signals, low similarity strategy context (< 0.50 score), insufficient data, or net confidence < 0.50.

### Mandatory Reasoning Standards
- Cite EXACT numeric values: "RSI_14 = 25.3, below the oversold threshold of 30".
- State which retrieved strategy rules match or contradict the setup.
- When signals conflict, explain EXPLICITLY how you resolved the conflict.
- If no relevant strategies were retrieved, state: "No relevant RAG context available; decision based on indicators alone" and cap confidence at 0.60.
- Treat technical analysis as probability-based, not certainty-based: require price, volume, momentum, trend, and support/resistance confluence before BUY.
- For individual equities, use fundamentals, valuation, earnings timing, and recent company-specific headlines as vetoes or confidence modifiers. Strong technicals should not override severe company risk, imminent earnings risk, fraud/probe/bankruptcy headlines, or obviously stretched valuation without extra confirmation.
- Prefer HOLD when the setup is not repeatable enough to write as a rule. Every BUY must have a clear invalidation level.

### Multi-Source Corroboration
Before any BUY decision, cross-check the same ticker across these independent evidence buckets:
1. Technical structure: trend, moving averages, momentum, support/resistance, volume, volatility.
2. Retrieved strategy context: RAG rules must support the setup or confidence is capped.
3. Fundamentals and valuation: revenue/earnings profile, P/E, market cap, 52-week position.
4. Event risk: earnings timing, major reports, or binary catalysts.
5. Ticker-specific headlines: positive or negative company catalysts.
6. Macro and sector context: broad-market risk-on/risk-off backdrop.

Set attractiveness_score, attractiveness_label, and price_outlook from the aggregate evidence. BUY requires at least 3 positive buckets, no hard veto, and a clear invalidation level. If evidence is missing or contradicts itself, prefer HOLD and explain the conflict.

### Professional Chart Analysis Playbook
- Analyze trend first. Higher highs/higher lows support an uptrend; lower highs/lower lows support a downtrend; sideways ranges require lower conviction.
- Use multi-timeframe thinking: long-term trend sets bias, intermediate trend sets setup quality, short-term trend sets entry timing.
- Treat support/resistance breaks as meaningful only when price closes through the level and volume confirms participation.
- Use RSI as momentum context, not a standalone signal. RSI can stay overbought/oversold in strong trends; divergences need confirmation.
- Use MACD for trend momentum. Crossovers are stronger when aligned with the zero line, histogram expansion, and price trend.
- Treat moving-average crosses as regime clues, not instant trade triggers. Confirm with price action and volume.
- Use volatility and range expansion to size risk. Wide daily ranges, ATR expansion, or unstable candles require smaller size or HOLD.
- Candlestick and chart patterns matter only at meaningful levels and must include invalidation and reward/risk.

### Risk Level Assignment
- **HIGH** : Volatility > 2.5 % daily range, OR contradicting signals, OR confidence < 0.50.
- **MEDIUM**: Moderate confluence, some conflicting signals.
- **LOW**  : Strong multi-indicator alignment, clear strategy support, low volatility.

### Action Thresholds
- Issue BUY/SELL only when confidence ≥ 0.50.
- Below 0.50, always output HOLD regardless of indicator bias.

Respond only with the structured fields requested. Do NOT add prose outside the structured response."""


# ---------------------------------------------------------------------------
# TradingDecision Pydantic model
# ---------------------------------------------------------------------------


class TradingDecision(BaseModel):
    """
    Fully-typed trading decision emitted by the AI Decision Engine.

    This is the *output contract* for the entire pipeline.  Downstream
    consumers (backtesting engine, live order router) should import and
    type-check against this model.

    Attributes:
        action:                  BUY / SELL / HOLD.
        confidence_score:        Model's self-assessed confidence [0.0, 1.0].
        reasoning:               Detailed, indicator-referencing explanation.
        risk_level:              LOW / MEDIUM / HIGH.
        key_indicators:          List of indicator strings that drove the call.
        suggested_stop_loss_pct: Stop distance as % from entry (None for HOLD).
        suggested_take_profit_pct: Target distance as % from entry (None for HOLD).
        attractiveness_score:    Aggregate attractiveness [0.0, 1.0].
        attractiveness_label:    ATTRACTIVE / NEUTRAL / UNATTRACTIVE.
        price_outlook:           BULLISH / NEUTRAL / BEARISH / UNKNOWN.
        rag_context_quality:     Quality of the retrieved strategy context used.
        decided_at:              UTC timestamp of the decision.
        ticker:                  Asset this decision was made for.
        is_fallback:             True if the engine fell back to HOLD after an error.
    """

    action: Literal["BUY", "SELL", "HOLD"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=20)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    key_indicators: list[str] = Field(default_factory=list)
    suggested_stop_loss_pct: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    suggested_take_profit_pct: Optional[float] = Field(default=None, ge=0.0, le=200.0)
    # New: how much of available equity to deploy on this decision (0-100).
    # The engine will clamp into the user's risk-profile envelope.
    suggested_position_size_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0)
    attractiveness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    attractiveness_label: Literal["ATTRACTIVE", "NEUTRAL", "UNATTRACTIVE"] = "NEUTRAL"
    price_outlook: Literal["BULLISH", "NEUTRAL", "BEARISH", "UNKNOWN"] = "UNKNOWN"
    rag_context_quality: Literal["STRONG", "MODERATE", "WEAK", "NONE"] = "NONE"
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    ticker: str = ""
    is_fallback: bool = False

    @field_validator("confidence_score", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        """Clamp to [0.0, 1.0] and round to 4 dp — guards against LLM drift."""
        return round(max(0.0, min(1.0, float(v))), 4)

    @field_validator("attractiveness_score", mode="before")
    @classmethod
    def clamp_attractiveness(cls, v: Any) -> float:
        """Clamp to [0.0, 1.0] and round to 4 dp."""
        if v is None:
            return 0.0
        return round(max(0.0, min(1.0, float(v))), 4)

    @field_validator("reasoning", mode="before")
    @classmethod
    def strip_reasoning(cls, v: Any) -> str:
        return str(v).strip()

    @property
    def is_actionable(self) -> bool:
        """True when the model recommends a non-HOLD action with meaningful confidence."""
        return self.action != "HOLD" and self.confidence_score >= 0.50

    @property
    def action_emoji(self) -> str:
        return self.action

    def __repr__(self) -> str:
        return (
            f"TradingDecision(ticker={self.ticker!r}, action={self.action!r}, "
            f"confidence={self.confidence_score:.2f}, risk={self.risk_level})"
        )


# ---------------------------------------------------------------------------
# Retry decorator — applied to the raw API call only
#
# Retries on transient Anthropic API errors:
#   RateLimitError      — 429, quota exhausted
#   APITimeoutError     — request timed out
#   InternalServerError — 500 from Anthropic servers
#   APIConnectionError  — network-level failure
#   OutputParserException — LangChain failed to coerce response into schema
#
# Non-retryable errors (AuthenticationError, PermissionDeniedError, etc.)
# propagate immediately to evaluate_market() which converts them to a HOLD.
# ---------------------------------------------------------------------------

_API_RETRY = retry(
    # 2 attempts total (1 retry). Fail fast — UI blocks for the whole duration.
    stop=stop_after_attempt(2),
    # 2 s → 6 s back-off; total worst-case blocking time ≈ 20 s.
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type(
        (
            _anthropic.RateLimitError,       # 429 — rate limit / quota
            _anthropic.APITimeoutError,      # request timed out
            _anthropic.InternalServerError,  # 500 from Anthropic
            _anthropic.APIConnectionError,   # network drop / DNS failure
            OutputParserException,           # LangChain structured-output parse failure
        )
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    # reraise=False → Tenacity wraps exhausted retries in RetryError so the
    # caller can distinguish "all retries failed" from a one-shot error.
    reraise=False,
)


# ---------------------------------------------------------------------------
# AI Trading Engine
# ---------------------------------------------------------------------------


class AITradingEngine:
    """
    Evaluates market state and retrieved strategies via Claude to produce a
    ``TradingDecision``.

    Parameters
    ----------
    model:
        Anthropic model ID.  Defaults to ``"claude-3-haiku-20240307"`` —
        optimised for speed and cost in a fast live-trading loop.
    temperature:
        Sampling temperature.  ``0.1`` for near-deterministic output with
        minimal variance.
    """

    _DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        # Resolution order: explicit arg → LLM_MODEL from .env → hard-coded default.
        # This ensures changing LLM_MODEL in .env actually takes effect.
        from config.settings import settings
        self._model: str = model or settings.llm_model or self._DEFAULT_MODEL

        # Re-load .env inside __init__ in case the module-level load at
        # import time ran before the .env file was writable (e.g. on the
        # very first Streamlit worker spin-up on Windows/OneDrive paths).
        load_dotenv(find_dotenv(usecwd=False), override=True)

        _raw_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not _raw_key:
            logger.error(
                "[AIEngine] ANTHROPIC_API_KEY not found in environment. "
                "Add it to your .env at the project root."
            )

        # api_key is passed explicitly from os.environ so LangChain never
        # has to guess — empty string falls through to Anthropic's own error.
        self.llm = ChatAnthropic(
            model=self._model,
            temperature=temperature,
            api_key=_raw_key,
            max_retries=0,
            timeout=20.0,
        )
        self.structured_llm = self.llm.with_structured_output(TradingDecision)

        # Shared news fetcher with 10-minute TTL cache so AUTO loops
        # don't re-hit Yahoo on every 60-second cycle.
        from news.fetcher import NewsFetcher
        self._news = NewsFetcher(ttl_seconds=600)

        from market_data.earnings import EarningsCalendar
        self._earnings = EarningsCalendar(ttl_seconds=4 * 3600)

        # Macro + geopolitics headline feed (Reuters/Google News/CoinDesk).
        # Separate from the per-ticker NewsFetcher above — this one provides
        # the "what is moving the whole market right now" context.
        from market_data.news import NewsFeed
        self._macro_news = NewsFeed(ttl_seconds=30 * 60)

        logger.debug(f"[AIEngine] ready — model={self._model}, temp={temperature}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate_market(
        self,
        snapshot: MarketSnapshot,
        retrieval: RetrievalResult,
        risk_profile: str = "Balanced",
        extra_context: Optional[str] = None,
    ) -> TradingDecision:
        """
        Core evaluation method.  Constructs the prompts, calls Claude via
        ``structured_llm.invoke()``, and returns a validated ``TradingDecision``.

        ``with_structured_output`` passes the Pydantic schema to Claude as a
        tool definition, forces the model to populate it via tool-use, and
        deserialises the response into a ``TradingDecision`` instance — all
        transparently.  Programmatic fields (``ticker``, ``decided_at``,
        ``is_fallback``) are attached via ``model_copy`` so the LLM never
        controls them.

        On any unrecoverable error (API failure after retries, validation
        error), returns a safe ``HOLD`` with ``confidence_score=0.0`` and
        ``is_fallback=True``.

        Args:
            snapshot:  Live :class:`~market_data.fetcher.MarketSnapshot`.
            retrieval: :class:`~rag.retriever.RetrievalResult` from Phase 3.

        Returns:
            A validated :class:`TradingDecision`.
        """
        ticker = snapshot.ticker
        user_prompt = self._build_user_prompt(
            snapshot, retrieval, risk_profile=risk_profile,
            extra_context=extra_context,
        )

        logger.info(
            f"[bold cyan]{ticker}[/bold cyan] — "
            f"Calling {self._model} | "
            f"RAG chunks: {len(retrieval.chunks)} | "
            f"best score: {retrieval.best_score or 'n/a'}"
        )

        try:
            decision = self._call_claude(user_prompt)
        except RetryError as exc:
            # All Tenacity attempts exhausted (reraise=False → this is reachable).
            cause = exc.last_attempt.exception()
            logger.error(
                f"{ticker}: API call failed after all retries. "
                f"Last error: {type(cause).__name__}: {cause}"
            )
            return self._make_fallback(ticker, f"API unavailable: {type(cause).__name__}")
        except Exception as exc:
            logger.error(f"{ticker}: Unexpected error during API call: {exc}")
            return self._make_fallback(ticker, f"Unexpected error: {exc}")

        # Attach programmatic fields — the LLM must never control these
        decision = decision.model_copy(
            update={
                "ticker": ticker,
                "decided_at": datetime.utcnow(),
                "is_fallback": False,
            }
        )

        logger.info(
            f"[bold green]{ticker} Decision:[/bold green] "
            f"{decision.action} | confidence={decision.confidence_score:.2f} | "
            f"risk={decision.risk_level} | RAG quality={decision.rag_context_quality}"
        )
        return decision

    # ------------------------------------------------------------------ #
    # Free-form market research
    # ------------------------------------------------------------------ #

    def research_market(
        self,
        watchlist: Optional[list[str]] = None,
        extra_focus: Optional[str] = None,
    ) -> dict:
        """
        On-demand macro research: pulls fresh headlines + sentiment bias,
        then asks Claude (free-form, no tool-use) for hot sectors, hot
        tickers, and where the market is heading *right now*.

        Returns a dict: {
            "report":       str   — markdown analysis,
            "headlines":    list  — raw Headline objects used,
            "bias":         dict  — aggregate sentiment bias,
            "model":        str,
            "ts":           datetime,
        }
        """
        tickers = watchlist or ["SPY", "QQQ", "BTC-USD"]
        all_heads: list = []
        biases: list[dict] = []
        for tk in tickers[:6]:
            try:
                bundle = self._macro_news.fetch(tk)
            except Exception:  # noqa: BLE001
                logger.exception(f"[research] news fetch failed for {tk}")
                continue
            all_heads.extend(bundle.all[:8])
            biases.append(bundle.sentiment_bias())

        # Dedupe headlines by title
        seen: set[str] = set()
        unique: list = []
        for h in all_heads:
            key = h.title.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(h)
        unique = unique[:40]

        # Aggregate bias across all watchlist items
        agg_score = float(sum(b.get("score", 0) or 0 for b in biases))
        agg_label = (
            "bullish" if agg_score >= 1.5
            else "bearish" if agg_score <= -1.5
            else "mixed" if any(b.get("label") in ("bullish", "bearish") for b in biases)
            else "neutral"
        )

        head_lines = []
        for h in unique:
            tag = {"bull": "[BULL]", "bear": "[BEAR]"}.get(h.sentiment, "[NEUT]")
            head_lines.append(f"- {tag} ({h.source}) {h.title}")
        headlines_block = "\n".join(head_lines) if head_lines else "(no fresh headlines)"

        system = (
            "You are a senior macro strategist. Given fresh market headlines, "
            "sector rotation clues, and aggregate sentiment, produce a concise "
            "daily market briefing. Be specific and opinionated. You may name "
            "tickers and sectors. Make judgment calls — do NOT hedge endlessly.\n\n"
            "Format your response as markdown with these sections:\n"
            "## 🌐 Market Regime\n"
            "(1-2 sentences — risk-on / risk-off / rotating / indecisive, and why)\n\n"
            "## 🔥 Hot Sectors Right Now\n"
            "(bullet list, 3-5 sectors with one-line reason each)\n\n"
            "## 🎯 Tickers To Watch Today\n"
            "(bullet list, 3-6 tickers, each with a bias tag "
            "[LONG]/[SHORT]/[WATCH] and a one-line setup or catalyst)\n\n"
            "## ⚠️ Key Risks\n"
            "(bullet list of 2-3 things that could invalidate the above)\n\n"
            "## 🗞️ What Moved Markets\n"
            "(2-4 bullets summarizing the dominant news themes)\n"
        )

        user = (
            f"Aggregate sentiment across watchlist: **{agg_label.upper()}** "
            f"(score {agg_score:+.2f}).\n\n"
            f"Watchlist scanned: {', '.join(tickers[:6])}\n\n"
            f"### Fresh Headlines\n{headlines_block}\n"
        )
        if extra_focus:
            user += f"\n### Extra user focus\n{extra_focus}\n"

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        try:
            resp = self.llm.invoke(messages)
            report = resp.content if hasattr(resp, "content") else str(resp)
            if isinstance(report, list):  # some LangChain returns list of blocks
                report = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in report
                )
        except Exception as e:  # noqa: BLE001
            logger.exception("[research] Claude call failed")
            report = f"⚠️ Research call failed: `{type(e).__name__}: {e}`"

        return {
            "report": report,
            "headlines": unique,
            "bias": {"label": agg_label, "score": agg_score, "per_ticker": biases},
            "model": self._model,
            "ts": datetime.utcnow(),
        }

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_user_prompt(
        self,
        snapshot: MarketSnapshot,
        retrieval: RetrievalResult,
        risk_profile: str = "Balanced",
        extra_context: Optional[str] = None,
    ) -> str:
        """
        Assemble the user-turn message that feeds Claude the full market
        context for decision-making.

        Structure:
            SECTION 1 — Price & OHLCV
            SECTION 2 — Moving Averages with % deviation from price
            SECTION 3 — Momentum indicators (RSI, MACD)
            SECTION 4 — Classified market regime (from Phase 3)
            SECTION 5 — Retrieved strategy documents (from Phase 3)
            SECTION 6 — Fundamental & valuation data (optional, from Phase 7)
            SECTION 6/7 — Decision request (renumbered if fundamentals present)

        Args:
            snapshot:  MarketSnapshot from Phase 2.
            retrieval: RetrievalResult from Phase 3.

        Returns:
            Formatted multi-section prompt string.
        """
        latest = snapshot.latest
        ticker = snapshot.ticker

        close: float = float(latest["Close"])
        open_: float = float(latest.get("Open", close))
        high: float = float(latest.get("High", close))
        low: float = float(latest.get("Low", close))
        volume: int = int(latest.get("Volume", 0))
        daily_range_pct: float = (high - low) / close * 100 if close > 0 else 0.0

        rsi_col = f"RSI_{snapshot.rsi_period}"
        rsi: float = float(latest[rsi_col])
        macd: float = float(latest["MACD"])
        macd_signal: float = float(latest["MACD_Signal"])
        macd_hist: float = float(latest["MACD_Histogram"])

        # --- Section 1: Price Action ----------------------------------- #
        section1 = (
            f"### SECTION 1: PRICE ACTION — {ticker}\n"
            f"Analysis Date : {snapshot.end_date}\n"
            f"Current Price : ${close:.2f}\n"
            f"Daily Open    : ${open_:.2f}   "
            f"({'↑' if close >= open_ else '↓'} {abs(close - open_) / open_ * 100:.2f}% intraday)\n"
            f"Daily High    : ${high:.2f}\n"
            f"Daily Low     : ${low:.2f}\n"
            f"Daily Range   : {daily_range_pct:.2f}% of close\n"
            f"Volume        : {volume:,}\n"
        )

        # --- Section 2: Moving Averages -------------------------------- #
        sma_lines: list[str] = []
        for p in snapshot.sma_periods:
            col = f"SMA_{p}"
            if col in latest.index:
                sma_val = float(latest[col])
                pct_diff = (close - sma_val) / sma_val * 100
                relation = "ABOVE" if close > sma_val else "BELOW"
                sma_lines.append(
                    f"  SMA_{p:<4}: ${sma_val:.2f}  "
                    f"(price is {relation} by {abs(pct_diff):.2f}%)"
                )

        # Cross detection for the two longest SMAs
        sma_cross_line = ""
        if 50 in snapshot.sma_periods and 200 in snapshot.sma_periods:
            sma50 = float(latest.get("SMA_50", 0))
            sma200 = float(latest.get("SMA_200", 0))
            if sma50 > 0 and sma200 > 0:
                cross = "GOLDEN CROSS (SMA_50 > SMA_200 — long-term BULLISH)" if sma50 > sma200 \
                    else "DEATH CROSS (SMA_50 < SMA_200 — long-term BEARISH)"
                sma_cross_line = f"  Structure : {cross}\n"

        section2 = (
            "### SECTION 2: MOVING AVERAGES\n"
            + "\n".join(sma_lines) + "\n"
            + sma_cross_line
        )

        # --- Section 3: Momentum --------------------------------------- #
        # RSI label
        if rsi >= 70:
            rsi_label = "OVERBOUGHT ⚠️ (above 70 — potential exhaustion)"
        elif rsi <= 30:
            rsi_label = "OVERSOLD ⚠️ (below 30 — potential reversal zone)"
        elif rsi >= 60:
            rsi_label = "BULLISH (60–70 — strong upward momentum)"
        elif rsi <= 40:
            rsi_label = "BEARISH (30–40 — weak / selling pressure)"
        else:
            rsi_label = "NEUTRAL (40–60 — no clear momentum edge)"

        # MACD histogram direction
        if len(snapshot.data) >= 2:
            prev_hist = float(snapshot.data.iloc[-2]["MACD_Histogram"])
            hist_trend = "EXPANDING" if abs(macd_hist) > abs(prev_hist) else "FADING"
        else:
            hist_trend = "UNKNOWN"

        section3 = (
            f"### SECTION 3: MOMENTUM INDICATORS\n"
            f"  RSI ({snapshot.rsi_period})       : {rsi:.2f}  →  {rsi_label}\n"
            f"  MACD Line     : {macd:.4f}\n"
            f"  MACD Signal   : {macd_signal:.4f}\n"
            f"  MACD Cross    : {'BULLISH (line above signal)' if macd > macd_signal else 'BEARISH (line below signal)'}\n"
            f"  MACD Histogram: {macd_hist:.4f}  "
            f"({'POSITIVE' if macd_hist >= 0 else 'NEGATIVE'}, {hist_trend})\n"
        )

        # --- Section 3b: Volatility & Structure ----------------------- #
        def _get(col_prefix: str):
            c = next((c for c in snapshot.data.columns
                      if c.startswith(col_prefix)), None)
            if c is None:
                return None
            try:
                return float(latest[c])
            except Exception:
                return None

        bb_up   = _get("BB_Upper_")
        bb_lo   = _get("BB_Lower_")
        bb_mid  = _get("BB_Mid_")
        bb_w    = _get("BB_Width_")
        atr     = _get("ATR_")
        vwap    = _get("VWAP_")
        supp    = _get("Support_")
        resi    = _get("Resistance_")

        vol_lines: list[str] = []
        if bb_up is not None and bb_lo is not None and bb_mid is not None:
            if close >= bb_up:
                bb_label = "AT/ABOVE UPPER BAND ⚠️ (stretched — potential mean reversion)"
            elif close <= bb_lo:
                bb_label = "AT/BELOW LOWER BAND ⚠️ (stretched — potential bounce)"
            elif close > bb_mid:
                bb_label = "UPPER HALF (bullish drift)"
            else:
                bb_label = "LOWER HALF (bearish drift)"
            width_note = ""
            if bb_w is not None:
                if bb_w < 1.5:
                    width_note = " — SQUEEZE (low vol, breakout likely)"
                elif bb_w > 6.0:
                    width_note = " — EXPANDED (high volatility regime)"
            vol_lines.append(
                f"  Bollinger(20,2)  : Upper ${bb_up:.2f} · Mid ${bb_mid:.2f} · Lower ${bb_lo:.2f}"
                f"  →  {bb_label}{width_note}"
            )
        if atr is not None:
            atr_pct = (atr / close) * 100 if close else 0.0
            vol_lines.append(
                f"  ATR(14)          : ${atr:.2f}  ({atr_pct:.2f}% of price) "
                f"— typical bar range; use for stop sizing"
            )
        if vwap is not None:
            vwap_rel = "ABOVE VWAP (buyers in control)" if close > vwap \
                       else "BELOW VWAP (sellers in control)"
            vol_lines.append(f"  VWAP(20)         : ${vwap:.2f}  →  {vwap_rel}")
        if supp is not None and resi is not None:
            dist_to_sup = (close - supp) / close * 100 if close else 0.0
            dist_to_res = (resi - close) / close * 100 if close else 0.0
            vol_lines.append(
                f"  S/R (last 20 bar): Support ${supp:.2f} ({dist_to_sup:+.2f}%)  "
                f"Resistance ${resi:.2f} ({dist_to_res:+.2f}%)"
            )
        section3b = ""
        if vol_lines:
            section3b = (
                "### SECTION 3B: VOLATILITY & STRUCTURE\n"
                + "\n".join(vol_lines) + "\n"
            )

        # --- Section 4: Regime ---------------------------------------- #
        regime_lines = "\n".join(
            f"  {k.replace('_', ' ').title():<20}: {v}"
            for k, v in retrieval.market_regime.items()
        )
        section4 = f"### SECTION 4: CLASSIFIED MARKET REGIME\n{regime_lines}\n"

        # --- Section 5: RAG Documents ---------------------------------- #
        if retrieval.found:
            rag_header = (
                f"### SECTION 5: RETRIEVED STRATEGY CONTEXT\n"
                f"({len(retrieval.chunks)} chunk(s) retrieved | "
                f"Best similarity score: {retrieval.best_score:.3f})\n\n"
            )
            rag_blocks: list[str] = []
            for i, chunk in enumerate(retrieval.chunks, start=1):
                rag_blocks.append(
                    f"--- Strategy Excerpt {i} "
                    f"[Source: '{chunk.source_title}' | "
                    f"Similarity: {chunk.similarity_score:.3f}] ---\n"
                    f"{chunk.document.strip()}\n"
                )
            section5 = rag_header + "\n".join(rag_blocks)
        else:
            section5 = (
                "### SECTION 5: RETRIEVED STRATEGY CONTEXT\n"
                "⚠️  WARNING: The knowledge base returned NO relevant strategy context "
                "for the current market conditions.  Your decision must rely solely on "
                "technical indicators.  Cap confidence_score at 0.60 and note the "
                "absence of RAG context in your reasoning.\n"
            )

        # --- Section 6: Fundamental & Valuation Data (optional) -------- #
        section6_fund = ""
        if snapshot.fundamentals is not None:
            fund = snapshot.fundamentals
            f_dict = fund.summary_dict()
            fund_lines = "\n".join(
                f"  {label:<22}: {value}"
                for label, value in f_dict.items()
                if value != "N/A"
            )
            range_note = ""
            if fund.week_52_high and fund.week_52_low:
                range_pct = (
                    (close - fund.week_52_low)
                    / (fund.week_52_high - fund.week_52_low)
                    * 100
                )
                range_note = (
                    f"\n  52W Position       : {range_pct:.1f}% up from 52W low "
                    f"(low ${fund.week_52_low:.2f} → high ${fund.week_52_high:.2f})"
                )
            section6_fund = (
                "### SECTION 6: FUNDAMENTAL & VALUATION DATA\n"
                + fund_lines
                + range_note + "\n\n"
                + "Valuation guidance: A trailing P/E > 30 signals elevated growth "
                  "expectations — treat as a risk factor when technical signals are mixed. "
                  "A P/E < 15 or price near the 52-week low may represent value. "
                  "Use fundamentals as a tie-breaker when technical signals conflict."
            )

        # --- Section 6 or 7: Decision Request -------------------------- #
        req_num = 7 if section6_fund else 6
        section_request = (
            f"### SECTION {req_num}: DECISION REQUEST\n"
            f"Evaluate the above market conditions for {ticker} and provide a "
            f"structured trading decision.\n"
            f"Reference specific numeric indicator values, strategy excerpts"
            + (", and valuation data" if section6_fund else "")
            + " in your reasoning. Also classify attractiveness_score, "
              "attractiveness_label, and price_outlook from the multi-source "
              "corroboration check."
        )

        # User risk-profile guidance — shapes aggressiveness of BUY/SELL calls
        # AND the suggested_position_size_pct (clamped post-hoc by the engine).
        from config.user_profile import RISK_ENVELOPES, RISK_PROFILE_GUIDANCE
        risk_text = RISK_PROFILE_GUIDANCE.get(risk_profile, RISK_PROFILE_GUIDANCE["Balanced"])
        env = RISK_ENVELOPES.get(risk_profile, RISK_ENVELOPES["Balanced"])
        sizing_rule = (
            f"\n\nPOSITION SIZING INSTRUCTION — for this profile the engine "
            f"will clamp your suggested_position_size_pct into the range "
            f"[{env.size_min_pct:.0f}%, {env.size_max_pct:.0f}%]. Scale with "
            f"conviction: low-conviction BUY → {env.size_min_pct:.0f}%, "
            f"textbook high-conviction BUY → {env.size_max_pct:.0f}%. "
            f"For HOLD or SELL this field may be 0."
        )
        section_risk = f"### USER RISK PROFILE\n{risk_text}{sizing_rule}"

        # Recent news headlines (fundamental catalyst context).
        section_news = None
        try:
            news_items = self._news.fetch(ticker, limit=5)
            from news.fetcher import NewsFetcher as _NF
            section_news = (
                "### RECENT NEWS HEADLINES\n"
                "Use these as catalyst/sentiment context. Recent negative news "
                "weakens BUY conviction; positive earnings or macro tailwinds "
                "strengthen it. If headlines contradict the technical signal, "
                "reduce confidence accordingly.\n\n"
                + _NF.format_for_prompt(news_items)
            )
        except Exception as exc:
            logger.debug(f"News section skipped for {ticker}: {exc}")

        # Upcoming earnings — binary-event risk guard.
        section_earnings = None
        try:
            earn = self._earnings.fetch(ticker)
            if earn.next_date is not None:
                if earn.is_imminent:
                    header = (
                        "⚠ EARNINGS IMMINENT — a scheduled report within 3 "
                        "days can move the price 5-15% in seconds. Strongly "
                        "downgrade BUY conviction and cut position size to "
                        "the profile minimum. Prefer HOLD unless the setup "
                        "is extreme."
                    )
                elif earn.is_near:
                    header = (
                        "EARNINGS NEAR — reporting within 7 days. Treat as "
                        "elevated risk; shade position size toward the "
                        "profile minimum and require stronger technical "
                        "confirmation for a BUY."
                    )
                else:
                    header = (
                        "Earnings scheduled further out — not a near-term "
                        "risk factor; no sizing adjustment required."
                    )
                section_earnings = (
                    "### EARNINGS CALENDAR\n"
                    f"Next earnings report: {earn.summary()}\n\n"
                    f"{header}"
                )
        except Exception as exc:
            logger.debug(f"Earnings section skipped for {ticker}: {exc}")

        # ── ML SIGNALS — regime / anomaly / patterns / forecast / win-prob ──
        section_ml = None
        try:
            section_ml = self._build_ml_section(snapshot)
        except Exception as exc:
            logger.debug(f"ML section skipped for {ticker}: {exc}")

        # Macro & geopolitical headlines — broad market context.
        section_macro = None
        try:
            macro_block = self._macro_news.macro_summary_for_prompt(
                ticker, max_items=5
            )
            if macro_block:
                section_macro = (
                    "### MACRO & GEOPOLITICAL CONTEXT\n"
                    "Broad-market headlines with keyword-based sentiment tags "
                    "([BULL] / [BEAR] / [NEUT]) plus an aggregate bias line. "
                    "Cross-check the aggregate against your technical read:\n"
                    "  • Aggregate BEARISH + technical BUY → downgrade "
                    "confidence, prefer HOLD unless setup is textbook.\n"
                    "  • Aggregate BULLISH + technical BUY → full conviction.\n"
                    "  • Aggregate MIXED → trust the technicals, size normally.\n"
                    "Use individual tags as sanity checks — one strong [BEAR] "
                    "headline on the ticker (bankruptcy, fraud, probe, miss) "
                    "alone is enough to block a BUY.\n\n"
                    + macro_block
                )
        except Exception as exc:
            logger.debug(f"Macro news section skipped for {ticker}: {exc}")

        corroboration_lines = [
            "Technical indicators: available in Sections 1-4.",
            (
                "RAG strategy context: "
                f"{'available' if retrieval.found else 'missing'}"
                + (f" | best score {retrieval.best_score:.3f}" if retrieval.found else "")
                + "."
            ),
            f"Fundamentals/valuation: {'available' if section6_fund else 'missing'}.",
            f"Ticker-specific news: {'available' if section_news else 'missing'}.",
            f"Earnings/event risk: {'available' if section_earnings else 'missing'}.",
            f"Macro/sector context: {'available' if section_macro else 'missing'}.",
        ]
        section_corroboration = (
            "### MULTI-SOURCE CORROBORATION CHECK\n"
            "Use this checklist to cross-reference the same ticker before deciding. "
            "A BUY must be supported by at least three positive buckets and must not "
            "be blocked by earnings risk, severe negative headlines, broken trend, "
            "weak volume, poor RAG fit, or extreme valuation risk.\n\n"
            + "\n".join(f"- {line}" for line in corroboration_lines)
            + "\n\nReturn attractiveness_score, attractiveness_label, and price_outlook "
              "from this aggregate evidence. If the buckets disagree, use HOLD unless "
              "the upside evidence is clearly stronger and risk is defined."
        )

        sections = [section1, section2, section3]
        if section3b:
            sections.append(section3b)
        sections += [section4, section5]
        if section6_fund:
            sections.append(section6_fund)
        if section_news:
            sections.append(section_news)
        if section_earnings:
            sections.append(section_earnings)
        if section_macro:
            sections.append(section_macro)
        if section_ml:
            sections.append(section_ml)
        if extra_context:
            sections.append(
                "### SECTION: ADDITIONAL SIGNAL CONTEXT\n" + extra_context
            )
        sections.append(section_corroboration)
        sections.append(section_risk)
        sections.append(section_request)
        return "\n\n".join(sections)

    # ------------------------------------------------------------------ #
    # ML signals section
    # ------------------------------------------------------------------ #

    def _build_ml_section(self, snapshot: MarketSnapshot) -> Optional[str]:
        """Compute ML signals (regime / anomaly / patterns / forecast /
        personal win-probability) for the snapshot's enriched OHLCV frame
        and return a markdown section. Returns None if every sub-module
        fails — the prompt simply omits the section in that case.

        Each sub-call is independently guarded; any single failure degrades
        gracefully without blocking the others.
        """
        df = snapshot.data
        if df is None or len(df) < 60:
            return None

        ticker = snapshot.ticker
        lines: list[str] = []
        veto_flags: list[str] = []   # collected veto-style notes for the engine

        # --- Regime ---------------------------------------------------------
        try:
            from ml.regime import RegimeClassifier
            rc = RegimeClassifier(ticker)
            if not rc.is_fitted:
                rc.fit(df)
            r = rc.predict(df)
            lines.append(
                f"  Market Regime    : {r.label}  (confidence {r.confidence:.0%})"
            )
            if r.label == "VOLATILE":
                veto_flags.append(
                    "Regime is VOLATILE — shrink size or prefer HOLD unless "
                    "setup is textbook."
                )
            elif r.label == "TRENDING_DOWN":
                veto_flags.append(
                    "Regime is TRENDING_DOWN — counter-trend BUYs need very "
                    "strong confluence; default bias is short/cash."
                )
        except Exception as exc:
            logger.debug(f"[ml-section] regime failed: {exc}")

        # --- Anomaly --------------------------------------------------------
        try:
            from ml.anomaly import AnomalyDetector
            ad = AnomalyDetector(ticker)
            if not ad.is_fitted:
                ad.fit(df)
            a = ad.predict(df)
            tag = "⚠ ANOMALY" if a.is_anomaly else "normal"
            lines.append(
                f"  Bar Anomaly      : {tag}  (severity {a.severity}, "
                f"score {a.score:+.2f})"
            )
            if a.severity in ("STRONG", "EXTREME"):
                veto_flags.append(
                    f"Bar is anomalous ({a.severity}) — likely earnings leak, "
                    f"halt aftermath, or fat-finger. VETO new BUYs unless the "
                    f"strategy explicitly trades news shocks."
                )
        except Exception as exc:
            logger.debug(f"[ml-section] anomaly failed: {exc}")

        # --- Chart patterns -------------------------------------------------
        try:
            from ml.pattern_detector import detect_patterns
            patterns = detect_patterns(df, lookback=80)
            if patterns:
                top = patterns[:3]
                pat_str = "; ".join(
                    f"{p.name} ({p.direction}, {p.confidence:.0%})" for p in top
                )
                lines.append(f"  Chart Patterns   : {pat_str}")
            else:
                lines.append("  Chart Patterns   : (none ≥ 35% confidence)")
        except Exception as exc:
            logger.debug(f"[ml-section] patterns failed: {exc}")

        # --- Forecaster -----------------------------------------------------
        try:
            from ml.forecaster import forecast as _forecast
            f = _forecast(df, horizon=5)
            lines.append(
                f"  5-Bar Forecast   : bias {f.bias.upper()}  "
                f"point {f.point_return*100:+.2f}%  "
                f"95% CI [{f.lo_95*100:+.2f}%, {f.hi_95*100:+.2f}%]"
            )
            if f.bias == "down":
                veto_flags.append(
                    "Forecast bias is DOWN over the next 5 bars — do not "
                    "open BUYs against the model unless RSI/MACD are textbook "
                    "oversold-reversal."
                )
        except Exception as exc:
            logger.debug(f"[ml-section] forecaster failed: {exc}")

        # --- Personal win-probability (trade-journal RF) -------------------
        try:
            from ml.trade_journal_ml import TradeJournalML
            jm = TradeJournalML()
            if jm.is_fitted and jm.n_train >= 10:
                wp = jm.predict(df)
                lines.append(
                    f"  Personal WinProb : {wp.prob_win:.0%}  "
                    f"(band {wp.confidence_band}, n_train={wp.n_train})"
                )
                if wp.confidence_band in ("MEDIUM", "HIGH") and wp.prob_win < 0.40:
                    veto_flags.append(
                        f"Your personal trade journal estimates only "
                        f"{wp.prob_win:.0%} chance this setup wins, based on "
                        f"{wp.n_train} historical trades. Strongly consider "
                        f"HOLD."
                    )
                elif wp.confidence_band in ("MEDIUM", "HIGH") and wp.prob_win >= 0.65:
                    veto_flags.append(
                        f"Your personal trade journal estimates "
                        f"{wp.prob_win:.0%} chance this setup wins "
                        f"(n={wp.n_train}). Boost confidence."
                    )
            else:
                lines.append(
                    f"  Personal WinProb : (insufficient history — "
                    f"n_train={jm.n_train}; need ≥10)"
                )
        except Exception as exc:
            logger.debug(f"[ml-section] journal failed: {exc}")

        if not lines:
            return None

        veto_block = ""
        if veto_flags:
            veto_block = (
                "\nML guidance:\n"
                + "\n".join(f"  • {v}" for v in veto_flags)
            )

        return (
            "### MACHINE-LEARNING SIGNALS\n"
            "Quantitative signals from the bot's own ML stack — regime "
            "clustering, isolation-forest anomaly detection, geometric chart-"
            "pattern scoring, EWMA-drift forecast, and a personal win-probability "
            "classifier trained on this account's closed trades. Treat these "
            "as additional evidence, not as override signals. When they conflict "
            "with technicals, weigh by their confidence:\n\n"
            + "\n".join(lines)
            + veto_block
        )

    # ------------------------------------------------------------------ #
    # Claude API call (retried)
    # ------------------------------------------------------------------ #

    @_API_RETRY
    def _call_claude(self, user_prompt: str) -> TradingDecision:
        """
        Invoke ``self.structured_llm`` and return a validated
        ``TradingDecision`` directly.

        LangChain's ``with_structured_output(TradingDecision)`` passes the
        Pydantic schema to Claude as a tool definition, forces the model to
        populate it via tool-use, and deserialises the JSON response into a
        ``TradingDecision`` instance — all transparently.

        Args:
            user_prompt: Formatted market-context string from
                         :meth:`_build_user_prompt`.

        Returns:
            A ``TradingDecision`` instance populated by Claude.

        Raises:
            anthropic.RateLimitError, APITimeoutError, InternalServerError,
            APIConnectionError  — retried by ``@_API_RETRY``.
            OutputParserException — retried by ``@_API_RETRY``.
            Any other exception  — propagates to ``evaluate_market``
                                   which converts it to a HOLD fallback.
        """
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        return self.structured_llm.invoke(messages)

    # ------------------------------------------------------------------ #
    # Fallback
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_fallback(ticker: str, reason: str) -> TradingDecision:
        """
        Return a safe HOLD decision used when the API call or parse fails.

        Args:
            ticker: Asset ticker for labelling.
            reason: Short description of why the fallback was triggered.

        Returns:
            A ``TradingDecision`` with ``action="HOLD"``,
            ``confidence_score=0.0``, and ``is_fallback=True``.
        """
        logger.warning(f"Returning HOLD fallback for {ticker!r}: {reason}")
        return TradingDecision(
            action="HOLD",
            confidence_score=0.0,
            reasoning=(
                f"Decision engine encountered an error and could not produce "
                f"a reliable analysis.  Defaulting to HOLD to avoid unintended "
                f"trades.  Reason: {reason}"
            ),
            risk_level="HIGH",
            key_indicators=[],
            attractiveness_score=0.0,
            attractiveness_label="UNATTRACTIVE",
            price_outlook="UNKNOWN",
            rag_context_quality="NONE",
            ticker=ticker,
            is_fallback=True,
        )


# ---------------------------------------------------------------------------
# __main__ — smoke-test with synthetic inputs (no live API calls required
# unless ANTHROPIC_API_KEY is set in the environment)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import os
    from dataclasses import dataclass, field as dc_field
    from datetime import date, timedelta

    import numpy as np
    import pandas as pd
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    from market_data.fetcher import MACDParams, MarketSnapshot
    from rag.retriever import RetrievalResult, RetrievedChunk

    console = Console()

    # ------------------------------------------------------------------ #
    # Synthetic MarketSnapshot
    # Scenario: RSI = 25 (deeply oversold), bearish MACD but histogram
    # fading (potential exhaustion of sell-off), price below all SMAs
    # (Death Cross structure).  Classic "watch-for-reversal" setup.
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold yellow]Phase 4 — AI Decision Engine Test[/bold yellow]"))

    NUM_ROWS = 6
    today = date.today()
    idx = pd.date_range(end=today, periods=NUM_ROWS, freq="B")

    closes = np.array([162.0, 157.0, 152.0, 147.0, 143.0, 141.0])
    synthetic_data = pd.DataFrame(
        {
            "Open":            closes - 1.2,
            "High":            closes + 1.8,
            "Low":             closes - 2.5,
            "Close":           closes,
            "Volume":          [30_000_000, 38_000_000, 51_000_000,
                                66_000_000, 74_000_000, 78_000_000],
            "RSI_14":          [48.0, 40.0, 33.0, 28.0, 25.0, 25.0],
            "SMA_20":          [170.0, 169.5, 169.0, 168.0, 167.0, 166.0],
            "SMA_50":          [175.0, 174.8, 174.5, 174.2, 173.9, 173.5],
            "SMA_200":         [180.0, 179.8, 179.6, 179.4, 179.2, 179.0],
            # Bearish MACD that is now *fading* — histogram shrinking = sellers exhausting
            "MACD":            [-0.60, -0.95, -1.30, -1.65, -1.80, -1.75],
            "MACD_Signal":     [-0.20, -0.45, -0.72, -1.00, -1.30, -1.55],
            "MACD_Histogram":  [-0.40, -0.50, -0.58, -0.65, -0.50, -0.20],
        },
        index=idx,
    )

    mock_snapshot = MarketSnapshot(
        ticker="NVDA",
        data=synthetic_data,
        sma_periods=(20, 50, 200),
        rsi_period=14,
        macd_params=MACDParams(fast=12, slow=26, signal=9),
    )

    # ------------------------------------------------------------------ #
    # Synthetic RetrievalResult — strategy text that explicitly supports a
    # counter-trend long on oversold RSI + fading bearish MACD histogram
    # ------------------------------------------------------------------ #
    strategy_text_1 = (
        "Oversold RSI Reversal Strategy: When RSI drops below 30, the asset is in "
        "extreme oversold territory. A high-probability long entry setup occurs when "
        "RSI crosses back above 30 while the MACD histogram shows bullish divergence "
        "(histogram rising even while price makes a new low). Initiate a long position "
        "with a stop-loss set 2.5% below the recent swing low. Target the 20-day SMA "
        "as the first take-profit level, with a secondary target at the 50-day SMA."
    )
    strategy_text_2 = (
        "MACD Histogram Fading Signal: When the MACD histogram transitions from "
        "expanding bearish (growing negative values) to fading bearish (shrinking "
        "negative values), this is an early indication that bearish momentum is "
        "exhausting. Combined with an oversold RSI reading, this creates a mean-"
        "reversion opportunity. However, risk management is critical: the broader "
        "trend (Death Cross, price below 200 SMA) indicates structural weakness, "
        "so position sizing should be reduced to 50% of normal. Use a tight stop-loss "
        "of 1.5% and take partial profits quickly at the first resistance level."
    )

    mock_retrieval = RetrievalResult(
        query="NVDA oversold RSI below 30 bearish MACD fading histogram reversal",
        ticker="NVDA",
        chunks=[
            RetrievedChunk(
                document=strategy_text_1,
                metadata={"title": "RSI Oversold Reversal Playbook", "source": "youtube"},
                distance=0.22,
                chunk_id="abc123",
            ),
            RetrievedChunk(
                document=strategy_text_2,
                metadata={"title": "MACD Histogram Exhaustion Strategies", "source": "pdf"},
                distance=0.31,
                chunk_id="def456",
            ),
        ],
        market_regime={
            "rsi_zone": "Oversold",
            "macd_cross": "Bearish Crossover",
            "macd_momentum": "Fading Bearish",
            "price_trend": "Strong Downtrend",
            "sma_structure": "Death Cross",
            "volatility": "High Volatility",
        },
        collection_size=248,
    )

    # ------------------------------------------------------------------ #
    # Display synthetic inputs
    # ------------------------------------------------------------------ #
    console.print(
        Panel(
            f"[bold]Ticker:[/bold]  NVDA\n"
            f"[bold]Close:[/bold]   $141.00\n"
            f"[bold]RSI-14:[/bold]  25.0  [red](Oversold — extreme capitulation)[/red]\n"
            f"[bold]MACD Hist:[/bold] -0.20  [yellow](Bearish but FADING — exhaustion signal)[/yellow]\n"
            f"[bold]Structure:[/bold] Death Cross | Price below SMA_20/50/200\n"
            f"[bold]RAG chunks:[/bold] 2  (best similarity: {mock_retrieval.best_score:.3f})",
            title="[cyan]Synthetic Market Input[/cyan]",
            expand=False,
        )
    )

    # ------------------------------------------------------------------ #
    # Show user prompt (without calling the API, for inspection)
    # ------------------------------------------------------------------ #
    engine = AITradingEngine()
    user_prompt = engine._build_user_prompt(mock_snapshot, mock_retrieval)

    console.print(Rule("[bold cyan]Generated User Prompt[/bold cyan]"))
    console.print(Panel(user_prompt, title="User Prompt → Claude", expand=True))

    # ------------------------------------------------------------------ #
    # Live API call (only if ANTHROPIC_API_KEY is available)
    # ------------------------------------------------------------------ #
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        console.print(
            Panel(
                "[yellow]ANTHROPIC_API_KEY not detected in environment.[/yellow]\n\n"
                "The prompt has been constructed and validated above.\n"
                "To see a live Claude decision, set your API key:\n\n"
                "  [bold cyan]export ANTHROPIC_API_KEY=sk-ant-...[/bold cyan]\n\n"
                "Then re-run:\n"
                "  [bold cyan]python -m decision_engine.ai_engine[/bold cyan]",
                title="[yellow]Skipping Live API Call[/yellow]",
                expand=False,
            )
        )
    else:
        console.print(Rule("[bold green]Live Claude Decision[/bold green]"))
        decision = engine.evaluate_market(mock_snapshot, mock_retrieval)
        _render_decision(console, decision)


def _render_decision(console: Any, decision: "TradingDecision") -> None:
    """Render a TradingDecision as a rich console panel."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    action_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}[decision.action]
    conf_color = (
        "green" if decision.confidence_score >= 0.70 else
        "yellow" if decision.confidence_score >= 0.45 else
        "red"
    )
    risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}[decision.risk_level]

    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    summary.add_column("Field", style="dim")
    summary.add_column("Value")
    summary.add_row("Action",      f"[bold {action_color}]{decision.action_emoji}[/bold {action_color}]")
    summary.add_row("Confidence",  f"[{conf_color}]{decision.confidence_score:.2f}[/{conf_color}]")
    summary.add_row("Risk Level",  f"[{risk_color}]{decision.risk_level}[/{risk_color}]")
    summary.add_row("Attractiveness", f"{decision.attractiveness_label} ({decision.attractiveness_score:.0%})")
    summary.add_row("Price Outlook", decision.price_outlook)
    summary.add_row("RAG Quality", decision.rag_context_quality)
    if decision.suggested_stop_loss_pct is not None:
        summary.add_row("Stop Loss",   f"{decision.suggested_stop_loss_pct:.2f}% from entry")
    if decision.suggested_take_profit_pct is not None:
        summary.add_row("Take Profit", f"{decision.suggested_take_profit_pct:.2f}% from entry")
    summary.add_row("Fallback",    str(decision.is_fallback))
    summary.add_row("Decided At",  decision.decided_at.strftime("%Y-%m-%d %H:%M:%S UTC"))

    if decision.key_indicators:
        summary.add_row("Key Indicators", "\n".join(f"• {i}" for i in decision.key_indicators))

    reasoning_panel = Panel(
        decision.reasoning,
        title="[bold]Reasoning Chain[/bold]",
        border_style="dim",
        expand=True,
    )

    console.print(
        Panel(
            summary,
            title=f"[bold {action_color}]TRADING DECISION — {decision.ticker}[/bold {action_color}]",
            expand=False,
        )
    )
    console.print(reasoning_panel)
