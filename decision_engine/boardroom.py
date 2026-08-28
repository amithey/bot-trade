"""
Analyst Boardroom — a hedge-fund style investment committee.
=============================================================

Models a real fund's morning meeting. Eight specialists each study
ONLY their own briefing packet, at the same moment (parallel calls):

    📉 Maya Cohen    — Chart & Technical analyst: price action,
                       moving averages, momentum, volatility structure.
    🏦 David Levi    — Fundamental analyst: valuation, quality,
                       52-week context (humble on crypto).
    📰 Noa Barak     — News & Sentiment analyst: ticker-specific
                       headlines + aggregate sentiment bias.
    🤖 Leo Stern     — Quant strategist: the 38-indicator ensemble
                       tally, breadth and regime.
    🌍 Amir Mizrahi  — Global Macro strategist: world/geopolitics
                       headlines, rates-and-risk appetite read.
    🛡 Dana Weiss    — Chief Risk Officer: volatility regime, position
                       exposure, daily P&L, drawdown — her red flag
                       carries veto-level weight with the chairman.
    📊 Tomer Gal     — Volume & Flow analyst: volume regime, VWAP,
                       OBV direction, support/resistance proximity.
    😈 Yael Sharon   — Contrarian (devil's advocate): paid to argue
                       against the obvious trade and find the crowd's
                       blind spot.

Each casts BUY / SELL / HOLD with a conviction level and a short
opinion in their own voice.

Finally the chairman & CIO — 🪑 Rafael Adler — reads the full
transcript plus the portfolio state and makes the binding decision the
engine executes. If the chairman call fails, a deterministic
conviction-weighted majority is applied so the meeting always ends
with a ruling.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)

_ANALYST_TIMEOUT_SEC = 75


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalystVote(BaseModel):
    """Structured output every analyst must return."""
    vote: Literal["BUY", "SELL", "HOLD"]
    conviction: float = Field(..., ge=0.0, le=1.0,
                              description="How strongly you hold this view.")
    opinion: str = Field(..., min_length=20, max_length=600,
                         description="2-4 sentences in your own voice, "
                                     "grounded ONLY in your briefing packet.")


@dataclass
class AnalystOpinion:
    """One analyst's contribution to the meeting."""
    name: str
    emoji: str
    role: str
    vote: str            # BUY / SELL / HOLD / ABSTAIN
    conviction: float
    opinion: str
    ok: bool = True      # False when the analyst call failed (abstained)


@dataclass
class BoardroomRuling:
    """The full minutes of one boardroom meeting."""
    opinions: list[AnalystOpinion]
    decision: object                 # TradingDecision (chairman's ruling)
    chair_name: str
    chair_is_fallback: bool          # True → majority fallback, not Claude
    convened_at: datetime = field(default_factory=datetime.now)

    def tally(self) -> dict[str, int]:
        t = {"BUY": 0, "SELL": 0, "HOLD": 0, "ABSTAIN": 0}
        for o in self.opinions:
            t[o.vote] = t.get(o.vote, 0) + 1
        return t

    def summary_dict(self) -> dict:
        return {
            "members": [{
                "name": o.name, "emoji": o.emoji, "role": o.role,
                "vote": o.vote, "conviction": round(o.conviction, 2),
                "opinion": o.opinion, "ok": o.ok,
            } for o in self.opinions],
            "chair": {
                "name": self.chair_name,
                "action": self.decision.action,
                "confidence": round(self.decision.confidence_score, 2),
                "reasoning": self.decision.reasoning,
                "is_fallback": self.chair_is_fallback,
            },
            "tally": self.tally(),
            "convened_at": self.convened_at.strftime("%H:%M:%S"),
        }


# ---------------------------------------------------------------------------
# Briefing-packet builders (each sees ONLY its own slice of the world)
# ---------------------------------------------------------------------------

def _technical_packet(snapshot) -> str:
    last = snapshot.latest
    df = snapshot.data
    close = float(last["Close"])
    lines = [f"Asset: {snapshot.ticker}",
             f"Last price: ${close:,.2f}",
             f"Bars available: {len(df)} "
             f"({df.index[0]} → {df.index[-1]})"]
    for col in df.columns:
        if col in ("Open", "High", "Low", "Close", "Volume"):
            continue
        try:
            v = float(last[col])
            lines.append(f"{col}: {v:,.4f}")
        except Exception:
            continue
    # 10-bar momentum context
    if len(df) >= 11:
        chg10 = (close / float(df["Close"].iloc[-11]) - 1) * 100
        lines.append(f"Change over last 10 bars: {chg10:+.2f}%")
    return "\n".join(lines)


def _fundamental_packet(snapshot) -> str:
    f = snapshot.fundamentals
    ticker = snapshot.ticker
    close = float(snapshot.latest["Close"])
    if f is None:
        return (f"Asset: {ticker} (price ${close:,.2f})\n"
                "No fundamental data is available — this is likely a "
                "crypto asset or an index. Assess from an asset-class "
                "perspective: scarcity/utility narrative, typical "
                "valuation regime, and whether current price looks "
                "stretched relative to its 52-week behaviour. Be "
                "explicitly humble about the missing data.")
    lines = [f"Asset: {ticker} ({f.long_name or 'n/a'})",
             f"Price: ${close:,.2f}",
             f"Sector / Industry: {f.sector or 'n/a'} / {f.industry or 'n/a'}",
             f"P/E trailing: {f.pe_trailing or 'n/a'}   "
             f"P/E forward: {f.pe_forward or 'n/a'}",
             f"Market cap: {f.market_cap or 'n/a'}",
             f"52w high: {f.week_52_high or 'n/a'}   "
             f"52w low: {f.week_52_low or 'n/a'}",
             f"Dividend yield: {f.dividend_yield or 'n/a'}",
             f"Beta: {f.beta or 'n/a'}"]
    if f.week_52_high:
        lines.append(f"Discount from 52w high: "
                     f"{(close / f.week_52_high - 1) * 100:+.1f}%")
    return "\n".join(lines)


def _news_packet(ticker: str, bundle) -> str:
    if bundle is None:
        return (f"Asset: {ticker}\nNews feed unavailable right now. "
                "State that you cannot assess sentiment and vote HOLD "
                "with low conviction.")
    heads = list(bundle.ticker_specific) or list(bundle.all)
    if not heads:
        return (f"Asset: {ticker}\nNo fresh headlines found. State that "
                "the news cycle is quiet and weigh that accordingly.")
    bias = bundle.sentiment_bias()
    lines = [f"Asset: {ticker}",
             f"Aggregate sentiment: {bias.get('label', 'n/a')} "
             f"(score {bias.get('score', 0)})",
             "Fresh headlines:"]
    for h in heads[:14]:
        tag = {"bull": "[BULL]", "bear": "[BEAR]"}.get(h.sentiment, "[NEUT]")
        lines.append(f"  {tag} ({h.source}) {h.short(120)}")
    return "\n".join(lines)


def _macro_packet(ticker: str, bundle) -> str:
    if bundle is None or not bundle.macro:
        return (f"Asset under discussion: {ticker}\nNo global macro "
                "headlines available right now. Give your structural "
                "macro read (rates, risk appetite, season) from general "
                "principles, flag the missing feed, and keep conviction "
                "modest.")
    lines = [f"Asset under discussion: {ticker}",
             "Global macro / geopolitics headlines (not ticker-specific):"]
    for h in bundle.macro[:14]:
        tag = {"bull": "[BULL]", "bear": "[BEAR]"}.get(h.sentiment, "[NEUT]")
        lines.append(f"  {tag} ({h.source}) {h.short(120)}")
    lines.append("Judge ONLY the macro environment: is this a "
                 "risk-on or risk-off tape for the asset class?")
    return "\n".join(lines)


def _risk_packet(snapshot, in_position: bool,
                 entry_price: Optional[float],
                 daily_pnl_pct: Optional[float]) -> str:
    df = snapshot.data
    close = float(snapshot.latest["Close"])
    lines = [f"Asset: {snapshot.ticker}",
             f"Last price: ${close:,.2f}"]

    def _last(prefix):
        c = next((c for c in df.columns if c.startswith(prefix)), None)
        try:
            return float(snapshot.latest[c]) if c else None
        except Exception:
            return None

    atr = _last("ATR_")
    if atr:
        lines.append(f"ATR: ${atr:,.2f} ({atr / close * 100:.2f}% of price)")
    bbw = _last("BB_Width_")
    if bbw is not None:
        lines.append(f"Bollinger band width: {bbw:.2f}% "
                     f"({'expanding/volatile' if bbw > 6 else 'compressed'})")
    if len(df) >= 30:
        peak30 = float(df["Close"].iloc[-30:].max())
        lines.append(f"Distance from 30-bar high: "
                     f"{(close / peak30 - 1) * 100:+.2f}%")
        rets = df["Close"].iloc[-30:].pct_change().dropna()
        if len(rets):
            lines.append(f"30-bar realized volatility (per bar): "
                         f"{float(rets.std()) * 100:.2f}%")
    if in_position and entry_price:
        lines.append(f"OPEN POSITION: long from ${entry_price:,.2f} "
                     f"({(close / entry_price - 1) * 100:+.2f}% unrealized)")
    else:
        lines.append("Position: in CASH (no exposure)")
    if daily_pnl_pct is not None:
        lines.append(f"Portfolio P&L today: {daily_pnl_pct:+.2f}%")
    lines.append("Your mandate: capital preservation first. If risk is "
                 "elevated, say so loudly — your red flag carries "
                 "veto-level weight with the chairman.")
    return "\n".join(lines)


def _flow_packet(snapshot) -> str:
    df = snapshot.data
    close = float(snapshot.latest["Close"])
    lines = [f"Asset: {snapshot.ticker}",
             f"Last price: ${close:,.2f}"]
    if "Volume" in df.columns and len(df) >= 21:
        v_now = float(df["Volume"].iloc[-1])
        v_avg = float(df["Volume"].iloc[-21:-1].mean())
        if v_avg > 0:
            lines.append(f"Current bar volume vs 20-bar average: "
                         f"{v_now / v_avg:.2f}x")
        import numpy as np
        direction = np.sign(df["Close"].diff()).fillna(0)
        obv = (direction * df["Volume"]).cumsum()
        obv_slope = float(obv.iloc[-1] - obv.iloc[-10]) if len(obv) >= 10 else 0
        lines.append(f"OBV direction over last 10 bars: "
                     f"{'accumulation (rising)' if obv_slope > 0 else 'distribution (falling)'}")

    def _last(prefix):
        c = next((c for c in df.columns if c.startswith(prefix)), None)
        try:
            return float(snapshot.latest[c]) if c else None
        except Exception:
            return None

    vwap = _last("VWAP_")
    if vwap:
        lines.append(f"VWAP-20: ${vwap:,.2f} — price is "
                     f"{(close / vwap - 1) * 100:+.2f}% vs VWAP")
    sup, res = _last("Support_"), _last("Resistance_")
    if sup and res:
        lines.append(f"20-bar support ${sup:,.2f} ({(close / sup - 1) * 100:+.1f}% above) · "
                     f"resistance ${res:,.2f} ({(close / res - 1) * 100:+.1f}%)")
    lines.append("Judge ONLY participation: is real money confirming "
                 "this price, and which level breaks first?")
    return "\n".join(lines)


def _contrarian_packet(snapshot, bundle, verdict) -> str:
    df = snapshot.data
    close = float(snapshot.latest["Close"])
    chg10 = ((close / float(df["Close"].iloc[-11]) - 1) * 100
             if len(df) >= 11 else 0.0)
    sentiment = "n/a"
    if bundle is not None:
        try:
            sentiment = str(bundle.sentiment_bias().get("label", "n/a"))
        except Exception:
            pass
    quant = (f"{verdict.bulls} bull / {verdict.bears} bear "
             f"(score {verdict.score:+.2f})" if verdict is not None else "n/a")
    return (f"Asset: {snapshot.ticker} at ${close:,.2f}\n"
            f"What the room sees: 10-bar move {chg10:+.2f}%, news "
            f"sentiment '{sentiment}', systematic ensemble {quant}.\n"
            f"Your job: argue the OTHER side of whatever looks obvious. "
            f"If the room is bullish, build the strongest bear case; if "
            f"bearish, the strongest bull case; if the consensus is "
            f"mushy, attack the complacency. Then vote your honest view "
            f"AFTER steel-manning the opposite — sometimes the contrarian "
            f"conclusion is to agree, reluctantly.")


def _quant_packet(ticker: str, verdict) -> str:
    if verdict is None:
        return (f"Asset: {ticker}\nThe 38-indicator committee could not "
                "run (insufficient bars). Vote HOLD with low conviction.")
    cats = " · ".join(f"{c}: {s:+.2f}"
                      for c, s in sorted(verdict.category_scores.items()))
    return (f"Asset: {ticker}\n"
            f"Systematic ensemble of {verdict.total} technical indicators:\n"
            f"  BULL votes: {verdict.bulls}\n"
            f"  BEAR votes: {verdict.bears}\n"
            f"  NEUTRAL:    {verdict.neutrals}\n"
            f"  Net score:  {verdict.score:+.2f}  (range -1..+1)\n"
            f"  Category breakdown: {cats}\n"
            f"  Mechanical verdict: {verdict.action}\n"
            f"Leading agreeing signals: "
            f"{', '.join(verdict.top_contributors(6)) or '—'}")


# ---------------------------------------------------------------------------
# The boardroom
# ---------------------------------------------------------------------------

_PANEL = [
    # (name, emoji, role, packet_kind, persona_system)
    ("Maya Cohen", "📉", "Chart & Technical Analyst", "technical",
     "You are Maya Cohen, a veteran chart analyst. You trade purely off "
     "price action, moving averages, momentum and volatility structure. "
     "You distrust narratives — the tape is the truth."),
    ("David Levi", "🏦", "Fundamental Analyst", "fundamental",
     "You are David Levi, a buy-side fundamental analyst. You care about "
     "valuation, quality and margin of safety. You think in quarters and "
     "years, not minutes, and you say so when the horizon mismatches."),
    ("Noa Barak", "📰", "News & Sentiment Analyst", "news",
     "You are Noa Barak, a market news and sentiment specialist. You "
     "weigh fresh headlines, crowd positioning and narrative momentum. "
     "You know news fades fast and flag when a story is already priced in."),
    ("Leo Stern", "🤖", "Quant Strategist", "quant",
     "You are Leo Stern, a systematic quant. You trust ensembles of "
     "signals over any single human read. Interpret the committee tally "
     "statistically: breadth, regime, and signal agreement."),
    ("Amir Mizrahi", "🌍", "Global Macro Strategist", "macro",
     "You are Amir Mizrahi, a global macro strategist. You read rates, "
     "geopolitics and cross-asset risk appetite. You care whether the "
     "TAPE is risk-on or risk-off, not about any single chart."),
    ("Dana Weiss", "🛡", "Chief Risk Officer", "risk",
     "You are Dana Weiss, the fund's Chief Risk Officer. Your only job "
     "is capital preservation: volatility regime, exposure, drawdown. "
     "You are professionally paranoid — say 'no' when sizing or "
     "volatility is wrong, and say it plainly."),
    ("Tomer Gal", "📊", "Volume & Flow Analyst", "flow",
     "You are Tomer Gal, a market-microstructure and volume-flow "
     "analyst. Price without participation is noise: you track whether "
     "real money is accumulating or distributing, and which levels "
     "matter."),
    ("Yael Sharon", "😈", "Contrarian — Devil's Advocate", "contrarian",
     "You are Yael Sharon, the fund's designated contrarian. Your job "
     "is to attack the consensus before the market does. You earn your "
     "seat by finding the blind spot — but you vote honestly once "
     "you've steel-manned the other side."),
]

_CHAIR_NAME = "Rafael Adler"


class AnalystBoardroom:
    """
    Convenes the panel for one ticker and returns a BoardroomRuling.

    Parameters
    ----------
    llm:
        Chat model for the eight analysts (the AITradingEngine's ``.llm``).
    chair_llm:
        Optional separate model for the chairman. The panel is ~9 calls per
        cycle, so running every seat on one model means any upgrade costs
        nine times as much. Each analyst returns a single structured vote
        from a narrow packet and stays cheap; the chairman weighs the whole
        transcript and casts the binding ruling, so that is the one seat
        worth spending on. Defaults to *llm* — a single-model setup behaves
        exactly as before.
    """

    def __init__(self, llm, chair_llm=None) -> None:
        self._llm = llm
        self._chair_llm = chair_llm if chair_llm is not None else llm
        self._vote_llm = llm.with_structured_output(AnalystVote)
        from market_data.news import NewsFeed
        self._news = NewsFeed(ttl_seconds=15 * 60)

    # -- individual analyst ------------------------------------------------

    def _ask_analyst(self, name: str, emoji: str, role: str,
                     persona: str, packet: str,
                     position_note: str) -> AnalystOpinion:
        prompt = (
            f"{persona}\n\n"
            f"You are in a live investment-committee meeting. Below is "
            f"YOUR briefing packet — it is the ONLY information you may "
            f"use. Other specialists cover the other angles; do not "
            f"speculate outside your lane.\n\n"
            f"--- BRIEFING PACKET ---\n{packet}\n"
            f"--- END PACKET ---\n\n"
            f"Portfolio note: {position_note}\n\n"
            f"Cast your vote (BUY / SELL / HOLD), your conviction (0-1), "
            f"and a 2-4 sentence opinion in your own professional voice."
        )
        try:
            v: AnalystVote = self._vote_llm.invoke(prompt)
            return AnalystOpinion(name=name, emoji=emoji, role=role,
                                  vote=v.vote,
                                  conviction=float(v.conviction),
                                  opinion=v.opinion.strip(), ok=True)
        except Exception as exc:
            logger.warning(f"Boardroom analyst {name} failed: {exc}")
            return AnalystOpinion(
                name=name, emoji=emoji, role=role, vote="ABSTAIN",
                conviction=0.0,
                opinion=f"(Could not present — {type(exc).__name__})",
                ok=False)

    # -- chairman ------------------------------------------------------------

    @staticmethod
    def _majority_ruling(ticker: str, opinions: list[AnalystOpinion]):
        """Deterministic conviction-weighted majority — chairman fallback."""
        from decision_engine.ai_engine import TradingDecision
        w = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        for o in opinions:
            if o.vote in w:
                w[o.vote] += max(0.05, o.conviction)
        action = max(w, key=w.get)
        total = sum(w.values()) or 1.0
        # Ties or weak pluralities resolve to HOLD
        if action != "HOLD" and w[action] <= total / 2:
            action = "HOLD"
        conf = min(1.0, 0.5 + w[action] / total / 2)
        lines = [f"{o.emoji} {o.name} ({o.vote}, {o.conviction:.0%}): "
                 f"{o.opinion}" for o in opinions]
        return TradingDecision(
            action=action,
            confidence_score=round(conf, 3),
            reasoning=("Chairman unavailable — conviction-weighted majority "
                       f"of the panel applied.\n" + "\n".join(lines)),
            risk_level="MEDIUM",
            key_indicators=[f"{o.name}: {o.vote}" for o in opinions][:5],
            rag_context_quality="NONE",
            is_fallback=False,
        )

    def _ask_chairman(self, ticker: str, opinions: list[AnalystOpinion],
                      position_note: str, risk_profile: str):
        from decision_engine.ai_engine import TradingDecision
        chair_llm = self._chair_llm.with_structured_output(TradingDecision)
        transcript = "\n\n".join(
            f"{o.emoji} {o.name} — {o.role}\n"
            f"Vote: {o.vote} (conviction {o.conviction:.0%})\n"
            f"Opinion: {o.opinion}"
            for o in opinions
        )
        prompt = (
            f"You are {_CHAIR_NAME}, chairman of the investment committee "
            f"and the final decision-maker. Your panel has just presented "
            f"on {ticker}.\n\n"
            f"--- MEETING TRANSCRIPT ---\n{transcript}\n"
            f"--- END TRANSCRIPT ---\n\n"
            f"Portfolio note: {position_note}\n"
            f"Risk profile: {risk_profile}\n\n"
            f"Rules of the house:\n"
            f"- Long-only book: SELL means exit to cash, never short.\n"
            f"- A BUY needs genuine support from at least three "
            f"specialists with decent conviction.\n"
            f"- Your Chief Risk Officer's red flag carries veto-level "
            f"weight: never BUY over a credible risk objection from her.\n"
            f"- The contrarian's job is to attack consensus — engage her "
            f"argument on the merits; do not dismiss it, do not "
            f"automatically follow it.\n"
            f"- One loud analyst is not a majority. Weigh conviction AND "
            f"track-record realism of each argument.\n"
            f"- When the panel is split or lukewarm, HOLD is the "
            f"professional answer.\n\n"
            f"Deliver your binding ruling. In `reasoning`, briefly weigh "
            f"each analyst by name, state who convinced you and why, then "
            f"give the ruling."
        )
        decision: "TradingDecision" = chair_llm.invoke(prompt)
        return decision.model_copy(update={
            "ticker": ticker,
            "decided_at": datetime.utcnow(),
            "is_fallback": False,
        })

    # -- public --------------------------------------------------------------

    def convene(self, snapshot, verdict=None, in_position: bool = False,
                entry_price: Optional[float] = None,
                risk_profile: str = "Balanced",
                daily_pnl_pct: Optional[float] = None) -> BoardroomRuling:
        """
        Hold one full committee meeting for *snapshot.ticker*.

        ``verdict`` is an optional pre-computed 38-indicator
        CommitteeVerdict for the quant seat. ``daily_pnl_pct`` feeds the
        Chief Risk Officer's packet.
        """
        ticker = snapshot.ticker
        close = float(snapshot.latest["Close"])
        if in_position and entry_price:
            pnl = (close / entry_price - 1) * 100
            position_note = (f"We currently HOLD a long position from "
                             f"${entry_price:,.2f} ({pnl:+.2f}% unrealized).")
        elif in_position:
            position_note = "We currently HOLD a long position."
        else:
            position_note = "We are in CASH — no open position."

        # One news fetch feeds both the sentiment and macro desks.
        try:
            bundle = self._news.fetch(ticker)
        except Exception as exc:
            logger.warning(f"Boardroom news fetch failed: {exc}")
            bundle = None

        packets = {
            "technical":   _technical_packet(snapshot),
            "fundamental": _fundamental_packet(snapshot),
            "news":        _news_packet(ticker, bundle),
            "quant":       _quant_packet(ticker, verdict),
            "macro":       _macro_packet(ticker, bundle),
            "risk":        _risk_packet(snapshot, in_position,
                                        entry_price, daily_pnl_pct),
            "flow":        _flow_packet(snapshot),
            "contrarian":  _contrarian_packet(snapshot, bundle, verdict),
        }

        # All analysts study their packets at the same moment.
        with ThreadPoolExecutor(max_workers=len(_PANEL),
                                thread_name_prefix="boardroom") as pool:
            futures = [
                pool.submit(self._ask_analyst, name, emoji, role,
                            persona, packets[kind], position_note)
                for name, emoji, role, kind, persona in _PANEL
            ]
            opinions = [f.result(timeout=_ANALYST_TIMEOUT_SEC)
                        for f in futures]

        # The chairman rules.
        try:
            decision = self._ask_chairman(ticker, opinions, position_note,
                                          risk_profile)
            chair_fallback = False
        except Exception as exc:
            logger.warning(f"Boardroom chairman failed: {exc} — "
                           f"using majority fallback")
            decision = self._majority_ruling(ticker, opinions)
            chair_fallback = True

        return BoardroomRuling(
            opinions=opinions,
            decision=decision,
            chair_name=_CHAIR_NAME,
            chair_is_fallback=chair_fallback,
        )
