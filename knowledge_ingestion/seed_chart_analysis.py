"""
Seed Chart Analysis & Market Understanding — Direct ChromaDB Ingestion
=======================================================================
Extends the strategy knowledge base with deep chart-reading and
market-understanding material. Content is synthesised from authoritative
2026 sources (Investopedia, StockCharts ChartSchool, CFA curriculum,
Wyckoff Analytics, TradingView educational articles) and rewritten for
LLM consumption — long, self-contained, keyword-rich prose that embeds
cleanly.

Run once to populate:
    python -m knowledge_ingestion.seed_chart_analysis

Idempotent — re-running skips chunks already in the collection.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.hf_quiet import configure_quiet_hf, quiet_model_load

configure_quiet_hf()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from rag.ownership import SHARED_OWNER
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge library — (source_id, title, full_text)
# Each book is deliberately long and prose-heavy so the sentence-transformer
# embedder has enough context to produce discriminative vectors.
# ─────────────────────────────────────────────────────────────────────────────

CHART_ANALYSIS_LIBRARY: list[tuple[str, str, str]] = [

# ── 1. Reversal Chart Patterns ────────────────────────────────────────────────
("chart_patterns_reversal_v1",
 "Reversal Chart Patterns — Head & Shoulders, Double Tops/Bottoms, Triple Tops",
 """
Reversal Chart Patterns — Complete Reading Guide

Reversal chart patterns mark points where the dominant trend runs out of fuel
and a new trend begins. They are among the highest-conviction setups because
they encode a visible shift in order flow: buyers who previously bought every
dip stop showing up, or sellers who previously sold every rally capitulate.
The reliability of these patterns comes from three factors that must coexist:
a clear preceding trend, a clean structural shift, and volume that confirms
the new direction.

HEAD AND SHOULDERS (Bearish Reversal at the top of an uptrend)

The head-and-shoulders pattern is the single most studied reversal in all of
technical analysis. It consists of three consecutive peaks: the left shoulder,
a higher middle peak (the head), and a right shoulder that is lower than the
head and roughly aligned with the left shoulder. The lows between the peaks
define a line — the "neckline" — that may be horizontal or gently sloped.

Psychology: Buyers push to a new high (the left shoulder) with strong volume.
They push again (the head) to a higher high, but volume already declines —
a sign of waning participation. The rally to the right shoulder fails to
exceed the head; volume is noticeably weaker. Sellers step in. When price
decisively closes below the neckline, the pattern completes.

Entry, stop, target:
- Entry on the neckline break, or on a retest of the broken neckline as
  new resistance.
- Stop above the right shoulder.
- Target = height from the head down to the neckline, projected below
  the neckline.
- Historical reliability: roughly 83-89% when volume confirms the breakdown
  and the prior uptrend is at least 10-20% old.

INVERSE HEAD AND SHOULDERS (Bullish Reversal at the bottom of a downtrend)

Mirror image: three troughs with the deepest in the middle, then a break
above the neckline. Volume should expand on the rally off the head and
explode on the neckline breakout. Very common at major swing lows in
equities and crypto. Targets are measured the same way, projected upward.

DOUBLE TOP (Bearish Reversal)

Two attempts to rally to the same price ceiling, separated by a moderate
pullback. The second peak often shows a failure pattern — a shooting star,
bearish engulfing, or evening star — at the exact level of the first peak.
Volume on the second peak is almost always lower than on the first. When
price breaks the intervening pullback low ("the valley"), the pattern is
confirmed and a new downtrend typically begins. Reliability rises when
the two peaks are separated by 4-8 weeks on a daily chart.

DOUBLE BOTTOM (Bullish Reversal)

Mirror image. Two tests of the same floor, separated by a relief rally.
Second test often holds slightly higher than the first — a subtle "failure
to make a new low" that is extremely bullish. Break above the intervening
peak completes the pattern. This is historically the second most reliable
reversal pattern, with win rates around 85-88%.

TRIPLE TOP AND TRIPLE BOTTOM

Three roughly equal peaks or troughs. Rarer than double patterns but even
higher conviction when they appear, because each failed attempt represents
absorbed buying/selling pressure. Each subsequent test should show declining
volume. Once the structure breaks, the move is typically larger than from
a double top because more trapped liquidity is released.

ROUNDING BOTTOM (SAUCER)

A slow, gradual curve — often 6-12 months on a daily chart — that marks the
transition from a long downtrend through accumulation into a new uptrend.
Volume typically forms a matching saucer: high during the initial decline,
minimal at the bottom, rising as the right side develops. This pattern is
favoured by long-horizon investors because it signals durable institutional
accumulation.

V-BOTTOM AND V-TOP

Sharp, one-bar reversals. Unlike the patient patterns above, V-reversals
happen in days, usually triggered by a macro surprise or capitulation event.
They are harder to trade because entries are reactive and stops must be
wide. Confirmation tools: RSI bullish/bearish divergence, volume spikes,
hammer or shooting-star candles, and failure at historical support/resistance.

PATTERN-QUALITY CHECKLIST (apply to every reversal you identify):
1. Is there a clear prior trend at least 5-8 swing points long?
2. Is volume behaving as theory predicts (declining into top, expanding on
   breakdown)?
3. Does the neckline/break level coincide with a prior swing low/high
   or a moving average (SMA 50 or 200)?
4. Are higher-timeframe indicators (daily RSI/MACD) aligning?
5. Is position sizing calibrated so the stop is a small fraction of account?
If fewer than 3 of these are true, treat the pattern as speculative rather
than actionable.
"""),

# ── 2. Continuation Chart Patterns ────────────────────────────────────────────
("chart_patterns_continuation_v1",
 "Continuation Chart Patterns — Flags, Pennants, Triangles, Rectangles, Wedges",
 """
Continuation Chart Patterns — Trading the Pause in a Trend

Continuation patterns form during pauses in an established trend. Unlike
reversal patterns, they do not signal a change of direction — they signal
that the trend is compressing energy before its next leg. The practical
edge: continuation patterns are easier to trade than reversals because
you are aligning with momentum rather than fighting it.

BULL FLAG (Bullish Continuation)

After a sharp advance (the flagpole), price consolidates in a small,
downward-sloping rectangle or channel. Volume collapses during the flag —
profit-taking dries up, new buyers wait, sellers have already sold. The
breakout to new highs, accompanied by a volume surge, resumes the uptrend.
- Entry: breakout above the flag's upper trendline.
- Stop: below the flag's lower trendline or the most recent swing low.
- Target: flagpole height projected from the breakout point.
- Best context: strong markets, recent breakouts from bases, earnings
  runners. Historical reliability in trending regimes: 80-85%.

BEAR FLAG (Bearish Continuation)

Mirror of the bull flag. A sharp decline (flagpole down) followed by an
upward-sloping consolidation channel on shrinking volume, then a break to
new lows on expanded volume. Short on the breakdown; stop above the flag
high; target = flagpole depth projected down.

PENNANT

Visually similar to a flag but symmetrical — converging trendlines forming
a small triangle after a sharp move. Very short duration (1-3 weeks on a
daily chart). Volume dries up in the pennant, explodes on the breakout.
The target is measured the same as a flag (prior pole projected).

SYMMETRICAL TRIANGLE

Higher lows meet lower highs, compressing into an apex. Unlike pennants,
symmetrical triangles can last for weeks or months. They are bilateral —
the breakout direction is unknown until it happens — but in a prevailing
trend the probability favours continuation. Wait for a close beyond the
apex with volume expansion; avoid trading pre-breakout because false moves
back into the triangle are common.

ASCENDING TRIANGLE (Bullish Bias)

Flat horizontal ceiling, rising support line. Buyers repeatedly bid price
higher off each pullback but fail to clear the ceiling — until eventually
they do. A break of the horizontal ceiling with volume is one of the
highest-reliability continuation patterns in trending markets (roughly
75-83% when volume confirms). Measure target as the height at the widest
point, projected up from the breakout.

DESCENDING TRIANGLE (Bearish Bias)

Flat horizontal floor, declining resistance line. Sellers absorb each
bounce, buyers can't make new highs, and eventually the floor breaks.
Target measured the same way, projected down.

RECTANGLE (Trading Range)

Horizontal support and resistance over an extended period. Neither side
wins — price oscillates between the two lines. In an uptrend, rectangles
typically resolve upward; in a downtrend, downward. Advanced traders
fade the range edges inside rectangles (sell the top, buy the bottom)
with tight stops, then switch to breakout-trading once either boundary
gives way decisively.

FALLING WEDGE (Bullish Continuation or Reversal)

Both trendlines slope down, but the lower one declines faster — the
wedge narrows. Interpreted as selling exhaustion: each new low is
shallower than the last. Breakouts are to the upside in roughly 68-72%
of cases. Stronger when occurring at a higher-timeframe support level.

RISING WEDGE (Bearish Continuation or Reversal)

Mirror: both trendlines slope up but upper rises faster. Buying is
losing steam. Usually resolves to the downside, especially when RSI is
divergent. In strong uptrends this is one of the earliest warnings that
a pullback is imminent.

CUP AND HANDLE (Bullish Continuation)

A rounded base ("cup") followed by a smaller downward pullback ("handle")
that consolidates on low volume, then a breakout above the cup's rim.
Popularised by William O'Neil — very common in stocks coming out of
multi-month consolidations. Minimum cup depth: 12-33%; handle depth should
not exceed 1/3 of the cup. Target: cup depth projected from the breakout.

DIAMOND AND BROADENING FORMATIONS

Widening then narrowing price swings that often mark emotional extremes
(diamonds) or growing instability (broadening tops). Less reliable than
cleaner patterns; require strict confirmation before acting.

COMMON MISTAKES WHEN TRADING CONTINUATION PATTERNS:
- Forcing a pattern onto a choppy market. If it's not obvious at a glance,
  it's not a pattern.
- Ignoring volume. A breakout without volume is likely a trap.
- Trading too early — waiting for a close beyond the pattern edge filters
  out most false breakouts.
- Placing stops inside the pattern (too tight). Use the opposite boundary
  of the pattern as stop reference.
"""),

# ── 3. Candlestick Patterns & Psychology ─────────────────────────────────────
("candlestick_psychology_v1",
 "Candlestick Patterns — Market Psychology Encoded in Price",
 """
Candlestick Patterns — What Each Candle Reveals About Buyer/Seller Balance

Every candlestick encodes four data points (open, high, low, close) into a
visual story of the battle between buyers and sellers during that period.
A long green body says buyers dominated from open to close with little
pushback. A long lower wick says sellers pushed price down but buyers
reclaimed the territory by close — a footprint of absorption. Pattern
reading is not about memorising shapes; it's about decoding whose hand
is on the tiller at decision points.

SINGLE-CANDLE REVERSAL SIGNALS

Hammer (bullish at downtrend lows): Small real body near the top of the
range, long lower wick at least 2× the body, little or no upper wick.
Story: sellers drove price to a new low, buyers stepped in hard and closed
the candle near the high. Requires confirmation from the next candle
(close above hammer high) and preferably occurs at prior support or a
high-timeframe moving average.

Shooting Star (bearish at uptrend highs): Mirror of hammer. Small real
body near the bottom of the range, long upper wick. Buyers failed to hold
their gains; sellers closed the candle near the lows. Bearish divergence
on RSI multiplies the signal.

Doji: Open and close are essentially equal. Pure indecision. A doji after
an extended trend at overbought/oversold RSI is a yellow flag — neither
side controls price any more. Types to know:
- Gravestone doji (long upper wick, close at low) — bearish at a top.
- Dragonfly doji (long lower wick, close at high) — bullish at a bottom.
- Long-legged doji (both wicks) — maximum indecision, often precedes a
  volatility expansion.

Spinning Top: Small body, both wicks moderate. Similar to doji but less
extreme. Its significance depends entirely on context (trend maturity,
prior candle structure, volume).

TWO-CANDLE REVERSAL SIGNALS

Bullish Engulfing: Small red candle followed by a larger green candle
whose body completely engulfs the prior body. Buyers overwhelmed the
previous bar's sellers. Best at oversold levels or prior support.

Bearish Engulfing: Mirror. A large red body engulfs the prior green
body. At overbought levels, this is one of the most reliable short
triggers — think of it as capitulation of the buyers.

Piercing Line (bullish): Red candle, then a green candle that gaps down
but closes above the midpoint of the red candle's body. Signals buyers
stepping in aggressively after overnight weakness.

Dark Cloud Cover (bearish): Green candle, then red candle gaps up but
closes below the midpoint of the green body. Story: optimism at the
open, sellers take control, distribution begins.

THREE-CANDLE REVERSAL SIGNALS

Morning Star (bullish): Three candles — a long red, a small-bodied
middle candle (pause/indecision), then a long green that reclaims most
of the first red body. A high-quality bottoming signal when it appears
at oversold RSI or a key support level.

Evening Star (bearish): Mirror. Long green, small-bodied middle, long
red that erases most of the green. One of the cleanest short triggers
at the top of extended rallies.

Three White Soldiers (bullish continuation/reversal): Three consecutive
long green candles with each closing near its high. Strong accumulation
signal; at the end of a downtrend it marks a high-probability reversal.

Three Black Crows (bearish): Three long red candles with lower closes.
After extended rallies, marks institutional distribution.

CONTINUATION CANDLESTICK PATTERNS

Rising Three Methods: In an uptrend, a long green candle followed by 3-4
smaller red candles that stay within the first candle's range, then
another long green. Represents a healthy pause; the trend resumes.

Falling Three Methods: Mirror in a downtrend.

Mat Hold: Similar to rising three methods but with a small gap up after
the long green candle. Strong bullish continuation.

PSYCHOLOGICAL PRINCIPLES:
1. Candles near the extremes of a move carry more weight than candles
   in the middle of a trend.
2. Long wicks signal rejection of price — the market is telling you that
   level matters.
3. Small bodies near decision zones (prior highs/lows, moving averages,
   fib levels) warn of an imminent direction choice.
4. Volume multiplies every candle's signal. A hammer on 3× average
   volume is very different from a hammer on a quiet day.
5. No candle pattern is self-contained. Always read it in conjunction
   with the trend, RSI/MACD state, and key horizontal levels.

COMMON MISTAKES:
- Taking single-candle signals without confirmation.
- Counter-trend candle trading in strong momentum (e.g. shorting a
  hammer after three green days — trend beats candle).
- Ignoring gap behaviour — a morning star with a breakaway gap is
  stronger than one without.
- Using candles on too-low timeframes where noise dominates psychology
  (1-minute candlestick signals are unreliable in most assets).
"""),

# ── 4. Wyckoff Method ────────────────────────────────────────────────────────
("wyckoff_method_v1",
 "Wyckoff Method — Accumulation, Markup, Distribution, Markdown",
 """
The Wyckoff Method — Reading Market Cycles Through Smart-Money Behaviour

Richard Wyckoff's framework, developed in the early 1900s, remains the
most complete model for reading how institutional capital moves through
an asset. The core insight: retail traders react to price, institutions
cause it. If you can identify where professional operators are buying
(accumulation) or selling (distribution), you can position alongside
them instead of against them. Every asset — equities, crypto, forex,
commodities — moves through the same four-phase cycle: accumulation,
markup, distribution, markdown.

THE THREE LAWS OF WYCKOFF

Law of Supply and Demand: Price rises when demand exceeds supply and
falls when supply exceeds demand. Every candle is a snapshot of the
current supply/demand balance; every trading range is a battle to
establish new control.

Law of Cause and Effect: The length and magnitude of a future trend are
proportional to the time spent in accumulation or distribution. A
multi-month accumulation range produces a multi-month markup. A brief
consolidation produces a brief follow-through. This is where point-and-
figure counting originates: wider causes yield larger effects.

Law of Effort vs Result: Volume (effort) should align with price movement
(result). When effort increases but result decreases, something is wrong
— smart money is absorbing supply without letting price run, typically
near the end of a decline (accumulation) or the end of a rally
(distribution). This divergence is the single most important read in
Wyckoff.

ACCUMULATION PHASE — DETAILED SCHEMATIC

Phase A: Stopping Action. The prior downtrend slows. Preliminary support
(PS) forms. A selling climax (SC) creates a sharp sell-off on huge volume
— often the absolute low of the entire cycle — followed by an automatic
rally (AR) that defines the top of the range. A secondary test (ST)
retests the SC low on lower volume, confirming that sellers are exhausted.

Phase B: Building the Cause. The range develops. Price oscillates between
AR high and ST low. Institutions quietly accumulate. Volume contracts
over time; each new test of support comes on lower volume. Retail
traders lose interest; news flow is boring or negative.

Phase C: The Spring (or Shakeout). A false breakdown below the range low
flushes out stop-losses from retail traders and weak longs. The spring
often drops briefly below support before reversing sharply back into
the range on rising volume. This is the textbook bullish trigger in
Wyckoff — smart money's final absorption of supply.

Phase D: Markup Begins. Price breaks above the range on expanding volume.
Pullbacks now hold above the prior range ceiling, which has flipped to
support. Signs of Strength (SOS) candles appear. A last point of support
(LPS) marks the optimal buy — low-risk entry with stops below the LPS.

Phase E: Trend Continuation. Markup accelerates. Each pullback is shallow
and short; trend followers pile in. This is when retail finally sees
the opportunity — but the best risk-reward entries have passed.

DISTRIBUTION PHASE — MIRROR STRUCTURE

Phase A: Preliminary supply (PSY), buying climax (BC) on huge volume,
automatic reaction (AR) sets the range floor, secondary test (ST)
revisits BC on lower volume.

Phase B: Range builds. Smart money sells into strength while keeping
price elevated. Volume on rallies weakens; volume on declines grows
subtly.

Phase C: Upthrust (UT) or Upthrust After Distribution (UTAD). A false
breakout above the range traps breakout buyers. Price reverses sharply
back inside the range. This is the textbook bearish trigger.

Phase D: Markdown Begins. Price breaks the range floor. Former support
becomes resistance. Signs of Weakness (SOW) candles appear.

Phase E: Downtrend accelerates. Bounces are weak; selling dominates.

VOLUME ANALYSIS IN WYCKOFF

- Climax volume marks absolute turning points (selling climax at lows,
  buying climax at highs).
- Volume should expand in the direction of the underlying trend bias.
  If price rises on weak volume, the advance is suspect — distribution,
  not accumulation.
- Dry volume at the bottom of a range means sellers have given up.
- Effort-result divergences (big volume, tiny price progress) signal
  absorption by the opposite side.

PRACTICAL APPLICATION:
1. Identify the phase by locating the most recent climax candle and
   tracking the range that forms after it.
2. Wait for Phase C confirmation (spring for accumulation, UT for
   distribution) before entering.
3. Enter on the LPS / LPSY pullback with stop beyond the spring/UT
   extreme.
4. Manage position through Phase D; scale out as Phase E matures.
5. Respect that not every range is accumulation or distribution —
   some are simply sideways noise. The presence of specific climax
   events and the effort-result behaviour are what separates a true
   Wyckoff pattern from a random sideways chop.

Wyckoff works on any timeframe but is most reliable on daily and
weekly charts where institutional flows dominate noise.
"""),

# ── 5. Support, Resistance & Trendlines ──────────────────────────────────────
("support_resistance_trendlines_v1",
 "Support, Resistance & Trendlines — The Foundation of Price-Action Analysis",
 """
Support, Resistance, and Trendlines — The Structural Skeleton of Every Chart

Before any indicator, pattern, or strategy, a chart is read through its
levels. Support and resistance are not abstractions — they are zones
where real orders have clustered historically and where market
participants remember decisions. Understanding how to draw, interpret,
and trade these zones is the foundation every other technique sits on.

WHAT MAKES A LEVEL SIGNIFICANT

A horizontal price level gains significance from:
1. Number of touches. Two touches form a level; three or more confirm it.
2. Volume at the level. High-volume candles at a price mean many shares
   were transacted there; participants remember that zone.
3. Time since formation. Recent levels (1-3 months) matter most for
   short-term trading; older levels (1+ year) matter for position trades.
4. Round numbers. $100, $50,000 and similar round prices act as
   psychological magnets and pivot points.
5. Previous swing extremes. Major swing highs and lows are by definition
   where trends turned, which is why they persist as reference points.
6. Confluence with other tools. A level that overlaps a moving average,
   a fibonacci retracement, or a trendline is stronger than a standalone
   level.

SUPPORT — WHY PRICE BOUNCES

Support is the price zone where buying interest consistently overwhelms
selling pressure. Mechanically: limit bids stack up, algorithms buy the
dip, fundamental-driven investors add to positions they consider
undervalued. When price reaches support, several behaviours converge
and cause reversals. Once broken decisively, former support flips to
resistance because the buyers who bid the zone before now see their
positions underwater and sell into any retest.

RESISTANCE — WHY PRICE STALLS

Resistance is the mirror: a zone where supply overwhelms demand.
Participants who bought at higher prices and are now underwater sell
into any rally that brings them near breakeven. Traders who missed the
last top place sell orders anticipating a retest. Once broken, former
resistance flips to support for the same reason in reverse.

DYNAMIC SUPPORT AND RESISTANCE — TRENDLINES

Trendlines connect a series of rising lows (bullish trendline / dynamic
support) or declining highs (bearish trendline / dynamic resistance).
They represent the rate of change of buyer or seller dominance.

Drawing rules:
- Minimum two touches to define; third touch confirms.
- Ignore the occasional wick below/above the line — close beyond the
  line is what matters for breakouts.
- Slope matters: steep trendlines (>45°) are unsustainable and almost
  always break; moderate slopes (20-35°) are more durable.
- The longer a trendline has held, the more significant its eventual
  break.

MOVING AVERAGES AS DYNAMIC S/R

- 20-period SMA/EMA: short-term trend anchor, common pullback zone in
  strong uptrends.
- 50-period: medium-term trend anchor; where institutional money often
  adds in bull markets.
- 200-period: long-term trend divider. Above = secular bull; below =
  secular bear. Crossing the 200 from below (golden cross when 50 > 200)
  or above (death cross when 50 < 200) are macro regime signals.

MARKET STRUCTURE — HIGHER HIGHS & HIGHER LOWS

A trend is formally defined by the sequence of its swing points:
- Uptrend: successively higher highs (HH) and higher lows (HL).
- Downtrend: successively lower highs (LH) and lower lows (LL).
- Sideways: overlapping highs and lows, no clear directional bias.

Structure breaks are critical pivots. If price in an uptrend fails to
make a new higher high and then breaks below the most recent higher low,
the uptrend structure has broken. The next leg is typically a transition
into range or reversal. Until that sequence is clearly broken, assume
the trend continues (trend-following discipline).

CONFLUENCE — WHY PROS WAIT FOR IT

Single-factor levels work sometimes. Multi-factor confluence works much
more often. The highest-quality zones align:
- A prior swing high/low (horizontal support/resistance)
- A key moving average (20/50/200)
- A fibonacci level (38.2, 50, 61.8 from the last leg)
- A trendline touchpoint
- A round number
- High historical volume (visible via a volume-profile chart)

When four or more of these align at the same price, the odds of a
reaction are extremely high. This is where professional traders place
their largest entries.

BREAKOUT vs FALSE BREAKOUT — THE TRADER'S HARDEST PROBLEM

Not every level-break is a continuation. False breakouts are common
because institutional traders deliberately push price beyond levels to
hunt stops. Filters to distinguish genuine breakouts:
- Close beyond the level, not just a wick.
- Expanding volume on the breakout candle.
- Follow-through in the next 1-3 periods — no immediate reversal.
- Prior structure supports the direction (don't trust a bullish breakout
  in a broken downtrend without other confirmations).

The retest trade: once a breakout is confirmed, a pullback to the broken
level — which has flipped polarity — offers the highest-reward entry
with the tightest possible stop.

ACTION CHECKLIST FOR EVERY TRADE:
1. Draw the 3-5 most significant horizontal levels on your primary
   timeframe.
2. Identify current trend via market structure (HH/HL or LH/LL).
3. Check higher-timeframe levels (daily/weekly) even when trading a
   lower timeframe.
4. Plan the trade from confluence to confluence — entry at one zone,
   target at the next.
5. Never enter blind. If you can't identify the level you're bouncing
   off, you're guessing.
"""),

# ── 6. Volume Analysis ───────────────────────────────────────────────────────
("volume_analysis_v1",
 "Volume Analysis — Reading the Commitment Behind Every Price Move",
 """
Volume Analysis — Price Tells You What Happened, Volume Tells You Why

Volume is the second dimension of every chart. Price without volume is
a claim; price with volume is proof. Every breakout, reversal, and
continuation is more or less reliable depending on the volume behind it.
Masters of volume analysis read the commitment, conviction, and
participation behind every move — and avoid the low-conviction moves
that bait retail traders.

CORE PRINCIPLES

1. Volume confirms direction. A breakout from a range on 2-3× average
   volume is far more reliable than one on average or below-average
   volume. The latter is often a stop-hunt that reverses within days.

2. Volume precedes price. Large hidden accumulation typically shows up
   as rising cumulative volume even while price remains sideways. Smart
   money is buying; retail hasn't noticed yet.

3. Volume climaxes mark turning points. Capitulation lows in
   downtrends and euphoric highs in uptrends produce volume spikes of
   3-10× recent average. These candles, especially when they have long
   wicks in the opposite direction, are prime reversal markers.

4. Volume dry-up at support is bullish. When selling interest
   evaporates (volume falling as price tests the low), sellers are
   exhausted. The next buying impulse meets little resistance.

5. Rising volume in ranges signals preparation. Boring sideways
   charts with gently rising volume often precede major breakouts —
   this is Phase B Wyckoff accumulation in action.

ON-BALANCE VOLUME (OBV)

A cumulative volume indicator: add the period's volume if close was up,
subtract if down. Rising OBV with rising price confirms the trend.
Rising OBV with flat price signals hidden accumulation — bullish. Flat
OBV with rising price signals hidden distribution — bearish (buyers are
running out while price still drifts up). OBV divergences are classic
early-warning signals of trend exhaustion.

ACCUMULATION/DISTRIBUTION LINE (A/D)

A volume-weighted indicator that considers where the close fell within
the period's range. A close near the high = accumulation bias; close
near the low = distribution bias. Divergences between A/D and price
often precede reversals by weeks.

CHAIKIN MONEY FLOW (CMF)

Measures volume-weighted money flow over a lookback (typically 20
periods). Values above 0 = buying pressure dominates; below 0 = selling
pressure. Persistent positive CMF supports bullish trend continuation.

VOLUME PROFILE / VOLUME-BY-PRICE

A horizontal histogram showing how much volume transacted at each price
over a period. High-volume nodes (HVNs) act as magnets and strong S/R.
Low-volume nodes (LVNs) are price vacuums — price moves through them
quickly when they are crossed. The point of control (POC) is the single
price with the most volume; it is one of the most reliable levels in
any asset.

VWAP (VOLUME-WEIGHTED AVERAGE PRICE)

The benchmark institutional traders use to assess execution quality.
Price above VWAP = day's buyers winning; below = sellers winning.
Pullbacks to VWAP in a trending session are high-probability entry
zones. VWAP resets daily in most charting tools but is also computed
weekly, monthly, anchored to events, etc.

VOLUME BEHAVIOUR IN CHART PATTERNS

- Bull flags: volume collapses in the flag, expands on breakout.
- Head and shoulders: volume highest at the left shoulder, lower at the
  head, lowest at the right shoulder; expanding on the neckline break.
- Double tops/bottoms: volume lower on the second test than the first.
- Triangles: volume decreases through the formation, expands on the
  breakout.
If volume doesn't match the textbook behaviour, treat the pattern with
skepticism — it may be a trap.

CONTEXT MATTERS:
- Volume significance is relative. Compare today's volume to 20-day
  average, not absolute numbers.
- Pre-market and after-hours volume in equities is thin and often noisy;
  rely on regular-session data for pattern confirmation.
- In crypto, volume varies wildly by exchange — use aggregated or
  primary-exchange data.
- Weekend and holiday volume in traditional markets is unusually low;
  signals from those periods are often unreliable.

COMMON VOLUME MISTAKES:
- Ignoring volume entirely (trading on price alone).
- Chasing volume spikes without context (climaxes can mark both tops
  and bottoms).
- Assuming every breakout needs massive volume — some low-float or
  crypto assets run on ordinary volume.
- Confusing volume of contracts/shares with dollar volume in thinly
  priced assets.
"""),

# ── 7. Market Cycles & Macro Regimes ─────────────────────────────────────────
("market_cycles_regimes_v1",
 "Market Cycles, Regimes & Sentiment — Knowing Where You Are in the Cycle",
 """
Market Cycles and Regimes — The Macro Context Every Trade Lives In

Every individual chart lives inside a much larger cycle. Ignoring that
cycle is the single most common reason retail traders fail: they buy
extended leaders at the top of bull markets and sell exhausted laggards
at the bottom of bear markets. Understanding where you are in the
macro cycle biases every decision correctly.

THE FOUR MACRO PHASES

1. Accumulation (after a bear market). Sentiment is wrecked; headlines
   are bearish; volume is low. Smart money starts buying quietly. The
   market bottoms gradually, often over months. Trading bias: patient
   long exposure on clear reversal patterns at major S/R.

2. Markup / Expansion (bull market). The economy improves; credit is
   cheap; risk appetite rebuilds. Price advances with shallow
   corrections. Each pullback is bought. Trading bias: trend-follow
   longs, avoid counter-trend shorts.

3. Distribution (after an extended advance). Sentiment is euphoric;
   retail participation peaks; valuations stretch. Smart money quietly
   sells into strength. The market tops in a rolling, sector-by-sector
   fashion. Breadth deteriorates even while indices make new highs.
   Trading bias: tighten stops, reduce size, rotate to defensives.

4. Markdown / Contraction (bear market). Headlines turn negative;
   credit tightens; leveraged positions unwind. Bounces fail. Trading
   bias: capital preservation, short rallies with tight risk, avoid
   catching falling knives.

IDENTIFYING REGIMES IN REAL TIME

Key instruments and signals:
- Major index trend relative to 200-day SMA: above and rising = bull;
  below and falling = bear; oscillating around = transition.
- Breadth indicators: percentage of stocks above 50/200-day SMA,
  advance-decline line, new-highs vs new-lows. Divergences between
  price and breadth warn of phase transitions.
- Yield curve: persistent inversion is historically one of the most
  reliable recession signals.
- Credit spreads (high-yield minus investment-grade): widening =
  risk-off stress; tightening = risk appetite recovering.
- VIX: sustained low readings accompany bull phases; sustained high
  or rising readings accompany distribution and markdown.
- Sector leadership rotation: defensives (staples, utilities,
  healthcare) outperforming cyclicals (tech, discretionary,
  industrials) is a late-cycle warning.

SENTIMENT AND CONTRARIAN THINKING

Extreme sentiment marks extremes in price. When everyone is bullish,
buying demand has already been spent — no marginal buyer remains.
When everyone is bearish, selling has washed out — no marginal seller
remains. Indicators to track:
- Put/call ratios
- Investors Intelligence survey
- Fear and Greed Index
- Retail margin debt
- Mutual fund flows into equity/bond
At extremes, fade the consensus. In neutral zones, respect the trend.

CRYPTO-SPECIFIC REGIME ANALYSIS

Crypto cycles have historically aligned with Bitcoin halving events
(every ~4 years), though recent cycles show weaker coupling. Within
each cycle:
- Accumulation: 12-18 months post-halving, sentiment crushed.
- Markup: 18-30 months post-halving, euphoric highs.
- Distribution: 6-9 months of topping.
- Markdown: 9-15 month decline.
Within this frame, altcoin seasons (when capital rotates from BTC to
smaller assets) typically begin 2-4 months into markup phase and peak
near the macro top.

REGIME-AWARE POSITION SIZING

In markup phases: size up on trend-aligned breakouts; pullbacks to
dynamic support (20/50 MA) are buying opportunities; counter-trend
shorts have poor risk-reward.

In distribution: size down; require higher-quality setups; prefer
range-fading over breakout-chasing; watch for breadth divergences as
early exit signals.

In markdown: shift to cash/defensives; only trade highest-quality
setups; accept that most trades will be stopped out; capital
preservation trumps return.

In accumulation: begin scaling back in on oversold multi-month
consolidations; volume dry-up and sentiment washout signal the
transition.

CHECKLIST FOR EVERY TRADING DAY:
1. What phase is the broad market in?
2. Which sectors are leading? Which are lagging?
3. What is the VIX doing today?
4. Are credit spreads widening or tightening?
5. Does my individual setup align with the prevailing regime, or am I
   fighting it?
If #5 is 'fighting it', the position should be smaller, the stop tighter,
and the target closer.

Trading aligned with the macro wind is easier than fighting it. Read the
regime first; choose setups second.
"""),

# ── 8. Multi-Timeframe Analysis ──────────────────────────────────────────────
("multi_timeframe_analysis_v1",
 "Multi-Timeframe Analysis — Aligning Signals Across Scales",
 """
Multi-Timeframe Analysis — Context, Confirmation, Execution

Every timeframe tells a different story, and none of them is complete
on its own. A bullish reversal on a 15-minute chart might be a trivial
bounce within a daily downtrend. A bearish rejection on the 4-hour
might be noise within a weekly uptrend. Professional traders operate
across at least three timeframes simultaneously: one for context, one
for confirmation, and one for execution.

THE THREE-TIMEFRAME FRAMEWORK

CONTEXT (highest): This is the macro view. For position traders: weekly
and daily charts. For swing traders: daily and 4-hour. For day traders:
daily and 1-hour. The context chart answers: "What is the dominant
trend? Where are the major levels? What phase of the cycle am I in?"

CONFIRMATION (middle): This is where you look for alignment — a setup
forming that fits the context. For swing traders: 4-hour and 1-hour.
For day traders: 1-hour and 15-minute. The confirmation chart answers:
"Is there a pattern or momentum shift consistent with the context?"

EXECUTION (lowest): This is timing the entry. For swing traders:
1-hour or 30-minute. For day traders: 5-minute and 1-minute. The
execution chart answers: "Where is the optimal trigger? Where is the
stop? What is the realistic first target?"

ALIGNMENT RULES

1. Never trade against the context timeframe's trend without a very
   high-conviction pattern. The vast majority of losing trades come
   from lower-timeframe signals that contradict higher-timeframe
   structure.

2. Lower timeframes should echo higher-timeframe signals. If the daily
   is breaking out, 4-hour should show strength (higher highs) and
   1-hour should show the pattern that triggered the daily break.

3. If higher and lower timeframes conflict, either wait for them to
   align or reduce size significantly. Conflict is noise, not signal.

4. Execute on the lowest timeframe, but stop-loss and target using the
   confirmation timeframe structure. A day-trader's 5-minute stop
   placed under the nearest 5-minute swing low will often get hunted;
   placing it under the 1-hour swing low is structurally sounder even
   if it sacrifices a bit of R:R.

APPLYING MULTI-TF TO COMMON SETUPS

Breakout:
- Context: price in an uptrend on the daily, consolidating in a
  rectangle near resistance.
- Confirmation: 4-hour forming a bull flag against that resistance.
- Execution: 30-minute candle closing above flag high on expanded
  volume = entry.

Reversal:
- Context: daily RSI showing bullish divergence at a major support
  zone (confluence of 200 SMA and prior swing low).
- Confirmation: 4-hour morning-star or hammer at that support.
- Execution: 1-hour close above the preceding swing high = entry.

Retest trade (post-breakout):
- Context: daily just broke out of a 3-month base on expanding volume.
- Confirmation: 4-hour pulling back to broken resistance (now support)
  with declining volume.
- Execution: 15-minute bullish engulfing candle at the retest level
  with volume = entry.

COMMON MISTAKES

- Trading the signal without checking the context. Countless "perfect"
  1-hour bullish patterns fail because the daily is clearly in a
  distribution phase.
- Using too many timeframes. Three is enough. Five creates paralysis.
- Overweighting lower-timeframe noise. The more you stare at 1-minute
  charts, the more every bar feels important — most aren't.
- Trading a higher-timeframe setup but managing it with lower-timeframe
  emotion. If you entered for a daily target, don't panic out on a
  1-hour dip.

PRACTICAL WORKFLOW

1. Start at the highest timeframe. Mark major trend and key levels.
2. Move to the middle timeframe. Identify current setup or lack
   thereof.
3. Move to the lowest timeframe only once a higher-timeframe setup is
   present. Execution-timeframe analysis without context is gambling.
4. Set stops/targets using middle-timeframe structure.
5. Review winners AND losers across all timeframes to refine the
   alignment discipline.

Multi-timeframe discipline is what separates patient edge-trading from
random chart-staring. Context gates everything.
"""),

# ── 9. Micro-Scalping & Short-Timeframe Edge ─────────────────────────────────
("micro_scalping_v1",
 "Micro-Scalping — Capturing Short Intraday Swings Inside Larger Trends",
 """
Micro-Scalping — Capturing Small Moves with Tight Discipline

Micro-scalping is the practice of entering and exiting within a short
window (seconds to hours), aiming for small gains (0.2-1.5%) repeatedly
rather than large gains occasionally. It works because markets rarely
trend cleanly — most of the time price is oscillating within
short-range noise. In a broader downtrend, there are still hourly
bounces of 0.5-1% that can be captured. In a broader uptrend, there
are still shallow pullbacks that precede quick resumptions. Scalping
harvests this micro-structure while respecting the macro bias.

CORE PRINCIPLES

1. High win rate, small reward per trade. Scalping edges are typically
   55-65% win rate with 1:1 or 1.2:1 reward:risk. Volume of trades
   compensates for size of each.

2. Tight stops, tight targets. Because the entry is reactive to
   short-term structure, the stop is close. If you're wrong, you know
   quickly; if you're right, you take profit quickly.

3. Execution speed matters. Slippage, spread, and latency can eat a
   scalper's edge. Choose liquid assets (major crypto pairs, large-cap
   equities, index ETFs) and avoid illiquid times.

4. Alignment with micro-trend. Scalp in the direction of the 1-hour
   or 4-hour trend. Counter-trend scalping is possible but requires
   much more skill — most losing scalpers are fighting the micro-
   trend they can't see.

HIGH-QUALITY MICRO-SCALP SETUPS

1. VWAP Pullback in Trending Session. Price is above VWAP and trending
   up intraday. On each pullback to VWAP, watch for a 1-minute bullish
   reversal candle + volume pickup -> long with stop below VWAP or the
   1-minute swing low, target = prior intraday high. Mirror for shorts
   below VWAP.

2. Range Fade at Clear Levels. Price is oscillating between well-
   defined intraday support and resistance with no directional bias.
   Sell at resistance rejections, buy at support holds, tight stops
   beyond the level, target the opposite side.

3. Breakout Retest. Price breaks out of a multi-hour consolidation on
   strong volume. The first pullback to the broken level (now flipped
   polarity) is a high-probability continuation scalp. Entry on the
   retest hold, stop beyond the level, target = prior breakout
   extension.

4. Momentum Continuation After News. A catalyst pushes price hard;
   after an initial spike, price consolidates in a small range; the
   next breakout in the catalyst's direction is typically a momentum
   scalp with 3-5 minutes of duration.

5. Bounce in Oversold RSI at Intraday Support. On 5-minute chart, RSI
   < 30 at a prior intraday low + bullish divergence on the next test
   -> long with 1% stop, 1-2% target.

AVOIDING SCALP TRAPS

- Never scalp into major macro events (FOMC, NFP, CPI release). Spreads
  widen and whipsaws are lethal.
- Avoid scalping near daily pivots unless volume supports direction.
- Don't double down on losing scalps. If stopped, take the loss and
  wait for the next setup.
- Avoid scalping thin pre-market or post-market sessions in equities.
- Position sizing must account for the fact that losses come in
  streaks. Risk only 0.25-0.5% of account per scalp.

SCALPING TOOLS AND INDICATORS

- 1m, 5m, 15m charts as primary execution; 1h and 4h for context.
- VWAP (anchored to session open or to key intraday events).
- Short-period moving averages (9-EMA, 21-EMA) for micro-trend.
- RSI (5-period or 9-period) for quick overbought/oversold.
- Volume bars compared to 20-period average on the same timeframe.
- Order-flow / tape for high-liquidity assets (advanced).

REGIME-AWARE SCALPING

In a broad downtrend: favour shorting rips into resistance. Longs are
counter-trend — smaller size, faster profit-taking, wider caution.

In a broad uptrend: favour buying dips to dynamic support. Shorts are
counter-trend — smaller size, faster profit-taking.

In sideways regimes: range-fade both sides; breakout-scalp if a clean
break occurs with volume.

In high-volatility regimes (VIX elevated, crypto funding extreme):
shrink size, widen stops minimally, accept fewer trades. Volatility
expands both wins and losses; risk discipline compensates.

LEARNING TRAJECTORY FOR SCALPERS

New scalpers should start by paper-trading or micro-sizing for weeks
to build pattern recognition. Track every trade with screenshots and
notes on why the entry was taken, where the stop was, and whether the
plan was followed. Journaling is what turns scalping from gambling
into a repeatable edge. Without journaling, every scalper eventually
regresses to random behaviour.

The scalper's edge is discipline, not prediction. Small, repeatable
wins; small, controlled losses; relentless execution of a proven
playbook. Any single scalp is insignificant — the equity curve over
500 trades is what defines a scalper's skill.
"""),

# ── 10. News, Catalysts & Sentiment-Driven Moves ─────────────────────────────
("news_catalysts_sentiment_v1",
 "News Catalysts & Sentiment — How External Events Move Price",
 """
News, Catalysts, and Sentiment — The Fundamental Fuel Behind Technical Moves

Technical patterns don't materialise in a vacuum. Beneath every
breakout, reversal, and consolidation sits a stream of news, earnings,
macroeconomic data, geopolitical events, and crowd sentiment. Traders
who ignore this layer trade on incomplete information; traders who read
it can anticipate which chart setups are loaded with fuel and which
are likely to fail.

CATALYST CATEGORIES

1. Scheduled economic data. Non-farm payrolls (NFP), CPI, PCE, FOMC
   rate decisions, GDP, PMIs. These releases reprice entire asset
   classes in seconds. Equities, bonds, currencies, and commodities
   all react; the cross-asset coherence of those reactions often
   reveals which narrative the market is currently pricing.

2. Earnings reports. Individual stock catalysts. Pre-earnings
   positioning, implied volatility, and post-earnings drift are all
   patterns that repeat. Avoid pattern-trading stocks within 3-5 days
   of earnings unless you explicitly want event exposure.

3. Central bank commentary. Fed/ECB/BOJ speeches can shift the
   aggregate rate narrative without a formal policy move. Terminal-
   rate expectations drive duration assets, credit, and the dollar.

4. Geopolitical shocks. Wars, sanctions, trade policy, regime changes.
   These are tail events — infrequent but massive. They flip
   correlations (e.g., equities and bonds can both fall as USD rallies
   during risk-off) and invalidate short-term charts until the new
   regime is priced.

5. Crypto-specific catalysts. Protocol upgrades, exchange hacks,
   regulatory actions, stablecoin events, major whale movements.
   Crypto markets trade 24/7, so weekend moves are common.

6. Corporate actions. M&A, buybacks, splits, dividends, management
   changes. Usually asset-specific but can reshape entire sectors.

PRE-EVENT vs POST-EVENT POSITIONING

Pre-event: Markets price expectations. Implied volatility rises into
the event, spreads widen, directional bets have unfavourable skew. If
your setup requires a specific outcome, remember you're trading the
surprise vs expectations, not the raw outcome.

Post-event: Markets digest the actual result. Initial spike often
reverses partially (the "first move is often the false move"), then
the durable move emerges 1-3 hours later as the crowd settles. This
is typically a cleaner technical entry than the initial whipsaw.

SENTIMENT INDICATORS

Sentiment measures are contrarian tools. Extreme bullishness marks
tops; extreme bearishness marks bottoms. Key tools:

- Fear & Greed Index (CNN / crypto versions): composite of volatility,
  momentum, safe-haven demand, put/call ratios.
- AAII Bull/Bear Survey (equities): retail sentiment. Extreme bullish
  readings (>60%) often precede 5-10% index pullbacks.
- Commitment of Traders (COT) reports: institutional positioning in
  futures. Extreme long/short crowding by commercials vs non-commercials
  foreshadows reversals.
- Social media sentiment (Twitter/X, Reddit): dangerous when uniform —
  retail often peaks before institutions.
- Google Trends: spikes in search interest for "bitcoin" or "should I
  buy stocks" align with tops.

NEWS-FLOW READING

Good traders skim rather than read. The headline's first clause usually
carries the tradeable information. Specific things to scan for:
- Direction of surprise vs consensus.
- Revisions to prior data.
- Forward guidance (future-tense language in corporate releases).
- Policy language shifts (a single word change in Fed statement can
  move bonds 10 bps).
- Cross-asset reactions (if S&P rallies but 10Y yield also spikes,
  the move is not a risk-off recovery).

GEOPOLITICAL ANALYSIS FRAMEWORK

When a geopolitical event breaks:
1. Classify impact: local, regional, global.
2. Identify affected asset classes: equities, currencies, commodities,
   credit.
3. Consider duration: one-day shock vs regime change.
4. Map second-order effects: sanctions ripple into commodities, which
   ripple into inflation, which ripples into rates, which ripples into
   equities.
5. Align trades with the regime-level move, not the first-minute
   reaction.

NEWS + TECHNICAL CONFLUENCE

The best trades combine:
- A technical setup that is already high-quality on its own.
- A catalyst that points in the same direction.
- Sentiment that isn't already saturated in that direction.

Example: a stock is in a multi-month base with clear accumulation
(Wyckoff Phase B), approaching its breakout level. Earnings are due in
two weeks. Analyst sentiment has been neutral. The technical setup is
loaded with fuel — a positive surprise will likely send it through
resistance; a negative surprise will likely be quickly bought into the
base. Asymmetric risk/reward.

WHEN NOT TO TRADE THE NEWS

- Into a binary event with no edge on outcome direction (FOMC, for a
  retail trader without rate-pricing expertise).
- When the market has already priced the outcome (e.g., earnings
  where the stock rallied 15% into the print).
- When liquidity is thin (post-event minutes in illiquid assets).
- When your position size is too large for the expected volatility
  spike.

TRADE JOURNALING FOR NEWS-DRIVEN MOVES

Track every news-driven trade with: catalyst type, expected outcome,
actual outcome, initial reaction, final daily close, your entry/exit,
and lessons. Over time, patterns emerge: certain types of catalysts
align with certain types of moves for certain types of assets. This
meta-knowledge is where sustained edge compounds.

Fundamentals set the fuel. Technicals set the timing. Sentiment sets
the odds. The intersection is where conviction lives.
"""),

("professional_multi_source_stock_analysis_v1",
 "Professional Multi-Source Stock Analysis — Technical, Fundamental, News, and Event Confluence",
 """
Professional Multi-Source Stock Analysis Playbook

This playbook is synthesized from reputable investor-education and market
structure sources, including Fidelity technical-analysis education,
Investor.gov 10-K guidance, and SEC EDGAR API documentation. It is designed
for autonomous trading decisions that must cross-check multiple evidence
buckets before declaring a stock attractive.

CORE PRINCIPLE

No single indicator, article, or data point is enough to justify a BUY. A
professional decision cross-references independent information streams:
technical structure, volume confirmation, strategy fit, valuation,
fundamentals, earnings risk, company-specific headlines, and macro/sector
context. A BUY requires alignment across several buckets and no hard veto.
When the buckets disagree, the default action is HOLD.

TECHNICAL STRUCTURE

Read the trend before reading indicators. An uptrend is built from higher
highs and higher lows. A downtrend is built from lower highs and lower lows.
A sideways range means supply and demand are balanced, so breakout trades
need more proof and mean-reversion trades need tighter targets. The longer
timeframe sets the bias. The intermediate timeframe defines setup quality.
The short timeframe only helps with entry timing.

Support and resistance are supply-demand zones, not magic price lines. A
support break is bearish only when price closes through the zone and sellers
show participation. A resistance break is bullish only when price closes
above the zone and volume expands. Low-volume breakouts are lower quality
because they may reflect thin liquidity rather than committed buyers.

MOMENTUM AND VOLUME

RSI measures momentum and can identify overbought or oversold conditions, but
it is not a standalone trade signal. In strong trends, RSI can remain extreme
for extended periods. RSI is most useful when it confirms trend strength,
shows divergence at a meaningful support/resistance zone, or aligns with a
clear reversal pattern.

MACD is a trend-momentum tool. Bullish crossovers are stronger when they
occur with improving histogram momentum, price holding above key moving
averages, and a broader uptrend. Bearish crossovers are stronger when price
is losing support and histogram momentum is expanding negatively. MACD is
less reliable inside choppy sideways ranges and should be discounted there.

Volume confirms participation. A breakout through resistance on expanding
volume has better quality than the same breakout on weak volume. A support
break on expanding volume shows stronger selling pressure. Volume divergence
can warn that a trend is losing sponsorship before price fully reverses.

FUNDAMENTAL AND VALUATION CHECK

For individual stocks, fundamentals can veto a chart setup. Review the
business, risk factors, management discussion, audited financial statements,
cash flow, debt, margins, revenue trend, earnings trend, and valuation. A
high P/E can be justified by durable growth, but it becomes a risk when
technicals are mixed or growth headlines deteriorate. A low P/E is not
automatically attractive if the company has declining revenue, legal risk,
balance-sheet stress, or shrinking margins.

SEC filings and EDGAR data are authoritative sources for U.S. public-company
filing history and XBRL financial data. Use them as ground truth when
available. Market-data APIs and news feeds are useful, but company filings
are the formal disclosure layer.

EVENT AND NEWS RISK

Earnings can invalidate normal technical analysis because price can move
sharply on information that is not visible in the chart. If earnings are
imminent, reduce size or prefer HOLD unless there is a defined event strategy.
Negative company-specific headlines such as fraud allegations, bankruptcy
risk, regulatory probes, major guidance cuts, failed trials, or accounting
issues should block BUY even if short-term indicators look oversold.

ATTRACTIVENESS SCORING

Classify the stock as ATTRACTIVE only when at least three independent
buckets support upside and no hard veto exists. Examples of positive buckets:
uptrend or constructive base, bullish momentum, confirmed volume, strong RAG
strategy fit, reasonable valuation, healthy fundamentals, positive company
catalyst, supportive macro/sector backdrop, and no near-term binary event.

Classify NEUTRAL when evidence is mixed, incomplete, or setup quality is not
repeatable. Classify UNATTRACTIVE when trend is broken, volume confirms
selling, valuation/fundamentals are poor, headlines are negative, earnings
risk is high, or the reward/risk cannot be defined.

FORECAST DISCIPLINE

The price outlook is not a promise. Use BULLISH, NEUTRAL, or BEARISH as a
probabilistic forecast from available evidence. Every BULLISH outlook needs
an invalidation level. Every BUY needs a stop, a take-profit area, and a
reason why the setup is better than simply waiting.
"""),

]

# ─────────────────────────────────────────────────────────────────────────────
# Ingestion engine (mirrors seed_strategies.ingest_all for consistency)
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk_id(source_id: str, idx: int, text: str) -> str:
    fingerprint = f"{source_id}:{idx}:{text[:64]}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def ingest_all() -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    with quiet_model_load():
        embed_fn = SentenceTransformerEmbeddingFunction(
            # No trust_remote_code — see rag/retriever.py, which loads the
            # same model without granting a repo arbitrary code execution.
            model_name=settings.embedding_model,
        )

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total_added = 0
    total_skipped = 0

    for source_id, title, text in CHART_ANALYSIS_LIBRARY:
        chunks = splitter.split_text(text.strip())
        ids       = [_make_chunk_id(source_id, i, c) for i, c in enumerate(chunks)]
        documents = chunks
        metadatas = [
            {
                "source":       "seed_chart_analysis",
                # Operator-curated: every account may retrieve it.
                "owner":        SHARED_OWNER,
                "source_id":    source_id,
                "title":        title,
                "type":         "knowledge",
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "chunk_size":   len(c),
            }
            for i, c in enumerate(chunks)
        ]

        try:
            existing = collection.get(ids=ids, include=[])
            existing_set = set(existing["ids"])
        except Exception:
            existing_set = set()

        new_ids   = [i for i in ids if i not in existing_set]
        new_docs  = [documents[ids.index(i)] for i in new_ids]
        new_metas = [metadatas[ids.index(i)] for i in new_ids]
        skipped   = len(ids) - len(new_ids)

        if new_ids:
            collection.upsert(ids=new_ids, documents=new_docs, metadatas=new_metas)

        total_added   += len(new_ids)
        total_skipped += skipped
        logger.info(
            f"  [{source_id}] '{title[:55]}' "
            f"-> {len(new_ids)} chunks added, {skipped} already existed"
        )

    count = collection.count()
    logger.info(
        f"\n{'='*60}\n"
        f"  Chart-analysis ingestion complete.\n"
        f"  Added: {total_added} chunks\n"
        f"  Skipped (already existed): {total_skipped} chunks\n"
        f"  Total collection size: {count} chunks\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  BotTrade — Seeding Chart Analysis Knowledge Base")
    print("=" * 60)
    ingest_all()
