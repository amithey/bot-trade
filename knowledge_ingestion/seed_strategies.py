"""
Seed Trading Strategies — Direct ChromaDB Ingestion
=====================================================
Inserts curated, comprehensive trading strategy knowledge directly into
ChromaDB without relying on YouTube transcript availability.

Run once to populate the knowledge base:
    python -m knowledge_ingestion.seed_strategies

All content is organised into named "books" — each book is a self-contained
strategy guide that the RAG engine will surface when market conditions match.
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
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Strategy library — each entry is (source_id, title, full_text)
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_LIBRARY: list[tuple[str, str, str]] = [

# ── 1. RSI Complete Playbook ──────────────────────────────────────────────────
("rsi_playbook_v1", "RSI Trading Strategy — Complete Playbook", """
RSI (Relative Strength Index) Trading Strategy — Complete Playbook

The Relative Strength Index (RSI) is a momentum oscillator that measures the speed
and change of price movements on a scale of 0 to 100. Developed by J. Welles Wilder,
it remains one of the most widely-used indicators in both equity and crypto markets.

CORE INTERPRETATION RULES:
- RSI above 70: Overbought territory. Price has risen rapidly and may be due for a
  pullback or consolidation. Do NOT automatically sell — overbought can persist in
  strong uptrends (crypto especially). Look for bearish divergence or momentum failure.
- RSI below 30: Oversold territory. Price has fallen sharply. Potential reversal zone.
  Do NOT automatically buy — oversold can persist in strong downtrends. Look for
  bullish divergence or momentum exhaustion signals.
- RSI 40–60: Neutral zone. No clear momentum edge. Defer to trend direction and MACD.
- RSI 60–70: Strong bullish momentum. Trend-following long positions are viable.
  This zone in an uptrend is where professional traders ADD to positions.
- RSI 30–40: Bearish momentum zone. Selling pressure dominates. Avoid new longs.

RSI DIVERGENCE — THE HIGHEST PROBABILITY SIGNAL:
Bullish Divergence: Price makes a new LOW but RSI makes a HIGHER LOW.
  → Selling pressure is exhausting despite new price lows. Strong reversal signal.
  → Entry: RSI crosses back above 30 with price confirmation candle.
  → Stop Loss: 2–3% below the recent swing low.
  → Take Profit: Previous swing high or SMA_20.

Bearish Divergence: Price makes a new HIGH but RSI makes a LOWER HIGH.
  → Buying pressure is exhausting despite new price highs. Strong reversal warning.
  → Consider tightening stops on longs. Not a standalone short signal.

RSI STRATEGY FOR CRYPTO (Bitcoin / BTC-USD):
Bitcoin exhibits higher volatility than equities. Overbought (RSI>70) often continues
in bull markets for extended periods. Key adjustments:
  - Use 40 as the bull market support zone (not 30). If RSI holds above 40 in
    an uptrend, the trend is intact. A break below 40 = momentum shift warning.
  - RSI below 30 in crypto with high volume = capitulation event. High-probability
    mean-reversion bounce. Enter small position, add on confirmation.
  - RSI above 70 in crypto: Do NOT short. Trim longs by 25–30%, trail stop.

ENTRY RULES (BUY):
1. RSI falls below 30 and then crosses back above 30 (oversold exit confirmation)
2. Volume is above 20-day average on the reversal candle
3. Price is above the 200-day SMA (long-term uptrend confirmation)
4. MACD histogram begins to turn positive (momentum confirmation)
Confidence: HIGH (0.75+) if all 4 conditions met. MEDIUM (0.60) if 3 met.

EXIT RULES (SELL):
1. RSI reaches 65–70 zone AND shows momentum divergence
2. Price has reached the nearest major resistance level
3. MACD histogram starts turning negative (momentum fading)
Take Profit: Set at the next resistance. Stop Loss: 2–2.5% below entry.

RISK MANAGEMENT:
- Never risk more than 2% of total capital per trade
- In strong downtrends (price below SMA_200, Death Cross), reduce position size 50%
- RSI signals are more reliable on higher timeframes (daily > hourly > 5-minute)
"""),

# ── 2. MACD Complete Playbook ─────────────────────────────────────────────────
("macd_playbook_v1", "MACD Trading Strategy — Complete Playbook", """
MACD (Moving Average Convergence Divergence) — Complete Trading Playbook

MACD was developed by Gerald Appel and is one of the most reliable trend-following
momentum indicators. It shows the relationship between two exponential moving averages
of price, making it ideal for catching sustained directional moves.

COMPONENTS:
- MACD Line: 12-period EMA minus 26-period EMA
- Signal Line: 9-period EMA of the MACD Line
- Histogram: MACD Line minus Signal Line (positive = bullish, negative = bearish)

MACD CROSSOVER SIGNALS:
Bullish Crossover (Golden Cross of MACD):
  → MACD Line crosses ABOVE the Signal Line
  → Histogram turns from negative to positive
  → Meaning: Short-term momentum is accelerating faster than long-term momentum
  → Action: Enter long position. Strongest when it occurs below the zero line
    (meaning price was in oversold territory before the momentum shift).

Bearish Crossover (Dead Cross of MACD):
  → MACD Line crosses BELOW the Signal Line
  → Histogram turns from positive to negative
  → Meaning: Short-term momentum is decelerating faster than long-term
  → Action: Close longs, consider short position only in confirmed downtrend.
    Strongest when it occurs above the zero line.

MACD HISTOGRAM — THE MOMENTUM ACCELEROMETER:
The histogram is the most underutilised part of MACD. Its direction reveals
whether momentum is ACCELERATING or DECELERATING:

  Expanding Positive Histogram: Bullish momentum accelerating. Bulls are gaining
    strength. This is the BEST time to hold or add to long positions.

  Shrinking Positive Histogram (FADING BULLISH): Bullish momentum is decelerating.
    Bulls are losing steam. Prepare to exit longs. Do NOT enter new longs.
    This is an early warning signal — often appears before the actual price peak.

  Expanding Negative Histogram: Bearish momentum accelerating. Bears control.
    Avoid longs. Short positions viable in confirmed downtrends.

  Shrinking Negative Histogram (FADING BEARISH): Bearish momentum exhausting.
    Bears are losing power. This precedes bullish reversals. Start watching for
    RSI confirmation. Potential long entry setup forming.

MACD ZERO LINE SIGNIFICANCE:
  - MACD above zero: Asset is in a bullish regime (short-term trend above long-term)
  - MACD below zero: Asset is in a bearish regime
  - Crossing zero line = confirmation of trend change (stronger than crossover signal)

MACD DIVERGENCE:
  Bullish Divergence: Price makes lower low but MACD histogram makes higher low.
    → Bears are exhausted. Strong bullish reversal signal. High conviction.
    → Entry: First green histogram bar after divergence. Stop: Below recent low.

  Bearish Divergence: Price makes higher high but MACD histogram makes lower high.
    → Bulls are exhausted. Strong bearish reversal warning. Trail stops.

MACD FOR BITCOIN (CRYPTO) TRADING:
Bitcoin's high volatility means MACD signals generate more false positives on
short timeframes (5m, 15m). Best practices:
  - Use 5-minute MACD for timing, but only trade in direction of daily MACD.
  - If daily MACD is bullish crossover: Only take 5m bullish crossover signals.
  - If daily MACD is bearish crossover: Wait. Do not fight the daily trend.
  - Histogram expanding negative on 5m during a daily uptrend = dip opportunity.

COMBINED MACD + RSI SIGNAL MATRIX:
HIGH CONFIDENCE BUY (confidence 0.75+):
  → MACD bullish crossover + RSI oversold exit (RSI crosses above 30) + price above SMA_20
  → This combination is one of the most reliable entry signals in technical analysis.

HIGH CONFIDENCE SELL/EXIT (confidence 0.75+):
  → MACD bearish crossover + RSI overbought divergence + MACD histogram fading
  → Exit full position. Do not wait for confirmation.

MEDIUM CONFIDENCE (confidence 0.55-0.70):
  → MACD bullish crossover but RSI neutral (40-60) and price mixed relative to SMAs
  → Enter 50% position size. Add more on confirmation.

HOLD / NO TRADE:
  → MACD bearish crossover but RSI neutral/bullish — conflicting signals
  → MACD histogram fading in both directions — momentum unclear
  → Confidence below 0.55: Always output HOLD regardless of signal bias.
"""),

# ── 3. Moving Averages & Trend Structure ─────────────────────────────────────
("ma_trend_v1", "Moving Average Trend Structure — SMA_20 SMA_50 SMA_200 Death Cross Golden Cross", """
Moving Average Trend Structure — Complete Guide

Moving averages smooth price data to reveal the underlying trend direction.
They serve as dynamic support and resistance levels and are the foundation
of all trend-following strategies.

SMA HIERARCHY (The Trend Stack):
  SMA_20  (20-period): Short-term trend. Fast-moving. Shows current momentum.
  SMA_50  (50-period): Medium-term trend. Institutional reference level.
  SMA_200 (200-period): Long-term trend. The definitive bull/bear dividing line.

GOLDEN CROSS — The Bullish Structural Signal:
  Definition: SMA_50 crosses ABOVE SMA_200
  Meaning: Medium-term trend has turned bullish relative to the long-term trend.
  Historical significance: Major bull market initiator signal. When confirmed with
    volume, it often marks the beginning of sustained uptrends lasting months.

  Entry Rules after Golden Cross:
  1. Wait for price to pull back to SMA_50 (do not chase breakout)
  2. RSI should be 40-60 on the pullback (not overbought)
  3. MACD should be positive or crossing bullish
  4. Enter at SMA_50 touch with stop below SMA_200
  Stop Loss: 3% below the SMA_200
  Take Profit: 15-20% move, or RSI reaches 70

DEATH CROSS — The Bearish Structural Signal:
  Definition: SMA_50 crosses BELOW SMA_200
  Meaning: Medium-term trend has turned bearish relative to the long-term trend.
  Historical significance: Strong warning signal for sustained downtrends.

  Implications when Death Cross is active:
  1. All long positions carry elevated risk. Reduce position sizes by 50%.
  2. Do NOT buy pullbacks aggressively — they may be bull traps.
  3. Short-term rallies to SMA_50 or SMA_200 are resistance, not support.
  4. For a BUY signal to be HIGH CONFIDENCE, it must overcome the Death Cross bearish bias.
     Require RSI < 35 AND MACD bullish crossover AND price reclaiming SMA_20.
  5. Death Cross does not mean immediate collapse — price often rallies 10-15%
     after the cross before continuing lower (bull trap rally to SMA_50).

  Death Cross Risk Management:
  - Capital allocation: Maximum 25% of normal position size for any long
  - Stop Loss: Tighter than normal — 1.5% below entry (not 2.5%)
  - Exit any long if price closes below SMA_20 for 2 consecutive candles

PRICE POSITION RELATIVE TO SMAS:
  Price > SMA_20 > SMA_50 > SMA_200: Perfect uptrend structure. All SMAs aligned.
    → Strong bullish bias. Buy pullbacks to SMA_20. High confidence longs.

  Price > SMA_20 > SMA_50 but BELOW SMA_200 (current BTC situation):
    → Short-term bullish but long-term bearish. MIXED structure.
    → The SMA_200 acts as overhead resistance. Rally may stall here.
    → Reduced position sizes. Require stronger confirmation before buying.
    → Confidence cap: 0.65 for BUY signals in this structure.

  Price below SMA_20 but above SMA_50:
    → Short-term pullback in medium-term uptrend. Watch SMA_50 as support.
    → If RSI oversold + MACD bullish divergence at SMA_50 = buy setup.

  Price < SMA_20 < SMA_50 < SMA_200: Perfect downtrend structure.
    → Strong bearish bias. Do NOT buy. Any rally is a short opportunity.
    → Only consider long after RSI hits extreme oversold (<25) with volume spike.

SMA AS DYNAMIC SUPPORT AND RESISTANCE:
  In uptrends: Price tends to bounce off SMA_20, then SMA_50 on deeper dips.
  In downtrends: SMA_20 and SMA_50 become resistance (ceilings, not floors).
  The SMA_200 is the ultimate support in bull markets and resistance in bear markets.

BITCOIN-SPECIFIC SMA OBSERVATIONS:
  Bitcoin's SMA_200 is the most-watched institutional level.
  - Price above SMA_200 = Bitcoin "bull market" designation by analysts
  - Price below SMA_200 = Bitcoin "bear market" designation
  - The SMA_200 acts as a strong magnet — price often reverts to it after extremes
  - Bitcoin Death Cross occurred historically before significant drops (2018, 2022)
  - Bitcoin Golden Cross historically preceded major bull runs (2019, 2020, 2023)
  - Current Death Cross context: Wait for price to reclaim SMA_200 before
    increasing position sizes. A weekly close above SMA_200 is the key level.
"""),

# ── 4. Crypto-Specific Trading Rules ─────────────────────────────────────────
("crypto_rules_v1", "Bitcoin Crypto-Specific Trading Rules and Risk Management", """
Bitcoin and Crypto-Specific Trading Rules

Bitcoin (BTC-USD) trades 24 hours a day, 7 days a week, unlike traditional
equity markets. This creates unique opportunities and risks that require
crypto-specific adaptations to standard technical analysis strategies.

CRYPTO 24/7 TRADING IMPLICATIONS:
  - Gaps (common in equities) don't occur on weekends — Bitcoin's price is
    continuous. Weekend moves are fully visible and tradeable.
  - Asian session (00:00-08:00 UTC) often has lower volume but can initiate moves.
  - US session (13:00-21:00 UTC) has highest volume and most reliable signals.
  - Major moves often begin on weekends when institutional desks are closed and
    retail sentiment drives price — be cautious of weekend signals.

BITCOIN VOLATILITY CONTEXT:
  - Bitcoin's average daily range is 3-5% in normal conditions, 8-15% in volatile periods.
  - A "Low Volatility" reading for BTC (daily range < 1%) is actually unusual and
    often precedes a significant move (either direction). Coil before spring.
  - Stop losses for Bitcoin should be wider than equities: 2-4% for swing trades.
  - RSI overbought (>70) and oversold (<30) signals are more extreme in crypto:
    RSI can stay above 80 for weeks in a bull market.
    RSI can stay below 20 for weeks in a bear market (crypto winters).

BITCOIN MARKET CYCLE PHASES:
  Phase 1 — Accumulation (RSI 30-45, flat SMAs, low volume):
    → Smart money accumulating. Best buying opportunity. Risk is low.
    → Strategy: Buy when RSI bounces from 30-35, MACD turns bullish.

  Phase 2 — Early Bull (RSI 45-65, rising SMAs, increasing volume):
    → Trend confirmed. Golden Cross may form. Most reliable trend following.
    → Strategy: Buy pullbacks to SMA_20 in RSI 40-55 zone. Full position size.

  Phase 3 — Late Bull / Euphoria (RSI 70-90, steep price, extreme volume):
    → FOMO zone. Risky to enter. More risk than reward.
    → Strategy: Trail stops aggressively. Take 25-50% profits. Do not add.

  Phase 4 — Distribution / Bear (RSI falling from 70, MACD crossing bearish):
    → Smart money selling. Do not buy rallies.
    → Strategy: Exit remaining longs. Move to cash. Wait for Death Cross confirmation.

ENTRY SIZING BY MARKET PHASE:
  Accumulation phase: 100% of normal position size (best risk/reward)
  Early Bull (Golden Cross): 75-100% of normal position
  Late Bull (RSI>70, price parabolic): 25% maximum, trailing stop only
  Death Cross active: 25% maximum for any long
  Full downtrend (price < all SMAs): 0% — stay in cash

BTC SUPPORT AND RESISTANCE LEVELS TO WATCH:
  Key psychological levels: $60,000 / $70,000 / $80,000 / $100,000
  Technical levels: SMA_200 (institutional benchmark)
  Previous all-time highs act as resistance until broken convincingly.

STOP LOSS AND TAKE PROFIT FOR BITCOIN:
  Conservative approach (Death Cross active):
    → Stop Loss: 1.5% below entry
    → Take Profit: 3% above entry (2:1 ratio)

  Moderate approach (mixed signals, neutral RSI):
    → Stop Loss: 2.5% below entry
    → Take Profit: 6% above entry (2.4:1 ratio)

  Aggressive approach (strong confluence, Golden Cross):
    → Stop Loss: 3% below entry
    → Take Profit: 10-15% above entry (trailing stop after 5% gain)

WHEN NOT TO TRADE (CONFIDENCE BELOW 0.55 — OUTPUT HOLD):
  1. Mixed RSI (40-60) + mixed MACD + Death Cross = too many conflicting signals
  2. Price stuck between SMA_20 and SMA_200 (range-bound = no clear direction)
  3. Volume is very low (below average) = no institutional conviction
  4. MACD histogram shrinking in both directions = momentum unclear
  5. Before major macroeconomic events (Fed rate decisions, CPI prints)
  Always output HOLD when confidence is below 0.55 — preserving capital IS a trade.
"""),

# ── 5. Trend Confirmation & Entry Timing ─────────────────────────────────────
("entry_timing_v1", "Trade Entry Timing and Confluence Confirmation Rules", """
Trade Entry Timing and Confluence — The Professional Approach

Profitable trading is not about predicting direction — it is about finding
situations where multiple independent signals CONFIRM each other, reducing
the probability of a false signal.

THE CONFLUENCE SCORING SYSTEM:
Score each of the following factors. BUY/SELL only with 3+ points.
Each factor worth 1 point:

  BUY CONFLUENCE POINTS:
  ✓ +1: RSI below 40 or coming off oversold (<30) reversal
  ✓ +1: MACD bullish crossover (line above signal)
  ✓ +1: MACD histogram positive AND expanding (momentum accelerating)
  ✓ +1: Price above SMA_20 (short-term bullish)
  ✓ +1: Price above SMA_50 (medium-term bullish)
  ✓ +1: Price above SMA_200 (long-term bull market confirmed)
  ✓ +1: SMA_50 above SMA_200 (Golden Cross — structural bull)
  ✓ +1: RSI showing bullish divergence (higher low while price lower low)
  ✓ +1: Volume above 20-day average on the entry candle

  Total BUY score → confidence:
  8-9 points: 0.90+ confidence → STRONG BUY
  6-7 points: 0.75–0.85 confidence → BUY
  4-5 points: 0.60–0.70 confidence → BUY (reduce position size by 30%)
  2-3 points: 0.50–0.55 confidence → HOLD (insufficient confluence)
  0-1 points: 0.30 confidence → HOLD

  SELL CONFLUENCE POINTS:
  ✓ +1: RSI above 65 or coming off overbought (>70) divergence
  ✓ +1: MACD bearish crossover (line below signal)
  ✓ +1: MACD histogram negative AND expanding (bearish momentum accelerating)
  ✓ +1: Price below SMA_20 (short-term bearish)
  ✓ +1: Price below SMA_50 (medium-term bearish)
  ✓ +1: Price below SMA_200 (bear market structure)
  ✓ +1: SMA_50 below SMA_200 (Death Cross — structural bear)

TIMING ENTRIES — WAITING FOR CONFIRMATION:
  Never enter at the first signal. Wait for the CONFIRMATION:

  For RSI oversold bounce:
    1. RSI drops below 30 (signal)
    2. Wait for RSI to cross BACK above 30 (confirmation — momentum has turned)
    3. Wait for the NEXT candle to close green (price confirmation)
    4. Enter at open of the 2nd green candle after RSI crosses 30

  For MACD bullish crossover:
    1. MACD line crosses signal line (signal)
    2. Histogram turns from red/negative to green/positive (confirmation)
    3. Enter at close of the first green histogram candle

  For SMA Golden Cross:
    1. SMA_50 crosses above SMA_200 (signal)
    2. Wait for price to pull back to the SMA_50 level (retest)
    3. RSI should be 40-55 on the pullback (not overbought)
    4. Enter at the SMA_50 retest if RSI confirms

FALSE SIGNAL AVOIDANCE:
  The most common trading mistake is entering on a signal that immediately reverses.
  Key filters to avoid false signals:

  1. Do not trade against the daily timeframe trend on intraday signals.
     If daily is bearish (MACD negative, below SMA_200), ignore hourly BUY signals.

  2. Do not enter if MACD histogram is FADING (shrinking) even if it's positive.
     A shrinking histogram means the recent momentum is decelerating — wait for reset.

  3. Do not enter during Death Cross unless RSI is below 35 (extreme oversold only).
     During Death Cross, the probability of BUY failures is 2x higher than normal.

  4. Volume confirmation: A signal without above-average volume is suspect.
     Institutional moves ALWAYS come with volume. Low-volume breakouts often fail.

CURRENT MARKET STRUCTURE ANALYSIS FRAMEWORK:
  Step 1: Define the long-term trend (SMA_200 position + Death/Golden Cross)
  Step 2: Define the medium-term trend (SMA_50 position + MACD above/below zero)
  Step 3: Define the short-term momentum (RSI zone + MACD crossover direction)
  Step 4: Count confluence points (target: 4+ for medium confidence, 6+ for high)
  Step 5: Execute or HOLD based on confluence score and confidence threshold

  If long-term bearish (Death Cross) AND medium-term mixed AND short-term bullish:
  → Conflicting multi-timeframe signals → HOLD → Confidence: 0.50-0.55

  If long-term bullish (Golden Cross) AND medium-term bullish AND short-term bullish:
  → Full alignment → BUY → Confidence: 0.75-0.90

  If long-term bearish AND short-term oversold (RSI<30) with MACD divergence:
  → Counter-trend trade opportunity → BUY (50% size) → Confidence: 0.60-0.70
"""),

# ── 6. BTC-USD Current Pattern Recognition ───────────────────────────────────
("btc_patterns_v1", "Bitcoin Pattern Recognition — Death Cross Recovery and Range Trading", """
Bitcoin Pattern Recognition — Death Cross Recovery and Range-Bound Conditions

RECOGNISING THE DEATH CROSS RECOVERY PATTERN:
When Bitcoin enters a Death Cross (SMA_50 crosses below SMA_200), it often goes
through a predictable sequence:

Phase A — Initial Drop (days 1-30 after Death Cross):
  - Price often drops 15-25% in the weeks following Death Cross
  - RSI falls toward 30-40 zone
  - MACD histogram expands negative (accelerating downside)
  - Action: HOLD / Cash. Do not buy this phase.

Phase B — Dead Cat Bounce (days 30-60):
  - Price rallies 10-20% after initial drop
  - RSI recovers to 50-60 zone
  - MACD crosses bullish but BELOW zero line (weak signal)
  - This rally often fails at the SMA_50 or SMA_200 (now resistance)
  - Action: Do NOT buy. This is a bull trap. Watch for MACD histogram to peak and fade.

Phase C — Retest and Consolidation:
  - Price retests lows or makes slightly lower lows
  - RSI divergence often appears (higher low on RSI despite price lower low)
  - MACD histogram: bearish but fading
  - Action: This is the highest-probability long entry in the Death Cross cycle.
  - Enter small position (25-30% of normal size). Stop: 3% below swing low.

Phase D — Recovery (if Golden Cross eventually forms):
  - Price reclaims SMA_200 with conviction (high volume, closes above)
  - SMA_50 begins curling up toward SMA_200
  - RSI establishes support at 50 (former resistance becomes support)
  - MACD crosses bullish and rises above zero line
  - Action: ADD to position. This is early bull phase. High confidence.

RANGE-BOUND BITCOIN (Price Between SMA_50 and SMA_200):
When Bitcoin price is trading ABOVE SMA_50 but BELOW SMA_200 with no clear trend:
  - This is a compression zone. Price coiling for directional move.
  - MACD near zero (line and signal close together) = indecision.
  - RSI 45-55 = neutral, no momentum edge.
  - Strategy: HOLD. Wait for a decisive break of either SMA_50 (support)
    or SMA_200 (resistance) on HIGH VOLUME.
  - A daily close above SMA_200 with RSI>60 and MACD bullish = BUY signal.
  - A daily close below SMA_50 with RSI<45 and MACD bearish = SELL signal.

VOLUME ANALYSIS FOR BITCOIN:
  Volume is the "fuel" behind price moves. Without volume, signals are unreliable.

  High Volume Breakout (above SMA_200): Very bullish. High probability continuation.
    → Confidence +0.15 on top of indicator-based confidence.

  Low Volume Breakout: Suspect. Often fails. Wait for volume to confirm.
    → Confidence -0.10. Wait for volume confirmation before acting.

  Volume Spike on Decline (panic selling): Often marks the short-term bottom.
    → Look for RSI oversold + volume spike + MACD histogram fading negative.
    → This combination = capitulation event. Counter-trend long opportunity.

  Declining Volume in Downtrend: Bears losing conviction. Reversal may be near.
    → Do NOT short declining volume drops. Cover shorts, look for long entry.

PRACTICAL SIGNAL COMBINATIONS FOR CURRENT DEATH CROSS CONDITIONS:

  BUY signal IN Death Cross (requires higher bar):
  - RSI must be below 35 (deeper oversold than normal threshold of 40)
  - MACD must show bullish crossover with histogram turning positive
  - Price must be above SMA_20 (short-term momentum aligned)
  - Confidence cap: 0.70 (Death Cross prevents high-confidence longs)
  - Position size: 30-40% of normal

  HOLD signal IN Death Cross (most common):
  - RSI 40-60: neutral zone, no clear oversold bounce
  - MACD bearish OR mixed: confirms bearish medium-term
  - Price between SMA_50 and SMA_200: compression zone
  - Confidence: 0.45-0.55 → HOLD (below 0.55 threshold)
  - Action: Preserve capital. Wait for better setup.

  SELL signal IN Death Cross (clear exit point):
  - RSI was overbought (>65) and now rolling over
  - MACD crosses bearish after bullish rally (Phase B bull trap ending)
  - Price fails at SMA_200 resistance with bearish candle pattern
  - Confidence: 0.65-0.75 -> SELL (if holding position from earlier entry)
"""),

# ── 5. Candlestick & Chart Pattern Playbook ──────────────────────────────────
("candlestick_patterns_v1", "Candlestick & Chart Pattern Recognition - Day Trader's Playbook", """
Candlestick Patterns and Chart Formations - Complete Recognition Playbook

Candlestick patterns encode battles between buyers and sellers within a single period.
For day trading and swing trading, recognizing high-probability patterns is essential.

SINGLE-CANDLE REVERSAL PATTERNS:

Hammer (bullish reversal at bottom):
  - Small body at top, long lower shadow (2x+ body), little/no upper shadow.
  - Appears after a downtrend. Signals sellers exhausted, buyers stepped in.
  - Entry: Buy above the hammer's high next candle. Stop: below hammer's low.
  - Confirmation: Next candle closes green above hammer's body.
  - Confidence: HIGH when hammer appears at prior support or SMA_200.

Shooting Star (bearish reversal at top):
  - Small body at bottom, long upper shadow (2x+ body), little/no lower shadow.
  - Appears after an uptrend. Signals buyers rejected at highs.
  - Entry (short or exit long): Below shooting star low. Stop: above its high.

Doji (indecision):
  - Open ~= Close. Long shadows both sides.
  - Neutral alone - meaningful only at trend extremes or key levels.
  - Doji at resistance after strong rally = potential reversal warning.
  - Doji at support after strong decline = potential bottom forming.

Engulfing Patterns (two-candle, highest reliability):
  Bullish Engulfing: Small red candle followed by large green candle whose body
    completely engulfs the red one's body. Strong reversal signal at bottoms.
  Bearish Engulfing: Small green followed by large red that engulfs. Top signal.
  - Requires volume on the engulfing candle (above 20-day average) for confirmation.

CHART PATTERNS (multi-candle, swing trading):

Double Bottom (W-shape):
  - Two distinct lows at similar price, separated by a moderate rally in between.
  - Confirmed when price breaks above the middle peak ("neckline").
  - Target: Measure distance from lows to neckline, project upward from breakout.
  - Stop: Below the second low.

Double Top (M-shape): Mirror of double bottom. Bearish.

Head and Shoulders (classic reversal):
  - Three peaks: middle (head) higher than left and right shoulders.
  - Neckline: line connecting the two troughs between peaks.
  - Confirmed on break below neckline with volume.
  - Target: Height from head to neckline, projected downward.

Triangles (continuation or breakout):
  Ascending: Flat resistance + rising support. Bullish breakout bias.
  Descending: Flat support + falling resistance. Bearish breakout bias.
  Symmetrical: Both converging. Trade the breakout direction, not the pattern itself.
  - Volume typically contracts inside the triangle, expands on breakout.

Flag and Pennant (continuation, short-term):
  - Sharp move ("flagpole") followed by a tight consolidation (flag or pennant).
  - Breakout in the direction of the flagpole is the entry.
  - Target: Length of flagpole projected from breakout point.

KEY RULES FOR PATTERN TRADING:
1. Never trade a pattern in isolation - confirm with volume and trend context.
2. Patterns in the direction of the higher-timeframe trend have higher win rates.
3. Failed patterns (breakout that reverses) are often more profitable in opposite direction.
4. Stop loss placement is non-negotiable - always define risk BEFORE entry.
5. Aim for minimum 2:1 reward-to-risk ratio. Skip trades that don't offer it.
"""),

# ── 6. Support, Resistance & Trendlines ──────────────────────────────────────
("support_resistance_v1", "Support, Resistance, and Trendline Trading", """
Support, Resistance, and Trendline Analysis - Professional Framework

Support and resistance are the foundation of price action trading. Every indicator
is derivative of price. S/R levels are where real buying and selling pressure lives.

DEFINITIONS:
- Support: Price level where buying pressure historically exceeds selling pressure,
  halting declines. Acts as a "floor."
- Resistance: Level where selling pressure halts rallies. Acts as a "ceiling."
- Once broken, support becomes resistance and vice versa (polarity principle).

IDENTIFYING S/R LEVELS (ranked by importance):
1. Prior swing highs/lows (most reliable)
2. Round psychological numbers ($100, $50k for BTC)
3. Prior day/week/month highs and lows
4. SMA_50, SMA_200 - dynamic support/resistance
5. Volume Profile POC (Point of Control) - high-volume price nodes
6. Fibonacci retracement levels (38.2%, 50%, 61.8%)

TRADING S/R LEVELS:

Bounce Trade (reversal at level):
  - Entry: Limit order just above support (long) or below resistance (short).
  - Stop: Below support with buffer (0.5-1% or 1 ATR).
  - Target: Next S/R level in profit direction.
  - Confirmation: Rejection candle (pin bar, engulfing) at the level.

Breakout Trade:
  - Entry: After clear break with volume expansion (1.5x 20-day avg).
  - Beware fakeouts: Wait for close above resistance (not just wick).
  - Retest entries are safer - enter on pullback to broken level that holds.
  - Stop: Back below the broken level.

TRENDLINES:
- Uptrend: Connect at least 2 higher lows. Third touch validates the line.
- Downtrend: Connect at least 2 lower highs.
- Steeper trendlines break faster; gentler ones more durable.
- Trendline break alone is NOT a reversal - requires confirmation (lower high/higher low).

MULTI-TIMEFRAME ALIGNMENT:
- Daily trend = context (bullish, bearish, range).
- 4h/1h trend = setup identification.
- 15m/5m = execution timing.
- Best setups: all timeframes aligned. Fading higher timeframes = low success rate.

COMMON MISTAKES:
- Drawing too many levels - only the CLEAN, obvious ones work.
- Ignoring volume confirmation at breakouts.
- Entering on first test - third test weakens a level (more likely to break).
- Not moving stops to breakeven after price travels to next S/R.
"""),

# ── 7. Fundamental Analysis Essentials ───────────────────────────────────────
("fundamental_analysis_v1", "Fundamental Analysis for Stock Selection", """
Fundamental Analysis - Essential Framework for Equities

Technical analysis tells you WHEN. Fundamental analysis tells you WHAT to trade.
For equities (stocks), fundamentals determine intrinsic value and long-term direction.
For crypto, fundamentals are different (on-chain, adoption, macro).

KEY VALUATION METRICS FOR STOCKS:

P/E Ratio (Price-to-Earnings):
  - P/E = Stock Price / Earnings Per Share
  - Lower = cheaper (relative to peers). Higher = growth expectations priced in.
  - S&P 500 historical average: 15-20. Tech/growth often 25-40+.
  - Context matters: AAPL at P/E 30 is normal; a utility at P/E 30 is expensive.

PEG Ratio (Price/Earnings to Growth):
  - PEG = P/E / Annual EPS Growth Rate (%).
  - PEG < 1.0 = potentially undervalued relative to growth.
  - PEG > 2.0 = growth already priced in or overvalued.

Price-to-Sales (P/S):
  - Useful for unprofitable growth companies where P/E is meaningless.
  - P/S under 2 = value. P/S over 10 = pricey unless high growth (20%+).

Free Cash Flow (FCF) and FCF Yield:
  - FCF Yield = FCF / Market Cap. Above 5% is attractive.
  - FCF is harder to manipulate than earnings. Preferred by institutional investors.

Debt-to-Equity:
  - Below 1.0 = healthy. Above 2.0 = risky in rising rate environments.

QUARTERLY EARNINGS - THE SINGLE BIGGEST CATALYST:
- Earnings beats drive rallies; misses drive crashes.
- What matters more than the number: FORWARD GUIDANCE.
- Reactions are often counter-intuitive: beat + lower guidance = drop.
- Post-earnings drift: stocks that gap up on strong earnings often continue for weeks.

MACRO CONTEXT (affects ALL stocks):

Federal Reserve Interest Rates:
  - Rising rates compress valuations (higher discount rates). Bearish for growth stocks.
  - Falling rates expand valuations. Bullish for growth, tech, crypto.

CPI / Inflation Data:
  - Hot CPI = more rate hikes = bearish risk assets.
  - Cooling CPI = dovish Fed pivot = bullish equities.

Employment (NFP, Unemployment Rate):
  - Stronger than expected = Fed stays hawkish = short-term bearish.
  - Weaker = rate cuts priced in = bullish.

SECTOR ROTATION:
- Early cycle (recession end): Financials, Consumer Discretionary
- Mid cycle (expansion): Technology, Industrials
- Late cycle (peak): Energy, Materials
- Recession: Utilities, Consumer Staples, Healthcare (defensive)

CRYPTO FUNDAMENTALS (different framework):
- On-chain: Active addresses, exchange flows, stablecoin supply.
- Halving cycles (Bitcoin): Historically bullish 6-18 months post-halving.
- Regulatory news: SEC actions, ETF approvals = major catalysts.
- Network fees / activity: Measures real demand (Ethereum gas, BTC fees).

COMBINING FUNDAMENTAL + TECHNICAL:
1. Use fundamentals to build a watchlist of quality names.
2. Use technicals to time entries and exits.
3. Never fight the trend - even a "cheap" stock in a downtrend keeps falling.
4. Catalyst trading: Enter before earnings ONLY if setup is clean; otherwise wait.
"""),

# ── 8. Day Trading Rules & Risk Management ───────────────────────────────────
("day_trading_rules_v1", "Day Trading Rules, Position Sizing & Psychology", """
Day Trading Playbook - Rules, Position Sizing, and Psychology

Day trading has the lowest win rate among retail activities (80%+ lose). The 20%
that succeed follow strict rules. This is not about prediction - it's about process.

POSITION SIZING - THE MOST IMPORTANT RULE:

The 1% Rule:
  - Never risk more than 1% of total capital on any single trade.
  - Position Size = (Account * 0.01) / (Entry Price - Stop Loss Price).
  - Example: $10,000 account, entry $100, stop $98 (2% risk per share).
    Position = ($10,000 * 0.01) / ($100 - $98) = $100 / $2 = 50 shares.

Kelly Criterion (advanced):
  - Optimal position size = (Win Rate * Avg Win) - (Loss Rate * Avg Loss) / Avg Win.
  - Most traders use "half Kelly" to reduce volatility.

DRAWDOWN LIMITS:
- Daily loss limit: -3% of account. Hit it -> stop trading for the day.
- Weekly loss limit: -6%. Hit it -> stop trading for the week, review mistakes.
- Monthly loss limit: -10%. Hit it -> reduce size by 50% until recovered.

ENTRY RULES (all must be true):
1. Setup matches a pre-defined playbook pattern (no improvisation).
2. Higher-timeframe trend aligns with trade direction.
3. Defined stop-loss BEFORE entry (never after).
4. Reward-to-risk ratio >= 2:1.
5. Position size pre-calculated and within account rules.
6. Market open long enough (avoid first/last 15 min unless specific setup).

EXIT RULES:

Stop Loss:
  - ALWAYS placed at entry. Never widen a stop to "give it room."
  - Mental stops are not stops. Use limit or stop orders.
  - Trail to breakeven after 1R profit (risk amount) reached.

Take Profit:
  - Scale out: 50% at 1R (cover risk), 25% at 2R, 25% runner with trailing stop.
  - Time-based exit: If trade doesn't work within expected timeframe (1 hour for
    scalp, 1 day for swing), exit regardless.

DAILY WORKFLOW:
Pre-market (1 hour before open):
  - Review overnight news and futures.
  - Mark key levels on watchlist (S/R, prior day H/L).
  - Identify catalysts (earnings, economic data).
  - Define 2-3 setups for the day. Only trade those.

Market hours:
  - First 15 minutes: Observe, don't trade (unless opening range strategy).
  - 10:00-11:30 ET: Primary setup window for momentum trades.
  - 11:30-14:00 ET: Chop / lunch. Lowest volume. Avoid unless trend day.
  - 14:00-16:00 ET: Afternoon trend resumption. Good for continuation trades.

Post-market:
  - Journal EVERY trade: entry reason, exit reason, emotion, lesson.
  - Weekly review: Win rate, average R, top 3 mistakes.

PSYCHOLOGY RULES:
- Revenge trading = account death. After a loss, walk away for 15 minutes minimum.
- FOMO trading = entering without setup. Just say no.
- Overconfidence after wins is more dangerous than fear after losses.
- The goal is NOT to be right. The goal is to make money when right and
  lose little when wrong. Process over outcome.

KEY MINDSET SHIFTS:
- Think in probabilities, not predictions. Every setup has a ~50-60% win rate max.
- Accept losses as cost of doing business. They are not failures.
- Consistency beats brilliance. Boring, repeated execution of edge = profits.
"""),

]

# ─────────────────────────────────────────────────────────────────────────────
# Ingestion engine
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
            model_name=settings.embedding_model,
            trust_remote_code=True,
        )

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total_added = 0
    total_skipped = 0

    for source_id, title, text in STRATEGY_LIBRARY:
        chunks = splitter.split_text(text.strip())
        ids        = [_make_chunk_id(source_id, i, c) for i, c in enumerate(chunks)]
        documents  = chunks
        metadatas  = [
            {
                "source":       "seed",
                "source_id":    source_id,
                "title":        title,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "chunk_size":   len(c),
            }
            for i, c in enumerate(chunks)
        ]

        # Check existing
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
            f"→ {len(new_ids)} chunks added, {skipped} already existed"
        )

    count = collection.count()
    logger.info(
        f"\n{'='*60}\n"
        f"  Seed ingestion complete.\n"
        f"  Added: {total_added} chunks\n"
        f"  Skipped (already existed): {total_skipped} chunks\n"
        f"  Total collection size: {count} chunks\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  BotTrade — Seeding Trading Strategy Knowledge Base")
    print("=" * 60)
    ingest_all()
