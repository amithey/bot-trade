"""
Seed Knowledge — Direct ChromaDB Ingestion (Bypass YouTube)
=========================================================
This script directly populates the ChromaDB 'trading_strategies' collection 
with 4 highly detailed, text-based technical analysis strategies for Bitcoin.

Usage:
    python seed_knowledge.py
"""

from rag.ownership import SHARED_OWNER
from utils.hf_quiet import configure_quiet_hf, quiet_model_load

configure_quiet_hf()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
import hashlib

# ---------------------------------------------------------------------------
# 4 Detailed Trading Strategies for Bitcoin
# ---------------------------------------------------------------------------

STRATEGIES = [
    {
        "id": "rsi_strategy_v1",
        "title": "RSI Oversold/Overbought Playbook for Bitcoin",
        "content": """
RSI (Relative Strength Index) Strategy for Bitcoin

The Relative Strength Index (RSI) is a momentum oscillator that measures the speed
and change of price movements on a scale of 0 to 100.

CORE RULES:
- RSI above 70: Overbought territory. Price has risen rapidly and may be due for a
  pullback or consolidation. In strong Bitcoin bull markets, RSI can stay above 70 
  for weeks; look for bearish divergence (price makes higher high, RSI makes lower high) 
  as the actual exit signal.
- RSI below 30: Oversold territory. Price has fallen sharply. Potential reversal zone.
  In Bitcoin, RSI below 30 often indicates a capitulation event and a high-probability
  mean-reversion bounce.

ENTRY RULES (BUY):
1. RSI falls below 30 and then crosses back above 30 (oversold exit).
2. Volume on the reversal candle is higher than the previous 5 candles.
3. Price is above the 200-period SMA (long-term trend confirmation).

EXIT RULES (SELL):
1. RSI reaches 70 or higher.
2. Bearish divergence appears on the 1-hour or 4-hour timeframe.
3. Price closes below the 20-period SMA.
"""
    },
    {
        "id": "macd_strategy_v1",
        "title": "MACD Bullish/Bearish Crossover Strategy",
        "content": """
MACD (Moving Average Convergence Divergence) Strategy

MACD is a trend-following momentum indicator that shows the relationship between 
two moving averages of an asset’s price.

CORE COMPONENTS:
- MACD Line: 12-day EMA minus 26-day EMA.
- Signal Line: 9-day EMA of the MACD Line.
- Histogram: MACD Line minus Signal Line.

BULLISH SIGNAL (BUY):
- MACD Line crosses ABOVE the Signal Line.
- Histogram turns from negative to positive.
- Strongest when the crossover happens below the zero line (oversold momentum shift).

BEARISH SIGNAL (SELL):
- MACD Line crosses BELOW the Signal Line.
- Histogram turns from positive to negative.
- Strongest when the crossover happens above the zero line (overbought momentum shift).

CONFIRMATION:
Always wait for the candle to CLOSE before acting on a crossover. False 'pokes'
across the signal line are common in low-volatility environments.
"""
    },
    {
        "id": "sma_cross_v1",
        "title": "SMA 50/200 Death/Golden Cross Strategy",
        "content": """
Moving Average Structural Strategy (Golden & Death Cross)

This strategy uses the 50-period SMA and 200-period SMA to define long-term 
market structure for Bitcoin.

GOLDEN CROSS (BULLISH):
- The 50-period SMA crosses ABOVE the 200-period SMA.
- This indicates the medium-term trend is accelerating faster than the long-term trend.
- Historical Significance: Often marks the beginning of major multi-month bull runs.
- Strategy: Look for long entries on pullbacks to the 50-SMA after the cross occurs.

DEATH CROSS (BEARISH):
- The 50-period SMA crosses BELOW the 200-period SMA.
- This indicates a structural shift to a bear market regime.
- Historical Significance: Precedes major Bitcoin 'winters' and deep corrections.
- Strategy: Reduce position sizes or move to cash. Long entries require extreme
  oversold RSI (<25) to justify the risk against the structural downtrend.
"""
    },
    {
        "id": "price_action_vol_v1",
        "title": "Price Action & Volume Confirmation",
        "content": """
Price Action and Volume Confirmation Rules

Indicators alone can be deceptive. Professional Bitcoin traders use price action 
and volume to confirm what the indicators are suggesting.

VOLUME RULES:
- Breakout Confirmation: A price break above resistance (like the 200-SMA) is 
  only valid if accompanied by a significant spike in volume (at least 50% 
  above the 20-day average).
- Low Volume Rallies: If price is rising but volume is falling, the move is 
  unsupported by 'smart money' and is likely a bull trap.

CANDLESTICK CONFIRMATION:
- Bullish Engulfing: A large green candle that completely 'engulfs' the previous 
  red candle at a support level.
- Hammer / Pin Bar: A candle with a long lower wick, showing that buyers 
  rejected lower prices.

ENTRY CONFLUENCE:
Only enter a trade when at least THREE signals align. For example:
1. RSI is coming off an oversold bounce.
2. MACD shows a bullish crossover.
3. Price confirms with a Hammer candle on high volume.
"""
    }
]

def seed():
    print(f"Connecting to ChromaDB at {settings.chroma_persist_dir}...")
    
    # Initialize Embedding Function
    with quiet_model_load():
        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    
    # Initialize Chroma Client
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    
    # Get or Create Collection
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Splitter for text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print(f"Seeding {len(STRATEGIES)} strategies into '{settings.chroma_collection_name}'...")
    
    for strategy in STRATEGIES:
        chunks = splitter.split_text(strategy["content"].strip())
        
        ids = [f"{strategy['id']}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": "seed_script",
                # Operator-curated: every account may retrieve it.
                "owner": SHARED_OWNER,
                "title": strategy["title"],
                "strategy_id": strategy["id"],
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        
        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"  - Added '{strategy['title']}' ({len(chunks)} chunks)")

    print(f"\nSuccess! Collection now has {collection.count()} documents.")

if __name__ == "__main__":
    seed()
