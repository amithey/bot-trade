"""Chart extraction must preserve market values and trade overlays."""
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from dashboard.charts import market_chart
from dashboard.theme import BG_DEEP, C_BUY, C_SELL


def test_chart_preserves_candles_and_trade_markers():
    df = pd.DataFrame(
        {"Open": [100, 102, 101], "High": [103, 104, 105],
         "Low": [99, 100, 100], "Close": [102, 101, 104],
         "Volume": [1000, 1200, 1500]},
        index=pd.date_range("2026-01-05", periods=3),
    )
    trade = SimpleNamespace(ticker="AAPL", action="BUY", price=102,
                            executed_at=datetime(2026, 1, 5))
    figure = market_chart(df, "AAPL", [trade])
    candles = figure.data[0]
    assert candles.type == "candlestick"
    assert list(candles.close) == [102, 101, 104]
    assert candles.increasing.fillcolor == C_BUY
    assert candles.decreasing.fillcolor == C_SELL
    marker = next(trace for trace in figure.data if trace.name == "BUY")
    assert list(marker.y) == [102]
    volume = next(trace for trace in figure.data if trace.name == "Volume")
    assert volume.yaxis == "y2"
    assert figure.layout.uirevision == "AAPL"
    assert figure.layout.plot_bgcolor == BG_DEEP


def test_chart_keeps_indicator_panels_and_crypto_weekends():
    df = pd.DataFrame(
        {"Open": [1, 2], "High": [2, 3], "Low": [.5, 1], "Close": [2, 2.5],
         "RSI_14": [50, 55], "MACD": [.1, .2],
         "MACD_Signal": [.05, .1], "MACD_Histogram": [.05, .1]},
        index=pd.date_range("2026-01-03", periods=2, tz="UTC"),
    )
    figure = market_chart(df, "BTC-USD")
    rsi = next(trace for trace in figure.data if trace.name == "RSI")
    macd = next(trace for trace in figure.data if trace.name == "MACD")
    assert rsi.yaxis == "y3"
    assert macd.yaxis == "y4"
    assert not figure.layout.xaxis.rangebreaks
    assert df.index.tz is not None  # Rendering must not mutate source data.
