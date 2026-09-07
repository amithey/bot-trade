"""Chart builders independent of Streamlit session state."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.theme import BG_DEEP, C_BUY, C_SELL, CYAN, GRID, TEXT, TEXT_DIM, register_chart_theme


def market_chart(df, ticker, trades=(), *, overlays=True, show_trades=True):
    """Candles, volume, RSI and MACD with a persistent pan/zoom viewport."""
    register_chart_theme()
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.56, 0.12, 0.16, 0.16],
        subplot_titles=("", "Volume", "RSI-14", "MACD"),
    )
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    fig.add_trace(go.Candlestick(
        x=idx, open=df["Open"], high=df["High"], low=df["Low"],
        close=df["Close"], name="Price",
        increasing_line_color=C_BUY, decreasing_line_color=C_SELL,
        increasing_fillcolor=C_BUY, decreasing_fillcolor=C_SELL,
        showlegend=False,
    ), row=1, col=1)

    # Bollinger band envelope (soft fill behind the SMAs)
    if overlays and "BB_Upper_20" in df.columns and "BB_Lower_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=idx, y=df["BB_Upper_20"], name="BB±2σ",
            line=dict(color="rgba(0,183,255,0.28)", width=0.8),
            legendgroup="bb", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=df["BB_Lower_20"], name="BB lower",
            line=dict(color="rgba(0,183,255,0.28)", width=0.8),
            fill="tonexty", fillcolor="rgba(0,183,255,0.045)",
            legendgroup="bb", showlegend=False,
        ), row=1, col=1)

    for col_name, clr, lbl, w in [
        ("SMA_20",  "#f39c12", "SMA20", 1.1),
        ("SMA_50",  "#3498db", "SMA50", 1.1),
        ("SMA_200", "#e74c3c", "SMA200", 1.6),
    ]:
        if overlays and col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=idx, y=df[col_name], name=lbl,
                line=dict(color=clr, width=w), opacity=0.85,
            ), row=1, col=1)

    # Trade markers for this ticker
    import pandas as pd
    # Prevent old fills stretching the current chart's visible time window.
    lower = pd.Timestamp(df.index[0])
    upper = pd.Timestamp(df.index[-1])
    step = df.index[-1] - df.index[-2] if len(df) > 1 else pd.Timedelta(days=1)
    lower = lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC")
    upper = (upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC")) + step
    def visible_trade(trade):
        stamp = pd.Timestamp(trade.executed_at)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        return lower <= stamp < upper
    relevant = [t for t in trades if show_trades and t.ticker == ticker and visible_trade(t)]
    buys  = [t for t in relevant if t.action == "BUY"]
    sells = [t for t in relevant if "SELL" in t.action or t.action == "FORCE_CLOSE"]
    if buys:
        fig.add_trace(go.Scatter(
            x=[t.executed_at for t in buys],
            y=[t.price for t in buys],
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=16, color=C_BUY,
                        line=dict(width=2, color="white")),
            hovertemplate="BUY $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if sells:
        fig.add_trace(go.Scatter(
            x=[t.executed_at for t in sells],
            y=[t.price for t in sells],
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=16, color=C_SELL,
                        line=dict(width=2, color="white")),
            hovertemplate="SELL $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # Volume bars colored by candle direction
    if "Volume" in df.columns:
        vol_colors = [C_BUY if c >= o else C_SELL
                      for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=idx, y=df["Volume"], name="Volume", showlegend=False,
            marker_color=vol_colors, opacity=0.5,
        ), row=2, col=1)

    rsi_col = next((c for c in df.columns if c.startswith("RSI_")), None)
    if rsi_col:
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,71,87,0.07)",
                      line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,212,170,0.07)",
                      line_width=0, row=3, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=df[rsi_col], name="RSI",
            line=dict(color="#ab47bc", width=1.4),
        ), row=3, col=1)

    if "MACD" in df.columns:
        hist = df["MACD_Histogram"]
        fig.add_trace(go.Bar(
            x=idx, y=hist, name="Hist", showlegend=False,
            marker_color=[C_BUY if v >= 0 else C_SELL for v in hist],
            opacity=0.65,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=df["MACD"], name="MACD",
            line=dict(color=CYAN, width=1.1),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#f39c12", width=1.1),
        ), row=4, col=1)

    is_crypto = "-USD" in ticker.upper()
    fig.update_layout(
        template="bottrade", height=620,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_DEEP,
        margin=dict(l=8, r=58, t=28, b=16),
        hovermode="x unified", dragmode="pan",
        uirevision=ticker,
        newshape=dict(line_color=CYAN, line_width=1.5),
        font=dict(family="Inter, Segoe UI, sans-serif", size=11, color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(rangeslider_visible=False,
                   rangebreaks=[] if is_crypto else [dict(bounds=["sat","mon"])],
                   showgrid=True, gridcolor=GRID),
        yaxis=dict(side="right", showgrid=True, gridcolor=GRID),
        yaxis2=dict(side="right", showgrid=False),
        yaxis3=dict(side="right", range=[0, 100], showgrid=True,
                    gridcolor=GRID),
        yaxis4=dict(side="right", showgrid=True, gridcolor=GRID),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 10
        ann.font.color = TEXT_DIM
    fig.update_xaxes(showspikes=True, spikecolor=TEXT_DIM, spikethickness=1, spikemode="across")
    return fig
