"""Analytics package — performance metrics, equity curves, drawdown,
trade attribution, and holdings-risk computations for the BotTrade
virtual portfolio."""
from analytics.performance import (
    PerformanceMetrics,
    compute_metrics,
    equity_curve,
    drawdown_series,
    benchmark_equity,
)
from analytics.attribution import (
    pnl_by_ticker,
    pnl_by_weekday,
    pnl_by_hour,
    win_rate_by_ticker,
    trade_durations,
)
from analytics.holdings_risk import (
    concentration,
    correlation_matrix,
)

__all__ = [
    "PerformanceMetrics",
    "compute_metrics",
    "equity_curve",
    "drawdown_series",
    "benchmark_equity",
    "pnl_by_ticker",
    "pnl_by_weekday",
    "pnl_by_hour",
    "win_rate_by_ticker",
    "trade_durations",
    "concentration",
    "correlation_matrix",
]
