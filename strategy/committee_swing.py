"""Hourly COMMITTEE research adapter with explicit completed-bucket rules."""
import pandas as pd


def completed_hourly_bars(df):
    """Input and output use bar-open timestamps; omit incomplete UTC hours.

    US equity sessions start on the half hour, so their boundary half-hours
    are deliberately excluded. Missing source candles invalidate that hour.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Need unique chronological five-minute candles")
    if (df.index.asi8 % pd.Timedelta(minutes=5).value != 0).any():
        raise ValueError("Expected aligned five-minute candles")
    grouped = df.resample("1h", closed="left", label="left")
    hourly = grouped.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return hourly.loc[grouped.Close.count() == 12]


def swing_inputs(df):
    """Hourly signals, five-minute execution and protection on EVERY candle."""
    from strategy.committee import IndicatorCommittee
    from strategy.research import entry_features
    from strategy.trend_controls import trend_control_features
    hourly = completed_hourly_bars(df)
    available = hourly.index + pd.Timedelta(hours=1)
    signal_times = df.index + pd.Timedelta(minutes=5)

    def align(frame):
        frame = frame.copy()
        frame.index = available
        result = frame.reindex(signal_times, method="ffill")
        result.index = df.index
        return result

    features = align(entry_features(hourly))
    # Only one entry opportunity per freshly completed hour. A signal
    # preceding an overnight/data gap cannot open exposure next session.
    next_open = df.index.to_series().shift(-1)
    features["eligible"] = features.eligible.eq(True) & signal_times.isin(available) & next_open.eq(pd.Series(signal_times, index=df.index))
    return features, align(IndicatorCommittee().vote_matrix(hourly)).fillna(0), align(trend_control_features(hourly))
