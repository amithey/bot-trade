"""BotTrade ML package — machine-learning enhancements for the AI engine.

Modules
-------
features          Shared feature-extraction helpers (OHLCV → np.ndarray)
regime            Unsupervised regime classifier (KMeans on feature space)
anomaly           Anomaly detector (IsolationForest on returns+volume)
pattern_detector  Chart-pattern scorer (Head&Shoulders, Double Bottom, etc.)
trade_journal_ml  Self-learning outcome classifier (RandomForest)
forecaster        Short-horizon return forecaster (EWMA-drift + volatility band)
model_store       Pickle helpers used by all of the above
"""
from ml.features import extract_features, FEATURE_NAMES  # re-export for convenience

__all__ = ["extract_features", "FEATURE_NAMES"]
