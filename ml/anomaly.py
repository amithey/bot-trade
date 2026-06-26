"""Anomaly detector — Isolation Forest on returns + volume.

Fits on ~500 recent bars and flags the current bar as anomalous if its
features land in the tail of the learned distribution. Useful to veto
entries during suspicious spikes (earnings leak, fat-finger, halt
aftermath).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml import features as _feat
from ml import model_store
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float       # higher = more anomalous (normalized ~[-1, 1])
    severity: str      # NONE / MILD / STRONG / EXTREME


_COLS = ["ret_1", "ret_5", "vol_20", "atr_pct", "volume_zscore", "range_pct"]


class AnomalyDetector:
    """IsolationForest wrapper. Persisted under ``anomaly_<ticker>``."""

    def __init__(self, ticker: str, contamination: float = 0.05) -> None:
        self.ticker = ticker.upper()
        self.contamination = contamination
        self._model: IsolationForest | None = None
        self._load()

    def _store_name(self) -> str:
        return f"anomaly_{self.ticker}"

    def _load(self) -> None:
        blob = model_store.load(self._store_name())
        if blob is not None:
            self._model = blob.get("model")
            self.contamination = blob.get("contamination", self.contamination)

    def save(self) -> None:
        model_store.save(self._store_name(), {
            "model": self._model,
            "contamination": self.contamination,
        })

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        feats = _feat.extract_features(df)[_COLS].iloc[30:]
        if len(feats) < 50:
            raise ValueError(f"anomaly fit needs >=50 rows, got {len(feats)}")
        self._model = IsolationForest(
            n_estimators=120,
            contamination=self.contamination,
            random_state=42,
        ).fit(feats.to_numpy(dtype=float))
        self.save()
        logger.info(f"[anomaly] {self.ticker}: fitted on {len(feats)} rows")
        return self

    def predict(self, df: pd.DataFrame) -> AnomalyResult:
        if not self.is_fitted:
            self.fit(df)
        row = _feat.extract_features(df)[_COLS].iloc[-1:].to_numpy(dtype=float)
        # Higher score_samples => more NORMAL. We flip the sign so that
        # a higher reported score means MORE anomalous (more intuitive).
        raw = float(self._model.score_samples(row)[0])
        pred = int(self._model.predict(row)[0])   # -1 = anomaly, +1 = normal

        # Map raw to a severity bucket by quantiles of recent history.
        feats = _feat.extract_features(df)[_COLS].iloc[30:]
        bg = self._model.score_samples(feats.to_numpy(dtype=float))
        thresh_mild = np.quantile(bg, 0.05)
        thresh_strong = np.quantile(bg, 0.01)
        thresh_extreme = np.quantile(bg, 0.002)

        if raw <= thresh_extreme:
            severity = "EXTREME"
        elif raw <= thresh_strong:
            severity = "STRONG"
        elif raw <= thresh_mild:
            severity = "MILD"
        else:
            severity = "NONE"

        return AnomalyResult(
            is_anomaly=(pred == -1),
            score=float(-raw),   # flip so higher=more anomalous
            severity=severity,
        )
