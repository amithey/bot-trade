"""Unsupervised regime classifier.

KMeans on a carefully-labeled feature subspace (momentum + volatility)
produces 4 clusters that map cleanly onto intuitive regimes:

    - TRENDING_UP      (high +momentum, moderate vol)
    - TRENDING_DOWN    (high -momentum, moderate vol)
    - RANGING          (near-zero momentum, low vol)
    - VOLATILE         (any momentum, high vol)

The mapping cluster_idx -> label is computed once at fit time by
inspecting the cluster centroids — we don't hard-code which KMeans label
is which, so it's stable across retrains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ml import features as _feat
from ml import model_store
from utils.logger import get_logger

logger = get_logger(__name__)

RegimeLabel = Literal["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"]

# Subset of features used *only* for regime clustering — we want the
# model to focus on momentum + volatility, not mean-reversion noise.
_REGIME_COLS = ["ret_5", "ret_20", "vol_20", "atr_pct", "sma50_vs_sma200"]


@dataclass
class RegimeResult:
    label: RegimeLabel
    confidence: float          # softmax-ish, 1.0 = dead-center of cluster
    distances: dict[str, float]


class RegimeClassifier:
    """4-regime KMeans model, persisted under ``regime_<ticker>``."""

    _N_CLUSTERS = 4

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker.upper()
        self._scaler: StandardScaler | None = None
        self._km: KMeans | None = None
        self._mapping: dict[int, RegimeLabel] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _store_name(self) -> str:
        return f"regime_{self.ticker}"

    def _load(self) -> None:
        blob = model_store.load(self._store_name())
        if blob is None:
            return
        self._scaler = blob.get("scaler")
        self._km = blob.get("km")
        self._mapping = blob.get("mapping") or {}

    def save(self) -> None:
        model_store.save(self._store_name(), {
            "scaler":  self._scaler,
            "km":      self._km,
            "mapping": self._mapping,
        })

    @property
    def is_fitted(self) -> bool:
        return self._km is not None and self._scaler is not None

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "RegimeClassifier":
        """Fit on the *enriched* OHLCV frame."""
        feats = _feat.extract_features(df)
        X = feats[_REGIME_COLS].to_numpy(dtype=float)
        # Skip warm-up rows where momentum is still 0.
        X = X[20:]
        if len(X) < 50:
            raise ValueError(
                f"regime fit needs >=50 rows of features, got {len(X)}"
            )

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._km = KMeans(n_clusters=self._N_CLUSTERS, n_init=10,
                          random_state=42).fit(Xs)
        self._mapping = self._label_clusters(self._km.cluster_centers_)
        self.save()
        logger.info(
            f"[regime] {self.ticker}: fitted 4 clusters on {len(X)} rows, "
            f"mapping={self._mapping}"
        )
        return self

    def _label_clusters(self, centers: np.ndarray) -> dict[int, RegimeLabel]:
        """Assign intuitive labels by inspecting cluster centroids.

        Centers are in the *scaled* space; sign still carries meaning.
        Columns (scaled): [ret_5, ret_20, vol_20, atr_pct, sma50_vs_sma200]
        """
        mapping: dict[int, RegimeLabel] = {}
        used: set[RegimeLabel] = set()

        # Score each cluster on each regime signature; assign greedily.
        candidates = []
        for i, c in enumerate(centers):
            mom = 0.5 * c[0] + 0.5 * c[1] + 0.5 * c[4]   # ret_5 + ret_20 + SMA slope
            vol = 0.5 * c[2] + 0.5 * c[3]                # vol_20 + atr
            candidates.append((i, mom, vol))

        # Highest vol first -> VOLATILE
        v_idx = max(candidates, key=lambda t: t[2])[0]
        mapping[v_idx] = "VOLATILE"
        used.add("VOLATILE")

        remaining = [c for c in candidates if c[0] not in mapping]
        # Highest positive momentum -> TRENDING_UP
        up_idx = max(remaining, key=lambda t: t[1])[0]
        mapping[up_idx] = "TRENDING_UP"
        remaining = [c for c in remaining if c[0] not in mapping]

        # Most negative momentum -> TRENDING_DOWN
        dn_idx = min(remaining, key=lambda t: t[1])[0]
        mapping[dn_idx] = "TRENDING_DOWN"
        remaining = [c for c in remaining if c[0] not in mapping]

        # Whatever is left -> RANGING
        if remaining:
            mapping[remaining[0][0]] = "RANGING"

        return mapping

    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame) -> RegimeResult:
        """Predict the regime of the *latest* row of ``df``."""
        if not self.is_fitted:
            # Fit on whatever data we have on first call.
            self.fit(df)

        feats = _feat.extract_features(df).iloc[-1:][_REGIME_COLS]
        Xs = self._scaler.transform(feats.to_numpy(dtype=float))
        centers = self._km.cluster_centers_
        dists = np.linalg.norm(centers - Xs, axis=1)
        idx = int(np.argmin(dists))
        label = self._mapping.get(idx, "RANGING")

        # Confidence = inverse-distance softmax, temperature = mean(d)
        temp = dists.mean() or 1.0
        logits = -dists / temp
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        return RegimeResult(
            label=label,
            confidence=float(probs[idx]),
            distances={
                self._mapping.get(i, f"cluster_{i}"): float(d)
                for i, d in enumerate(dists)
            },
        )
