"""Self-learning trade journal.

Every time a trade closes (SELL / FORCE_CLOSE with realized_pnl), we
can reconstruct the feature vector that was present at entry and record
whether the trade ultimately won or lost. Over time this builds a
labeled dataset: (features_at_entry, won?) → 1/0.

A RandomForest classifier trained on that history tells us, for any
*new* candidate entry, how likely it is to produce a winning trade
*given this bot's recent track record*.

The model is personal — what works for one user's watchlist + risk
profile may not work for another's.

Training data
-------------
``portfolio/virtual_account.py`` persists trades as ``TradeRecord``
objects. We reconstruct entries by walking the log in chronological
order, pairing BUYs with matching SELLs. For each pair we look up the
bar that was *closest* to the executed_at of the BUY and compute the
feature vector from that bar's context.

Persistence
-----------
Model + fitted encoder are saved under ``journal_ml``.
A small JSON history of training runs is kept alongside.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml import features as _feat
from ml import model_store
from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "journal_ml"
_METADATA_PATH = Path("data/ml_models/journal_ml_meta.json")


@dataclass
class WinProbability:
    prob_win: float           # 0.0 .. 1.0 — P(trade wins | features)
    n_train: int              # how many historical trades the model has seen
    confidence_band: str      # LOW / MEDIUM / HIGH, based on n_train


class TradeJournalML:
    """Personal-edge classifier. Call :meth:`fit_from_portfolio` after a
    trade closes; call :meth:`predict` before a new entry."""

    def __init__(self) -> None:
        self._model: RandomForestClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._n_train: int = 0
        self._last_fit: str | None = None
        self._load()

    def _load(self) -> None:
        blob = model_store.load(_MODEL_NAME)
        if blob is None:
            return
        self._model = blob.get("model")
        self._scaler = blob.get("scaler")
        self._n_train = int(blob.get("n_train") or 0)
        self._last_fit = blob.get("last_fit")

    def save(self) -> None:
        model_store.save(_MODEL_NAME, {
            "model":    self._model,
            "scaler":   self._scaler,
            "n_train":  self._n_train,
            "last_fit": self._last_fit,
        })
        _METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        _METADATA_PATH.write_text(json.dumps({
            "n_train": self._n_train,
            "last_fit": self._last_fit,
        }, indent=2), encoding="utf-8")

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def n_train(self) -> int:
        return self._n_train

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pair_trades(trade_log: list[Any]) -> list[tuple[Any, Any]]:
        """Walk the log and pair each BUY with its next SELL on the same
        ticker. Returns list of (buy_record, sell_record)."""
        pairs: list[tuple[Any, Any]] = []
        open_buys: dict[str, Any] = {}   # ticker -> BUY record
        for rec in trade_log:
            action = getattr(rec, "action", None) or (rec.get("action") if isinstance(rec, dict) else None)
            ticker = getattr(rec, "ticker", None) or (rec.get("ticker") if isinstance(rec, dict) else None)
            if action == "BUY":
                open_buys[ticker] = rec
            elif action in ("SELL", "FORCE_CLOSE") and ticker in open_buys:
                pairs.append((open_buys.pop(ticker), rec))
        return pairs

    @staticmethod
    def _rec_time(rec: Any) -> datetime | None:
        v = getattr(rec, "executed_at", None) or (rec.get("executed_at") if isinstance(rec, dict) else None)
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _rec_pnl(rec: Any) -> float:
        v = getattr(rec, "realized_pnl", None) or (rec.get("realized_pnl") if isinstance(rec, dict) else None)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def fit_from_portfolio(
        self,
        trade_log: list[Any],
        fetch_history: "callable",
    ) -> int:
        """Rebuild the training set from the portfolio's trade log and refit.

        Parameters
        ----------
        trade_log
            Iterable of ``TradeRecord`` (or dicts).
        fetch_history
            ``callable(ticker, end_dt) -> pd.DataFrame`` — returns an
            enriched OHLCV frame ending at or before ``end_dt``.
            The caller is responsible for caching / rate-limiting.

        Returns
        -------
        Number of (X, y) training pairs used.
        """
        pairs = self._pair_trades(list(trade_log))
        X_rows: list[np.ndarray] = []
        y_rows: list[int] = []

        for buy, sell in pairs:
            t_entry = self._rec_time(buy)
            ticker = getattr(buy, "ticker", None) or buy.get("ticker")
            pnl = self._rec_pnl(sell)
            if t_entry is None or not ticker:
                continue
            try:
                df = fetch_history(ticker, t_entry)
            except Exception:   # noqa: BLE001
                logger.exception(f"[journal_ml] fetch_history failed for {ticker}@{t_entry}")
                continue
            if df is None or df.empty:
                continue
            feats = _feat.extract_latest(df)
            if feats is None or len(feats) == 0:
                continue
            X_rows.append(feats)
            y_rows.append(1 if pnl > 0 else 0)

        if len(X_rows) < 10:
            logger.info(
                f"[journal_ml] only {len(X_rows)} labeled trades — need >=10 to fit. "
                f"Skipping."
            )
            self._n_train = len(X_rows)
            self._last_fit = None
            self.save()
            return len(X_rows)

        X = np.vstack(X_rows)
        y = np.asarray(y_rows, dtype=int)

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ).fit(Xs, y)
        self._n_train = len(X)
        self._last_fit = datetime.utcnow().isoformat(timespec="seconds")
        self.save()
        wins = int(y.sum())
        logger.info(
            f"[journal_ml] fitted on {len(X)} trades "
            f"({wins} wins / {len(X)-wins} losses)"
        )
        return len(X)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(self, df_enriched: pd.DataFrame) -> WinProbability:
        """Return the predicted probability of win for a new candidate entry.

        If the model has never been fitted (or was fitted on < 10 trades),
        returns a neutral 0.5 with a LOW confidence band.
        """
        if not self.is_fitted or self._scaler is None:
            return WinProbability(prob_win=0.5, n_train=self._n_train,
                                  confidence_band="LOW")

        feats = _feat.extract_latest(df_enriched).reshape(1, -1)
        Xs = self._scaler.transform(feats)
        try:
            prob = float(self._model.predict_proba(Xs)[0, 1])
        except Exception:   # noqa: BLE001
            logger.exception("[journal_ml] predict failed")
            return WinProbability(prob_win=0.5, n_train=self._n_train,
                                  confidence_band="LOW")

        if self._n_train >= 50:
            band = "HIGH"
        elif self._n_train >= 20:
            band = "MEDIUM"
        else:
            band = "LOW"
        return WinProbability(prob_win=prob, n_train=self._n_train,
                              confidence_band=band)
