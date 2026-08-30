"""
Tests for the ml/ package — shared feature extraction, and the
regime/anomaly/pattern/forecaster/trade-journal models built on it.

All models are fit on small synthetic OHLCV frames (no real market data,
no network) with just enough rows to satisfy each model's minimum-sample
guard. model_store is redirected to tmp_path so nothing touches the real
``data/ml_models/`` directory on disk.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import ml
from ml import features as feat_mod
from ml.features import extract_features, extract_latest, batch_matrix, FEATURE_NAMES, N_FEATURES
from ml import model_store
from ml.anomaly import AnomalyDetector, AnomalyResult
from ml.regime import RegimeClassifier, RegimeResult
from ml.forecaster import forecast, ForecastResult
from ml.pattern_detector import detect_patterns, _rel_diff, _peaks_troughs
from ml.trade_journal_ml import TradeJournalML, WinProbability
from portfolio.virtual_account import TradeRecord


# --------------------------------------------------------------------------- #
# Synthetic enriched OHLCV fixture
# --------------------------------------------------------------------------- #
def make_enriched_df(n: int = 300, seed: int = 7, drift: float = 0.0003,
                      vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    log_rets = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({"Open": close, "High": high, "Low": low,
                        "Close": close, "Volume": volume}, index=idx)
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean().bfill()
    df["RSI_14"] = 50 + 10 * np.sin(np.linspace(0, 20, n))  # bounded oscillator stand-in
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["MACD_Histogram"] = macd - signal
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Upper_20"] = mid + 2 * std
    df["BB_Lower_20"] = mid - 2 * std
    tr = (df["High"] - df["Low"]).abs()
    df["ATR_14"] = tr.rolling(14).mean()
    df["VWAP_20"] = (df["Close"] * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
    return df


@pytest.fixture(autouse=True)
def _isolate_model_store(tmp_path, monkeypatch):
    """Redirect ml.model_store's persistence directory to a tmp dir so
    tests never touch (or depend on) the real data/ml_models/ folder."""
    monkeypatch.setattr(model_store, "_STORE_DIR", tmp_path / "ml_models")


# --------------------------------------------------------------------------- #
# ml/__init__.py
# --------------------------------------------------------------------------- #
def test_package_reexports_extract_features_and_feature_names():
    assert ml.extract_features is extract_features
    assert ml.FEATURE_NAMES == FEATURE_NAMES


# --------------------------------------------------------------------------- #
# features.py
# --------------------------------------------------------------------------- #
def test_extract_features_returns_expected_columns_and_no_nans():
    df = make_enriched_df()
    feats = extract_features(df)
    assert list(feats.columns) == FEATURE_NAMES
    assert not feats.isna().any().any()
    assert len(feats) == len(df)


def test_extract_features_empty_df_returns_empty_frame_with_columns():
    feats = extract_features(pd.DataFrame())
    assert feats.empty
    assert list(feats.columns) == FEATURE_NAMES


def test_extract_features_none_input_returns_empty_frame():
    feats = extract_features(None)
    assert feats.empty


def test_extract_features_degrades_gracefully_without_indicator_columns():
    idx = pd.bdate_range("2025-01-01", periods=40)
    df = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0 + np.arange(40) * 0.1,
        "Volume": 1_000_000.0,
    }, index=idx)
    feats = extract_features(df)
    # Columns that need SMA/BB/VWAP/RSI/MACD fall back to their neutral default.
    assert (feats["sma20_vs_sma50"] == 0.0).all()
    assert (feats["bb_position"] == 0.5).all()
    assert (feats["vwap_deviation"] == 0.0).all()
    assert (feats["rsi_centered"] == 0.0).all()
    assert (feats["macd_hist_norm"] == 0.0).all()
    # atr_pct falls back to (high-low)/close when ATR_14 is absent.
    assert (feats["atr_pct"] > 0).all()


def test_extract_latest_returns_zero_vector_when_empty():
    vec = extract_latest(pd.DataFrame())
    assert vec.shape == (N_FEATURES,)
    assert (vec == 0.0).all()


def test_extract_latest_matches_last_row_of_extract_features():
    df = make_enriched_df()
    vec = extract_latest(df)
    row = extract_features(df).iloc[-1].to_numpy(dtype=float)
    np.testing.assert_allclose(vec, row)


def test_batch_matrix_drops_warmup_head():
    df = make_enriched_df(n=100)
    mat = batch_matrix(df, min_rows=30)
    assert mat.shape == (70, N_FEATURES)


def test_batch_matrix_returns_everything_when_shorter_than_min_rows():
    df = make_enriched_df(n=10)
    mat = batch_matrix(df, min_rows=30)
    assert mat.shape[0] == 10


def test_safe_div_guards_against_zero_and_none():
    assert feat_mod._safe_div(10.0, 0) == 0.0
    assert feat_mod._safe_div(10.0, None) == 0.0
    assert feat_mod._safe_div(10.0, 5.0) == 2.0


# --------------------------------------------------------------------------- #
# model_store.py
# --------------------------------------------------------------------------- #
def test_model_store_round_trips_an_object():
    model_store.save("thing", {"a": 1})
    assert model_store.exists("thing")
    assert model_store.load("thing") == {"a": 1}


def test_model_store_load_missing_returns_default():
    assert model_store.load("nope", default="fallback") == "fallback"
    assert model_store.exists("nope") is False


def test_model_store_delete_removes_file_and_reports_result():
    model_store.save("temp", 42)
    assert model_store.delete("temp") is True
    assert model_store.exists("temp") is False
    assert model_store.delete("temp") is False  # already gone


def test_model_store_load_survives_a_corrupt_file(tmp_path, monkeypatch):
    store_dir = tmp_path / "ml_models"
    monkeypatch.setattr(model_store, "_STORE_DIR", store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "corrupt.joblib").write_bytes(b"not a real pickle")
    assert model_store.load("corrupt", default="safe") == "safe"


# --------------------------------------------------------------------------- #
# anomaly.py
# --------------------------------------------------------------------------- #
def test_anomaly_detector_is_not_fitted_initially():
    det = AnomalyDetector("aapl")
    assert det.is_fitted is False
    assert det.ticker == "AAPL"


def test_anomaly_fit_raises_on_too_few_rows():
    det = AnomalyDetector("AAPL")
    df = make_enriched_df(n=40)
    with pytest.raises(ValueError, match=">=50"):
        det.fit(df)


def test_anomaly_fit_then_predict_returns_a_result():
    det = AnomalyDetector("AAPL")
    df = make_enriched_df(n=200)
    det.fit(df)
    assert det.is_fitted
    result = det.predict(df)
    assert isinstance(result, AnomalyResult)
    assert result.severity in ("NONE", "MILD", "STRONG", "EXTREME")
    assert isinstance(result.is_anomaly, bool)


def test_anomaly_predict_auto_fits_when_not_fitted():
    det = AnomalyDetector("AAPL")
    df = make_enriched_df(n=200)
    result = det.predict(df)  # should trigger an implicit fit()
    assert det.is_fitted
    assert isinstance(result, AnomalyResult)


def test_anomaly_detector_persists_and_reloads_via_model_store():
    det = AnomalyDetector("AAPL", contamination=0.1)
    df = make_enriched_df(n=200)
    det.fit(df)
    reloaded = AnomalyDetector("AAPL")
    assert reloaded.is_fitted
    assert reloaded.contamination == 0.1


def test_anomaly_a_sharp_price_spike_scores_more_anomalous_than_normal_bar():
    det = AnomalyDetector("AAPL")
    df = make_enriched_df(n=250)
    det.fit(df)
    spiked = df.copy()
    spiked.loc[spiked.index[-1], "Close"] *= 1.5  # a violent 50% single-bar spike
    spiked.loc[spiked.index[-1], "High"] *= 1.5
    normal_result = det.predict(df)
    spiked_result = det.predict(spiked)
    assert spiked_result.score >= normal_result.score


# --------------------------------------------------------------------------- #
# regime.py
# --------------------------------------------------------------------------- #
def test_regime_classifier_is_not_fitted_initially():
    clf = RegimeClassifier("qqq")
    assert clf.is_fitted is False
    assert clf.ticker == "QQQ"


def test_regime_fit_raises_on_too_few_rows():
    clf = RegimeClassifier("QQQ")
    df = make_enriched_df(n=40)
    with pytest.raises(ValueError, match=">=50"):
        clf.fit(df)


def test_regime_fit_then_predict_returns_a_valid_label():
    clf = RegimeClassifier("QQQ")
    df = make_enriched_df(n=200)
    clf.fit(df)
    result = clf.predict(df)
    assert isinstance(result, RegimeResult)
    assert result.label in ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE")
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.distances.keys()) <= {
        "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"
    } | {f"cluster_{i}" for i in range(4)}


def test_regime_predict_auto_fits_when_not_fitted():
    clf = RegimeClassifier("QQQ")
    df = make_enriched_df(n=200)
    clf.predict(df)
    assert clf.is_fitted


def test_regime_label_clusters_assigns_all_four_labels_uniquely():
    clf = RegimeClassifier("QQQ")
    # Hand-craft 4 well-separated centroids in scaled space:
    # columns: [ret_5, ret_20, vol_20, atr_pct, sma50_vs_sma200]
    centers = np.array([
        [ 2.0,  2.0, -1.0, -1.0,  2.0],   # strong up momentum, low vol -> TRENDING_UP
        [-2.0, -2.0, -1.0, -1.0, -2.0],   # strong down momentum, low vol -> TRENDING_DOWN
        [ 0.0,  0.0,  3.0,  3.0,  0.0],   # high vol -> VOLATILE
        [ 0.0,  0.0, -1.0, -1.0,  0.0],   # flat, low vol -> RANGING
    ])
    mapping = clf._label_clusters(centers)
    assert set(mapping.values()) == {"TRENDING_UP", "TRENDING_DOWN", "VOLATILE", "RANGING"}
    assert mapping[0] == "TRENDING_UP"
    assert mapping[1] == "TRENDING_DOWN"
    assert mapping[2] == "VOLATILE"
    assert mapping[3] == "RANGING"


def test_regime_persists_and_reloads_via_model_store():
    clf = RegimeClassifier("QQQ")
    df = make_enriched_df(n=200)
    clf.fit(df)
    reloaded = RegimeClassifier("QQQ")
    assert reloaded.is_fitted
    assert reloaded._mapping == clf._mapping


# --------------------------------------------------------------------------- #
# forecaster.py
# --------------------------------------------------------------------------- #
def test_forecast_returns_flat_zero_result_on_insufficient_data():
    df = make_enriched_df(n=10)
    result = forecast(df, horizon=5, ewm_span=20)
    assert result.bias == "flat"
    assert result.point_return == 0.0
    assert result.sigma == 0.0


def test_forecast_none_df_returns_flat_result():
    result = forecast(None)
    assert result.bias == "flat"


def test_forecast_strong_uptrend_produces_up_bias():
    df = make_enriched_df(n=200, drift=0.01, vol=0.001, seed=1)
    result = forecast(df, horizon=5)
    assert result.bias == "up"
    assert result.point_return > 0
    assert result.lo_95 < result.point_return < result.hi_95


def test_forecast_strong_downtrend_produces_down_bias():
    df = make_enriched_df(n=200, drift=-0.01, vol=0.001, seed=2)
    result = forecast(df, horizon=5)
    assert result.bias == "down"
    assert result.point_return < 0


def test_forecast_band_widens_with_horizon():
    df = make_enriched_df(n=200, seed=3)
    short = forecast(df, horizon=1)
    long_ = forecast(df, horizon=20)
    assert (long_.hi_95 - long_.lo_95) > (short.hi_95 - short.lo_95)


def test_forecast_result_is_a_plain_dataclass_with_expected_fields():
    r = ForecastResult(horizon=5, point_return=0.01, lo_95=-0.02, hi_95=0.04,
                        bias="up", sigma=0.015)
    assert r.horizon == 5 and r.bias == "up"


# --------------------------------------------------------------------------- #
# pattern_detector.py
# --------------------------------------------------------------------------- #
def test_rel_diff_zero_for_identical_values():
    assert _rel_diff(100.0, 100.0) == 0.0


def test_rel_diff_zero_when_both_are_zero():
    assert _rel_diff(0.0, 0.0) == 0.0


def test_rel_diff_symmetric():
    assert _rel_diff(100.0, 110.0) == pytest.approx(_rel_diff(110.0, 100.0))


def test_peaks_troughs_finds_a_clean_sine_waves_extrema():
    x = np.sin(np.linspace(0, 6 * np.pi, 300)) * 10 + 100
    peaks, troughs = _peaks_troughs(x)
    assert len(peaks) >= 2
    assert len(troughs) >= 2


def test_detect_patterns_returns_empty_for_too_short_series():
    df = pd.DataFrame({"Close": [100.0] * 10})
    assert detect_patterns(df) == []


def test_detect_patterns_returns_empty_for_none():
    assert detect_patterns(None) == []


def test_detect_patterns_finds_a_double_bottom():
    # W shape: down, up (peak), down to a similar low, breakout up.
    close = np.concatenate([
        np.linspace(100, 80, 15),    # first leg down to trough 1
        np.linspace(80, 95, 10),     # up to the peak between bottoms
        np.linspace(95, 80.5, 15),   # back down to trough 2 (~same level)
        np.linspace(80.5, 105, 15),  # breakout above the neckline
    ])
    df = pd.DataFrame({"Close": close})
    matches = detect_patterns(df, lookback=80)
    names = [m.name for m in matches]
    assert "DOUBLE_BOTTOM" in names
    m = next(m for m in matches if m.name == "DOUBLE_BOTTOM")
    assert m.direction == "bullish"
    assert 0.0 <= m.confidence <= 1.0


def test_detect_patterns_finds_a_double_top():
    close = np.concatenate([
        np.linspace(80, 100, 15),
        np.linspace(100, 88, 10),
        np.linspace(88, 99.5, 15),
        np.linspace(99.5, 75, 15),
    ])
    df = pd.DataFrame({"Close": close})
    matches = detect_patterns(df, lookback=80)
    names = [m.name for m in matches]
    assert "DOUBLE_TOP" in names
    m = next(m for m in matches if m.name == "DOUBLE_TOP")
    assert m.direction == "bearish"


def test_detect_patterns_results_are_sorted_by_confidence_descending():
    close = np.concatenate([
        np.linspace(100, 80, 15),
        np.linspace(80, 95, 10),
        np.linspace(95, 80.5, 15),
        np.linspace(80.5, 105, 15),
    ])
    df = pd.DataFrame({"Close": close})
    matches = detect_patterns(df, lookback=80)
    confidences = [m.confidence for m in matches]
    assert confidences == sorted(confidences, reverse=True)


def test_detect_patterns_all_matches_meet_the_confidence_floor():
    df = make_enriched_df(n=150)
    matches = detect_patterns(df)
    assert all(m.confidence >= 0.35 for m in matches)


# --------------------------------------------------------------------------- #
# trade_journal_ml.py
# --------------------------------------------------------------------------- #
def _trade(action, ticker, dt, pnl=0.0):
    return TradeRecord(
        executed_at=dt, action=action, ticker=ticker, quantity=1.0, price=100.0,
        gross_value=100.0, fee=0.0, net_value=100.0, realized_pnl=pnl,
        cash_after=0.0, portfolio_value=10_000.0,
    )


def test_pair_trades_matches_buy_with_next_sell_same_ticker():
    t0 = datetime(2026, 1, 1)
    log = [
        _trade("BUY", "AAPL", t0),
        _trade("SELL", "AAPL", t0 + timedelta(days=1), pnl=50.0),
    ]
    pairs = TradeJournalML._pair_trades(log)
    assert len(pairs) == 1
    assert pairs[0][0].action == "BUY"
    assert pairs[0][1].action == "SELL"


def test_pair_trades_ignores_unmatched_sell():
    log = [_trade("SELL", "AAPL", datetime(2026, 1, 1), pnl=10.0)]
    assert TradeJournalML._pair_trades(log) == []


def test_pair_trades_accepts_plain_dicts():
    log = [
        {"action": "BUY", "ticker": "MSFT", "executed_at": "2026-01-01T00:00:00"},
        {"action": "FORCE_CLOSE", "ticker": "MSFT", "executed_at": "2026-01-02T00:00:00",
         "realized_pnl": -5.0},
    ]
    pairs = TradeJournalML._pair_trades(log)
    assert len(pairs) == 1


def test_rec_time_parses_string_and_datetime_and_handles_bad_input():
    assert TradeJournalML._rec_time({"executed_at": "2026-01-01T00:00:00"}) == datetime(2026, 1, 1)
    assert TradeJournalML._rec_time({"executed_at": "not a date"}) is None
    assert TradeJournalML._rec_time({}) is None
    dt = datetime(2026, 1, 1)
    assert TradeJournalML._rec_time(_trade("BUY", "AAPL", dt)) == dt


def test_rec_pnl_handles_missing_and_invalid_values():
    assert TradeJournalML._rec_pnl({"realized_pnl": "12.5"}) == 12.5
    assert TradeJournalML._rec_pnl({}) == 0.0
    assert TradeJournalML._rec_pnl({"realized_pnl": "bad"}) == 0.0


def test_journal_not_fitted_returns_neutral_prediction():
    ml_journal = TradeJournalML()
    df = make_enriched_df(n=60)
    pred = ml_journal.predict(df)
    assert pred.prob_win == 0.5
    assert pred.confidence_band == "LOW"


def test_journal_fit_from_portfolio_skips_when_fewer_than_10_labeled_trades():
    ml_journal = TradeJournalML()
    t0 = datetime(2026, 1, 1)
    log = [_trade("BUY", "AAPL", t0), _trade("SELL", "AAPL", t0 + timedelta(days=1), pnl=10.0)]

    def fetch_history(ticker, end_dt):
        return make_enriched_df(n=60)

    n = ml_journal.fit_from_portfolio(log, fetch_history)
    assert n == 1
    assert ml_journal.is_fitted is False
    assert ml_journal.n_train == 1


def test_journal_fit_from_portfolio_fits_with_enough_labeled_trades():
    ml_journal = TradeJournalML()
    t0 = datetime(2026, 1, 1)
    log = []
    for i in range(12):
        buy_t = t0 + timedelta(days=2 * i)
        sell_t = buy_t + timedelta(days=1)
        pnl = 10.0 if i % 2 == 0 else -10.0
        log.append(_trade("BUY", "AAPL", buy_t))
        log.append(_trade("SELL", "AAPL", sell_t, pnl=pnl))

    def fetch_history(ticker, end_dt):
        return make_enriched_df(n=60)

    n = ml_journal.fit_from_portfolio(log, fetch_history)
    assert n == 12
    assert ml_journal.is_fitted is True
    assert ml_journal.n_train == 12


def test_journal_fit_from_portfolio_skips_trades_whose_fetch_history_raises():
    ml_journal = TradeJournalML()
    t0 = datetime(2026, 1, 1)
    log = [_trade("BUY", "AAPL", t0), _trade("SELL", "AAPL", t0 + timedelta(days=1), pnl=5.0)]

    def fetch_history(ticker, end_dt):
        raise RuntimeError("network down")

    n = ml_journal.fit_from_portfolio(log, fetch_history)
    assert n == 0


def test_journal_fit_from_portfolio_skips_trades_with_empty_history():
    ml_journal = TradeJournalML()
    t0 = datetime(2026, 1, 1)
    log = [_trade("BUY", "AAPL", t0), _trade("SELL", "AAPL", t0 + timedelta(days=1), pnl=5.0)]

    def fetch_history(ticker, end_dt):
        return pd.DataFrame()

    n = ml_journal.fit_from_portfolio(log, fetch_history)
    assert n == 0


def test_journal_predict_after_fit_returns_a_confidence_band():
    ml_journal = TradeJournalML()
    t0 = datetime(2026, 1, 1)
    log = []
    for i in range(25):
        buy_t = t0 + timedelta(days=2 * i)
        sell_t = buy_t + timedelta(days=1)
        pnl = 10.0 if i % 3 else -10.0
        log.append(_trade("BUY", "AAPL", buy_t))
        log.append(_trade("SELL", "AAPL", sell_t, pnl=pnl))

    def fetch_history(ticker, end_dt):
        return make_enriched_df(n=60, seed=5)

    ml_journal.fit_from_portfolio(log, fetch_history)
    pred = ml_journal.predict(make_enriched_df(n=60))
    assert 0.0 <= pred.prob_win <= 1.0
    assert pred.confidence_band == "MEDIUM"  # 20 <= n_train(25) < 50


def test_journal_persists_and_reloads_via_model_store():
    ml_journal = TradeJournalML()
    t0 = datetime(2026, 1, 1)
    log = []
    for i in range(12):
        buy_t = t0 + timedelta(days=2 * i)
        sell_t = buy_t + timedelta(days=1)
        pnl = 10.0 if i % 2 == 0 else -10.0
        log.append(_trade("BUY", "AAPL", buy_t))
        log.append(_trade("SELL", "AAPL", sell_t, pnl=pnl))

    ml_journal.fit_from_portfolio(log, lambda ticker, end_dt: make_enriched_df(n=60))

    reloaded = TradeJournalML()
    assert reloaded.is_fitted
    assert reloaded.n_train == 12


def test_journal_writes_a_metadata_json_file(tmp_path, monkeypatch):
    from ml import trade_journal_ml as journal_mod
    meta_path = tmp_path / "journal_meta.json"
    monkeypatch.setattr(journal_mod, "_METADATA_PATH", meta_path)

    ml_journal = TradeJournalML()
    t0 = datetime(2026, 1, 1)
    log = []
    for i in range(11):
        buy_t = t0 + timedelta(days=2 * i)
        sell_t = buy_t + timedelta(days=1)
        log.append(_trade("BUY", "AAPL", buy_t))
        log.append(_trade("SELL", "AAPL", sell_t, pnl=10.0 if i % 2 else -5.0))

    ml_journal.fit_from_portfolio(log, lambda ticker, end_dt: make_enriched_df(n=60))
    assert meta_path.exists()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["n_train"] == 11
    assert data["last_fit"] is not None
