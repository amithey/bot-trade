import pandas as pd
import pytest

from tests.test_committee_policy import replay_data
from tools import forward_committee as forward


def test_append_only_closed_bars_and_preserve_history():
    df, _, _ = replay_data()
    merged = forward.append_closed(df.iloc[:750], df.iloc[740:760], df.index[755])
    pd.testing.assert_frame_equal(merged, df.iloc[:755])
    changed = df.iloc[740:760].copy()
    changed.loc[changed.index[0], "Volume"] += 1
    with pytest.raises(ValueError, match="revised"):
        forward.append_closed(df.iloc[:750], changed, df.index[755])
    with pytest.raises(ValueError, match="overlap"):
        forward.append_closed(df.iloc[:750], df.iloc[751:760], df.index[755])
    kept = forward.append_closed(df.iloc[:750], changed, df.index[755], frozen_before=df.index[750])
    pd.testing.assert_frame_equal(kept, df.iloc[:755])
    with pytest.raises(ValueError, match="revised"):
        forward.append_closed(df.iloc[:750], changed, df.index[755], frozen_before=df.index[740])


def test_freeze_rejects_overwrite_code_drift_and_seed_changes(tmp_path, monkeypatch):
    df, _, _ = replay_data()
    source, study = tmp_path / "source", tmp_path / "study"
    source.mkdir()
    for ticker in forward.TICKERS:
        df.to_csv(source / f"{ticker}.csv")
    now = df.index[-1] + pd.Timedelta(minutes=10)
    monkeypatch.setattr(forward, "fingerprint", lambda: "frozen-code")
    forward.initialize(study, source, now)
    _, report = forward.observe(study)
    assert report["status"] == "insufficient_forward_evidence"
    assert report["results"] == [] and report["coverage_days"] == 0
    with pytest.raises(FileExistsError):
        forward.initialize(study, source, now)
    monkeypatch.setattr(forward, "fingerprint", lambda: "different-code")
    with pytest.raises(ValueError, match="changed"):
        forward.observe(study)
    monkeypatch.setattr(forward, "fingerprint", lambda: "frozen-code")
    (study / "BTC-USD.seed.csv").write_text("modified")
    with pytest.raises(ValueError, match="Seed modified"):
        forward.observe(study)
