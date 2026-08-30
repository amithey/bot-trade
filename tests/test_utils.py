"""
Tests for the utils/ package: market-hours logic, HF-quiet startup
suppression, the crash reporter, and the centralised logger.

market_logic's `datetime.now()` is monkeypatched with a small stand-in
whose `.now()` returns a *real* `datetime.datetime` instance (never a
substitute class) so nothing downstream that touches real datetime
machinery (astimezone, replace, isinstance) is disturbed.

crash_reporter installs process-wide hooks (sys.excepthook,
threading.excepthook) and caches a singleton logger — every test that
touches install()/_record() resets those module globals and restores the
original hooks afterwards so no test leaks state into another.
"""
from __future__ import annotations

import io
import logging
import sys
import threading
from datetime import datetime as real_datetime
from unittest.mock import MagicMock

import pytest
import pytz

import utils.market_logic as market_logic_mod
from utils.market_logic import get_market_status, is_market_open

import utils.crash_reporter as crash_mod

import utils.hf_quiet as hf_quiet_mod
from utils.hf_quiet import configure_quiet_hf, quiet_model_load

import utils.logger as logger_mod
from utils.logger import (
    get_logger,
    log_api_call,
    log_rag_retrieval,
    log_trade_execution,
)


# --------------------------------------------------------------------------- #
# market_logic.py
# --------------------------------------------------------------------------- #
_ET = pytz.timezone("US/Eastern")


class _FixedDatetime:
    """Stand-in for the `datetime` class used only for `.now(tz)` — always
    returns a genuine `datetime.datetime` instance, so astimezone/replace/
    isinstance downstream all keep working normally."""

    def __init__(self, et_dt: real_datetime):
        self._et_dt = et_dt

    def now(self, tz):
        return self._et_dt.astimezone(pytz.utc)


def _patch_now(monkeypatch, year, month, day, hour, minute, weekday_override=None):
    naive = real_datetime(year, month, day, hour, minute)
    et_dt = _ET.localize(naive)
    monkeypatch.setattr(market_logic_mod, "datetime", _FixedDatetime(et_dt))


def test_crypto_ticker_is_always_open(monkeypatch):
    # A Sunday at 3am ET -- would be CLOSED for equities, but crypto ignores it.
    _patch_now(monkeypatch, 2026, 1, 4, 3, 0)  # 2026-01-04 is a Sunday
    is_open, msg = get_market_status("BTC-USD")
    assert is_open is True
    assert "24/7 Crypto" in msg


def test_crypto_detection_is_case_insensitive():
    assert market_logic_mod._is_crypto("btc-usd") is True
    assert market_logic_mod._is_crypto("AAPL") is False


def test_weekend_is_closed(monkeypatch):
    _patch_now(monkeypatch, 2026, 1, 3, 12, 0)  # Saturday
    is_open, msg = get_market_status("AAPL")
    assert is_open is False
    assert "Weekend" in msg
    assert "Saturday" in msg


def test_pre_market_is_closed(monkeypatch):
    _patch_now(monkeypatch, 2026, 1, 5, 8, 0)  # Monday, 8am ET
    is_open, msg = get_market_status("AAPL")
    assert is_open is False
    assert "Pre-Market" in msg


def test_after_hours_is_closed(monkeypatch):
    _patch_now(monkeypatch, 2026, 1, 5, 16, 30)  # Monday, 4:30pm ET
    is_open, msg = get_market_status("AAPL")
    assert is_open is False
    assert "After-Hours" in msg


def test_market_close_boundary_is_closed(monkeypatch):
    _patch_now(monkeypatch, 2026, 1, 5, 16, 0)  # exactly 16:00 ET
    is_open, _ = get_market_status("AAPL")
    assert is_open is False


def test_market_open_boundary_is_open(monkeypatch):
    _patch_now(monkeypatch, 2026, 1, 5, 9, 30)  # exactly 09:30 ET
    is_open, msg = get_market_status("AAPL")
    assert is_open is True
    assert "OPEN" in msg


def test_mid_session_reports_time_remaining(monkeypatch):
    _patch_now(monkeypatch, 2026, 1, 5, 14, 0)  # Monday, 2pm ET -> 2h left
    is_open, msg = get_market_status("AAPL")
    assert is_open is True
    assert "2h 0m remaining" in msg


def test_empty_ticker_defaults_to_equity_logic(monkeypatch):
    _patch_now(monkeypatch, 2026, 1, 3, 12, 0)  # Saturday
    is_open, msg = get_market_status("")
    assert is_open is False
    assert "Weekend" in msg


def test_is_market_open_wraps_get_market_status(monkeypatch):
    _patch_now(monkeypatch, 2026, 1, 5, 14, 0)
    assert is_market_open("AAPL") is True
    assert is_market_open("ETH-USD") is True  # crypto, no patch needed


# --------------------------------------------------------------------------- #
# hf_quiet.py
# --------------------------------------------------------------------------- #
def test_configure_quiet_hf_sets_expected_env_vars(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    configure_quiet_hf()
    import os
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"


def test_configure_quiet_hf_lowers_noisy_logger_levels():
    configure_quiet_hf()
    for name in ("huggingface_hub", "sentence_transformers", "transformers"):
        assert logging.getLogger(name).level == logging.ERROR


def test_configure_quiet_hf_never_raises_if_optional_imports_fail(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _blow_up_on_hf(name, *a, **kw):
        if name in ("huggingface_hub.utils", "transformers"):
            raise ImportError("simulated missing package")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _blow_up_on_hf)
    configure_quiet_hf()  # must not raise


def test_quiet_model_load_suppresses_stdout_and_stderr(capsys):
    with quiet_model_load():
        print("this should not reach the captured stdout")
        sys.stderr.write("nor this\n")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_quiet_model_load_still_propagates_exceptions():
    with pytest.raises(ValueError, match="boom"):
        with quiet_model_load():
            raise ValueError("boom")


def test_quiet_model_load_restores_stdout_after_use():
    original_stdout = sys.stdout
    with quiet_model_load():
        pass
    assert sys.stdout is original_stdout


# --------------------------------------------------------------------------- #
# crash_reporter.py
# --------------------------------------------------------------------------- #
@pytest.fixture
def isolated_crash_reporter(tmp_path, monkeypatch):
    """Redirect the crash log to tmp_path and reset the module's singleton
    state, restoring the real excepthooks afterwards."""
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(crash_mod, "_LOGS_DIR", log_dir)
    monkeypatch.setattr(crash_mod, "_CRASH_LOG", log_dir / "crashes.log")
    monkeypatch.setattr(crash_mod, "_crash_logger", None)
    monkeypatch.setattr(crash_mod, "_post_callback", None)
    monkeypatch.setattr(crash_mod, "_installed", False)

    original_excepthook = sys.excepthook
    original_thread_excepthook = getattr(threading, "excepthook", None)
    try:
        yield log_dir / "crashes.log"
    finally:
        sys.excepthook = original_excepthook
        if original_thread_excepthook is not None:
            threading.excepthook = original_thread_excepthook


def test_is_installed_false_before_install(isolated_crash_reporter):
    assert crash_mod.is_installed() is False


def test_install_sets_installed_flag_and_hooks(isolated_crash_reporter):
    crash_mod.install()
    assert crash_mod.is_installed() is True
    assert sys.excepthook is not sys.__excepthook__
    assert threading.excepthook is not None


def test_report_manual_with_exception_writes_to_crash_log(isolated_crash_reporter):
    crash_log = isolated_crash_reporter
    try:
        raise RuntimeError("something broke")
    except RuntimeError as exc:
        crash_mod.report_manual("manual test", exc=exc, source="unit-test")
    text = crash_log.read_text(encoding="utf-8")
    assert "RuntimeError: something broke" in text
    assert "unit-test" in text


def test_report_manual_without_exception_uses_format_exc(isolated_crash_reporter):
    crash_log = isolated_crash_reporter
    crash_mod.report_manual("no exception here", source="unit-test")
    text = crash_log.read_text(encoding="utf-8")
    assert "no exception here" in text


def test_recent_crashes_returns_empty_list_when_no_log_exists(isolated_crash_reporter):
    assert crash_mod.recent_crashes() == []


def test_recent_crashes_returns_newest_first_and_respects_limit(isolated_crash_reporter):
    for i in range(3):
        try:
            raise ValueError(f"error {i}")
        except ValueError as exc:
            crash_mod.report_manual(f"crash {i}", exc=exc)
    lines = crash_mod.recent_crashes(limit=2)
    assert len(lines) == 2
    assert "error 2" in lines[0]  # newest first
    assert "error 1" in lines[1]


def test_post_callback_is_invoked_on_crash(isolated_crash_reporter):
    calls = []
    crash_mod.install(post_callback=lambda one, full: calls.append((one, full)))
    try:
        raise KeyError("missing")
    except KeyError as exc:
        crash_mod.report_manual("cb test", exc=exc)
    assert len(calls) == 1
    assert "KeyError" in calls[0][0]


def test_post_callback_exceptions_are_swallowed(isolated_crash_reporter):
    def _exploding_callback(one, full):
        raise RuntimeError("callback itself is broken")

    crash_mod.install(post_callback=_exploding_callback)
    try:
        raise ValueError("primary crash")
    except ValueError as exc:
        crash_mod.report_manual("swallow test", exc=exc)  # must not raise


def test_crash_log_path_returns_the_configured_path(isolated_crash_reporter):
    assert crash_mod.crash_log_path() == isolated_crash_reporter


def test_thread_excepthook_records_uncaught_thread_exceptions(isolated_crash_reporter):
    crash_log = isolated_crash_reporter
    crash_mod.install()

    def _boom():
        raise RuntimeError("thread crash")

    t = threading.Thread(target=_boom, name="worker-1")
    t.start()
    t.join()
    text = crash_log.read_text(encoding="utf-8")
    assert "thread:worker-1" in text
    assert "RuntimeError: thread crash" in text


# --------------------------------------------------------------------------- #
# logger.py
# --------------------------------------------------------------------------- #
def test_get_logger_returns_a_standard_logger_with_the_given_name():
    logger = get_logger("bottrade.test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "bottrade.test_module"


def test_get_logger_is_idempotent_across_repeated_calls():
    a = get_logger("bottrade.a")
    b = get_logger("bottrade.b")
    # Root handlers attached exactly once regardless of how many calls.
    root_handlers_after_a = len(logging.getLogger().handlers)
    get_logger("bottrade.c")
    assert len(logging.getLogger().handlers) == root_handlers_after_a


def test_get_logger_attaches_handlers_on_first_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(logger_mod, "_configured", False)
    monkeypatch.setattr(logger_mod, "_LOGS_DIR", tmp_path / "logs")
    root = logging.getLogger()
    before = list(root.handlers)
    for h in before:
        root.removeHandler(h)
    try:
        get_logger("bottrade.fresh")
        assert len(root.handlers) == 2  # Rich console + rotating file
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in before:
            root.addHandler(h)


def test_log_api_call_formats_debug_message_with_all_fields():
    fake_logger = MagicMock()
    log_api_call(fake_logger, model="claude-opus-5", ticker="AAPL",
                 prompt_tokens=1200, latency_ms=350.4)
    fake_logger.debug.assert_called_once()
    msg = fake_logger.debug.call_args[0][0]
    assert "model=claude-opus-5" in msg
    assert "ticker=AAPL" in msg
    assert "tokens~1200" in msg
    assert "latency=350ms" in msg


def test_log_api_call_omits_optional_fields_when_none():
    fake_logger = MagicMock()
    log_api_call(fake_logger, model="claude-opus-5", ticker="AAPL")
    msg = fake_logger.debug.call_args[0][0]
    assert "tokens~" not in msg
    assert "latency=" not in msg


def test_log_rag_retrieval_formats_score_as_na_when_none():
    fake_logger = MagicMock()
    log_rag_retrieval(fake_logger, ticker="AAPL", n_chunks=3,
                       best_score=None, query_snippet="RSI oversold bounce")
    fake_logger.debug.assert_called_once()
    args = fake_logger.debug.call_args[0]
    assert args[3] == "n/a"


def test_log_rag_retrieval_formats_score_to_three_decimals():
    fake_logger = MagicMock()
    log_rag_retrieval(fake_logger, ticker="AAPL", n_chunks=3,
                       best_score=0.876543, query_snippet="q")
    args = fake_logger.debug.call_args[0]
    assert args[3] == "0.877"


def test_log_trade_execution_passes_through_all_fields():
    fake_logger = MagicMock()
    log_trade_execution(fake_logger, action="BUY", ticker="AAPL", price=101.234,
                         quantity=10.0, cash_after=8_900.0, portfolio_value=10_000.0,
                         reasoning_snippet="strong RSI divergence")
    fake_logger.info.assert_called_once()
    args = fake_logger.info.call_args[0]
    assert args[1] == "BUY"
    assert args[2] == "AAPL"
    assert args[3] == pytest.approx(101.234)
