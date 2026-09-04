"""
Tests for notifications/ — config persistence and the dispatcher that fans a
trading event out to Telegram, a generic webhook, and the in-app toast queue.

notifications/ shipped with zero tests. The behaviors documented in its own
docstring — non-blocking, best-effort, de-duplicated — are exactly the kind
that degrade silently: a broken dedup window doesn't crash anything, it just
starts spamming Telegram or swallowing real alerts, and nobody notices until
it matters.

No test hits the real network. `_send_telegram`/`_send_webhook` spawn a
background daemon thread on the real `notify()` path; the `sync_thread`
fixture replaces `threading.Thread` with a synchronous stand-in so the send
happens deterministically before an assertion runs, and `urllib.request.
urlopen` is monkeypatched so nothing ever leaves the process.
"""
from __future__ import annotations

import json
import time
import urllib.parse

import pytest

from notifications import dispatcher as dispatcher_mod
from notifications.config import NotificationConfig
from notifications.dispatcher import NotificationDispatcher, drain_toast_queue


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def clean_toast_queue():
    """The toast queue is a module-level singleton shared by every dispatcher."""
    drain_toast_queue()
    yield
    drain_toast_queue()


class _SyncThread:
    """Stand-in for threading.Thread that runs the target immediately and
    synchronously, so a background send completes before the test asserts
    on it instead of racing a real OS thread."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self) -> None:
        self._target(*self._args, **self._kwargs)

    def join(self, timeout=None) -> None:
        pass


@pytest.fixture()
def sync_threads(monkeypatch):
    monkeypatch.setattr(dispatcher_mod.threading, "Thread", _SyncThread)


class _FakeResponse:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_urlopen(monkeypatch, status=200, exc=None):
    """Patch urlopen to either return a fake response or raise, and record
    every call's Request object for inspection."""
    calls = []

    def fake(req, timeout=None):
        calls.append(req)
        if exc is not None:
            raise exc
        return _FakeResponse(status)

    monkeypatch.setattr(dispatcher_mod.urllib.request, "urlopen", fake)
    return calls


TELEGRAM_CFG = NotificationConfig(
    telegram_enabled=True, telegram_bot_token="123:ABC", telegram_chat_id="999",
    webhook_enabled=False,
)
WEBHOOK_CFG = NotificationConfig(
    webhook_enabled=True, webhook_url="https://hooks.example.com/in",
    telegram_enabled=False,
)


# --------------------------------------------------------------------------- #
# NotificationConfig
# --------------------------------------------------------------------------- #
def test_defaults_are_a_sane_starting_point():
    cfg = NotificationConfig()
    assert cfg.toast_enabled is True
    assert cfg.notify_trade is True and cfg.notify_error is True
    assert cfg.notify_decision is False, "every-decision notifications are chatty by default"
    assert cfg.has_external_channel is False


def test_to_dict_round_trips_every_field():
    cfg = NotificationConfig(telegram_enabled=True, telegram_bot_token="t",
                             telegram_chat_id="1", notify_decision=True)
    d = cfg.to_dict()
    assert NotificationConfig(**d) == cfg


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "notifications.json"
    cfg = NotificationConfig(webhook_enabled=True, webhook_url="https://x.example",
                             notify_reflect=True)
    cfg.save(path)
    loaded = NotificationConfig.load(path)
    assert loaded == cfg


def test_load_returns_defaults_when_file_is_missing(tmp_path):
    loaded = NotificationConfig.load(tmp_path / "does_not_exist.json")
    assert loaded == NotificationConfig()


def test_load_returns_defaults_on_corrupt_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert NotificationConfig.load(path) == NotificationConfig()


def test_load_ignores_unknown_keys_for_forward_compatibility(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({
        "telegram_enabled": True, "telegram_bot_token": "t",
        "telegram_chat_id": "1", "a_field_from_a_future_version": "surprise",
    }), encoding="utf-8")
    loaded = NotificationConfig.load(path)
    assert loaded.telegram_enabled is True
    assert not hasattr(loaded, "a_field_from_a_future_version")


def test_has_external_channel_requires_all_three_telegram_fields():
    assert not NotificationConfig(telegram_enabled=True).has_external_channel
    assert not NotificationConfig(telegram_enabled=True,
                                  telegram_bot_token="t").has_external_channel
    assert NotificationConfig(telegram_enabled=True, telegram_bot_token="t",
                              telegram_chat_id="1").has_external_channel


def test_has_external_channel_requires_webhook_url():
    assert not NotificationConfig(webhook_enabled=True).has_external_channel
    assert NotificationConfig(webhook_enabled=True,
                              webhook_url="https://x").has_external_channel


def test_disabled_channel_with_fields_set_does_not_count():
    """A saved token that's toggled off must not silently count as active."""
    cfg = NotificationConfig(telegram_enabled=False, telegram_bot_token="t",
                             telegram_chat_id="1")
    assert cfg.has_external_channel is False


# --------------------------------------------------------------------------- #
# Toast queue
# --------------------------------------------------------------------------- #
def test_toast_queue_starts_empty():
    assert drain_toast_queue() == []


def test_notify_queues_a_toast_when_enabled():
    d = NotificationDispatcher(NotificationConfig(toast_enabled=True))
    d.notify("TRADE", "BUY BTC-USD")
    items = drain_toast_queue()
    assert len(items) == 1
    assert items[0]["category"] == "TRADE"
    assert items[0]["message"] == "BUY BTC-USD"


def test_notify_skips_the_toast_queue_when_disabled():
    d = NotificationDispatcher(NotificationConfig(toast_enabled=False))
    d.notify("TRADE", "BUY BTC-USD")
    assert drain_toast_queue() == []


def test_drain_empties_the_queue():
    d = NotificationDispatcher(NotificationConfig())
    d.notify("TRADE", "one")
    drain_toast_queue()
    assert drain_toast_queue() == []


def test_a_full_queue_drops_silently_instead_of_raising(monkeypatch):
    """maxsize=200 - notify() must never raise just because nobody drained."""
    d = NotificationDispatcher(NotificationConfig())
    for i in range(205):
        d.notify("TRADE", f"msg {i}", ticker=f"T{i}")   # unique keys: no dedup
    assert len(drain_toast_queue()) <= 200


# --------------------------------------------------------------------------- #
# Category filters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("category,flag", [
    ("TRADE", "notify_trade"), ("ERROR", "notify_error"),
    ("RISK", "notify_risk"), ("DECISION", "notify_decision"),
    ("REFLECT", "notify_reflect"),
])
def test_each_category_is_gated_by_its_own_flag(category, flag):
    cfg = NotificationConfig(**{flag: False})
    d = NotificationDispatcher(cfg)
    d.notify(category, "irrelevant")
    assert drain_toast_queue() == [], f"{category} should be suppressed when {flag}=False"

    cfg2 = NotificationConfig(**{flag: True})
    d2 = NotificationDispatcher(cfg2)
    d2.notify(category, "irrelevant")
    assert len(drain_toast_queue()) == 1, f"{category} should fire when {flag}=True"


def test_category_matching_is_case_insensitive():
    cfg = NotificationConfig(notify_trade=False)
    d = NotificationDispatcher(cfg)
    d.notify("trade", "lowercase category")
    assert drain_toast_queue() == []


def test_updating_config_takes_effect_immediately():
    d = NotificationDispatcher(NotificationConfig(notify_trade=False))
    d.notify("TRADE", "should be suppressed")
    assert drain_toast_queue() == []
    d.update_config(NotificationConfig(notify_trade=True))
    d.notify("TRADE", "should fire now")
    assert len(drain_toast_queue()) == 1


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #
def test_identical_messages_within_the_window_are_collapsed():
    d = NotificationDispatcher(NotificationConfig())
    d.notify("TRADE", "BUY BTC-USD", ticker="BTC-USD")
    d.notify("TRADE", "BUY BTC-USD", ticker="BTC-USD")
    d.notify("TRADE", "BUY BTC-USD", ticker="BTC-USD")
    assert len(drain_toast_queue()) == 1


def test_the_same_message_fires_again_after_the_window_passes(monkeypatch):
    d = NotificationDispatcher(NotificationConfig())
    clock = {"t": 1_000.0}
    monkeypatch.setattr(dispatcher_mod.time, "time", lambda: clock["t"])

    d.notify("TRADE", "BUY BTC-USD", ticker="BTC-USD")
    clock["t"] += 3.1     # just past the 3-second dedup window
    d.notify("TRADE", "BUY BTC-USD", ticker="BTC-USD")

    assert len(drain_toast_queue()) == 2


def test_dedup_key_includes_ticker_so_different_symbols_both_fire():
    d = NotificationDispatcher(NotificationConfig())
    d.notify("TRADE", "BUY", ticker="BTC-USD")
    d.notify("TRADE", "BUY", ticker="ETH-USD")
    assert len(drain_toast_queue()) == 2


def test_dedup_key_includes_category_so_different_categories_both_fire():
    d = NotificationDispatcher(NotificationConfig())
    d.notify("TRADE", "same text", ticker="BTC-USD")
    d.notify("ERROR", "same text", ticker="BTC-USD")
    assert len(drain_toast_queue()) == 2


def test_dedup_state_is_per_dispatcher_instance():
    """Two independent dispatchers (e.g. two accounts) must not dedup
    against each other's traffic."""
    a, b = NotificationDispatcher(NotificationConfig()), NotificationDispatcher(NotificationConfig())
    a.notify("TRADE", "BUY", ticker="BTC-USD")
    b.notify("TRADE", "BUY", ticker="BTC-USD")
    assert len(drain_toast_queue()) == 2


# --------------------------------------------------------------------------- #
# Telegram formatting + sending
# --------------------------------------------------------------------------- #
def test_format_telegram_includes_the_category_emoji_and_ticker():
    d = NotificationDispatcher(TELEGRAM_CFG)
    text = d._format_telegram({"category": "TRADE", "message": "BUY", "ticker": "BTC-USD"})
    assert "💸" in text and "TRADE" in text and "BTC-USD" in text and "BUY" in text


def test_format_telegram_omits_the_ticker_tag_when_absent():
    d = NotificationDispatcher(TELEGRAM_CFG)
    text = d._format_telegram({"category": "ERROR", "message": "boom", "ticker": ""})
    assert "<code>" not in text


def test_format_telegram_falls_back_to_a_generic_bell_for_unknown_categories():
    d = NotificationDispatcher(TELEGRAM_CFG)
    text = d._format_telegram({"category": "SOMETHING_NEW", "message": "x", "ticker": ""})
    assert "🔔" in text


def test_send_telegram_posts_to_the_bot_token_url(monkeypatch):
    calls = _mock_urlopen(monkeypatch, status=200)
    d = NotificationDispatcher(TELEGRAM_CFG)
    ok, detail = d._send_telegram({"category": "TRADE", "message": "BUY", "ticker": "BTC-USD"})
    assert ok is True
    assert len(calls) == 1
    assert calls[0].full_url == "https://api.telegram.org/bot123:ABC/sendMessage"
    body = urllib.parse.parse_qs(calls[0].data.decode())
    assert body["chat_id"] == ["999"]
    assert body["parse_mode"] == ["HTML"]


def test_send_telegram_reports_a_non_200_as_failure(monkeypatch):
    _mock_urlopen(monkeypatch, status=500)
    d = NotificationDispatcher(TELEGRAM_CFG)
    ok, detail = d._send_telegram({"category": "TRADE", "message": "x", "ticker": ""})
    assert ok is False
    assert "500" in detail


def test_send_telegram_never_raises_on_a_network_error(monkeypatch):
    _mock_urlopen(monkeypatch, exc=OSError("network unreachable"))
    d = NotificationDispatcher(TELEGRAM_CFG)
    ok, detail = d._send_telegram({"category": "TRADE", "message": "x", "ticker": ""})
    assert ok is False
    assert "network unreachable" in detail


def test_notify_spawns_telegram_only_when_all_three_fields_are_set(
    monkeypatch, sync_threads,
):
    calls = _mock_urlopen(monkeypatch, status=200)
    incomplete = NotificationConfig(telegram_enabled=True, telegram_bot_token="t")  # no chat id
    NotificationDispatcher(incomplete).notify("TRADE", "x", ticker="a")
    assert len(calls) == 0

    NotificationDispatcher(TELEGRAM_CFG).notify("TRADE", "x", ticker="b")
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Webhook formatting + sending
# --------------------------------------------------------------------------- #
def test_send_webhook_posts_json_to_the_configured_url(monkeypatch):
    calls = _mock_urlopen(monkeypatch, status=200)
    d = NotificationDispatcher(WEBHOOK_CFG)
    ok, detail = d._send_webhook({"category": "RISK", "message": "halted", "ticker": "SPY"})
    assert ok is True
    assert calls[0].full_url == "https://hooks.example.com/in"
    body = json.loads(calls[0].data.decode())
    assert body["category"] == "RISK" and body["ticker"] == "SPY"
    assert calls[0].headers.get("Content-type") == "application/json"


def test_send_webhook_treats_4xx_and_5xx_as_failure(monkeypatch):
    _mock_urlopen(monkeypatch, status=404)
    d = NotificationDispatcher(WEBHOOK_CFG)
    ok, detail = d._send_webhook({"category": "TRADE", "message": "x", "ticker": ""})
    assert ok is False
    assert "404" in detail


def test_send_webhook_never_raises_on_a_bad_url(monkeypatch):
    _mock_urlopen(monkeypatch, exc=ValueError("unknown url type"))
    d = NotificationDispatcher(WEBHOOK_CFG)
    ok, detail = d._send_webhook({"category": "TRADE", "message": "x", "ticker": ""})
    assert ok is False


def test_notify_does_not_spawn_webhook_when_disabled(monkeypatch, sync_threads):
    calls = _mock_urlopen(monkeypatch, status=200)
    cfg = NotificationConfig(webhook_enabled=False, webhook_url="https://x.example")
    NotificationDispatcher(cfg).notify("TRADE", "x")
    assert len(calls) == 0


# --------------------------------------------------------------------------- #
# notify() never blocks the trading loop on a slow/broken channel
# --------------------------------------------------------------------------- #
def test_notify_returns_immediately_even_though_channels_are_real_threads():
    """Without mocking threading.Thread, notify() must still return fast -
    the whole point is that it does not wait for the network call."""
    cfg = NotificationConfig(
        telegram_enabled=True, telegram_bot_token="123:ABC", telegram_chat_id="999",
        webhook_enabled=True, webhook_url="http://10.255.255.1/unreachable",
    )
    d = NotificationDispatcher(cfg)
    start = time.perf_counter()
    d.notify("TRADE", "BUY", ticker="BTC-USD")
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, "notify() must not block on the network call"


# --------------------------------------------------------------------------- #
# test_telegram() / test_webhook() — the Settings page "send test ping" buttons
# --------------------------------------------------------------------------- #
def test_test_telegram_requires_credentials():
    d = NotificationDispatcher(NotificationConfig())
    ok, msg = d.test_telegram()
    assert ok is False
    assert "required" in msg.lower()


def test_test_telegram_reports_success_on_a_real_200(monkeypatch):
    _mock_urlopen(monkeypatch, status=200)
    d = NotificationDispatcher(TELEGRAM_CFG)
    ok, msg = d.test_telegram()
    assert ok is True


def test_test_telegram_reports_failure_on_a_network_error(monkeypatch):
    """The bug this guards: _send_telegram swallows every exception itself,
    so a try/except around calling it can never fire - test_telegram used to
    report success unconditionally regardless of what actually happened."""
    _mock_urlopen(monkeypatch, exc=OSError("network unreachable"))
    d = NotificationDispatcher(TELEGRAM_CFG)
    ok, msg = d.test_telegram()
    assert ok is False
    assert "failed" in msg.lower()


def test_test_webhook_requires_a_url():
    d = NotificationDispatcher(NotificationConfig())
    ok, msg = d.test_webhook()
    assert ok is False


def test_test_webhook_reports_failure_on_a_bad_response(monkeypatch):
    _mock_urlopen(monkeypatch, status=500)
    d = NotificationDispatcher(WEBHOOK_CFG)
    ok, msg = d.test_webhook()
    assert ok is False
    assert "failed" in msg.lower()


def test_test_webhook_reports_success_on_a_real_200(monkeypatch):
    _mock_urlopen(monkeypatch, status=200)
    d = NotificationDispatcher(WEBHOOK_CFG)
    ok, msg = d.test_webhook()
    assert ok is True
