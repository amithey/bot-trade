"""Notification dispatcher — delivers messages to all enabled channels.

Design goals
------------
* **Non-blocking**: each delivery runs on a small thread pool so a slow
  Telegram round trip never stalls the trading loop.
* **Best-effort**: every error is swallowed and logged. A misconfigured
  webhook must never break the engine.
* **De-duplicated**: identical messages within a 3-second window are
  collapsed (prevents spam when Streamlit rerenders trigger an event twice).
* **In-process queue for toasts**: the Streamlit UI drains a queue on
  every rerun and surfaces them via ``st.toast``.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Optional

from notifications.config import NotificationConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Singleton queue for Streamlit toasts. Pages drain it on every rerun.
_toast_queue: "queue.Queue[dict]" = queue.Queue(maxsize=200)


def drain_toast_queue() -> list[dict]:
    """Pop all pending toasts. Called by Streamlit pages on every rerun."""
    out: list[dict] = []
    try:
        while True:
            out.append(_toast_queue.get_nowait())
    except queue.Empty:
        pass
    return out


class NotificationDispatcher:
    """Fan-out a single notification to all enabled channels."""

    def __init__(self, config: Optional[NotificationConfig] = None) -> None:
        self._config = config or NotificationConfig()
        self._dedup_lock = threading.Lock()
        self._recent: dict[str, float] = {}      # key → epoch seconds
        # Tiny worker pool — we don't need many threads, just enough to
        # absorb the latency of one HTTP call without blocking the trading loop.
        self._sender_lock = threading.Lock()

    @property
    def config(self) -> NotificationConfig:
        return self._config

    def update_config(self, cfg: NotificationConfig) -> None:
        self._config = cfg

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def notify(
        self,
        category: str,                # TRADE / ERROR / RISK / DECISION / REFLECT
        message: str,
        *,
        meta: Optional[dict] = None,
        ticker: str = "",
        level: str = "INFO",
    ) -> None:
        """Send a notification, respecting category filters and dedup."""
        cfg = self._config
        cat = category.upper()
        if cat == "TRADE" and not cfg.notify_trade: return
        if cat == "ERROR" and not cfg.notify_error: return
        if cat == "RISK" and not cfg.notify_risk: return
        if cat == "DECISION" and not cfg.notify_decision: return
        if cat == "REFLECT" and not cfg.notify_reflect: return

        # Dedup — collapse identical messages within 3 seconds
        key = f"{cat}|{ticker}|{message}"
        now = time.time()
        with self._dedup_lock:
            last = self._recent.get(key, 0.0)
            if now - last < 3.0:
                return
            self._recent[key] = now
            # Trim old entries
            if len(self._recent) > 200:
                cutoff = now - 30.0
                self._recent = {k: v for k, v in self._recent.items() if v >= cutoff}

        payload = {
            "category": cat,
            "message": message,
            "ticker": ticker,
            "level": level,
            "meta": meta or {},
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        # In-app toast queue (always cheap, runs on Streamlit thread)
        if cfg.toast_enabled:
            try:
                _toast_queue.put_nowait(payload)
            except queue.Full:
                pass  # Drop oldest by ignoring — non-critical

        # External channels — run on a daemon thread so we don't block
        if cfg.telegram_enabled and cfg.telegram_bot_token and cfg.telegram_chat_id:
            threading.Thread(
                target=self._send_telegram, args=(payload,), daemon=True,
            ).start()
        if cfg.webhook_enabled and cfg.webhook_url:
            threading.Thread(
                target=self._send_webhook, args=(payload,), daemon=True,
            ).start()

    # ─────────────────────────────────────────────────────────────────
    # Channel implementations
    # ─────────────────────────────────────────────────────────────────

    def _format_telegram(self, payload: dict) -> str:
        emoji = {
            "TRADE":    "💸",
            "ERROR":    "⛔",
            "RISK":     "🛡",
            "DECISION": "🤖",
            "REFLECT":  "📘",
        }.get(payload["category"], "🔔")
        head = f"{emoji} <b>{payload['category']}</b>"
        if payload.get("ticker"):
            head += f" · <code>{payload['ticker']}</code>"
        body = payload["message"]
        return f"{head}\n{body}"

    def _send_telegram(self, payload: dict) -> None:
        cfg = self._config
        url = (
            f"https://api.telegram.org/bot{cfg.telegram_bot_token.strip()}"
            f"/sendMessage"
        )
        data = urllib.parse.urlencode({
            "chat_id": cfg.telegram_chat_id.strip(),
            "text":    self._format_telegram(payload),
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Telegram notify non-200: %s", resp.status
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Telegram notify failed: %s", exc)

    def _send_webhook(self, payload: dict) -> None:
        cfg = self._config
        url = cfg.webhook_url.strip()
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status >= 400:
                    logger.warning("Webhook notify status %s", resp.status)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Webhook notify failed: %s", exc)

    # ─────────────────────────────────────────────────────────────────
    # Test ping — used by the Settings page "Send test" button
    # ─────────────────────────────────────────────────────────────────

    def test_telegram(self) -> tuple[bool, str]:
        cfg = self._config
        if not (cfg.telegram_bot_token and cfg.telegram_chat_id):
            return False, "Telegram bot token + chat ID required."
        payload = {
            "category": "TEST", "message": "BotTrade test ping ✅",
            "ticker": "", "level": "INFO", "meta": {},
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        try:
            self._send_telegram(payload)
            return True, "Telegram ping sent (check your chat)."
        except Exception as exc:  # noqa: BLE001
            return False, f"Telegram ping failed: {exc}"

    def test_webhook(self) -> tuple[bool, str]:
        cfg = self._config
        if not cfg.webhook_url:
            return False, "Webhook URL required."
        payload = {
            "category": "TEST", "message": "BotTrade test ping ✅",
            "ticker": "", "level": "INFO", "meta": {},
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        try:
            self._send_webhook(payload)
            return True, "Webhook ping sent."
        except Exception as exc:  # noqa: BLE001
            return False, f"Webhook ping failed: {exc}"
