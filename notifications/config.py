"""Persisted notifications config — Telegram + generic webhook + filters.

Stored at ``data/notifications.json``. Tokens / chat IDs are kept on
disk in plain text — same as ``.env`` already is. Don't commit them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_DEFAULT_PATH = Path("data/notifications.json")


@dataclass
class NotificationConfig:
    # Channels
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    webhook_enabled: bool = False
    webhook_url: str = ""

    toast_enabled: bool = True            # in-app Streamlit toasts

    # Filters — which event categories fire notifications
    notify_trade: bool = True             # BUY/SELL/FORCE_CLOSE
    notify_error: bool = True             # engine errors
    notify_risk: bool = True              # safety blocks, halts, panic
    notify_decision: bool = False         # every Claude decision (chatty)
    notify_reflect: bool = False          # post-trade lessons

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path | None = None) -> None:
        p = Path(path or _DEFAULT_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "NotificationConfig":
        p = Path(path or _DEFAULT_PATH)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    @property
    def has_external_channel(self) -> bool:
        return (
            (self.telegram_enabled and self.telegram_bot_token and self.telegram_chat_id)
            or (self.webhook_enabled and self.webhook_url)
        )
