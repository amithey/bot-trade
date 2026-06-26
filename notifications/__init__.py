"""Notifications — pluggable dispatch of trade / error events to external
channels (Telegram, webhook) and the in-app toast queue.

Usage
-----
    from notifications import NotificationDispatcher, NotificationConfig

    cfg = NotificationConfig.load()
    dispatcher = NotificationDispatcher(cfg)
    dispatcher.notify("TRADE", "🟢 BUY SPY @ 440.12", meta={"ticker":"SPY"})

The live engine wires the dispatcher into its ``_emit`` so every
PulseEvent of interest fans out automatically.
"""
from notifications.config import NotificationConfig
from notifications.dispatcher import NotificationDispatcher

__all__ = ["NotificationConfig", "NotificationDispatcher"]
