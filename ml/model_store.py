"""Pickle helpers shared by every ML module.

Models live under ``./data/ml_models/`` so they persist across restarts
and can be inspected / deleted by the user without digging through code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from utils.logger import get_logger

logger = get_logger(__name__)

_STORE_DIR = Path("data/ml_models")


def _path(name: str) -> Path:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR / f"{name}.joblib"


def save(name: str, obj: Any) -> Path:
    p = _path(name)
    joblib.dump(obj, p)
    logger.info(f"[ml] saved model -> {p} ({p.stat().st_size/1024:.1f} KB)")
    return p


def load(name: str, default: Any = None) -> Any:
    p = _path(name)
    if not p.exists():
        return default
    try:
        return joblib.load(p)
    except Exception:  # noqa: BLE001
        logger.exception(f"[ml] failed to load {p}")
        return default


def exists(name: str) -> bool:
    return _path(name).exists()


def delete(name: str) -> bool:
    p = _path(name)
    if p.exists():
        p.unlink()
        logger.info(f"[ml] deleted model {p}")
        return True
    return False
