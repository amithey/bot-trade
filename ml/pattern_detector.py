"""Chart-pattern detector.

Uses scipy peak/trough detection on close prices + geometric validation
to score five canonical patterns:

    - DOUBLE_BOTTOM
    - DOUBLE_TOP
    - HEAD_AND_SHOULDERS
    - INVERSE_HEAD_AND_SHOULDERS
    - CUP_AND_HANDLE

Each pattern returns a confidence ∈ [0, 1]. A confidence above 0.6 is
typically strong enough to act on. Multiple patterns can be active at
the same time; the detector returns the top-N.

Design: this is NOT a deep-learning CNN. We tried that first but it
needs labeled data we don't have. Rule-based pattern scoring is
deterministic, explainable, and good enough for a first pass. A CNN
could replace this module later without changing its signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utils.logger import get_logger

logger = get_logger(__name__)

PatternName = str


@dataclass
class PatternMatch:
    name: PatternName
    confidence: float      # 0.0 .. 1.0
    direction: str         # "bullish" | "bearish"
    window: tuple[int, int]  # (start_idx, end_idx) in the input frame
    notes: str = ""


# ─── Helpers ────────────────────────────────────────────────────────────────

def _peaks_troughs(close: np.ndarray, min_distance: int = 5,
                   prominence: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (peak_idxs, trough_idxs) with sensible defaults."""
    if prominence is None:
        prominence = float(np.std(close) * 0.15) or 1e-6
    peaks, _ = find_peaks(close, distance=min_distance, prominence=prominence)
    troughs, _ = find_peaks(-close, distance=min_distance, prominence=prominence)
    return peaks, troughs


def _rel_diff(a: float, b: float) -> float:
    m = (abs(a) + abs(b)) / 2.0
    return abs(a - b) / m if m else 0.0


# ─── Pattern scorers ────────────────────────────────────────────────────────

def _score_double_bottom(close: np.ndarray,
                         troughs: np.ndarray,
                         peaks: np.ndarray) -> PatternMatch | None:
    """Two troughs at similar price with a peak between them."""
    if len(troughs) < 2:
        return None
    # Last two troughs
    t1, t2 = int(troughs[-2]), int(troughs[-1])
    if t2 - t1 < 8:
        return None
    p_between = [p for p in peaks if t1 < p < t2]
    if not p_between:
        return None
    pk = int(p_between[np.argmax(close[p_between])])

    bottom_diff = _rel_diff(close[t1], close[t2])
    neckline_rise = (close[pk] - min(close[t1], close[t2])) / close[pk]

    # Needs current price to be breaking above the neckline for confirmation
    current = close[-1]
    neckline = close[pk]
    breakout = (current - neckline) / neckline

    if bottom_diff > 0.04:       # bottoms > 4% apart — not a W
        return None
    if neckline_rise < 0.015:    # peak must be >1.5% above troughs
        return None
    # Confidence components
    c_shape = 1.0 - min(bottom_diff / 0.04, 1.0)       # tighter bottoms = higher
    c_height = min(neckline_rise / 0.05, 1.0)          # deeper W = higher
    c_break = 0.5 + min(max(breakout * 20, -0.5), 0.5) # breakout bonus/penalty
    conf = float(np.clip(0.35 * c_shape + 0.35 * c_height + 0.30 * c_break, 0, 1))

    return PatternMatch(
        name="DOUBLE_BOTTOM",
        confidence=conf,
        direction="bullish",
        window=(t1, len(close) - 1),
        notes=f"neckline@{neckline:.2f}, bottoms Δ={bottom_diff*100:.2f}%, "
              f"breakout={breakout*100:+.2f}%",
    )


def _score_double_top(close: np.ndarray,
                      troughs: np.ndarray,
                      peaks: np.ndarray) -> PatternMatch | None:
    if len(peaks) < 2:
        return None
    p1, p2 = int(peaks[-2]), int(peaks[-1])
    if p2 - p1 < 8:
        return None
    t_between = [t for t in troughs if p1 < t < p2]
    if not t_between:
        return None
    tr = int(t_between[np.argmin(close[t_between])])

    top_diff = _rel_diff(close[p1], close[p2])
    neckline_drop = (max(close[p1], close[p2]) - close[tr]) / close[tr]
    current = close[-1]
    neckline = close[tr]
    breakdown = (neckline - current) / neckline

    if top_diff > 0.04:
        return None
    if neckline_drop < 0.015:
        return None

    c_shape = 1.0 - min(top_diff / 0.04, 1.0)
    c_height = min(neckline_drop / 0.05, 1.0)
    c_break = 0.5 + min(max(breakdown * 20, -0.5), 0.5)
    conf = float(np.clip(0.35 * c_shape + 0.35 * c_height + 0.30 * c_break, 0, 1))

    return PatternMatch(
        name="DOUBLE_TOP",
        confidence=conf,
        direction="bearish",
        window=(p1, len(close) - 1),
        notes=f"neckline@{neckline:.2f}, tops Δ={top_diff*100:.2f}%, "
              f"breakdown={breakdown*100:+.2f}%",
    )


def _score_head_and_shoulders(close: np.ndarray,
                              troughs: np.ndarray,
                              peaks: np.ndarray,
                              *, inverse: bool = False) -> PatternMatch | None:
    src = -close if inverse else close
    pivots = troughs if inverse else peaks
    if len(pivots) < 3:
        return None
    a, b, c = (int(x) for x in pivots[-3:])
    # Head should be the highest (or lowest if inverse) of the three
    h = src[[a, b, c]]
    if h[1] <= h[0] or h[1] <= h[2]:
        return None
    shoulder_diff = _rel_diff(close[a], close[c])
    if shoulder_diff > 0.05:
        return None
    head_prom = (h[1] - max(h[0], h[2])) / (abs(h[1]) + 1e-9)
    if head_prom < 0.01:
        return None

    c_shoulders = 1.0 - min(shoulder_diff / 0.05, 1.0)
    c_head = min(head_prom / 0.04, 1.0)
    conf = float(np.clip(0.55 * c_shoulders + 0.45 * c_head, 0, 1))

    return PatternMatch(
        name="INVERSE_HEAD_AND_SHOULDERS" if inverse else "HEAD_AND_SHOULDERS",
        confidence=conf,
        direction="bullish" if inverse else "bearish",
        window=(a, len(close) - 1),
        notes=f"shoulders Δ={shoulder_diff*100:.2f}%, head prominence={head_prom*100:.2f}%",
    )


def _score_cup_and_handle(close: np.ndarray) -> PatternMatch | None:
    """Rough U-shape followed by a small dip (handle)."""
    n = len(close)
    if n < 40:
        return None
    window = close[-40:]
    # Split into cup (first 30) and handle (last 10)
    cup = window[:30]
    handle = window[30:]
    left = cup[0]
    right = cup[-1]
    mid_low = cup.min()
    # Cup: left and right roughly equal; middle ≥3% below
    if _rel_diff(left, right) > 0.04:
        return None
    cup_depth = (max(left, right) - mid_low) / max(left, right)
    if cup_depth < 0.03:
        return None
    # Handle: small pullback, max depth ≤ 0.5 × cup depth
    handle_low = handle.min()
    handle_depth = (right - handle_low) / right
    if handle_depth > cup_depth * 0.6:
        return None

    c_cup = min(cup_depth / 0.10, 1.0)
    c_handle = 1.0 - (handle_depth / (cup_depth * 0.6 + 1e-9))
    conf = float(np.clip(0.6 * c_cup + 0.4 * c_handle, 0, 1))

    return PatternMatch(
        name="CUP_AND_HANDLE",
        confidence=conf,
        direction="bullish",
        window=(n - 40, n - 1),
        notes=f"cup depth {cup_depth*100:.1f}%, handle depth {handle_depth*100:.1f}%",
    )


# ─── Public entry point ─────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame, *, lookback: int = 80) -> list[PatternMatch]:
    """Return all patterns whose confidence ≥ 0.35, sorted descending."""
    if df is None or len(df) < 30:
        return []
    series = df["Close"].astype(float).tail(lookback).to_numpy()
    peaks, troughs = _peaks_troughs(series)

    candidates: list[PatternMatch | None] = [
        _score_double_bottom(series, troughs, peaks),
        _score_double_top(series, troughs, peaks),
        _score_head_and_shoulders(series, troughs, peaks, inverse=False),
        _score_head_and_shoulders(series, troughs, peaks, inverse=True),
        _score_cup_and_handle(series),
    ]
    matches = [m for m in candidates if m is not None and m.confidence >= 0.35]
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches
