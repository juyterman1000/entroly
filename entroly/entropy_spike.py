"""Entropy-Spike Slicing — detect semantic boundaries via information density shifts.

Slides a window over text and computes per-window Shannon entropy. Points where
entropy changes sharply (spike_threshold standard deviations above the mean
delta) indicate transitions between semantic regions — e.g., from imports to
class definitions, from docstrings to algorithmic code.

These boundary points feed into the chunker as preferred split locations,
producing chunks that align with semantic structure rather than arbitrary token
counts.

Uses the Rust engine's Shannon entropy when available, falls back to a pure
Python implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EntropySpike:
    """A detected entropy transition point."""

    position: int
    delta: float
    left_entropy: float
    right_entropy: float


def _shannon_entropy_py(text: str) -> float:
    """Pure Python Shannon entropy in bits per character."""
    if not text:
        return 0.0
    counts: dict[int, int] = {}
    for b in text.encode("utf-8"):
        counts[b] = counts.get(b, 0) + 1
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _shannon_entropy(text: str) -> float:
    """Use Rust entropy when available, else pure Python."""
    try:
        from entroly_core import py_shannon_entropy
        return py_shannon_entropy(text)
    except (ImportError, AttributeError):
        return _shannon_entropy_py(text)


def detect_entropy_spikes(
    text: str,
    *,
    window_chars: int = 200,
    stride_chars: int = 50,
    spike_threshold: float = 1.5,
) -> list[EntropySpike]:
    """Detect points where entropy changes sharply.

    Slides a window of ``window_chars`` characters with ``stride_chars``
    step size. At each position, computes entropy of the left and right
    halves. A spike is declared when |right - left| exceeds
    ``spike_threshold`` standard deviations above the mean absolute delta.

    Args:
        text: Source text to analyze.
        window_chars: Size of the analysis window.
        stride_chars: Step between consecutive windows.
        spike_threshold: Number of std deviations for spike detection.

    Returns:
        List of detected spikes sorted by delta magnitude (strongest first).
    """
    if len(text) < window_chars * 2:
        return []

    half = window_chars // 2
    deltas: list[tuple[int, float, float, float]] = []

    pos = 0
    while pos + window_chars <= len(text):
        left = text[pos : pos + half]
        right = text[pos + half : pos + window_chars]
        left_h = _shannon_entropy(left)
        right_h = _shannon_entropy(right)
        delta = abs(right_h - left_h)
        deltas.append((pos + half, delta, left_h, right_h))
        pos += stride_chars

    if not deltas:
        return []

    mean_delta = sum(d[1] for d in deltas) / len(deltas)
    variance = sum((d[1] - mean_delta) ** 2 for d in deltas) / len(deltas)
    std_delta = math.sqrt(variance) if variance > 0 else 0.0

    threshold = mean_delta + spike_threshold * std_delta
    if threshold <= 0:
        return []

    spikes = [
        EntropySpike(
            position=pos,
            delta=delta,
            left_entropy=left_h,
            right_entropy=right_h,
        )
        for pos, delta, left_h, right_h in deltas
        if delta >= threshold
    ]

    spikes.sort(key=lambda s: s.delta, reverse=True)
    return spikes


def find_split_points(
    text: str,
    *,
    max_splits: int = 10,
    min_segment_chars: int = 100,
    window_chars: int = 200,
    stride_chars: int = 50,
    spike_threshold: float = 1.5,
) -> list[int]:
    """Return character positions where the text should be split.

    Detects entropy spikes and returns their positions as split candidates,
    filtered to maintain a minimum segment size. Positions are sorted in
    ascending order.

    These positions can be used by the chunker as preferred split points
    within large blocks that exceed the chunk token limit.
    """
    spikes = detect_entropy_spikes(
        text,
        window_chars=window_chars,
        stride_chars=stride_chars,
        spike_threshold=spike_threshold,
    )

    points: list[int] = []
    for spike in spikes[:max_splits * 2]:
        pos = spike.position
        nl = text.rfind("\n", max(0, pos - 40), pos + 40)
        if nl > 0:
            pos = nl + 1

        if not points or all(abs(pos - p) >= min_segment_chars for p in points):
            points.append(pos)
            if len(points) >= max_splits:
                break

    points.sort()
    return points
