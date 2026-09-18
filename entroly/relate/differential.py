from __future__ import annotations

from .types import DifferentialSpan


def _bounds(a: str, b: str) -> tuple[int, int]:
    prefix = 0
    limit = min(len(a), len(b))
    while prefix < limit and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < (limit - prefix) and a[len(a) - 1 - suffix] == b[len(b) - 1 - suffix]:
        suffix += 1
    return prefix, len(a) - suffix


def extract_differential_spans(candidate_id: str, text: str, other_text: str, *, context_chars: int = 80) -> tuple[DifferentialSpan, ...]:
    """Return exact differing spans of text relative to other_text."""
    start, end = _bounds(text, other_text)
    if start == end and len(text) == len(other_text):
        return ()
    before = text[max(0, start - context_chars):start]
    after = text[end:min(len(text), end + context_chars)]
    return (DifferentialSpan(candidate_id, start, end, text[start:end], before, after),)
