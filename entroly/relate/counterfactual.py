from __future__ import annotations

import re

_PAIRS = [
    ("not enabled", "enabled"),
    ("enabled", "not enabled"),
    ("allowed", "denied"),
    ("denied", "allowed"),
    ("before", "after"),
    ("after", "before"),
    ("include", "exclude"),
    ("exclude", "include"),
]


def make_counterfactual(text: str) -> str | None:
    spans: list[tuple[int, int, str, str]] = []
    for a, b in _PAIRS:
        for m in re.finditer(rf"\b{re.escape(a)}\b", text, flags=re.I):
            spans.append((m.start(), m.end(), a, b))
    filtered = []
    for item in spans:
        s, e, _, _ = item
        if any(os <= s and e <= oe and (oe - os) > (e - s) for os, oe, _, _ in spans):
            continue
        filtered.append(item)
    if len(filtered) != 1:
        return None
    s, e, _a, b = filtered[0]
    return text[:s] + b + text[e:]
