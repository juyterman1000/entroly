"""Fail-closed JSON and log hotfixes for :mod:`entroly.universal_compress`.

The public compatibility module imports these functions and installs them into
its legacy implementation so every existing SDK caller receives the corrected
behavior without duplicating the mature non-log compressors.
"""

from __future__ import annotations

import re
from typing import Any


_LOAD_BEARING_KEY = re.compile(
    r"(?:^|_)(?:id|ids|uuid|guid|key|ref|reference|code|codes|status|state|"
    r"error|errors|message|msg|reason|detail|details|amount|total|balance|"
    r"price|cost|fee|tax|currency|sku|isbn|account|invoice|order|"
    r"timestamp|time|date|created|updated|expires|version|hash|digest|"
    r"signature|token|url|uri|path|email|phone|port|line|column|signal|"
    r"errno)(?:$|_)",
    re.IGNORECASE,
)


def _normalise_json_key(key: str) -> str:
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    return re.sub(r"[^A-Za-z0-9]+", "_", split).strip("_").lower()


def _is_load_bearing_key(key: str) -> bool:
    return bool(_LOAD_BEARING_KEY.search(_normalise_json_key(key)))


def _json_to_schema(
    obj: Any,
    depth: int = 0,
    max_depth: int = 4,
    key: str = "",
) -> Any:
    """Elide repetitive structure without deleting answer-bearing fields."""
    protected_key = _is_load_bearing_key(key)
    if depth > max_depth:
        return obj if protected_key else "..."
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for index, (child_key, value) in enumerate(obj.items()):
            if index >= 20 and not _is_load_bearing_key(str(child_key)):
                continue
            result[str(child_key)] = _json_to_schema(
                value,
                depth + 1,
                max_depth,
                key=str(child_key),
            )
        return result
    if isinstance(obj, list):
        if not obj:
            return []
        if protected_key:
            return obj
        return [
            _json_to_schema(obj[0], depth + 1, max_depth, key=key),
            f"... ({len(obj)} items)",
        ]
    if isinstance(obj, str):
        if len(obj) <= 50 or protected_key:
            return obj
        return f"<str:{len(obj)}>"
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    return str(type(obj).__name__)


# Normalize only values whose role is recognizably incidental. A broad
# ``\d+ -> *`` rule is forbidden because it merges HTTP 404 with 500, exit 1
# with 137, and $100 with $100000.
_LOG_VARIABLE = re.compile(
    r"""
      [0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}
    | 0x[0-9a-fA-F]+
    | \b[0-9a-fA-F]{16,}\b
    | \b\d{1,3}(?:\.\d{1,3}){3}\b
    | \b\d+(?:\.\d+)?(?:ms|us|ns|kb|mb|gb)\b
    | \b(?:retry|attempt|counter|sequence|seq|request[_ -]?id|trace[_ -]?id)
      \s*[=: #]?\s*\d+\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_LOG_CRITICAL_NUMBER = re.compile(
    r"""
      \bHTTP(?:/\d(?:\.\d)?)?\s+\d{3}\b
    | \b(?:http[_ -]?status|status(?:_code)?|exit(?:ed)?[_ -]?(?:code|status)|
          signal|errno|error[_ -]?code|amount(?:_cents)?|total|balance|price|
          cost|fee|tax|port|version|line|column)
      \s*[=: #]?\s*[-+]?\d+(?:\.\d+)?\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _log_template(line: str) -> str:
    """Return a safe event key while preserving critical numeric distinctions."""
    if _LOG_CRITICAL_NUMBER.search(line):
        return line
    return _LOG_VARIABLE.sub("*", line)


def _compress_log_universal(text: str) -> str:
    """Collapse only events with the same complete, safety-preserving key."""
    lines = text.split("\n")
    seen: dict[str, int] = {}
    representatives: list[str] = []
    timestamp_prefix = re.compile(r"^\S+\s+\S+\s+")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        key = _log_template(timestamp_prefix.sub("", stripped))
        if key in seen:
            seen[key] += 1
        else:
            seen[key] = 1
            representatives.append(line)

    result: list[str] = []
    for line in representatives:
        key = _log_template(timestamp_prefix.sub("", line.strip()))
        count = seen.get(key, 1)
        result.append(f"{line}  [×{count}]" if count > 1 else line)
    return "\n".join(result)
