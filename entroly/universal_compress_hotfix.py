"""Fail-closed JSON and log hotfixes for :mod:`entroly.universal_compress`.

The public compatibility module installs these helpers into the mature legacy
compressor. JSON compression preserves bounded, load-bearing evidence from the
entire source (including late fields and later list records); log compression
normalises only recognisably incidental values and never strips non-timestamp
content from an event key.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any


_LOAD_BEARING_KEY = re.compile(
    r"(?:^|_)(?:id|ids|uuid|guid|key|ref|reference|code|codes|status|state|"
    r"error|errors|message|msg|reason|detail|details|amount|total|balance|"
    r"price|cost|fee|tax|currency|sku|isbn|account|invoice|order|"
    r"timestamp|time|date|created|updated|expires|version|hash|digest|"
    r"signature|token|url|uri|path|email|phone|port|line|column|signal|"
    r"errno|success|failed|failure)(?:$|_)",
    re.IGNORECASE,
)

_JSON_PROTECTED_SUMMARY_KEY = "__entroly_protected_values__"
_JSON_MAX_SCAN_NODES = 4096
_JSON_MAX_PROTECTED_VALUES = 256
_JSON_MAX_PROTECTED_BYTES = 16_384
_JSON_HARD_DEPTH = 32


def _normalise_json_key(key: str) -> str:
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    return re.sub(r"[^A-Za-z0-9]+", "_", split).strip("_").lower()


def _is_load_bearing_key(key: str) -> bool:
    return bool(_LOAD_BEARING_KEY.search(_normalise_json_key(key)))


def _json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _contains_load_bearing_key(value: Any) -> bool:
    """Return true when a bounded subtree contains a protected key.

    If the scan budget is exhausted, return true. The caller then preserves the
    subtree rather than making a false-negative safety decision.
    """

    stack = [value]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        if visited > _JSON_MAX_SCAN_NODES:
            return True
        if isinstance(node, dict):
            for key, child in node.items():
                if _is_load_bearing_key(str(key)):
                    return True
                if isinstance(child, (dict, list)):
                    stack.append(child)
        elif isinstance(node, list):
            stack.extend(item for item in node if isinstance(item, (dict, list)))
    return False


def _collect_load_bearing_values(value: Any) -> dict[str, list[Any]] | None:
    """Collect distinct protected values from a bounded JSON subtree.

    ``None`` means the scan exceeded a safety bound. Callers must preserve the
    original subtree in that case rather than emit an incomplete summary.
    """

    collected: dict[str, list[Any]] = defaultdict(list)
    fingerprints: dict[str, set[str]] = defaultdict(set)
    stack: list[tuple[str, Any, bool]] = [("", value, False)]
    visited = 0
    value_count = 0
    serialized_bytes = 0

    while stack:
        path, node, inherited = stack.pop()
        visited += 1
        if visited > _JSON_MAX_SCAN_NODES:
            return None

        if isinstance(node, dict):
            for key, child in reversed(list(node.items())):
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                protected = inherited or _is_load_bearing_key(key_text)
                stack.append((child_path, child, protected))
            continue

        if isinstance(node, list):
            for child in reversed(node):
                stack.append((path, child, inherited))
            continue

        if not inherited or not _json_scalar(node):
            continue

        canonical = json.dumps(
            node,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical in fingerprints[path]:
            continue
        fingerprints[path].add(canonical)
        collected[path].append(node)
        value_count += 1
        serialized_bytes += len(canonical.encode("utf-8"))
        if (
            value_count > _JSON_MAX_PROTECTED_VALUES
            or serialized_bytes > _JSON_MAX_PROTECTED_BYTES
        ):
            return None

    return dict(collected)


def _json_to_schema(
    obj: Any,
    depth: int = 0,
    max_depth: int = 4,
    key: str = "",
) -> Any:
    """Elide repetitive structure without deleting protected evidence.

    Late dictionary entries are retained when either their own key or a bounded
    descendant key is load-bearing. Repetitive lists keep a representative item
    plus a compact index of every distinct protected scalar. If that index would
    exceed the safety bounds, the list is preserved verbatim.
    """

    protected_key = _is_load_bearing_key(key)
    if protected_key:
        return obj
    if depth > _JSON_HARD_DEPTH:
        return obj

    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for index, (child_key, value) in enumerate(obj.items()):
            child_key_text = str(child_key)
            should_keep = (
                index < 20
                or _is_load_bearing_key(child_key_text)
                or _contains_load_bearing_key(value)
            )
            if not should_keep:
                continue
            result[child_key_text] = _json_to_schema(
                value,
                depth + 1,
                max_depth,
                key=child_key_text,
            )
        return result

    if isinstance(obj, list):
        if not obj:
            return []
        protected_values = _collect_load_bearing_values(obj)
        if protected_values is None:
            return obj
        output: list[Any] = [
            _json_to_schema(obj[0], depth + 1, max_depth, key=key),
            f"... ({len(obj)} items)",
        ]
        if protected_values:
            output.append({_JSON_PROTECTED_SUMMARY_KEY: protected_values})
        return output

    if depth > max_depth:
        return "..."
    if isinstance(obj, str):
        return obj if len(obj) <= 50 else f"<str:{len(obj)}>"
    if isinstance(obj, (bool, int, float)) or obj is None:
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

_TIMESTAMP_PREFIX = re.compile(
    r"""
    ^(?:
        \[[^\]\r\n]*\d{2}:\d{2}:\d{2}(?:[.,]\d+)?[^\]\r\n]*\]
      | \d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?
        (?:Z|[+-]\d{2}:?\d{2})?
      | \d{2}:\d{2}:\d{2}(?:[.,]\d+)?
    )\s+
    """,
    re.VERBOSE,
)


def _strip_log_prefix(line: str) -> str:
    """Strip only a recognized timestamp, preserving level and event text."""

    return _TIMESTAMP_PREFIX.sub("", line, count=1)


def _log_template(line: str) -> str:
    """Return a safe event key while preserving critical numeric distinctions."""

    if _LOG_CRITICAL_NUMBER.search(line):
        return line
    return _LOG_VARIABLE.sub("*", line)


def _compress_log_universal(text: str) -> str:
    """Collapse only events with the same complete, safety-preserving key."""

    seen: dict[str, int] = {}
    representatives: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = _log_template(_strip_log_prefix(stripped))
        if key in seen:
            seen[key] += 1
        else:
            seen[key] = 1
            representatives.append(line)

    result: list[str] = []
    for line in representatives:
        key = _log_template(_strip_log_prefix(line.strip()))
        count = seen.get(key, 1)
        result.append(f"{line}  [×{count}]" if count > 1 else line)
    return "\n".join(result)
