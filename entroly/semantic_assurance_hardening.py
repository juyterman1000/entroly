"""Adversarial hardening for Entroly semantic assurance internals.

The base semantic layer intentionally exposes pure helpers. This module tightens
two edge contracts without duplicating the provider pipeline:

- a reminder region is removed only when its XML-like boundaries are provable;
  a regex may never consume across another reminder tag;
- unusual mapping-shaped historical tool results are converted to a content
  block list before evidence-preserving retirement.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping

from . import semantic_assurance as _semantic

_REMINDER_REGION = re.compile(
    r"<system-reminder(?:\s[^>]*)?>"
    r"(?:(?!</?system-reminder\b).)*?"
    r"</system-reminder>",
    re.IGNORECASE | re.DOTALL,
)


def _harness_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    position = 0
    matched = False
    while position < len(stripped):
        while position < len(stripped) and stripped[position].isspace():
            position += 1
        match = _REMINDER_REGION.match(stripped, position)
        if match is None:
            return False
        matched = True
        position = match.end()
        while position < len(stripped) and stripped[position].isspace():
            position += 1
    return matched


def _line_isolated_prefix(prefix: str) -> bool:
    if not prefix:
        return True
    line_tail = prefix.rsplit("\n", 1)[-1]
    return not line_tail.strip()


def conservative_purify_block(text: str) -> tuple[str, int]:
    """Strip only structurally isolated trailing/whole reminder regions."""
    if _harness_only(text):
        return "", len(list(_REMINDER_REGION.finditer(text)))

    current = text
    removed = 0
    while current:
        end = len(current.rstrip())
        candidate = current[:end]
        # Find only the last non-nested reminder region and require it to reach
        # the end. Text before it is preserved byte-for-byte except whitespace
        # on the reminder's own line.
        matches = list(_REMINDER_REGION.finditer(candidate))
        if not matches:
            break
        last = matches[-1]
        if last.end() != len(candidate):
            break
        prefix = candidate[: last.start()]
        # A closed reminder literal inside a user sentence is still user text.
        if not _line_isolated_prefix(prefix):
            break
        current = prefix.rstrip()
        removed += 1
    return current, removed


def install_semantic_assurance_hardening() -> None:
    purify = _semantic._purify_block
    if not hasattr(purify, "__entroly_conservative_intent_original__"):
        conservative_purify_block.__entroly_conservative_intent_original__ = purify
        _semantic._purify_block = conservative_purify_block

    result_repair = _semantic._historical_tool_result
    if not hasattr(result_repair, "__entroly_shape_safe_original__"):
        def shape_safe(block: Mapping[str, object]):
            content = block.get("content", "")
            if isinstance(content, Mapping):
                copied = copy.deepcopy(dict(block))
                copied["content"] = [copy.deepcopy(dict(content))]
                return result_repair(copied)
            return result_repair(block)

        shape_safe.__entroly_shape_safe_original__ = result_repair
        _semantic._historical_tool_result = shape_safe


__all__ = [
    "conservative_purify_block",
    "install_semantic_assurance_hardening",
]
