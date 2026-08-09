"""Framework-neutral, contract-preserving request compression."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RequestCompressionResult:
    payload: dict[str, Any]
    changed: bool
    messages_key: str | None
    estimated_tokens_before: int
    estimated_tokens_after: int


def _tokens(value: Any) -> int:
    return max(1, len(json.dumps(value, ensure_ascii=False, default=str)) // 4)


def compress_request_payload(
    payload: Mapping[str, Any],
    *,
    budget: int,
    preserve_last_n: int = 4,
) -> RequestCompressionResult:
    """Compress message content without changing tools or provider controls."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    output = copy.deepcopy(dict(payload))
    key = next(
        (
            candidate
            for candidate in ("messages", "input")
            if isinstance(output.get(candidate), list)
            and all(isinstance(item, dict) for item in output[candidate])
        ),
        None,
    )
    before = _tokens(output)
    if key is None:
        return RequestCompressionResult(output, False, None, before, before)

    from ..sdk import compress_messages

    original_messages = output[key]
    output[key] = compress_messages(
        original_messages,
        budget=budget,
        preserve_last_n=preserve_last_n,
        model=str(output.get("model") or ""),
    )
    after = _tokens(output)
    changed = output[key] != original_messages
    return RequestCompressionResult(output, changed, key, before, after)
