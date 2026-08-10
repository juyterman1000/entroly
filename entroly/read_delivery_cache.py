"""Session-isolated suppression for identical rendered file reads.

The cache sits *after* rendering.  It therefore suppresses a delivery only
when the source bytes, caller-visible read contract, and rendered output are
all identical to something already delivered in the same MCP session.

This is intentionally stricter than a path-only cache.  A query, budget, mode,
line range, diff baseline, file edit, client session, or explicit ``fresh``
read all prevent an unsafe cache hit.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _token_encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except (ImportError, ValueError):
        return None


def count_tokens(text: str) -> int:
    """Count o200k tokens, with an explicitly conservative local fallback."""
    if not text:
        return 0
    encoding = _token_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, (len(text) + 3) // 4)


def _contract_digest(contract: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return _sha256(canonical)


@dataclass(frozen=True)
class CacheDecision:
    text: str
    reference: str
    cache_hit: bool
    source_sha256: str
    output_sha256: str
    original_tokens: int
    delivered_tokens: int
    tokens_saved: int


@dataclass(frozen=True)
class _Entry:
    source_sha256: str
    output_sha256: str
    output_lines: int
    original_tokens: int


class ReadDeliveryCache:
    """Bounded delivery cache partitioned by MCP session and read contract."""

    def __init__(self, *, max_sessions: int = 64, max_entries_per_session: int = 256):
        if max_sessions < 1 or max_entries_per_session < 1:
            raise ValueError("cache bounds must be positive")
        self.max_sessions = max_sessions
        self.max_entries_per_session = max_entries_per_session
        self._sessions: OrderedDict[str, OrderedDict[str, _Entry]] = OrderedDict()

    @staticmethod
    def _reference(path: str, mode: str, entry: _Entry) -> str:
        del path, mode
        # Fifteen decimal digits encode the first 48 digest bits. Decimal
        # groups tokenize predictably (five groups plus ``~`` under o200k),
        # while 48 bits keeps accidental handle collisions negligible for a
        # bounded per-session cache. The full digest remains the lookup key;
        # this handle is only the agent-visible receipt.
        value = int(entry.output_sha256[:12], 16)
        return f"~{value:015d}"

    def deliver(
        self,
        *,
        session_id: str,
        path: str,
        mode: str,
        contract: Mapping[str, Any],
        source: str,
        output: str,
        fresh: bool = False,
    ) -> CacheDecision:
        """Return a compact reference only for an exact same-session delivery."""
        if not session_id:
            raise ValueError("session_id is required")

        source_digest = _sha256(source)
        output_digest = _sha256(output)
        original_tokens = count_tokens(output)
        key = _contract_digest(contract)

        entries = self._sessions.pop(session_id, OrderedDict())
        self._sessions[session_id] = entries
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

        prior = entries.get(key)
        exact_repeat = (
            not fresh
            and prior is not None
            and prior.source_sha256 == source_digest
            and prior.output_sha256 == output_digest
        )
        if exact_repeat:
            entries.move_to_end(key)
            reference = self._reference(path, mode, prior)
            delivered_tokens = count_tokens(reference)
            if delivered_tokens < original_tokens:
                return CacheDecision(
                    text=reference,
                    reference=reference,
                    cache_hit=True,
                    source_sha256=source_digest,
                    output_sha256=output_digest,
                    original_tokens=original_tokens,
                    delivered_tokens=delivered_tokens,
                    tokens_saved=original_tokens - delivered_tokens,
                )

        entries[key] = _Entry(
            source_sha256=source_digest,
            output_sha256=output_digest,
            output_lines=len(output.splitlines()),
            original_tokens=original_tokens,
        )
        entries.move_to_end(key)
        while len(entries) > self.max_entries_per_session:
            entries.popitem(last=False)

        entry = entries[key]
        reference = self._reference(path, mode, entry)
        return CacheDecision(
            text=output,
            reference=reference,
            cache_hit=False,
            source_sha256=source_digest,
            output_sha256=output_digest,
            original_tokens=original_tokens,
            delivered_tokens=original_tokens,
            tokens_saved=0,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "sessions": len(self._sessions),
            "entries": sum(len(entries) for entries in self._sessions.values()),
            "max_sessions": self.max_sessions,
            "max_entries_per_session": self.max_entries_per_session,
            "baseline": (
                "same-session rendered-output suppression; local token accounting, "
                "not a provider bill delta"
            ),
        }
