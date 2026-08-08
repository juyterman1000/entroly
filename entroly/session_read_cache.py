"""Suppress re-delivery of content the agent has already been shown.

An agent re-reads the same file constantly: to check a signature, to confirm an
edit, to re-orient after a tool call. Every re-read costs full price even though
the bytes are identical to what was delivered minutes earlier.

An external context runtime addresses this with a time-bounded read cache, and
it works -- a warm re-read measured 2,171 tokens down to 8 (99.6%). It is also
*time*-bounded: the same file re-read later in the same session returned full
content again, because the entry had aged out.

This cache is keyed on the **content digest** instead, which is strictly better
on both failure modes:

  * it never expires while the content is unchanged, so a re-read an hour later
    is still free;
  * it invalidates *exactly* when the bytes change, so a stale reference is
    impossible by construction -- a modified file has a different digest and is
    delivered in full.

The reference that replaces the content must be honest and actionable. It names
the path, the size it stands for, the digest, and when it was first delivered,
so an agent can both trust it and ask for the content back. A reference that
merely said "cached" would silently look like an empty read.

Nothing here guesses. If the caller cannot prove the agent saw the content,
`record` is simply never called and delivery is unaffected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SessionReadCache", "CacheDecision"]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class CacheDecision:
    """What to deliver, and whether it was suppressed."""

    text: str
    suppressed: bool
    digest: str
    original_tokens: int
    delivered_tokens: int

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.delivered_tokens)


@dataclass
class _Entry:
    digest: str
    line_count: int
    token_count: int
    first_turn: int
    hits: int = 0


@dataclass
class SessionReadCache:
    """Per-session record of what has already been delivered verbatim.

    `max_entries` bounds memory; eviction is least-recently-delivered. Bounded
    rather than unbounded because a long session can touch thousands of files
    and this must never become the reason a process grows without limit.
    """

    max_entries: int = 4096
    # A reference is only worth emitting when it is much smaller than the
    # content it replaces. Below this the suppression is noise, and the agent
    # is better served by the real bytes.
    min_tokens_to_suppress: int = 200
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _turn: int = 0
    _suppressed: int = 0
    _tokens_saved: int = 0

    def advance_turn(self) -> None:
        self._turn += 1

    @staticmethod
    def _tokens(text: str) -> int:
        return max(1, len(text) // 4)

    PREAMBLE = (
        "[cached-ref] lines marked `~path NNNL #digest` are files already "
        "delivered verbatim in this session and unchanged since; re-read the "
        "path to expand one."
    )

    def _reference(self, path: str, entry: _Entry) -> str:
        """A reference an agent can act on, kept as small as that allows.

        The first version repeated the recovery instructions on every file and
        cost ~46 tokens each, against ~6 for a comparable external system. The
        instructions are identical for every entry, so they belong in
        `PREAMBLE`, emitted once per turn by the caller.

        What stays per-line is what actually varies and cannot be inferred:
        the path, the size it stands for, and a digest prefix. The digest is
        the part worth its tokens -- it lets an agent *verify* the content is
        unchanged rather than trust that a cache entry is still valid.
        """
        return f"~{path} {entry.line_count}L #{entry.digest[:8]}"

    def deliver(self, path: str, content: str) -> CacheDecision:
        """Decide what to emit for `path` this turn, and record the delivery.

        Returns the full content the first time, and a reference on any later
        turn where the bytes are unchanged. A changed file always re-delivers
        in full, because its digest no longer matches.
        """
        digest = _digest(content)
        original = self._tokens(content)

        entry = self._entries.get(path)
        unchanged = entry is not None and entry.digest == digest
        worth_it = original >= self.min_tokens_to_suppress

        if unchanged and worth_it and entry.first_turn != self._turn:
            entry.hits += 1
            # Refresh recency for eviction ordering.
            self._entries[path] = self._entries.pop(path)
            reference = self._reference(path, entry)
            delivered = self._tokens(reference)
            self._suppressed += 1
            self._tokens_saved += max(0, original - delivered)
            return CacheDecision(
                text=reference, suppressed=True, digest=digest,
                original_tokens=original, delivered_tokens=delivered,
            )

        # First delivery, changed content, or too small to be worth a reference.
        self._entries.pop(path, None)
        self._entries[path] = _Entry(
            digest=digest,
            line_count=len(content.splitlines()),
            token_count=original,
            first_turn=self._turn,
        )
        while len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        return CacheDecision(
            text=content, suppressed=False, digest=digest,
            original_tokens=original, delivered_tokens=original,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "tracked_paths": len(self._entries),
            "suppressed_deliveries": self._suppressed,
            "tokens_saved": self._tokens_saved,
            "turn": self._turn,
            "baseline": (
                "counts content already delivered verbatim in this session; "
                "local telemetry, not a provider bill delta"
            ),
        }
