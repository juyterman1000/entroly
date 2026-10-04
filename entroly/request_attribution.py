"""Request-scoped causal attribution for proxy outcomes.

Belief outcome attribution used to read ``proxy._last_injected_claim_ids``, a
single instance attribute overwritten by every request. ``/outcome`` is a
*separate* HTTP endpoint, so the sequence

    request A  -> claims A
    request B  -> claims B      (overwrites)
    outcome A  -> attributes A's result to B's beliefs

applied a durable Bayesian confidence update to the wrong beliefs and, on
failure, enqueued the wrong beliefs for reverification. It broke under ordinary
sequential use, not only under concurrency: any client issuing two completions
before reporting an outcome hit it.

This module holds the per-request state instead. It is deliberately a
dependency-free leaf: it imports nothing from ``entroly``, so it adds no edge to
the package's large import component.

Design notes:

* No "latest request" fallback exists, by construction. A missing, unknown or
  expired id yields ``None`` and the caller abstains.
* State is in-memory only. After a restart attribution abstains, which is the
  correct failure direction for an optional feature: a lost update is better
  than a wrong one.
* ``claim`` -> mutate -> ``commit`` rather than pop-then-mutate. A throwing
  Bayesian update leaves the entry retryable instead of silently dropping the
  outcome.
* CONSUMED entries are retained inside the same bounded map, so they serve as
  the duplicate-delivery tombstone without a second data structure.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

# Client-supplied correlation ids are reflected into a response header, so they
# must not be able to inject one. Conservative allowlist: a header value cannot
# contain CR, LF, or non-printable bytes, and unbounded length is a memory and
# log-injection concern.
_SAFE_REQUEST_ID = re.compile(r"\A[A-Za-z0-9._:-]{1,128}\Z")

DEFAULT_TTL_SECONDS = 900.0
DEFAULT_MAX_ENTRIES = 2048

PENDING = "PENDING"
IN_FLIGHT = "IN_FLIGHT"
CONSUMED = "CONSUMED"


def safe_request_id(candidate: object) -> str:
    """Return a header-safe request id, generating one when input is unusable.

    Never raises and never echoes an unvalidated value: an absent, malformed,
    over-long or control-character-bearing id is replaced with a fresh Entroly
    id rather than reflected. This is the only place a client string becomes a
    response header value.
    """
    if isinstance(candidate, str):
        text = candidate.strip()
        if _SAFE_REQUEST_ID.match(text):
            return text
    return uuid.uuid4().hex[:12]


@dataclass
class AttributionContext:
    """What a single request injected, and whether its outcome was applied."""

    claim_ids: tuple[str, ...]
    created_at: float
    expires_at: float
    state: str = PENDING
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class RequestAttributionStore:
    """Bounded TTL map from request id to injected claim ids.

    Thread-safe. Eviction is deterministic: oldest insertion first, which keeps
    behaviour reproducible in tests rather than depending on dict ordering or a
    heuristic.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._ttl = float(ttl_seconds)
        self._max = max(1, int(max_entries))
        self._entries: OrderedDict[str, AttributionContext] = OrderedDict()
        self._lock = threading.Lock()

    # ── write path ────────────────────────────────────────────────────────
    def record(self, request_id: str, claim_ids: list[str] | tuple[str, ...]) -> None:
        """Associate the claims a request injected with that request's id."""
        ids = tuple(str(c) for c in claim_ids if c)
        if not request_id or not ids:
            return
        now = time.monotonic()
        with self._lock:
            self._evict_expired_locked(now)
            self._entries.pop(request_id, None)  # re-record replaces, stays FIFO
            self._entries[request_id] = AttributionContext(
                claim_ids=ids, created_at=now, expires_at=now + self._ttl
            )
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    # ── outcome path ──────────────────────────────────────────────────────
    def claim(self, request_id: str) -> tuple[str, ...] | None:
        """Transition PENDING -> IN_FLIGHT and return the claims to attribute.

        Returns ``None`` -- meaning *abstain* -- when the id is missing,
        unknown, expired, already consumed, or already being attributed by
        another worker. Exactly one caller can win this transition, which is
        what makes a concurrent duplicate outcome mutate once.
        """
        if not request_id:
            return None
        now = time.monotonic()
        with self._lock:
            self._evict_expired_locked(now)
            entry = self._entries.get(request_id)
            if entry is None or entry.expires_at <= now:
                return None
            if entry.state != PENDING:
                return None
            entry.state = IN_FLIGHT
            return entry.claim_ids

    def commit(self, request_id: str) -> None:
        """Mark an attributed request CONSUMED so a redelivery cannot reapply."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None and entry.state == IN_FLIGHT:
                entry.state = CONSUMED

    def release(self, request_id: str) -> None:
        """Return a failed attribution to PENDING so a retry can succeed."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None and entry.state == IN_FLIGHT:
                entry.state = PENDING

    # ── introspection (diagnostics and tests) ─────────────────────────────
    def state_of(self, request_id: str) -> str | None:
        with self._lock:
            entry = self._entries.get(request_id)
            return None if entry is None else entry.state

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _evict_expired_locked(self, now: float) -> None:
        # Entries are inserted in time order, so expiry is a prefix scan.
        while self._entries:
            key = next(iter(self._entries))
            if self._entries[key].expires_at > now:
                return
            self._entries.popitem(last=False)


__all__ = [
    "AttributionContext",
    "CONSUMED",
    "IN_FLIGHT",
    "PENDING",
    "RequestAttributionStore",
    "safe_request_id",
]
