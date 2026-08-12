"""Content-blind detection and prevention of optimizer prefix interference.

Provider prompt caches reuse an exact leading token sequence.  A request can
therefore contain fewer tokens and still cost more when an optimizer rewrites
old history.  This module compares the prefix an append-only client made
available with the prefix Entroly actually forwards.

Only fixed-size SHA-256 block digests and byte counts survive an observation.
Prompt text, code, paths, message bodies, raw request/conversation identifiers,
and model names are never retained by the tracker or returned by :meth:`stats`.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping


_PROMPT_FIELDS = (
    "system",
    "instructions",
    "systemInstruction",
    "tools",
    "tool_choice",
    "messages",
    "input",
    "contents",
)


def _prompt_surface(body: Mapping[str, Any], provider: str) -> dict[str, Any]:
    """Return the prompt-only structure consumed by the streaming encoder."""
    prompt = [
        (field, body[field])
        for field in _PROMPT_FIELDS
        if field in body
    ]
    return {"provider": provider.lower().strip(), "prompt": prompt}


def _stream_block_digests(
    surface: Mapping[str, Any],
    *,
    block_bytes: int,
) -> tuple[tuple[str, ...], tuple[int, ...], int]:
    """Hash canonical JSON in bounded blocks without building a full copy."""
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    pending = bytearray()
    digests: list[str] = []
    lengths: list[int] = []
    total = 0
    for piece in encoder.iterencode(surface):
        encoded = piece.encode("utf-8")
        total += len(encoded)
        offset = 0
        while offset < len(encoded):
            take = min(block_bytes - len(pending), len(encoded) - offset)
            pending.extend(encoded[offset : offset + take])
            offset += take
            if len(pending) == block_bytes:
                digests.append(hashlib.sha256(pending).hexdigest())
                lengths.append(len(pending))
                pending.clear()
    if pending:
        digests.append(hashlib.sha256(pending).hexdigest())
        lengths.append(len(pending))
    return tuple(digests), tuple(lengths), total


@dataclass(frozen=True, slots=True)
class PrefixFingerprint:
    """Irreversible block identity for one outbound prompt surface."""

    block_digests: tuple[str, ...]
    block_lengths: tuple[int, ...]
    total_bytes: int

    @property
    def estimated_tokens(self) -> int:
        return (self.total_bytes + 3) // 4


def fingerprint_prompt(
    body: Mapping[str, Any],
    provider: str,
    *,
    block_bytes: int = 256,
) -> PrefixFingerprint:
    if block_bytes < 64:
        raise ValueError("block_bytes must be at least 64")
    digests, lengths, total = _stream_block_digests(
        _prompt_surface(body, provider),
        block_bytes=block_bytes,
    )
    return PrefixFingerprint(
        block_digests=digests,
        block_lengths=lengths,
        total_bytes=total,
    )


def common_prefix_bytes(
    previous: PrefixFingerprint,
    current: PrefixFingerprint,
) -> int:
    """Return a conservative common-prefix byte count at block granularity."""
    total = 0
    for previous_digest, current_digest, previous_length, current_length in zip(
        previous.block_digests,
        current.block_digests,
        previous.block_lengths,
        current.block_lengths,
    ):
        if previous_digest != current_digest or previous_length != current_length:
            break
        total += previous_length
    return total


@dataclass(frozen=True, slots=True)
class PrefixGuardDecision:
    action: str
    cache_warm: bool
    baseline_reusable_tokens: int
    candidate_reusable_tokens: int
    estimated_tokens_at_risk: int

    @property
    def preserved_baseline(self) -> bool:
        return self.action == "preserve_warm_prefix"


@dataclass(frozen=True, slots=True)
class PrefixContinuityObservation:
    status: str
    raw_reusable_prefix_tokens: int | None
    outbound_reusable_prefix_tokens: int | None
    estimated_optimizer_interference_tokens: int


@dataclass(slots=True)
class _PrefixState:
    raw: PrefixFingerprint
    outbound: PrefixFingerprint


class PrefixContinuityGuard:
    """Bounded hash-only continuity tracker with a conservative warm-cache guard."""

    def __init__(
        self,
        *,
        max_conversations: int = 2_000,
        block_bytes: int = 256,
        material_loss_tokens: int = 64,
    ) -> None:
        if max_conversations < 1:
            raise ValueError("max_conversations must be positive")
        if block_bytes < 64:
            raise ValueError("block_bytes must be at least 64")
        if material_loss_tokens < 1:
            raise ValueError("material_loss_tokens must be positive")
        self.max_conversations = max_conversations
        self.block_bytes = block_bytes
        self.material_loss_tokens = material_loss_tokens
        self._states: OrderedDict[str, _PrefixState] = OrderedDict()
        self._lock = threading.RLock()
        self._observations = 0
        self._cold_starts = 0
        self._preserved = 0
        self._degraded = 0
        self._improved = 0
        self._interference_tokens = 0
        self._guard_interventions = 0
        self._guard_tokens_preserved = 0

    def _fingerprint(
        self,
        body: Mapping[str, Any],
        provider: str,
    ) -> PrefixFingerprint:
        return fingerprint_prompt(body, provider, block_bytes=self.block_bytes)

    @staticmethod
    def _state_key(conversation_id: str) -> str:
        return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()

    def choose(
        self,
        conversation_id: str,
        *,
        provider: str,
        baseline_body: Mapping[str, Any],
        candidate_body: Mapping[str, Any],
        cache_warm: bool,
    ) -> tuple[dict[str, Any], PrefixGuardDecision]:
        """Preserve the safer optional-transform candidate for a warm cache.

        ``baseline_body`` must already include required security, recovery, and
        emergency-rescue mutations.  The guard is intentionally unable to undo
        those changes; it arbitrates optional transformations only.
        """
        baseline = dict(baseline_body)
        candidate = dict(candidate_body)
        state_key = self._state_key(conversation_id)
        with self._lock:
            previous = self._states.get(state_key)
            if previous is None:
                return candidate, PrefixGuardDecision(
                    action="observe",
                    cache_warm=cache_warm,
                    baseline_reusable_tokens=0,
                    candidate_reusable_tokens=0,
                    estimated_tokens_at_risk=0,
                )
            baseline_common = common_prefix_bytes(
                previous.outbound,
                self._fingerprint(baseline, provider),
            ) // 4
            candidate_common = common_prefix_bytes(
                previous.outbound,
                self._fingerprint(candidate, provider),
            ) // 4
            at_risk = max(0, baseline_common - candidate_common)
            preserve = cache_warm and at_risk >= self.material_loss_tokens
            if preserve:
                self._guard_interventions += 1
                self._guard_tokens_preserved += at_risk
            return (baseline if preserve else candidate), PrefixGuardDecision(
                action="preserve_warm_prefix" if preserve else "allow_candidate",
                cache_warm=cache_warm,
                baseline_reusable_tokens=baseline_common,
                candidate_reusable_tokens=candidate_common,
                estimated_tokens_at_risk=at_risk,
            )

    def observe(
        self,
        conversation_id: str,
        *,
        provider: str,
        raw_body: Mapping[str, Any],
        outbound_body: Mapping[str, Any],
    ) -> PrefixContinuityObservation:
        """Record an outbound request without retaining its prompt content."""
        raw = self._fingerprint(raw_body, provider)
        outbound = self._fingerprint(outbound_body, provider)
        state_key = self._state_key(conversation_id)
        with self._lock:
            previous = self._states.get(state_key)
            self._observations += 1
            if previous is None:
                self._cold_starts += 1
                observation = PrefixContinuityObservation(
                    status="first_observation",
                    raw_reusable_prefix_tokens=None,
                    outbound_reusable_prefix_tokens=None,
                    estimated_optimizer_interference_tokens=0,
                )
            else:
                raw_common_bytes = common_prefix_bytes(previous.raw, raw)
                outbound_common_bytes = common_prefix_bytes(
                    previous.outbound,
                    outbound,
                )
                raw_common = raw_common_bytes // 4
                outbound_common = outbound_common_bytes // 4
                raw_disrupted = max(
                    0,
                    previous.raw.total_bytes - raw_common_bytes,
                ) // 4
                outbound_disrupted = max(
                    0,
                    previous.outbound.total_bytes - outbound_common_bytes,
                ) // 4
                interference = max(0, outbound_disrupted - raw_disrupted)
                material = interference >= self.material_loss_tokens
                if material:
                    status = "prefix_degraded"
                    self._degraded += 1
                    self._interference_tokens += interference
                elif raw_disrupted > outbound_disrupted + self.material_loss_tokens:
                    status = "prefix_improved"
                    self._improved += 1
                else:
                    status = "prefix_preserved"
                    self._preserved += 1
                observation = PrefixContinuityObservation(
                    status=status,
                    raw_reusable_prefix_tokens=raw_common,
                    outbound_reusable_prefix_tokens=outbound_common,
                    estimated_optimizer_interference_tokens=(
                        interference if material else 0
                    ),
                )
            self._states[state_key] = _PrefixState(raw=raw, outbound=outbound)
            self._states.move_to_end(state_key)
            while len(self._states) > self.max_conversations:
                self._states.popitem(last=False)
            return observation

    def stats(self) -> dict[str, int | float | str | bool]:
        with self._lock:
            comparable = self._preserved + self._degraded + self._improved
            return {
                "measurement": "local_hash_only_prefix_estimate",
                "content_retained": False,
                "identifiers_exposed": False,
                "observations": self._observations,
                "cold_starts": self._cold_starts,
                "comparable_transitions": comparable,
                "prefix_preserved": self._preserved,
                "prefix_degraded": self._degraded,
                "prefix_improved": self._improved,
                "continuity_rate": (
                    (self._preserved + self._improved) / comparable
                    if comparable
                    else 0.0
                ),
                "estimated_optimizer_interference_tokens": self._interference_tokens,
                "guard_interventions": self._guard_interventions,
                "estimated_prefix_tokens_preserved": self._guard_tokens_preserved,
                "block_bytes": self.block_bytes,
                "material_loss_tokens": self.material_loss_tokens,
            }


__all__ = [
    "PrefixContinuityGuard",
    "PrefixContinuityObservation",
    "PrefixFingerprint",
    "PrefixGuardDecision",
    "common_prefix_bytes",
    "fingerprint_prompt",
]
