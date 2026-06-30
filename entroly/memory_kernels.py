"""Production fallback kernels for Entroly Memory Fabric.

These kernels make MemoryFabric fully usable without a native `entroly_core`
wheel. Native Rust kernels remain the acceleration path, but the Python package
now has a deterministic first-party implementation for:

- SCHIPC-style redundant inter-agent traffic suppression,
- compliance gating for sensitive / injected memory traffic,
- pollination-style learned lesson sharing.

The implementations are intentionally dependency-free, deterministic, and small
enough to audit. They are not meant to beat the Rust kernels on throughput; they
are meant to provide a production-safe baseline everywhere Python runs.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable

from .memory import MemoryOS


def _fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def simhash64(text: str) -> int:
    """Return a deterministic 64-bit SimHash for text."""
    weights = [0] * 64
    tokens = [t for t in text.lower().replace("_", " ").split() if t]
    if not tokens:
        tokens = [text]
    for token in tokens:
        h = _fnv1a64(token.encode("utf-8", errors="ignore"))
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            out |= 1 << bit
    return out


def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


@dataclass(slots=True)
class SchipcBus:
    """Dependency-free SCHIPC-style redundant message suppressor."""

    novelty_threshold: int = 4
    window_size: int = 256
    softcap: float = 8.0
    _rings: dict[int, Deque[int]] = field(default_factory=dict)
    total_sent: int = 0
    total_delivered: int = 0
    total_suppressed: int = 0

    def register_receiver(self, agent_id: int) -> None:
        self._rings.setdefault(int(agent_id), deque(maxlen=self.window_size))

    def send(self, sender_id: int, receiver_id: int, content: str) -> dict[str, object]:
        self.total_sent += 1
        receiver_id = int(receiver_id)
        self.register_receiver(receiver_id)
        fp = simhash64(content)
        ring = self._rings[receiver_id]
        min_d = min((hamming64(fp, old) for old in ring), default=64)
        novelty = self.softcap * math.tanh(min_d / self.softcap)
        entropy_approx = novelty / 64.0
        delivered = novelty > float(self.novelty_threshold)
        if delivered:
            ring.append(fp)
            self.total_delivered += 1
            reason = "novel"
        else:
            self.total_suppressed += 1
            reason = "redundant"
        return {
            "delivered": delivered,
            "reason": reason,
            "hamming": min_d,
            "novelty_score": novelty,
            "entropy_approx": entropy_approx,
            "sender_id": int(sender_id),
            "receiver_id": receiver_id,
            "fingerprint": fp,
        }

    def broadcast(self, sender_id: int, content: str) -> list[dict[str, object]]:
        return [self.send(sender_id, rid, content) for rid in list(self._rings) if rid != sender_id]

    def flush_sketch(self, agent_id: int) -> None:
        self._rings.pop(int(agent_id), None)

    def delivery_rate(self) -> float:
        return 1.0 if self.total_sent == 0 else self.total_delivered / self.total_sent

    def suppression_rate(self) -> float:
        return 1.0 - self.delivery_rate()

    def stats(self) -> dict[str, object]:
        return {
            "total_sent": self.total_sent,
            "total_delivered": self.total_delivered,
            "total_suppressed": self.total_suppressed,
            "delivery_rate": self.delivery_rate(),
            "suppression_rate": self.suppression_rate(),
            "registered_agents": len(self._rings),
            "novelty_threshold": self.novelty_threshold,
            "softcap": self.softcap,
            "window_size": self.window_size,
            "memory_bytes": len(self._rings) * self.window_size * 8,
        }


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last_refill: float


@dataclass(slots=True)
class ComplianceKernel:
    """Production fallback compliance gate for memory/message traffic."""

    pii_blocking: bool = True
    injection_blocking: bool = True
    rate_limit_mps: float = 100.0
    audit_capacity: int = 10_000
    _buckets: dict[int, _Bucket] = field(default_factory=dict)
    _audit: Deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=10_000))
    total_allowed: int = 0
    total_blocked_pii: int = 0
    total_blocked_injection: int = 0
    total_blocked_rate: int = 0

    def __post_init__(self) -> None:
        self.rate_limit_mps = max(float(self.rate_limit_mps), 1.0)
        self._audit = deque(maxlen=self.audit_capacity)

    def check_message(self, sender_id: int, receiver_id: int, content: str) -> dict[str, object]:
        findings = MemoryOS(default_budget=1).scan_safety(content).findings
        pii_types = sorted({f.kind for f in findings})
        if self.injection_blocking and any(f.kind == "prompt_injection" for f in findings):
            self.total_blocked_injection += 1
            self._append_audit(sender_id, receiver_id, "injection_blocked", pii_types)
            return {"allowed": False, "reason": "injection_detected", "pii_types": pii_types}
        if self.pii_blocking and pii_types:
            self.total_blocked_pii += 1
            self._append_audit(sender_id, receiver_id, "pii_blocked", pii_types)
            return {"allowed": False, "reason": "pii_detected", "pii_types": pii_types}
        if not self._consume_rate_token(int(sender_id)):
            self.total_blocked_rate += 1
            self._append_audit(sender_id, receiver_id, "rate_limited", [])
            return {"allowed": False, "reason": "rate_limited", "pii_types": []}
        self.total_allowed += 1
        self._append_audit(sender_id, receiver_id, "allowed", [])
        return {"allowed": True, "reason": "ok", "pii_types": []}

    def stats(self) -> dict[str, object]:
        total = self.total_allowed + self.total_blocked_pii + self.total_blocked_injection + self.total_blocked_rate
        denial_rate = 0.0 if total == 0 else (total - self.total_allowed) / total
        return {
            "total_allowed": self.total_allowed,
            "blocked_pii": self.total_blocked_pii,
            "blocked_injection": self.total_blocked_injection,
            "blocked_rate": self.total_blocked_rate,
            "denial_rate": denial_rate,
            "audit_entries": len(self._audit),
            "implementation": "python_fallback",
        }

    def export_audit(self) -> list[dict[str, object]]:
        return list(self._audit)

    def delete_sender_audit(self, sender_id: int) -> None:
        sender_id = int(sender_id)
        self._audit = deque((e for e in self._audit if e["sender"] != sender_id), maxlen=self.audit_capacity)
        self._buckets.pop(sender_id, None)

    def _consume_rate_token(self, sender_id: int) -> bool:
        now = time.monotonic()
        bucket = self._buckets.setdefault(sender_id, _Bucket(tokens=10.0, last_refill=now))
        elapsed = max(0.0, now - bucket.last_refill)
        bucket.tokens = min(10.0, bucket.tokens + elapsed * self.rate_limit_mps)
        bucket.last_refill = now
        if bucket.tokens < 1.0:
            return False
        bucket.tokens -= 1.0
        return True

    def _append_audit(self, sender_id: int, receiver_id: int, decision: str, pii_types: Iterable[str]) -> None:
        self._audit.append(
            {
                "ts": time.time_ns(),
                "sender": int(sender_id),
                "receiver": int(receiver_id),
                "decision": decision,
                "pii": list(pii_types),
            }
        )


@dataclass(slots=True)
class _PollinatorState:
    raw_eagerness: float = 0.0
    share_probability: float = 0.5
    active_shares: dict[str, int] = field(default_factory=dict)
    total_shares: int = 0
    total_rewards: int = 0
    lessons_given: int = 0
    lessons_received: int = 0


@dataclass(slots=True)
class _Lesson:
    description: str
    success: bool
    surprise: float
    domain: str
    tick: int


@dataclass(slots=True)
class PollinationKernel:
    """Dependency-free TD(0)-style learned lesson sharing kernel."""

    alpha: float = 0.1
    gamma: float = 0.9
    temperature: float = 1.0
    agents: dict[str, _PollinatorState] = field(default_factory=dict)
    pending: dict[str, list[_Lesson]] = field(default_factory=dict)
    current_tick: int = 0
    total_lessons_shared: int = 0

    def register_agent(self, agent_id: str) -> None:
        self.agents.setdefault(agent_id, _PollinatorState())

    def should_share(self, agent_id: str, current_surprise: float = 0.0) -> bool:
        state = self.agents.get(agent_id)
        if state is None:
            return False
        random_val = (_fnv1a64(f"{agent_id}:{self.current_tick}".encode()) % 1000) / 1000.0
        effective = min(1.0, state.share_probability + current_surprise * 0.5)
        return random_val < effective

    def record_lesson(
        self,
        agent_id: str,
        description: str,
        success: bool = True,
        surprise: float = 0.0,
        domain: str = "general",
    ) -> None:
        self.register_agent(agent_id)
        self.pending.setdefault(agent_id, []).append(_Lesson(description, bool(success), float(surprise), domain, self.current_tick))
        self.agents[agent_id].lessons_given += 1

    def share(self, from_agent: str, to_agent: str) -> int:
        self.register_agent(from_agent)
        self.register_agent(to_agent)
        count = len(self.pending.get(from_agent, []))
        if count == 0:
            return 0
        self.agents[from_agent].active_shares[to_agent] = self.current_tick
        self.agents[from_agent].total_shares += 1
        self.agents[to_agent].lessons_received += count
        self.total_lessons_shared += count
        return count

    def reward(self, sharer_id: str, receiver_id: str, reward: float) -> None:
        state = self.agents.get(sharer_id)
        if state is None:
            return
        if receiver_id in state.active_shares:
            state.active_shares.pop(receiver_id, None)
        current_v = state.raw_eagerness
        td_error = float(reward) + self.gamma * current_v - current_v
        state.raw_eagerness += self.alpha * td_error
        state.share_probability = 1.0 / (1.0 + math.exp(-state.raw_eagerness / max(0.001, self.temperature)))
        state.total_rewards += 1

    def tick(self) -> None:
        self.current_tick += 1

    def share_probability(self, agent_id: str) -> float:
        return self.agents.get(agent_id, _PollinatorState()).share_probability

    def value(self, agent_id: str) -> float:
        return self.agents.get(agent_id, _PollinatorState()).raw_eagerness

    def stats(self) -> dict[str, object]:
        return {
            "total_agents": len(self.agents),
            "total_packs_created": len(self.pending),
            "total_packs_ingested": 0,
            "total_lessons_shared": self.total_lessons_shared,
            "current_tick": self.current_tick,
            "implementation": "python_fallback",
            "agents": [
                {
                    "agent_id": agent_id,
                    "share_probability": state.share_probability,
                    "raw_eagerness": state.raw_eagerness,
                    "total_shares": state.total_shares,
                    "total_rewards": state.total_rewards,
                    "lessons_given": state.lessons_given,
                    "lessons_received": state.lessons_received,
                }
                for agent_id, state in sorted(self.agents.items())
            ],
        }


__all__ = [
    "ComplianceKernel",
    "PollinationKernel",
    "SchipcBus",
    "hamming64",
    "simhash64",
]
