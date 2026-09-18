from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return hashlib.sha256(_canon(value).encode("utf-8")).hexdigest()


class ConstraintKind(str, Enum):
    OBLIGATION = "obligation"
    EXCLUSION = "exclusion"
    CONDITION = "condition"


@dataclass(frozen=True, order=True)
class ConstraintSpan:
    kind: ConstraintKind
    text: str
    start: int
    end: int
    source: str = "trusted_task"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid constraint span")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class AuthorityEnvelope:
    allowed: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    source: str = "external_policy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed", tuple(sorted(set(self.allowed))))
        object.__setattr__(self, "denied", tuple(sorted(set(self.denied))))

    def permits(self, capability: str) -> bool:
        return capability in self.allowed and capability not in self.denied

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": list(self.allowed), "denied": list(self.denied), "source": self.source}


@dataclass(frozen=True)
class QueryContract:
    task: str
    obligations: tuple[ConstraintSpan, ...]
    exclusions: tuple[ConstraintSpan, ...]
    conditions: tuple[ConstraintSpan, ...]
    authority: AuthorityEnvelope
    compiler_version: str = "relate-x.constraint-compiler.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "obligations": [x.to_dict() for x in self.obligations],
            "exclusions": [x.to_dict() for x in self.exclusions],
            "conditions": [x.to_dict() for x in self.conditions],
            "authority": self.authority.to_dict(),
            "compiler_version": self.compiler_version,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True)
class RelationVector:
    relevance: float = 0.0
    support: float = 0.0
    contradiction: float = 0.0
    exclusion_violation: float = 0.0
    trust: float = 1.0
    uncertainty: float = 1.0
    backend: str = "deterministic"
    calibrated: bool = False

    def __post_init__(self) -> None:
        for name in (
            "relevance", "support", "contradiction", "exclusion_violation", "trust", "uncertainty"
        ):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1]")

    def score_for_selection(self) -> float:
        return (self.relevance + self.support + self.trust) - (
            self.contradiction + self.exclusion_violation + self.uncertainty
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceCandidate:
    candidate_id: str
    text: str
    token_cost: int
    deterministic_score: float = 0.0
    relation: RelationVector = field(default_factory=RelationVector)
    dependencies: tuple[str, ...] = ()
    recoverable_ref: str | None = None

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "token_cost": self.token_cost,
            "deterministic_score": self.deterministic_score,
            "relation": self.relation.to_dict(),
            "dependencies": list(self.dependencies),
            "recoverable_ref": self.recoverable_ref,
        }


@dataclass(frozen=True)
class CollisionReport:
    escalated: bool
    reasons: tuple[str, ...]
    severity: float
    compared_ids: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DifferentialSpan:
    candidate_id: str
    start: int
    end: int
    text: str
    context_before: str
    context_after: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OmissionWitness:
    candidate_id: str
    omitted_hash: str
    safe_to_omit: bool
    reasons: tuple[str, ...]
    retained_candidate_ids: tuple[str, ...]
    contract_fingerprint: str
    recoverable_ref: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
