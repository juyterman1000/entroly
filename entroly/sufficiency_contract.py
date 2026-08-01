"""Typed contract for context-sufficiency evidence.

This module deliberately separates *what a certificate claims* from how a
selector computes it. Retrieval-score proxies must not silently masquerade as
proof that answer-bearing spans survived final serialization and trimming.

The contract is pure Python and import-safe without the optional Rust engine.
Rust/PyO3/WASM producers can emit the same JSON fields and callers can validate
them here before accepting compressed context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class CertificateVerdict(str, Enum):
    """Outcome reported by a sufficiency evaluator."""

    SUFFICIENT = "sufficient"
    DEGRADED = "degraded"
    UNCERTAIN = "uncertain"


class CertificateScope(str, Enum):
    """Strongest claim justified by the certificate's available evidence."""

    UNAVAILABLE = "unavailable"
    OPTIMIZER_PROXY = "optimizer_proxy"
    FILE_RETRIEVAL = "file_retrieval"
    CANDIDATE_UNITS = "candidate_units"
    SEMANTIC = "semantic"
    TASK_VERIFIED = "task_verified"


_SCOPE_RANK = {
    CertificateScope.UNAVAILABLE: 0,
    CertificateScope.OPTIMIZER_PROXY: 1,
    CertificateScope.FILE_RETRIEVAL: 1,
    CertificateScope.CANDIDATE_UNITS: 2,
    CertificateScope.SEMANTIC: 3,
    CertificateScope.TASK_VERIFIED: 4,
}


def parse_scope(value: object) -> CertificateScope:
    """Parse a scope fail-closed; unknown or missing values are unavailable."""
    try:
        return CertificateScope(str(value))
    except (TypeError, ValueError):
        return CertificateScope.UNAVAILABLE


def parse_verdict(value: object) -> CertificateVerdict:
    """Parse a verdict fail-closed; unknown or missing values are uncertain."""
    try:
        return CertificateVerdict(str(value))
    except (TypeError, ValueError):
        return CertificateVerdict.UNCERTAIN


def scope_satisfies(actual: CertificateScope, required: CertificateScope) -> bool:
    """Return whether ``actual`` is at least as strong as ``required``."""
    return _SCOPE_RANK[actual] >= _SCOPE_RANK[required]


@dataclass(frozen=True)
class SufficiencyCertificate:
    """Validated, auditable sufficiency claim.

    ``metrics`` is intentionally open-ended so Rust can add measured signals
    without changing the acceptance contract. Unknown fields are preserved but
    never promoted into a stronger verdict or scope.
    """

    verdict: CertificateVerdict
    scope: CertificateScope
    reasons: tuple[str, ...] = field(default=())
    metrics: Mapping[str, Any] = field(default_factory=dict)
    calibration_version: str | None = None
    dataset_fingerprint: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SufficiencyCertificate":
        if payload is None:
            return cls(
                verdict=CertificateVerdict.UNCERTAIN,
                scope=CertificateScope.UNAVAILABLE,
                reasons=("selector emitted no sufficiency certificate",),
            )

        raw_reasons = payload.get("reasons")
        if isinstance(raw_reasons, Sequence) and not isinstance(
            raw_reasons, (str, bytes, bytearray)
        ):
            reasons = tuple(str(reason) for reason in raw_reasons)
        elif raw_reasons:
            reasons = (str(raw_reasons),)
        else:
            reasons = ()

        known = {
            "verdict",
            "scope",
            "reasons",
            "calibration_version",
            "dataset_fingerprint",
            "metrics",
        }
        raw_metrics = payload.get("metrics")
        metrics: dict[str, Any] = (
            dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
        )
        metrics.update({key: value for key, value in payload.items() if key not in known})

        verdict = parse_verdict(payload.get("verdict"))
        scope = parse_scope(payload.get("scope"))
        if verdict is CertificateVerdict.SUFFICIENT and scope is CertificateScope.UNAVAILABLE:
            verdict = CertificateVerdict.UNCERTAIN
            reasons = (*reasons, "a sufficient verdict requires an explicit evidence scope")

        return cls(
            verdict=verdict,
            scope=scope,
            reasons=reasons,
            metrics=metrics,
            calibration_version=(
                str(payload["calibration_version"])
                if payload.get("calibration_version") is not None
                else None
            ),
            dataset_fingerprint=(
                str(payload["dataset_fingerprint"])
                if payload.get("dataset_fingerprint") is not None
                else None
            ),
        )

    def satisfies(self, required_scope: CertificateScope) -> bool:
        return (
            self.verdict is CertificateVerdict.SUFFICIENT
            and scope_satisfies(self.scope, required_scope)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "scope": self.scope.value,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "calibration_version": self.calibration_version,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
