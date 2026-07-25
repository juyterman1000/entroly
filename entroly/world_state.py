"""Verified world-state kernel for evidence-grounded autonomous agents.

The kernel deliberately separates three responsibilities:

* claims hold structured, versioned beliefs;
* prediction models forecast possible transitions but never mutate truth;
* deterministic verification controls promotion into the verified frontier.

The implementation is local-first and standard-library-only. It is designed as
an additive research surface: callers can use existing Entroly vault/receipt
systems for durable evidence while this module enforces state-transition rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

WORLD_STATE_SCHEMA = "entroly.world-state.v1"
_SHA256_HEX_LENGTH = 64


class WorldStateError(RuntimeError):
    """Base class for fail-closed world-state errors."""


class InvalidClaimTransition(WorldStateError):
    """A claim attempted an illegal epistemic state transition."""


class VerificationRequired(WorldStateError):
    """A trusted transition lacked an acceptable verification receipt."""


class SnapshotIntegrityError(WorldStateError):
    """A persisted snapshot is malformed or does not match its digest."""


class ClaimStatus(str, Enum):
    """Epistemic states; only VERIFIED belongs to the trusted frontier."""

    PROPOSED = "proposed"
    OBSERVED = "observed"
    SUPPORTED = "supported"
    VERIFIED = "verified"
    INVALIDATED = "invalidated"


_ALLOWED_TRANSITIONS: dict[ClaimStatus, frozenset[ClaimStatus]] = {
    ClaimStatus.PROPOSED: frozenset(
        {ClaimStatus.OBSERVED, ClaimStatus.SUPPORTED, ClaimStatus.INVALIDATED}
    ),
    ClaimStatus.OBSERVED: frozenset(
        {ClaimStatus.SUPPORTED, ClaimStatus.INVALIDATED}
    ),
    ClaimStatus.SUPPORTED: frozenset(
        {ClaimStatus.VERIFIED, ClaimStatus.INVALIDATED}
    ),
    ClaimStatus.VERIFIED: frozenset({ClaimStatus.INVALIDATED}),
    ClaimStatus.INVALIDATED: frozenset(),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_confidence(value: float, *, field_name: str = "confidence") -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0, 1]")


def _validate_digest(value: str) -> None:
    if len(value) != _SHA256_HEX_LENGTH:
        raise ValueError("digest must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("digest must contain only hexadecimal characters") from exc


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Content-addressed evidence used to support or invalidate a claim."""

    source_id: str
    digest: str
    repo_sha: str = ""
    kind: str = "observation"
    locator: str = ""

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.kind.strip():
            raise ValueError("kind must not be empty")
        _validate_digest(self.digest)

    @classmethod
    def from_text(
        cls,
        source_id: str,
        text: str,
        *,
        repo_sha: str = "",
        kind: str = "observation",
        locator: str = "",
    ) -> "EvidenceRef":
        return cls(
            source_id=source_id,
            digest=_sha256_text(text),
            repo_sha=repo_sha,
            kind=kind,
            locator=locator,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "digest": self.digest,
            "repo_sha": self.repo_sha,
            "kind": self.kind,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        return cls(
            source_id=str(value["source_id"]),
            digest=str(value["digest"]),
            repo_sha=str(value.get("repo_sha", "")),
            kind=str(value.get("kind", "observation")),
            locator=str(value.get("locator", "")),
        )


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    """Independent, evidence-backed result for one claim verification."""

    verifier: str
    passed: bool
    evidence: tuple[EvidenceRef, ...]
    repo_sha: str = ""
    details: str = ""

    def __post_init__(self) -> None:
        if not self.verifier.strip():
            raise ValueError("verifier must not be empty")
        if not self.evidence:
            raise ValueError("verification receipt requires evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "passed": self.passed,
            "evidence": [item.to_dict() for item in self.evidence],
            "repo_sha": self.repo_sha,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class WorldClaim:
    """One typed proposition in the dynamic world state."""

    claim_id: str
    subject: str
    predicate: str
    object_value: str
    status: ClaimStatus = ClaimStatus.PROPOSED
    confidence: float = 0.5
    evidence: tuple[EvidenceRef, ...] = ()
    depends_on: tuple[str, ...] = ()
    repo_sha: str = ""
    invalidated_by: str = ""

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError("claim_id must not be empty")
        if not self.subject.strip() or not self.predicate.strip():
            raise ValueError("subject and predicate must not be empty")
        _validate_confidence(self.confidence)
        if self.claim_id in self.depends_on:
            raise ValueError("a claim cannot depend on itself")
        if self.status is ClaimStatus.INVALIDATED and not self.invalidated_by:
            raise ValueError("invalidated claims require invalidated_by")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "depends_on": list(self.depends_on),
            "repo_sha": self.repo_sha,
            "invalidated_by": self.invalidated_by,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorldClaim":
        return cls(
            claim_id=str(value["claim_id"]),
            subject=str(value["subject"]),
            predicate=str(value["predicate"]),
            object_value=str(value.get("object_value", "")),
            status=ClaimStatus(str(value.get("status", ClaimStatus.PROPOSED.value))),
            confidence=float(value.get("confidence", 0.5)),
            evidence=tuple(
                EvidenceRef.from_dict(item) for item in value.get("evidence", ())
            ),
            depends_on=tuple(str(item) for item in value.get("depends_on", ())),
            repo_sha=str(value.get("repo_sha", "")),
            invalidated_by=str(value.get("invalidated_by", "")),
        )


@dataclass(frozen=True, slots=True)
class TransitionPrediction:
    """A model forecast. Predictions are advisory and never trusted facts."""

    prediction_id: str
    model_id: str
    action_id: str
    probabilities: tuple[tuple[str, float], ...]

    @classmethod
    def create(
        cls,
        prediction_id: str,
        model_id: str,
        action_id: str,
        probabilities: Mapping[str, float],
    ) -> "TransitionPrediction":
        if not prediction_id.strip() or not model_id.strip() or not action_id.strip():
            raise ValueError("prediction_id, model_id, and action_id are required")
        normalized: list[tuple[str, float]] = []
        for label, probability in probabilities.items():
            probability = float(probability)
            _validate_confidence(probability, field_name=f"probability[{label}]")
            if not str(label).strip():
                raise ValueError("prediction labels must not be empty")
            normalized.append((str(label), probability))
        return cls(
            prediction_id=prediction_id,
            model_id=model_id,
            action_id=action_id,
            probabilities=tuple(sorted(normalized)),
        )

    def as_mapping(self) -> dict[str, float]:
        return dict(self.probabilities)


@dataclass(frozen=True, slots=True)
class PredictionReconciliation:
    """Observed prediction error and the resulting model authority."""

    prediction_id: str
    brier_error: float
    previous_authority: float
    new_authority: float
    observed_labels: tuple[str, ...]


class VerifiedWorldState:
    """Fail-closed world-state controller.

    The class is intentionally deterministic: insertion order does not affect
    snapshots, digests, invalidation propagation, or frontier enumeration.
    """

    def __init__(self) -> None:
        self._claims: dict[str, WorldClaim] = {}
        self._model_authority: dict[str, float] = {}

    @property
    def claims(self) -> tuple[WorldClaim, ...]:
        return tuple(self._claims[key] for key in sorted(self._claims))

    def claim(self, claim_id: str) -> WorldClaim:
        try:
            return self._claims[claim_id]
        except KeyError as exc:
            raise KeyError(f"unknown claim: {claim_id}") from exc

    def add_claim(self, claim: WorldClaim) -> None:
        if claim.status not in {ClaimStatus.PROPOSED, ClaimStatus.OBSERVED}:
            raise VerificationRequired(
                "claims must enter as proposed or observed; trusted states require transitions"
            )
        if claim.status is ClaimStatus.OBSERVED and not claim.evidence:
            raise VerificationRequired("observed claims require evidence")
        self._ensure_evidence_current(claim.repo_sha, claim.evidence)
        missing = sorted(set(claim.depends_on) - self._claims.keys())
        if missing:
            raise ValueError(f"unknown claim dependencies: {', '.join(missing)}")
        existing = self._claims.get(claim.claim_id)
        if existing is not None:
            if existing != claim:
                raise ValueError(
                    "claim_id already exists with different content: "
                    f"{claim.claim_id}"
                )
            return
        self._claims[claim.claim_id] = claim

    def transition(
        self,
        claim_id: str,
        target: ClaimStatus,
        *,
        evidence: Iterable[EvidenceRef] = (),
        confidence: float | None = None,
        receipt: VerificationReceipt | None = None,
        reason: str = "",
    ) -> WorldClaim:
        current = self.claim(claim_id)
        if target not in _ALLOWED_TRANSITIONS[current.status]:
            raise InvalidClaimTransition(
                f"illegal claim transition: {current.status.value} -> {target.value}"
            )

        supplied_evidence = tuple(evidence)
        self._ensure_evidence_current(current.repo_sha, supplied_evidence)
        merged_evidence = self._merge_evidence(current.evidence, supplied_evidence)

        next_confidence = current.confidence if confidence is None else float(confidence)
        _validate_confidence(next_confidence)

        if target in {ClaimStatus.OBSERVED, ClaimStatus.SUPPORTED} and not merged_evidence:
            raise VerificationRequired(f"{target.value} transition requires evidence")

        if target is ClaimStatus.VERIFIED:
            if receipt is None or not receipt.passed:
                raise VerificationRequired("verified transition requires a passing receipt")
            if current.repo_sha and receipt.repo_sha != current.repo_sha:
                raise VerificationRequired("verification receipt repository SHA is stale")
            self._ensure_evidence_current(current.repo_sha, receipt.evidence)
            merged_evidence = self._merge_evidence(
                merged_evidence, receipt.evidence
            )
            if not merged_evidence:
                raise VerificationRequired("verified transition requires evidence")

        if target is ClaimStatus.INVALIDATED:
            if not reason.strip():
                raise VerificationRequired("invalidation requires a reason")
            if not merged_evidence:
                raise VerificationRequired("invalidation requires evidence")

        updated = replace(
            current,
            status=target,
            confidence=next_confidence,
            evidence=merged_evidence,
            invalidated_by=reason if target is ClaimStatus.INVALIDATED else "",
        )
        self._claims[claim_id] = updated

        if target is ClaimStatus.INVALIDATED:
            self._propagate_invalidation(claim_id)
        return updated

    @staticmethod
    def _evidence_key(item: EvidenceRef) -> tuple[str, str, str, str, str]:
        return (
            item.source_id,
            item.digest,
            item.repo_sha,
            item.kind,
            item.locator,
        )

    @classmethod
    def _merge_evidence(
        cls,
        *groups: Iterable[EvidenceRef],
    ) -> tuple[EvidenceRef, ...]:
        unique: dict[tuple[str, str, str, str, str], EvidenceRef] = {}
        for item in (entry for group in groups for entry in group):
            unique[cls._evidence_key(item)] = item
        return tuple(unique[key] for key in sorted(unique))

    @staticmethod
    def _ensure_evidence_current(
        repo_sha: str, evidence: Iterable[EvidenceRef]
    ) -> None:
        if not repo_sha:
            return
        stale = sorted(
            item.source_id for item in evidence if item.repo_sha != repo_sha
        )
        if stale:
            raise VerificationRequired(
                "repository-bound evidence is stale or unversioned: "
                + ", ".join(stale)
            )

    def _propagate_invalidation(self, root_claim_id: str) -> None:
        queue = [root_claim_id]
        visited = {root_claim_id}
        while queue:
            invalid_id = queue.pop(0)
            dependents = sorted(
                claim.claim_id
                for claim in self._claims.values()
                if invalid_id in claim.depends_on
                and claim.status is not ClaimStatus.INVALIDATED
            )
            for dependent_id in dependents:
                if dependent_id in visited:
                    continue
                dependent = self._claims[dependent_id]
                self._claims[dependent_id] = replace(
                    dependent,
                    status=ClaimStatus.INVALIDATED,
                    invalidated_by=f"dependency:{invalid_id}",
                )
                visited.add(dependent_id)
                queue.append(dependent_id)

    def verified_frontier(self) -> tuple[WorldClaim, ...]:
        return tuple(
            claim
            for claim in self.claims
            if claim.status is ClaimStatus.VERIFIED
            and all(
                self._claims[parent].status is ClaimStatus.VERIFIED
                for parent in claim.depends_on
            )
        )

    def planning_weight(self, model_id: str) -> float:
        return self._model_authority.get(model_id, 1.0)

    def reconcile_prediction(
        self,
        prediction: TransitionPrediction,
        observed_labels: Iterable[str],
        *,
        learning_rate: float = 1.0,
    ) -> PredictionReconciliation:
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and non-negative")
        observed = frozenset(str(label) for label in observed_labels)
        predicted = prediction.as_mapping()
        labels = sorted(set(predicted) | observed)
        if labels:
            error = sum(
                (predicted.get(label, 0.0) - (1.0 if label in observed else 0.0)) ** 2
                for label in labels
            ) / len(labels)
        else:
            error = 0.0
        previous = self.planning_weight(prediction.model_id)
        updated = previous * math.exp(-learning_rate * error)
        self._model_authority[prediction.model_id] = updated
        return PredictionReconciliation(
            prediction_id=prediction.prediction_id,
            brier_error=error,
            previous_authority=previous,
            new_authority=updated,
            observed_labels=tuple(sorted(observed)),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": WORLD_STATE_SCHEMA,
            "claims": [claim.to_dict() for claim in self.claims],
            "model_authority": {
                key: self._model_authority[key]
                for key in sorted(self._model_authority)
            },
        }

    def digest(self) -> str:
        return _sha256_text(_canonical_json(self._payload()))

    def snapshot(self) -> dict[str, Any]:
        payload = self._payload()
        return {**payload, "state_sha256": _sha256_text(_canonical_json(payload))}

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "VerifiedWorldState":
        if snapshot.get("schema") != WORLD_STATE_SCHEMA:
            raise SnapshotIntegrityError("unsupported world-state schema")
        payload = {
            "schema": snapshot.get("schema"),
            "claims": snapshot.get("claims", []),
            "model_authority": snapshot.get("model_authority", {}),
        }
        expected = str(snapshot.get("state_sha256", ""))
        actual = _sha256_text(_canonical_json(payload))
        if expected != actual:
            raise SnapshotIntegrityError("world-state snapshot digest mismatch")

        state = cls()
        for raw_claim in payload["claims"]:
            claim = WorldClaim.from_dict(raw_claim)
            if claim.claim_id in state._claims:
                raise SnapshotIntegrityError(f"duplicate claim_id: {claim.claim_id}")
            try:
                state._ensure_evidence_current(claim.repo_sha, claim.evidence)
            except VerificationRequired as exc:
                raise SnapshotIntegrityError(str(exc)) from exc
            state._claims[claim.claim_id] = claim
        for claim in state._claims.values():
            missing = sorted(set(claim.depends_on) - state._claims.keys())
            if missing:
                raise SnapshotIntegrityError(
                    f"claim {claim.claim_id} has missing dependencies: {', '.join(missing)}"
                )
        for model_id, authority in dict(payload["model_authority"]).items():
            authority = float(authority)
            _validate_confidence(authority, field_name=f"model_authority[{model_id}]")
            state._model_authority[str(model_id)] = authority
        return state

    def save(self, path: str | Path) -> Path:
        """Atomically persist a content-addressed checkpoint."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(self.snapshot(), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "VerifiedWorldState":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError(f"unable to load world-state snapshot: {exc}") from exc
        if not isinstance(payload, dict):
            raise SnapshotIntegrityError("world-state snapshot must be a JSON object")
        return cls.from_snapshot(payload)
