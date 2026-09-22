"""Entroly Assurance Envelope: one fail-closed decision for autonomous work.

The individual signals in this module are deliberately modest. The product
value is their composition: an action cannot inherit trust from one green
signal while another required evidence surface is missing. Every decision is
bound to canonical input hashes so it can be replayed and compared later.

The envelope is independent of model and provider names. Adapters supply the
same evidence records regardless of which runtime produced the answer.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .relate.compression_residual import measure_context_drift
from .relate.coverage_verification import audit_evidence_boundary
from .sufficiency import Candidate, build_obligation_budget_witness
from .vault import classify_source_boundary

EnvelopeDecision = Literal["allow", "hold", "block"]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AssuranceEnvelope:
    """Replayable decision over the evidence required for one operation."""

    schema: str
    decision: EnvelopeDecision
    reasons: tuple[str, ...]
    task_sha256: str
    evidence_sha256: str
    obligation_verdict: str
    exact_budget_cost: int | None
    reference_coverage: float
    unsourced_references: int
    context_reverification_required: bool
    source_classes: tuple[tuple[str, str], ...]
    skill_evidence_validated: bool | None

    @property
    def autonomous_execution_allowed(self) -> bool:
        return self.decision == "allow"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_classes"] = dict(self.source_classes)
        result["autonomous_execution_allowed"] = self.autonomous_execution_allowed
        return result


def build_assurance_envelope(
    *,
    task: str,
    candidates: Sequence[Candidate],
    obligations: Sequence[str],
    budget: int,
    answer: str,
    selected_sources: Sequence[Mapping[str, Any]],
    omitted_sources: Sequence[Mapping[str, Any]],
    omitted_fragment: str = "",
    retained_context: str = "",
    context_update: str = "",
    coverage: Mapping[str, Sequence[str]] | None = None,
    trusted_source_root: str | Path | None = None,
    skill_id: str | None = None,
    skill_evidence_validated: bool | None = None,
    max_cover_states: int = 65_536,
) -> AssuranceEnvelope:
    """Build an allow/hold/block decision from independent evidence surfaces.

    ``block`` means a concrete contradiction exists: required obligations are
    absent or over budget, answer references lack selected evidence, a selected
    source escapes policy, or a requested skill has invalid benchmark binding.

    ``hold`` means evidence is incomplete: bounded cover search abstained,
    answer references could not be measured, context changed after omission,
    source provenance is unknown, or a requested skill has not been validated.

    ``allow`` requires every applicable surface to pass. It authorizes only
    this exact evidence envelope; it is not global trust in an agent or model.
    """
    if not task.strip():
        raise ValueError("task must be nonempty")
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    if skill_id is None and skill_evidence_validated is not None:
        raise ValueError("skill evidence requires a skill_id")

    budget_result = build_obligation_budget_witness(
        candidates,
        obligations,
        budget,
        coverage=dict(coverage) if coverage is not None else None,
        max_states=max_cover_states,
    )
    reference_result = audit_evidence_boundary(
        answer,
        [dict(source) for source in selected_sources],
        [dict(source) for source in omitted_sources],
    )
    drift_result = measure_context_drift(
        omitted_fragment,
        retained_context,
        context_update,
    )

    source_classes: list[tuple[str, str]] = []
    for index, source in enumerate(selected_sources):
        raw_path = str(source.get("source_path") or source.get("source") or "")
        label = raw_path or f"selected:{index}"
        source_classes.append(
            (label, classify_source_boundary(raw_path, trusted_source_root))
        )

    blockers: list[str] = []
    holds: list[str] = []

    if budget_result.verdict in {"missing_evidence", "insufficient_budget"}:
        blockers.append(f"obligation:{budget_result.verdict}")
    elif budget_result.verdict != "cover_found":
        holds.append(f"obligation:{budget_result.verdict}")

    if reference_result.entities_unsourced:
        blockers.append("answer:unsourced_reference")
    elif reference_result.honest is None:
        holds.append("answer:no_measurable_reference")

    if drift_result.requires_reverification:
        holds.append("context:reverification_required")

    classes = {trust for _, trust in source_classes}
    if "untrusted" in classes:
        blockers.append("source:outside_policy")
    if "unknown" in classes:
        holds.append("source:unknown")

    if skill_id is not None:
        if skill_evidence_validated is False:
            blockers.append("skill:stale_or_invalid_evidence")
        elif skill_evidence_validated is None:
            holds.append("skill:validation_missing")

    reasons = tuple(sorted(set(blockers or holds)))
    decision: EnvelopeDecision = "block" if blockers else "hold" if holds else "allow"

    candidate_records = [
        {
            "unit_id": candidate.unit_id,
            "cost": candidate.cost,
            "utility": candidate.utility,
            "selected": candidate.selected,
        }
        for candidate in candidates
    ]
    evidence_record = {
        "task": task,
        "candidates": candidate_records,
        "obligations": list(obligations),
        "budget": budget,
        "coverage": coverage,
        "answer": answer,
        "selected_sources": list(selected_sources),
        "omitted_sources": list(omitted_sources),
        "omitted_fragment_sha256": hashlib.sha256(
            omitted_fragment.encode("utf-8")
        ).hexdigest(),
        "retained_context_sha256": hashlib.sha256(
            retained_context.encode("utf-8")
        ).hexdigest(),
        "context_update_sha256": hashlib.sha256(
            context_update.encode("utf-8")
        ).hexdigest(),
        "source_classes": source_classes,
        "skill_id": skill_id,
        "skill_evidence_validated": skill_evidence_validated,
    }
    return AssuranceEnvelope(
        schema="entroly.assurance-envelope.v1",
        decision=decision,
        reasons=reasons,
        task_sha256=hashlib.sha256(task.encode("utf-8")).hexdigest(),
        evidence_sha256=_canonical_sha256(evidence_record),
        obligation_verdict=budget_result.verdict,
        exact_budget_cost=budget_result.minimum_cover_cost,
        reference_coverage=reference_result.coverage_ratio,
        unsourced_references=reference_result.entities_unsourced,
        context_reverification_required=drift_result.requires_reverification,
        source_classes=tuple(source_classes),
        skill_evidence_validated=skill_evidence_validated,
    )


def score_assurance_envelope(envelope: AssuranceEnvelope) -> float:
    """Stable ordering signal for dashboards; never overrides the decision.

    The score deliberately has no promotion threshold. It exposes gradation
    within a decision class while the categorical gate remains authoritative.
    """
    if not math.isfinite(envelope.reference_coverage):
        return 0.0
    base = {"block": 0.0, "hold": 0.5, "allow": 1.0}[envelope.decision]
    penalty = min(0.49, 0.05 * len(envelope.reasons))
    return round(max(0.0, base - penalty) * envelope.reference_coverage, 6)
