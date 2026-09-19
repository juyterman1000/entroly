"""Compositional omission safety and dimension-aware witness.

Individual omission checks cannot detect pairwise-independence violations:
two individually-safe omissions can be jointly unsafe when they eliminate
coverage of distinct information dimensions.

This module provides:

1. ``verify_omission_with_dimensions`` — wraps the per-fragment witness and
   overrides lexical-only false positives on summary queries when the
   retained set still covers enough information dimensions.  Hard safety
   checks (constraint carrier, contradiction, value/action loss, exclusion
   target) are NEVER overridden.

2. ``verify_joint_omission_safety`` — checks a SET of proposed omissions
   together, catching compositional failures that per-fragment witnesses miss.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .dimensions import DimensionCoverage, check_dimension_coverage, extract_dimensions
from .omission import verify_omission_safety
from .types import EvidenceCandidate, OmissionWitness, QueryContract

_SUMMARY_RE = re.compile(
    r"\b(?:summarize|summarise|describe|explain|overview|outline|review)\b", re.I,
)

_SOFT_PREFIX = "obligation_support_may_depend_on_omitted"


@dataclass(frozen=True)
class JointOmissionWitness:
    safe_to_omit: bool
    individual_witnesses: tuple[OmissionWitness, ...]
    dimension_coverage: DimensionCoverage | None
    reasons: tuple[str, ...]


def _is_summary_query(task: str) -> bool:
    return bool(_SUMMARY_RE.search(task))


def _has_only_soft_reasons(witness: OmissionWitness) -> bool:
    if witness.safe_to_omit:
        return True
    return all(r.startswith(_SOFT_PREFIX) for r in witness.reasons)


def _build_dimensions(all_evidence: tuple[EvidenceCandidate, ...]) -> list[frozenset[int]]:
    return extract_dimensions([c.text for c in all_evidence])


def verify_omission_with_dimensions(
    candidate: EvidenceCandidate,
    retained: tuple[EvidenceCandidate, ...],
    contract: QueryContract,
    *,
    all_evidence: tuple[EvidenceCandidate, ...],
) -> OmissionWitness:
    """Omission check with dimension-aware override for summary queries.

    If the base witness blocks the omission with ONLY soft reasons
    (lexical obligation support) AND the query is a summary type AND
    the retained set still covers enough information dimensions,
    override the block.

    Hard safety checks are NEVER overridden.
    """
    base = verify_omission_safety(candidate, retained, contract)

    if base.safe_to_omit:
        return base

    if not _has_only_soft_reasons(base):
        return base

    if not _is_summary_query(contract.task):
        return base

    dims = _build_dimensions(all_evidence)
    id_to_idx = {c.candidate_id: i for i, c in enumerate(all_evidence)}
    omit_idx = id_to_idx.get(candidate.candidate_id)
    if omit_idx is None:
        return base

    retained_indices = set(range(len(all_evidence))) - {omit_idx}
    coverage = check_dimension_coverage(retained_indices, dims)

    if not coverage.sufficient:
        return base

    return OmissionWitness(
        candidate.candidate_id,
        candidate.content_hash,
        True,
        (),
        tuple(x.candidate_id for x in retained),
        contract.fingerprint,
        candidate.recoverable_ref,
    )


def verify_joint_omission_safety(
    to_omit: list[EvidenceCandidate],
    all_candidates: list[EvidenceCandidate],
    contract: QueryContract,
) -> JointOmissionWitness:
    """Check whether a SET of omissions is jointly safe.

    Runs individual checks (with dimension override) and then verifies
    that the retained set as a whole still covers enough dimensions.
    Any hard safety failure on an individual fragment blocks the entire
    joint omission.
    """
    omit_ids = {c.candidate_id for c in to_omit}
    retained = tuple(c for c in all_candidates if c.candidate_id not in omit_ids)
    all_ev = tuple(all_candidates)

    individual: list[OmissionWitness] = []
    for cand in to_omit:
        w = verify_omission_with_dimensions(
            cand, retained, contract, all_evidence=all_ev,
        )
        individual.append(w)

    dims = _build_dimensions(all_ev)
    id_to_idx = {c.candidate_id: i for i, c in enumerate(all_candidates)}
    retained_indices = {
        id_to_idx[c.candidate_id]
        for c in all_candidates
        if c.candidate_id not in omit_ids
    }
    coverage = check_dimension_coverage(retained_indices, dims)

    reasons: list[str] = []
    for w in individual:
        if not w.safe_to_omit:
            reasons.extend(w.reasons)

    if not coverage.sufficient:
        reasons.append(f"joint_coverage_insufficient:{coverage.reason}")

    safe = not reasons
    return JointOmissionWitness(safe, tuple(individual), coverage, tuple(reasons))
