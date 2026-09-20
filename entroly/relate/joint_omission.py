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

from .compression_residual import conditional_residual
from .dimensions import DimensionCoverage, check_dimension_coverage, extract_dimensions
from .omission import verify_omission_safety
from .types import EvidenceCandidate, OmissionWitness, QueryContract

_SUMMARY_RE = re.compile(
    r"\b(?:summarize|summarise|describe|explain|overview|outline|review)\b", re.I,
)

_SOFT_PREFIX = "obligation_support_may_depend_on_omitted"

# Directional containment thresholds.  A fragment is subsumed by the
# retained set when the retained set costs substantially more given the
# fragment than the fragment costs given the retained set -- i.e. the
# retained set carries strictly more information.  Both conditions are
# required: the gap establishes direction, the ceiling establishes that the
# fragment is genuinely cheap rather than merely relatively cheaper.
#
# Calibrated on the soft-only population of both frozen datasets, defined
# by the BASE witness so the measurement survives this override going live
# (see benchmarks/relate/analyze_subsumption.py):
#
#   safe gaps   : 0.2898, 0.2026, 0.1358, 0.0975
#   unsafe gaps : 0.0165, -0.0165, -0.2088
#   separating band: (0.0165, 0.0975), margin 0.0810
#
# 0.10 sits just above the band, which is deliberately conservative: it
# rejects the weakest safe case rather than risk admitting an unsafe one.
# That case is recovered by the independent dimension path, so the cost is
# nil while the safety margin against the unsafe cluster stays ~6x.
# Calibrated on 7 points across two datasets -- still small; treat the
# exact value as provisional until a larger population exists.
_CONTAINMENT_GAP = 0.10
_CONTAINMENT_CEILING = 0.60


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


def is_subsumed(candidate: EvidenceCandidate, retained: tuple[EvidenceCandidate, ...]) -> bool:
    """Directional containment test: does the retained set carry strictly more?

    Uses the asymmetry of the conditional compression residual.  A positive
    gap means the retained set is expensive given the fragment while the
    fragment is cheap given the retained set -- the signature of the
    retained set subsuming the fragment.  A negative gap means the fragment
    is the richer of the two and must not be dropped.

    This is a recoverability test only.  It cannot detect constraint or
    authority loss: two contradictory sentences with near-identical surface
    form compress well against each other.  Callers must run it strictly
    behind the hard structural checks.
    """
    if not retained:
        return False
    retained_text = "\n".join(r.text for r in retained)
    forward = conditional_residual(candidate.text, retained_text)
    reverse = conditional_residual(retained_text, candidate.text)
    return (reverse - forward) > _CONTAINMENT_GAP and forward < _CONTAINMENT_CEILING


def verify_omission_with_dimensions(
    candidate: EvidenceCandidate,
    retained: tuple[EvidenceCandidate, ...],
    contract: QueryContract,
    *,
    all_evidence: tuple[EvidenceCandidate, ...],
) -> OmissionWitness:
    """Omission check with two override paths for soft-only blocks.

    When the base witness blocks an omission citing ONLY soft reasons
    (lexical obligation support), two independent arguments can clear it:

    A. **Directional containment** -- the retained set demonstrably carries
       strictly more information than the fragment.  Any query type.
    B. **Dimension coverage** -- enough distinct information dimensions
       survive the omission.  Summary queries only.

    Hard safety checks are NEVER overridden by either path.
    """
    base = verify_omission_safety(candidate, retained, contract)

    if base.safe_to_omit:
        return base

    # Hard structural failures -- constraint, contradiction, state conflict,
    # value/action loss, exclusion target, recoverability -- are never
    # overridden by either path below.
    if not _has_only_soft_reasons(base):
        return base

    # Path A: directional containment.  Valid for any query type, because
    # it certifies that this specific fragment's content is carried by the
    # retained set rather than reasoning about topic coverage.
    if is_subsumed(candidate, retained):
        return _approved(candidate, retained, contract)

    # Path B: dimension coverage.  Summary queries only -- it argues about
    # breadth of topics retained, which is what a summary must preserve.
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

    return _approved(candidate, retained, contract)


def _approved(
    candidate: EvidenceCandidate,
    retained: tuple[EvidenceCandidate, ...],
    contract: QueryContract,
) -> OmissionWitness:
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
