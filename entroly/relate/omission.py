from __future__ import annotations

from .types import EvidenceCandidate, OmissionWitness, QueryContract


def verify_omission_safety(candidate: EvidenceCandidate, retained: tuple[EvidenceCandidate, ...], contract: QueryContract) -> OmissionWitness:
    """Conservative omission witness.

    Safe only when the omitted item is recoverable, carries no contradiction or
    exclusion signal, and every compiled obligation still has lexical support in
    retained evidence. This is a falsifiable baseline, not a final theorem.
    """
    reasons: list[str] = []
    retained_text = "\n".join(x.text.lower() for x in retained)
    if not candidate.recoverable_ref:
        reasons.append("omitted_fragment_not_recoverable")
    if candidate.relation.contradiction > 0.0:
        reasons.append("omitted_fragment_contains_contradiction_signal")
    if candidate.relation.exclusion_violation > 0.0:
        reasons.append("omitted_fragment_contains_exclusion_signal")
    for obligation in contract.obligations:
        terms = [t for t in obligation.text.lower().split() if len(t) > 3]
        if terms and not any(t in retained_text for t in terms):
            reasons.append(f"obligation_support_may_depend_on_omitted:{obligation.text[:40]}")
    safe = not reasons
    return OmissionWitness(candidate.candidate_id, candidate.content_hash, safe, tuple(reasons), tuple(x.candidate_id for x in retained), contract.fingerprint, candidate.recoverable_ref)
