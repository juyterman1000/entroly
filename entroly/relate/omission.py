from __future__ import annotations

from .collision import detect_semantic_collision
from .info_residual import (
    compute_residual,
    detect_state_conflict,
    task_asks_for_value,
    task_is_action,
)
from .types import EvidenceCandidate, OmissionWitness, QueryContract


def verify_omission_safety(
    candidate: EvidenceCandidate,
    retained: tuple[EvidenceCandidate, ...],
    contract: QueryContract,
) -> OmissionWitness:
    """Omission witness with information-residual and causal checks.

    Safe only when the omitted item is recoverable, carries no unique
    constraints, hides no contradictions, and every compiled obligation
    still has support in retained evidence.
    """
    reasons: list[str] = []
    retained_text = "\n".join(x.text.lower() for x in retained)

    # --- Tier 1: structural prerequisites ---
    if not candidate.recoverable_ref:
        reasons.append("omitted_fragment_not_recoverable")
    if candidate.relation.contradiction > 0.0:
        reasons.append("omitted_fragment_contains_contradiction_signal")
    if candidate.relation.exclusion_violation > 0.0:
        reasons.append("omitted_fragment_contains_exclusion_signal")

    # --- Tier 2: information residual ---
    residual = compute_residual(candidate.text, retained_text)

    if residual.has_constraint_residual:
        reasons.append(
            f"omission_loses_unique_constraint:{','.join(residual.unique_constraints)}"
        )

    if task_asks_for_value(contract.task) and residual.has_value_residual:
        values = residual.unique_numbers + residual.unique_temporals
        reasons.append(
            f"omission_loses_task_relevant_value:{','.join(values)}"
        )

    if task_is_action(contract.task) and residual.has_path_residual:
        reasons.append(
            f"omission_loses_action_arguments:{','.join(residual.unique_paths)}"
        )

    # --- Tier 3: exclusion target preservation ---
    for exc in contract.exclusions:
        exc_lower = exc.text.lower()
        if exc_lower in candidate.text.lower() and exc_lower not in retained_text:
            reasons.append(
                f"omission_loses_exclusion_target:{exc.text[:40]}"
            )

    # --- Tier 4: contradiction via state/collision detection ---
    state_conflicts = detect_state_conflict(
        candidate.text, [r.text for r in retained]
    )
    for sc in state_conflicts:
        reasons.append(f"omission_hides_{sc}")

    for ret in retained:
        collision = detect_semantic_collision(contract, candidate, ret)
        if collision.escalated:
            conflict_types = [
                r for r in collision.reasons
                if r in (
                    "near_duplicate_negation_conflict",
                    "antonym_conflict",
                    "numeric_conflict",
                )
            ]
            if conflict_types:
                reasons.append(
                    f"omission_hides_collision:{','.join(conflict_types)}"
                )

    # --- Tier 5: lexical obligation support ---
    for obligation in contract.obligations:
        terms = [t for t in obligation.text.lower().split() if len(t) > 3]
        if terms and not any(t in retained_text for t in terms):
            reasons.append(
                f"obligation_support_may_depend_on_omitted:{obligation.text[:40]}"
            )

    safe = not reasons
    return OmissionWitness(
        candidate.candidate_id,
        candidate.content_hash,
        safe,
        tuple(reasons),
        tuple(x.candidate_id for x in retained),
        contract.fingerprint,
        candidate.recoverable_ref,
    )
