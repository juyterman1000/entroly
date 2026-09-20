from __future__ import annotations

from itertools import combinations
from .types import EvidenceCandidate


def plan_evidence_set(candidates: list[EvidenceCandidate], *, budget_tokens: int, pin_conflicts: bool = True) -> tuple[EvidenceCandidate, ...]:
    """Deterministic research selector over a declared utility, not a proof of optimality."""
    feasible = [c for c in candidates if c.token_cost <= budget_tokens]

    def utility(items: tuple[EvidenceCandidate, ...]) -> float:
        return sum(c.deterministic_score + c.relation.score_for_selection() for c in items)

    best: tuple[EvidenceCandidate, ...] = ()
    best_score = float("-inf")
    if len(feasible) <= 16:
        for r in range(len(feasible) + 1):
            for subset in combinations(feasible, r):
                if sum(c.token_cost for c in subset) <= budget_tokens:
                    score = utility(subset)
                    key = tuple(c.candidate_id for c in subset)
                    if score > best_score or (score == best_score and key < tuple(c.candidate_id for c in best)):
                        best, best_score = subset, score
        return best
    out: list[EvidenceCandidate] = []
    remaining = budget_tokens
    for c in sorted(feasible, key=lambda x: (-(x.deterministic_score + x.relation.score_for_selection()), x.candidate_id)):
        if c.token_cost <= remaining:
            out.append(c)
            remaining -= c.token_cost
    return tuple(out)
