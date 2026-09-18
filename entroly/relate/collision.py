from __future__ import annotations

import re
from .types import CollisionReport, EvidenceCandidate, QueryContract

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
_NUM = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?")
_NEG = {"no", "not", "never", "without", "cannot", "can't", "isn't", "aren't", "doesn't", "don't", "didn't"}
_ANTONYMS = {
    ("enabled", "disabled"), ("enable", "disable"), ("allowed", "denied"),
    ("allow", "deny"), ("success", "failure"), ("before", "after"),
    ("increase", "decrease"), ("higher", "lower"), ("true", "false"),
    ("public", "private"), ("include", "exclude"), ("present", "absent"),
}
_PAIRMAP = {a: b for a, b in _ANTONYMS} | {b: a for a, b in _ANTONYMS}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD.finditer(text)}


def _jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def _neg_parity(text: str) -> int:
    return sum(1 for t in _tokens(text) if t in _NEG or t.endswith("n't")) % 2


def _antonym_conflict(a: str, b: str) -> bool:
    x, y = _tokens(a), _tokens(b)
    return any((_PAIRMAP.get(t) in y) for t in x)


def _numbers(text: str) -> set[str]:
    return {m.group(0) for m in _NUM.finditer(text)}


def detect_semantic_collision(
    query_contract: QueryContract,
    left: EvidenceCandidate,
    right: EvidenceCandidate,
    *,
    score_margin_threshold: float = 0.08,
) -> CollisionReport:
    reasons: list[str] = []
    overlap = _jaccard(left.text, right.text)
    if overlap >= 0.55 and _neg_parity(left.text) != _neg_parity(right.text):
        reasons.append("near_duplicate_negation_conflict")
    if overlap >= 0.35 and _antonym_conflict(left.text, right.text):
        reasons.append("antonym_conflict")
    ln, rn = _numbers(left.text), _numbers(right.text)
    if ln and rn and ln != rn and overlap >= 0.35:
        reasons.append("numeric_conflict")
    if abs(left.deterministic_score - right.deterministic_score) <= score_margin_threshold:
        reasons.append("small_score_margin")
    for exc in query_contract.exclusions:
        exc_text = exc.text.lower()
        if (exc_text in left.text.lower()) != (exc_text in right.text.lower()):
            reasons.append("explicit_exclusion_split")
            break
    severity = min(1.0, 0.25 * len(set(reasons)))
    return CollisionReport(bool(reasons), tuple(sorted(set(reasons))), severity, (left.candidate_id, right.candidate_id))
