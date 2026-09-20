from __future__ import annotations

from collections.abc import Callable
from .types import RelationVector

NLIBackend = Callable[[str, str], tuple[str, float]]


def local_nli_backend() -> NLIBackend | None:
    """Return Entroly local NLI backend only if importable; never download silently."""
    try:
        from entroly.verifiers.local_nli import nli_score  # type: ignore
    except Exception:
        return None
    return nli_score


def score_relation(premise: str, hypothesis: str, *, backend: NLIBackend | None = None) -> RelationVector:
    if backend is None:
        return RelationVector(relevance=0.0, support=0.0, contradiction=0.0, uncertainty=1.0, backend="unavailable", calibrated=False)
    label, confidence = backend(premise, hypothesis)
    confidence = max(0.0, min(1.0, float(confidence)))
    if label == "entailment":
        return RelationVector(relevance=confidence, support=confidence, contradiction=0.0, uncertainty=1.0-confidence, backend="local_nli", calibrated=False)
    if label == "contradiction":
        return RelationVector(relevance=confidence, support=0.0, contradiction=confidence, uncertainty=1.0-confidence, backend="local_nli", calibrated=False)
    return RelationVector(relevance=0.3, support=0.0, contradiction=0.0, uncertainty=max(0.5, confidence), backend="local_nli", calibrated=False)
