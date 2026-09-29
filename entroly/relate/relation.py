from __future__ import annotations

from collections.abc import Callable
from .types import RelationVector

NLIBackend = Callable[[str, str], tuple[str, float]]


def local_nli_backend() -> NLIBackend | None:
    """Return the local NLI backend if the module imports, else ``None``.

    This checks importability only. It does **not** guarantee the model is
    present: ``nli_score`` resolves ``cross-encoder/nli-deberta-v3-small``
    through the Hugging Face cache on first call and fetches it if absent, so
    the first scored pair may reach the network. Callers that must stay offline
    should set ``HF_HUB_OFFLINE=1``, which makes the load fail and every pair
    report ``"unavailable"`` rather than a fabricated neutral.

    (The previous wording here promised "never download silently", which was
    not true of the function it returns.)
    """
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
    if label == "unavailable":
        # The backend existed but its model never loaded, so nothing was
        # assessed. That is the same epistemic state as having no backend at
        # all, and must report the same vacuous vector: all mass on "unknown",
        # none on support or contradiction. Falling through to the neutral
        # branch instead claimed `backend="local_nli"` with uncertainty 0.5 --
        # asserting half the information had been obtained, and attributing it
        # to a model that never ran.
        return RelationVector(relevance=0.0, support=0.0, contradiction=0.0,
                              uncertainty=1.0, backend="unavailable", calibrated=False)
    if label == "entailment":
        return RelationVector(relevance=confidence, support=confidence, contradiction=0.0, uncertainty=1.0-confidence, backend="local_nli", calibrated=False)
    if label == "contradiction":
        return RelationVector(relevance=confidence, support=0.0, contradiction=confidence, uncertainty=1.0-confidence, backend="local_nli", calibrated=False)
    return RelationVector(relevance=0.3, support=0.0, contradiction=0.0, uncertainty=max(0.5, confidence), backend="local_nli", calibrated=False)
