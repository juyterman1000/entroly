"""
Local NLI module — zero-API-cost entailment using DeBERTa-v3-small.

Uses `cross-encoder/nli-deberta-v3-small` (~80 MB, CPU-friendly).
Model is loaded lazily on first call and cached for the process lifetime.

Mathematical role in WITNESS
-----------------------------
For each (claim, evidence) pair the cross-encoder scores three classes:
    contradiction  →  C  ∈ [0, 1]
    neutral        →  N  ∈ [0, 1]
    entailment     →  E  ∈ [0, 1]   (C + N + E = 1 after softmax)

We map these to the WITNESS NLIVerdict schema:
    E > θ_ent   →  "entailment",    confidence = E
    C > θ_con   →  "contradiction", confidence = C
    otherwise   →  "neutral",       confidence = N

Cascade role
------------
The model is expensive (~30–80 ms/pair on CPU). WITNESS runs it only when
the deterministic local_pav verdict is "neutral" AND the claim risk from
the continuous path is in the uncertain band (0.25–0.75).  This limits
NLI calls to the ~30% of claims where it actually changes the verdict.

Enabling for all MCP / pip / npm users
---------------------------------------
    from entroly import WitnessAnalyzer
    analyzer = WitnessAnalyzer(use_local_nli=True)   # downloads model once

Or set the environment variable:
    ENTROLY_LOCAL_NLI=1

References
----------
He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced
BERT with disentangled attention. ICLR 2021.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ── Lazy model singleton ─────────────────────────────────────────────

_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_LOCK = threading.Lock()
_pipeline = None          # sentence_transformers CrossEncoder
_load_attempted = False
_load_failed = False


def _load_model() -> bool:
    """Load the model once. Returns True if loaded successfully."""
    global _pipeline, _load_attempted, _load_failed
    if _load_attempted:
        return not _load_failed
    with _LOCK:
        if _load_attempted:
            return not _load_failed
        _load_attempted = True
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            _pipeline = CrossEncoder(
                _MODEL_NAME,
                max_length=512,
                device="cpu",
            )
            logger.info("[local_nli] Loaded %s (cpu)", _MODEL_NAME)
            return True
        except Exception as e:
            _load_failed = True
            logger.warning("[local_nli] Could not load %s: %s — falling back to local PAV", _MODEL_NAME, e)
            return False


# ── NLI label indices (DeBERTa NLI label order) ──────────────────────
# cross-encoder/nli-deberta-v3-small uses: 0=contradiction, 1=entailment, 2=neutral
# (verified against model card)
_IDX_CONTRADICTION = 0
_IDX_ENTAILMENT    = 1
_IDX_NEUTRAL       = 2

_THRESHOLD_ENTAILMENT    = 0.60
_THRESHOLD_CONTRADICTION = 0.65


def nli_score(
    premise: str,
    hypothesis: str,
) -> tuple[str, float]:
    """
    Return (label, confidence) for (premise, hypothesis).

    label ∈ {"entailment", "contradiction", "neutral"}
    confidence ∈ [0, 1]

    Returns ("neutral", 0.5) if model unavailable.
    """
    if not _load_model():
        return "neutral", 0.5

    try:
        import numpy as np
        from scipy.special import softmax

        raw = _pipeline.predict([(premise, hypothesis)], apply_softmax=False)
        probs = softmax(raw[0])

        e = float(probs[_IDX_ENTAILMENT])
        c = float(probs[_IDX_CONTRADICTION])
        n = float(probs[_IDX_NEUTRAL])

        if e >= _THRESHOLD_ENTAILMENT and e >= c and e >= n:
            return "entailment", e
        if c >= _THRESHOLD_CONTRADICTION and c >= e and c >= n:
            return "contradiction", c
        return "neutral", n

    except Exception as exc:
        logger.debug("[local_nli] Scoring failed: %s", exc)
        return "neutral", 0.5


def batch_nli_scores(
    premise: str,
    hypotheses: list[str],
) -> list[tuple[str, float]]:
    """
    Score multiple hypotheses against the same premise.
    More efficient than calling nli_score() in a loop.
    """
    if not hypotheses:
        return []
    if not _load_model():
        return [("neutral", 0.5)] * len(hypotheses)

    try:
        import numpy as np
        from scipy.special import softmax

        pairs = [(premise, h) for h in hypotheses]
        raw = _pipeline.predict(pairs, apply_softmax=False)

        results = []
        for row in raw:
            probs = softmax(row)
            e = float(probs[_IDX_ENTAILMENT])
            c = float(probs[_IDX_CONTRADICTION])
            n = float(probs[_IDX_NEUTRAL])
            if e >= _THRESHOLD_ENTAILMENT and e >= c and e >= n:
                results.append(("entailment", e))
            elif c >= _THRESHOLD_CONTRADICTION and c >= e and c >= n:
                results.append(("contradiction", c))
            else:
                results.append(("neutral", n))
        return results

    except Exception as exc:
        logger.debug("[local_nli] Batch scoring failed: %s", exc)
        return [("neutral", 0.5)] * len(hypotheses)


def is_available() -> bool:
    """Return True if the NLI model is (or can be) loaded."""
    return _load_model()
