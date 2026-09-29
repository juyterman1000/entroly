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

When no posterior exists at all -- the model could not be loaded, or scoring
raised -- the result is "unavailable" with confidence 0, never "neutral". The
three classes above partition the claim/evidence relation; "did not check" is
not a member of that partition, and returning a neutral for it asserts a
measured N of 0.5 that no model produced. Downstream that matters: the vacuous
state is the identity under evidence combination, while a fabricated 0.5
dilutes whatever real evidence it is fused with.

When this runs: every claim, not a cascade
-------------------------------------------
This section previously described a cascade -- that WITNESS ran the model "only
when the deterministic local_pav verdict is neutral AND the claim risk from the
continuous path is in the uncertain band (0.25-0.75)", limiting it to "~30% of
claims". **No such gate exists.** `witness.py`'s `elif self.use_local_nli:`
branch is unconditional, and `local_pav`'s verdict is never consulted in the
condition. Measured 2026-09-29: 125 NLI calls for 125 claims, 100.0%.

It was never implemented rather than removed later: `git log -S` for the
condition returns nothing across the file's history, and the paragraph describing
it arrived in 2ee2b6aa -- the commit that added this module. The design was
written down and the gate was not built.

The cost is therefore roughly three times what that paragraph implied. Measured
on HaluEval, n=200 per slice, `force_python=True`, model already loaded, against
the same run with the flag off: 178 ms/sample vs 3.3 (QA), 331 vs 6.2
(Dialogue), 1302 vs 94.9 (Summarization) -- 14x to 54x.

Do not "fix" this by adding the gate. Enabling this model measurably *lowers*
WITNESS discrimination, and firing it less often would not change why:

    slice            AUROC off   AUROC on    delta
    HaluEval-QA         0.7686     0.7253   -0.0434
    HaluEval-Dialogue   0.5704     0.5340   -0.0364
    HaluEval-Summ.      0.6465     0.6399   -0.0067

(tie-corrected AUROC, 97 hallucinated of 200 per slice, paired on identical
samples.) A confident NLI verdict makes `use_continuous` false in
`_certify_claim`, so the claim leaves the continuous risk model for the
discrete-bucket path, where risk collapses to coarse constants -- distinct risk
values fell 75->39, 70->52, 68->54. The model does not merely fail to help; it
discards a better-calibrated signal. A cascade would reduce how often that
happens, not whether it happens.

Enabling it (off by default, and the measurement above is why)
---------------------------------------------------------------
    from entroly import WitnessAnalyzer
    analyzer = WitnessAnalyzer(use_local_nli=True)   # downloads model once

Or set the environment variable:
    ENTROLY_LOCAL_NLI=1

Either path resolves the model through the Hugging Face cache and fetches it if
absent, so the first scored pair may reach the network. `HF_HUB_OFFLINE=1` keeps
it local, at the cost of every pair reporting "unavailable".

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

    label ∈ {"entailment", "contradiction", "neutral", "unavailable"}
    confidence ∈ [0, 1]

    ``"unavailable"`` means the check did not run -- the model could not be
    loaded, or scoring raised. It is deliberately not ``"neutral"``: neutral is
    a verdict about the pair, and returning one for a check that never happened
    makes an unverified claim indistinguishable from a verified one. Callers
    that treat every non-entailment, non-contradiction label as neutral still
    degrade the same way as before; callers that care can now tell.
    """
    if not _load_model():
        return "unavailable", 0.0

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
        # Scoring raised, so no verdict was produced. Same reasoning as an
        # unloadable model: this is a fact about the check, not about the pair.
        logger.debug("[local_nli] Scoring failed: %s", exc)
        return "unavailable", 0.0


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
        return [("unavailable", 0.0)] * len(hypotheses)

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
        return [("unavailable", 0.0)] * len(hypotheses)


def is_available() -> bool:
    """Return True if the NLI model is (or can be) loaded."""
    return _load_model()
