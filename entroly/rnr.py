"""
RNR* — Retrieval Necessity Ratio (EICV Layer 3)
================================================

RNR* = I(Ŷ; S) — mutual information between the ESG verdict Ŷ and the
evidence-source indicator S ∈ {real=1, shuffled=0}.

Motivation
----------
A hallucination detector that fires on *claim-internal* lexical cues
(rarity of vocabulary, entity frequency, sentence structure) will produce
the same score whether the evidence is real or randomly substituted. Its
predictions carry no *retrieval necessity* — the retrieval step is unused.

RNR* measures exactly this: how much does using the *correct* evidence
(vs. a random corpus sample) change the ESG verdict distribution?

    I(Ŷ; S) = H(Ŷ) - H(Ŷ | S)

where:
  Ŷ = discretized ESG tension (K=5 bins, edges at [0, 0.2, 0.4, 0.6, 0.8, 1.0])
  S = 1 if evidence is the real retrieval, 0 if evidence is shuffled
  H = Shannon entropy (nats)

Higher RNR* → the evidence source matters more → the detector is genuinely
retrieval-grounded, not claim-internal.

Supplementary metrics
---------------------
- AUROC_real: AUROC of T(G) on real-evidence pairs
- AUROC_null: AUROC of T(G) on shuffled-evidence pairs
- AUROC_gap:  AUROC_real − AUROC_null  (simpler MI proxy)
- necessity_fraction: fraction of (evidence, claim) pairs where T(G)_real
  differs from T(G)_null by ≥ delta (default 0.15) — the "retrieval
  actually changed the verdict" rate

The I(Ŷ; S) estimator is discrete-coarsened to avoid the curse of
dimensionality in a continuous MI estimate. The K=5 binning is chosen
so each bin contains at least 10–20 samples at N=400 per condition.

References
----------
- Cover & Thomas, 2006. Elements of Information Theory, §2.2.
- Ross, 2014. Mutual Information between Discrete and Continuous Data
  Sets. PLOS ONE.
- EICV_PREREGISTRATION.md §3.3.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


# ── Binning spec ─────────────────────────────────────────────────────

_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_N_BINS = len(_BIN_EDGES) - 1   # 5 bins


def _bin(t: float) -> int:
    """Map T(G) ∈ [0, 1] to a bin index ∈ {0, …, _N_BINS-1}."""
    for i in range(_N_BINS - 1):
        if t < _BIN_EDGES[i + 1]:
            return i
    return _N_BINS - 1   # t == 1.0


def _entropy_nats(counts: list[int]) -> float:
    """Shannon entropy H in nats from a count vector."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


# ── Result dataclass ──────────────────────────────────────────────────


@dataclass
class RNRResult:
    """Output of RNRAnalyzer.compute().

    Primary signal: `mutual_information` I(Ŷ; S) in nats.
    Supplementary: AUROC gap, necessity fraction.
    """
    # Primary signal
    mutual_information: float   # I(Ŷ; S) in nats — higher = more retrieval-grounded

    # Supplementary
    auroc_real: float           # AUROC of T(G) with real evidence
    auroc_null: float           # AUROC of T(G) with shuffled evidence
    auroc_gap: float            # auroc_real − auroc_null

    # Empirical necessity: fraction of pairs where the verdict changed
    necessity_fraction: float   # frac where |T_real - T_null| >= delta

    # Entropy components (for decomposition)
    h_yhat: float               # H(Ŷ) — marginal entropy of discretized T(G)
    h_yhat_given_real: float    # H(Ŷ | S=real)
    h_yhat_given_null: float    # H(Ŷ | S=shuffled)

    # Sample counts
    n_real: int
    n_null: int

    # Config echo
    n_bins: int = _N_BINS
    necessity_delta: float = 0.15

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── AUROC ─────────────────────────────────────────────────────────────


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels))
    n0 = sum(1 for _, y in pairs if y == 0)
    n1 = sum(1 for _, y in pairs if y == 1)
    if n0 == 0 or n1 == 0:
        return 0.5
    rank_sum = sum(r for r, (_, y) in enumerate(pairs, 1) if y == 1)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)


# ── Main analyser ─────────────────────────────────────────────────────


class RNRAnalyzer:
    """Computes RNR* = I(Ŷ; S) for a hallucination detector.

    Args:
        scorer: callable (evidence: str, claim: str) -> float in [0,1].
            Higher score = more likely hallucinated. Typically ESGAnalyzer.tension.
        necessity_delta: threshold for declaring a verdict change between
            real and null evidence (default 0.15).
        seed: RNG seed for shuffling (default 42).
    """

    def __init__(
        self,
        scorer: "callable[[str, str], float]",
        necessity_delta: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.scorer = scorer
        self.necessity_delta = necessity_delta
        self._rng = random.Random(seed)

    def compute(
        self,
        items: Sequence[tuple[str, str, int]],
        *,
        n_shuffles: int = 1,
    ) -> RNRResult:
        """Compute RNR* over a dataset.

        Args:
            items: sequence of (evidence, claim, label) triples.
                label 0 = grounded, 1 = hallucinated.
            n_shuffles: number of shuffled-evidence pairings per item
                (default 1 for speed; ≥3 for publication runs).

        Returns:
            RNRResult with I(Ŷ; S) and supporting statistics.
        """
        n = len(items)
        if n == 0:
            return self._empty_result()

        evidences = [ev for ev, _, _ in items]
        claims   = [cl for _, cl, _ in items]
        labels   = [lb for _, _, lb in items]

        # Score real pairs
        t_real = [self.scorer(ev, cl) for ev, cl in zip(evidences, claims)]

        # Score null pairs: shuffle evidence pool, reassign to each claim
        t_null_all: list[list[float]] = []
        for _ in range(n_shuffles):
            shuffled_evs = list(evidences)
            self._rng.shuffle(shuffled_evs)
            # Avoid any accidentally identical pairing (best-effort; n is usually large)
            for i in range(n):
                if shuffled_evs[i] == evidences[i] and n > 1:
                    j = (i + 1) % n
                    shuffled_evs[i], shuffled_evs[j] = shuffled_evs[j], shuffled_evs[i]
            t_null_run = [self.scorer(ev, cl) for ev, cl in zip(shuffled_evs, claims)]
            t_null_all.append(t_null_run)

        # Average null scores across shuffles for stability
        t_null = [
            sum(t_null_all[s][i] for s in range(n_shuffles)) / n_shuffles
            for i in range(n)
        ]

        # ── AUROC ────────────────────────────────────────────────────
        auroc_real = _auroc(t_real, labels)
        auroc_null = _auroc(t_null, labels)
        auroc_gap  = auroc_real - auroc_null

        # ── Discrete MI: I(Ŷ; S) ─────────────────────────────────────
        # Build joint count table: rows=bin index, cols=S∈{0=null,1=real}
        joint_counts = [[0] * 2 for _ in range(_N_BINS)]
        for i in range(n):
            br = _bin(t_real[i])
            bn = _bin(t_null[i])
            joint_counts[br][1] += 1   # S = real
            joint_counts[bn][0] += 1   # S = null

        # Marginal Ŷ
        yhat_counts = [joint_counts[b][0] + joint_counts[b][1] for b in range(_N_BINS)]
        h_yhat = _entropy_nats(yhat_counts)

        # Conditional entropies H(Ŷ|S=real) and H(Ŷ|S=null)
        real_counts = [joint_counts[b][1] for b in range(_N_BINS)]
        null_counts = [joint_counts[b][0] for b in range(_N_BINS)]
        h_real = _entropy_nats(real_counts)
        h_null = _entropy_nats(null_counts)

        # H(Ŷ|S) = P(S=real)*H(Ŷ|S=real) + P(S=null)*H(Ŷ|S=null)
        # Both halves contribute equally (n real, n null)
        h_yhat_given_s = 0.5 * h_real + 0.5 * h_null

        mi = max(0.0, h_yhat - h_yhat_given_s)

        # ── Necessity fraction ────────────────────────────────────────
        changed = sum(
            1 for r, nu in zip(t_real, t_null)
            if abs(r - nu) >= self.necessity_delta
        )
        necessity_frac = changed / n

        return RNRResult(
            mutual_information=round(mi, 6),
            auroc_real=round(auroc_real, 4),
            auroc_null=round(auroc_null, 4),
            auroc_gap=round(auroc_gap, 4),
            necessity_fraction=round(necessity_frac, 4),
            h_yhat=round(h_yhat, 6),
            h_yhat_given_real=round(h_real, 6),
            h_yhat_given_null=round(h_null, 6),
            n_real=n,
            n_null=n * n_shuffles,
            necessity_delta=self.necessity_delta,
        )

    def _empty_result(self) -> RNRResult:
        return RNRResult(
            mutual_information=0.0,
            auroc_real=0.5, auroc_null=0.5, auroc_gap=0.0,
            necessity_fraction=0.0,
            h_yhat=0.0, h_yhat_given_real=0.0, h_yhat_given_null=0.0,
            n_real=0, n_null=0,
        )


# ── Module-level convenience ──────────────────────────────────────────


def compute_rnr(
    items: Sequence[tuple[str, str, int]],
    scorer: "callable[[str, str], float] | None" = None,
    *,
    n_shuffles: int = 1,
    seed: int = 42,
) -> RNRResult:
    """Compute RNR* on (evidence, claim, label) triples.

    Args:
        items: sequence of (evidence, claim, label) tuples.
        scorer: callable (evidence, claim) -> float ∈ [0,1].
            Defaults to ESGAnalyzer.tension.
        n_shuffles: number of shuffle repetitions (default 1).
        seed: RNG seed.

    Returns:
        RNRResult with I(Ŷ; S) and supporting statistics.
    """
    if scorer is None:
        from entroly.esg import compute_tension as scorer  # type: ignore[assignment]
    return RNRAnalyzer(scorer=scorer, seed=seed).compute(items, n_shuffles=n_shuffles)
