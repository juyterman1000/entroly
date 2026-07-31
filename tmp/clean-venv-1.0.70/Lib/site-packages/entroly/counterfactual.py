"""
Falsification Gradient Γ — EICV Layer 4
=========================================

Γ(x) = mean_{m ∈ M} |S(x) - S(x + δ_m)|

where:
  S(x)       = T(G)(evidence, claim) — the ESG tension score
  δ_m        = a unit perturbation on manifold m ∈ M
  M          = 5 perturbation manifolds (entity, numeric, compositional,
               temporal, retrieval-shuffle)

Lower Γ → the score is stable under perturbation → the detector is
measuring structural hallucination defects, not surface lexical artifacts.
Higher Γ → the score is brittle → surface-level, artifact-prone.

Relation to Fisher information
--------------------------------
For a continuous score S(x) with random perturbation δ ~ N(0, σ²I):
  E_δ[|S(x+δ) - S(x)|] / σ = √Fisher_x(S)   [Amari, 1998]

Our discrete perturbations approximate this in the five directions that
matter for hallucination detection. Low Γ on all 5 manifolds is a
necessary condition for robustness to distribution shift (the C1-C4
protocol measures exactly this at the dataset level).

The 5 manifolds
----------------
M1 entity    — swap one named entity with a plausible alternative
               (e.g. "Einstein" → "Bohr", "Paris" → "London")
M2 numeric   — perturb one number by ±(1 + 10% of value)
               (e.g. "1915" → "1916", "0.73" → "0.80")
M3 compose   — reorder clauses joined by coordinators
               (e.g. "A but B" → "B but A")
M4 temporal  — change tense / temporal markers
               (e.g. "developed" → "develops", "in 1915" → "by 1915")
M5 retrieval — substitute evidence with a passage from a different topic
               (in practice: shuffle evidence with another item from the batch)

Result: GammaResult with per-manifold |ΔT| and aggregate Γ.

References
----------
- Amari & Nagaoka, 2000. Methods of Information Geometry. AMS.
- EICV_PREREGISTRATION.md §3.4.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from typing import Sequence


# ── Perturbation helpers ──────────────────────────────────────────────

_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_NUMBER_RE  = re.compile(r"\b\d+(?:\.\d+)?\b")
_WORD_RE    = re.compile(r"[A-Za-z]+")

# Plausible entity substitutions (closed list, same domain classes)
_ENTITY_POOL = [
    "Newton", "Bohr", "Curie", "Darwin", "Turing", "Planck", "Dirac",
    "Lorentz", "Maxwell", "Faraday", "Kelvin", "Gauss", "Euler",
    "London", "Berlin", "Vienna", "Madrid", "Rome", "Warsaw", "Prague",
    "Tokyo", "Sydney", "Toronto", "Geneva", "Brussels",
    "Company", "Institute", "University", "Foundation", "Council",
]

_TENSE_SWAPS = {
    "is": "was", "was": "is",
    "are": "were", "were": "are",
    "has": "had", "had": "has",
    "have": "had",
    "developed": "develops", "discovered": "discovers",
    "found": "finds", "showed": "shows",
    "produced": "produces", "created": "creates",
    "won": "wins", "lost": "loses",
    "called": "calls", "named": "names",
    "known": "regarded",
}

# Temporal swaps are split into STYLISTIC (meaning-preserving) and
# SEMANTIC (meaning-changing). The Γ benchmark uses only stylistic ones
# so high Γ on M4 unambiguously indicates brittleness.
_TEMPORAL_STYLISTIC = {
    # Aspect / tense markers — these are essentially meaning-preserving
    # when the time reference is otherwise clear from context.
    "is": "was", "was": "is",
    "are": "were", "were": "are",
}

# (Kept for reference but not used in M4 anymore — these are SEMANTIC)
_TEMPORAL_SEMANTIC = {
    "before": "after", "after": "before",
    "during": "following", "following": "during",
    "since": "until", "until": "since",
}

# Public alias used by _perturb_temporal
_TEMPORAL_SWAPS = _TEMPORAL_STYLISTIC

_COORD_RE = re.compile(
    r"\b(and|but|or|however|although|while|whereas|yet|moreover"
    r"|furthermore|additionally|nevertheless|consequently)\b",
    re.I,
)


def _perturb_entity(text: str, rng: random.Random) -> str:
    """M1: swap one named entity with a plausible substitute."""
    ents = list({m.group() for m in _ENTITY_RE.finditer(text)})
    if not ents:
        return text
    target = rng.choice(ents)
    # Pick a replacement that is NOT the original
    replacements = [e for e in _ENTITY_POOL if e.lower() != target.lower()]
    if not replacements:
        return text
    replacement = rng.choice(replacements)
    return text.replace(target, replacement, 1)


def _perturb_numeric(text: str, rng: random.Random) -> str:
    """M2: perturb one number by ±(1 + 10%)."""
    nums = list({m.group() for m in _NUMBER_RE.finditer(text)})
    if not nums:
        return text
    target = rng.choice(nums)
    try:
        val = float(target)
    except ValueError:
        return text
    delta = max(1.0, abs(val) * 0.10)
    new_val = val + rng.choice([-1, 1]) * delta
    if "." not in target:
        new_str = str(int(round(new_val)))
    else:
        decimals = len(target.split(".")[-1])
        new_str = f"{new_val:.{decimals}f}"
    return text.replace(target, new_str, 1)


def _perturb_composition(text: str, rng: random.Random) -> str:
    """M3: swap clauses around a coordinator ("A but B" → "B but A")."""
    m = _COORD_RE.search(text)
    if m is None:
        return text
    coord = m.group()
    before = text[: m.start()].strip()
    after  = text[m.end() :].strip()
    if before and after:
        return f"{after} {coord} {before}"
    return text


def _perturb_temporal(text: str, rng: random.Random) -> str:
    """M4: swap one tense word or temporal preposition."""
    tokens = _WORD_RE.findall(text)
    candidates = []
    for t in tokens:
        tl = t.lower()
        if tl in _TENSE_SWAPS or tl in _TEMPORAL_SWAPS:
            candidates.append(t)
    if not candidates:
        return text
    target = rng.choice(candidates)
    tl = target.lower()
    replacement = _TENSE_SWAPS.get(tl) or _TEMPORAL_SWAPS.get(tl, target)
    # Preserve capitalisation
    if target[0].isupper():
        replacement = replacement.capitalize()
    return text.replace(target, replacement, 1)


def _perturb_retrieval(
    evidence: str,
    pool: "Sequence[str]",
    rng: random.Random,
) -> str:
    """M5: substitute evidence with a random item from the pool.

    If pool is empty or all items are identical to evidence, return evidence.
    """
    others = [e for e in pool if e != evidence]
    if not others:
        return evidence
    return rng.choice(others)


# ── Result dataclasses ────────────────────────────────────────────────


@dataclass
class ManifoldGamma:
    """Per-manifold |ΔT(G)| statistics."""
    manifold: str      # "entity", "numeric", "compose", "temporal", "retrieval"
    mean_delta: float  # mean |T(original) - T(perturbed)| over all items
    fraction_changed: float  # fraction with |ΔT| >= threshold (default 0.05)
    n_perturbable: int  # number of items where perturbation was possible


@dataclass
class GammaResult:
    """Output of FalsificationGradient.compute().

    Primary signal: `gamma` = mean_{m} ManifoldGamma.mean_delta
    Lower is better (more robust).
    """
    gamma: float                           # aggregate Γ across all manifolds
    manifolds: list[ManifoldGamma] = field(default_factory=list)
    n_items: int = 0
    change_threshold: float = 0.05

    @property
    def is_robust(self) -> bool:
        """True if Γ < 0.15 (heuristic robustness threshold)."""
        return self.gamma < 0.15

    def as_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "is_robust": self.is_robust,
            "n_items": self.n_items,
            "change_threshold": self.change_threshold,
            "manifolds": [m.__dict__ for m in self.manifolds],
        }


# ── Main class ────────────────────────────────────────────────────────


class FalsificationGradient:
    """Computes Γ(x) = mean_{m} |S(x) - S(x + δ_m)|.

    Args:
        scorer: callable (evidence: str, claim: str) -> float ∈ [0,1].
            Typically ESGAnalyzer.tension.
        change_threshold: |ΔT| must exceed this to count as "changed"
            for the fraction_changed statistic (default 0.05).
        seed: RNG seed (default 42).
    """

    def __init__(
        self,
        scorer: "callable[[str, str], float]",
        change_threshold: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.scorer = scorer
        self.change_threshold = change_threshold
        self._rng = random.Random(seed)

    def compute(
        self,
        items: Sequence[tuple[str, str]],
    ) -> GammaResult:
        """Compute Γ over a list of (evidence, claim) pairs.

        Args:
            items: sequence of (evidence, claim) tuples.

        Returns:
            GammaResult with per-manifold and aggregate Γ.
        """
        n = len(items)
        if n == 0:
            return GammaResult(gamma=0.0, n_items=0)

        evidences = [ev for ev, _ in items]
        claims    = [cl for _, cl in items]

        # Baseline scores
        t_base = [self.scorer(ev, cl) for ev, cl in zip(evidences, claims)]

        manifold_results: list[ManifoldGamma] = []

        def _eval_manifold(
            name: str,
            perturb_fn: "callable[[str, str, random.Random], tuple[str, str]]",
        ) -> ManifoldGamma:
            deltas: list[float] = []
            changed = 0
            n_perturbable = 0
            for i, (ev, cl) in enumerate(items):
                new_ev, new_cl = perturb_fn(ev, cl, self._rng)
                # Only score if something actually changed
                if new_ev != ev or new_cl != cl:
                    n_perturbable += 1
                    t_new = self.scorer(new_ev, new_cl)
                    delta = abs(t_base[i] - t_new)
                    deltas.append(delta)
                    if delta >= self.change_threshold:
                        changed += 1
            mean_d = sum(deltas) / max(len(deltas), 1)
            frac_c = changed / max(n_perturbable, 1)
            return ManifoldGamma(
                manifold=name,
                mean_delta=round(mean_d, 4),
                fraction_changed=round(frac_c, 4),
                n_perturbable=n_perturbable,
            )

        # M1: entity
        def _m1(ev, cl, rng):
            return (ev, _perturb_entity(cl, rng))
        manifold_results.append(_eval_manifold("entity", _m1))

        # M2: numeric
        def _m2(ev, cl, rng):
            cl2 = _perturb_numeric(cl, rng)
            if cl2 == cl:
                cl2 = _perturb_numeric(ev, rng)
                return (cl2, cl)   # fallback: perturb ev number
            return (ev, cl2)
        manifold_results.append(_eval_manifold("numeric", _m2))

        # M3: compositional
        def _m3(ev, cl, rng):
            return (ev, _perturb_composition(cl, rng))
        manifold_results.append(_eval_manifold("compose", _m3))

        # M4: temporal
        def _m4(ev, cl, rng):
            return (ev, _perturb_temporal(cl, rng))
        manifold_results.append(_eval_manifold("temporal", _m4))

        # M5: retrieval shuffle
        def _m5(ev, cl, rng):
            return (_perturb_retrieval(ev, evidences, rng), cl)
        manifold_results.append(_eval_manifold("retrieval", _m5))

        # Aggregate Γ: mean of per-manifold mean_delta
        # (only manifolds with n_perturbable > 0 contribute)
        active = [m for m in manifold_results if m.n_perturbable > 0]
        gamma = sum(m.mean_delta for m in active) / max(len(active), 1)

        return GammaResult(
            gamma=round(gamma, 6),
            manifolds=manifold_results,
            n_items=n,
            change_threshold=self.change_threshold,
        )


# ── Convenience ───────────────────────────────────────────────────────


def compute_gamma(
    items: Sequence[tuple[str, str]],
    scorer: "callable[[str, str], float] | None" = None,
    *,
    seed: int = 42,
) -> GammaResult:
    """Module-level convenience: compute Γ on (evidence, claim) pairs.

    Args:
        items: sequence of (evidence, claim) tuples.
        scorer: callable (ev, cl) -> float. Defaults to ESGAnalyzer.tension.
        seed: RNG seed.
    """
    if scorer is None:
        from entroly.esg import compute_tension as scorer  # type: ignore[assignment]
    return FalsificationGradient(scorer=scorer, seed=seed).compute(items)
