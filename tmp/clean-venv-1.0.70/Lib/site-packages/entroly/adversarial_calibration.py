"""
Adversarial Calibration via Group-DRO — EICV Phase 2
======================================================

Group Distributionally Robust Optimization (Sagawa et al. ICLR 2020) over
the 5 perturbation manifolds:

    θ* = argmin_θ max_{m ∈ M} L_m(θ)

where L_m(θ) = 1 - AUROC_m(θ) is the per-manifold loss under hyperparameter
configuration θ.

This is fundamentally different from ERM (average minimization):
  ERM: θ_ERM = argmin_θ (1/|M|) Σ_m L_m(θ)         — may sacrifice C2 for C1
  DRO: θ_DRO = argmin_θ max_m L_m(θ)                 — worst-case robust

The Fusion-4 collapse (C1=0.89, C4=0.45) is exactly the ERM failure mode:
average AUROC is high but worst-manifold is catastrophic. Group-DRO selects
the configuration whose worst-case is best.

ESG hyperparameters (θ)
------------------------
  lambda_contradict       ∈ {0.3, 0.5, 0.7, 0.9}   weight of K(u,v) in T(G)
  num_penalty_weight      ∈ {0.2, 0.4, 0.6}        coefficient in S(u,v)
  ent_penalty_weight      ∈ {0.2, 0.3, 0.4}        coefficient in S(u,v)
  ent_bonus_weight        ∈ {0.1, 0.2, 0.3}        coefficient in S(u,v)

Grid size: 4 × 3 × 3 × 3 = 108 configurations. Evaluated on 5 manifolds
× ~200 items = ~108,000 ESG calls. Should run in ~5 minutes.

The output is a JSON file recording:
  - the chosen θ*
  - per-manifold AUROC at θ*
  - per-manifold AUROC at the ERM baseline (for comparison)
  - the worst-case improvement (which validates DRO's value)

Result is written to benchmarks/results/adversarial_calibration.json
and the chosen θ* is applied to ESGAnalyzer's defaults (in a follow-up commit
or via the env-var ESG_CONFIG_PATH).

References
----------
- Sagawa et al., 2020. Distributionally Robust Neural Networks for Group
  Shifts. ICLR 2020.
- Duchi & Namkoong, 2021. Learning models with uniform performance via
  distributionally robust optimization. Annals of Statistics.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass, field
from typing import Sequence


# ── Hyperparameter dataclass ──────────────────────────────────────────


@dataclass(frozen=True)
class ESGConfig:
    """Hyperparameters for the ESGAnalyzer / scoring functions."""
    lambda_contradict: float = 0.5
    num_penalty_weight: float = 0.4   # support: number mismatch coefficient
    ent_penalty_weight: float = 0.30  # support: entity mismatch coefficient
    ent_bonus_weight: float = 0.20    # support: entity match bonus
    num_bonus_weight: float = 0.15    # support: number match bonus

    def as_tuple(self) -> tuple:
        return (
            self.lambda_contradict, self.num_penalty_weight,
            self.ent_penalty_weight, self.ent_bonus_weight,
            self.num_bonus_weight,
        )

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ── Configurable scorer ───────────────────────────────────────────────


def _idf_weight(word: str, ref_text: str) -> float:
    return 1.0 / (1.0 + math.log(1.0 + ref_text.lower().count(word)))


def _support_with_config(
    ev_atom: str, claim_atom: str, cfg: ESGConfig,
) -> float:
    """Configurable variant of esg._support_score."""
    from entroly.esg import _content_words, _named_entities, _extract_numbers

    claim_words = _content_words(claim_atom)
    if not claim_words:
        return 0.5
    ev_lower = ev_atom.lower()
    ev_words = _content_words(ev_atom)
    shared = claim_words & ev_words
    if not shared:
        idf_jaccard = 0.0
    else:
        num = sum(_idf_weight(w, ev_lower) for w in shared)
        den = sum(_idf_weight(w, ev_lower) for w in claim_words)
        idf_jaccard = num / max(den, 1e-9)

    claim_ents = _named_entities(claim_atom)
    if claim_ents:
        matched = sum(1 for e in claim_ents if e in ev_lower)
        if matched == len(claim_ents):
            ent_bonus = cfg.ent_bonus_weight
        else:
            missing = len(claim_ents) - matched
            ent_bonus = -cfg.ent_penalty_weight * missing / len(claim_ents)
    else:
        ent_bonus = 0.0

    claim_nums = _extract_numbers(claim_atom)
    if claim_nums:
        ev_nums = _extract_numbers(ev_atom)
        n_miss = sum(1 for n in claim_nums if n not in ev_nums)
        if n_miss == 0:
            num_bonus = cfg.num_bonus_weight
        else:
            num_bonus = -cfg.num_penalty_weight * n_miss / len(claim_nums)
    else:
        num_bonus = 0.0

    raw = idf_jaccard + ent_bonus + num_bonus
    return float(max(0.0, min(1.0, raw)))


def _contradiction_with_config(ev_atom: str, claim_atom: str) -> float:
    """Contradiction function is not tuned in this DRO sweep — uses
    the default ESG contradiction logic."""
    from entroly.esg import _contradiction_score
    return _contradiction_score(ev_atom, claim_atom)


def score_with_config(evidence: str, claim: str, cfg: ESGConfig) -> float:
    """Compute T(G) with custom hyperparameters."""
    from entroly.esg import _split_atoms

    ev_atoms = _split_atoms(evidence, min_words=3)
    claim_atoms = _split_atoms(claim, min_words=2)
    if not claim_atoms:
        return 0.0
    if not ev_atoms:
        return 1.0

    contrib_total = 0.0
    for ca in claim_atoms:
        best_sup = 0.0
        best_con = 0.0
        for ea in ev_atoms:
            s = _support_with_config(ea, ca, cfg)
            k = _contradiction_with_config(ea, ca)
            if s > best_sup:
                best_sup = s
            if k > best_con:
                best_con = k
        unsupported = max(0.0, 1.0 - best_sup)
        contrib_total += unsupported + cfg.lambda_contradict * best_con

    tension = contrib_total / len(claim_atoms)
    return float(max(0.0, min(1.0, tension)))


# ── AUROC ─────────────────────────────────────────────────────────────


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    from .metrics import tie_corrected_auroc
    return tie_corrected_auroc(scores, labels)


# ── Manifold construction ─────────────────────────────────────────────


def _build_c2_entity_controlled(
    items: list[tuple[str, str, str]],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """C2: shuffle named entities among right/halu pairs to neutralise
    the entity-gap signal."""
    import re
    _ENT_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")

    def _shuffle_entities(text: str) -> str:
        ents = list({m.group() for m in _ENT_RE.finditer(text)})
        if len(ents) < 2:
            return text
        mapping = dict(zip(ents, rng.sample(ents, len(ents))))
        out = text
        for orig, new in mapping.items():
            if orig != new:
                out = out.replace(orig, new)
        return out

    return [
        (ctx, _shuffle_entities(right), _shuffle_entities(halu))
        for ctx, right, halu in items
    ]


def _build_c3_paraphrase(
    items: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """C3: light paraphrase of the right answer (synonym swaps)."""
    _SWAPS = {
        "is": "was", "are": "were", "has": "had", "have": "had",
        "shows": "demonstrates", "found": "discovered",
        "known": "regarded", "called": "named", "used": "employed",
    }

    def _para(text: str) -> str:
        out = []
        for w in text.split():
            out.append(_SWAPS.get(w.lower(), w))
        return " ".join(out)

    return [(ctx, _para(right), halu) for ctx, right, halu in items]


def _build_c4_realistic(
    items: list[tuple[str, str, str]],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """C4: realistic = paraphrase right + entity-control halu."""
    c3 = _build_c3_paraphrase(items)
    return [
        (ctx, right, _build_c2_entity_controlled([(ctx, right, halu)], rng)[0][2])
        for (ctx, right, _), (_, _, halu) in zip(c3, items)
    ]


# ── DRO Search ────────────────────────────────────────────────────────


@dataclass
class CalibrationResult:
    best_config: ESGConfig
    erm_config: ESGConfig
    best_manifold_aurocs: dict[str, float]
    erm_manifold_aurocs: dict[str, float]
    worst_auroc_best: float    # worst across manifolds at best_config
    worst_auroc_erm: float     # worst across manifolds at erm_config
    mean_auroc_best: float
    mean_auroc_erm: float
    n_configs_evaluated: int
    n_items_per_manifold: int

    def as_dict(self) -> dict:
        return {
            "best_config": self.best_config.as_dict(),
            "erm_config": self.erm_config.as_dict(),
            "best_manifold_aurocs": self.best_manifold_aurocs,
            "erm_manifold_aurocs": self.erm_manifold_aurocs,
            "worst_auroc_best": self.worst_auroc_best,
            "worst_auroc_erm": self.worst_auroc_erm,
            "mean_auroc_best": self.mean_auroc_best,
            "mean_auroc_erm": self.mean_auroc_erm,
            "dro_improvement_worst_case": (
                self.worst_auroc_best - self.worst_auroc_erm
            ),
            "n_configs_evaluated": self.n_configs_evaluated,
            "n_items_per_manifold": self.n_items_per_manifold,
        }


def _eval_config_on_manifold(
    cfg: ESGConfig,
    pairs: list[tuple[str, str, str]],
) -> float:
    """AUROC of T(G) under cfg on the C-manifold pairs."""
    scores = []
    labels = []
    for ctx, right, halu in pairs:
        scores += [
            score_with_config(ctx, right, cfg),
            score_with_config(ctx, halu, cfg),
        ]
        labels += [0, 1]
    return _auroc(scores, labels)


def group_dro_search(
    items: list[tuple[str, str, str]],
    *,
    grid_size: str = "medium",
    seed: int = 42,
) -> CalibrationResult:
    """Group-DRO grid search over ESG hyperparameters.

    Args:
        items: list of (context, right_answer, hallucinated_answer) triples.
        grid_size: "small" (16 configs), "medium" (108), "large" (324).
        seed: RNG seed.
    """
    rng = random.Random(seed)

    # Grid definitions
    if grid_size == "small":
        grid = {
            "lambda_contradict": [0.3, 0.7],
            "num_penalty_weight": [0.3, 0.5],
            "ent_penalty_weight": [0.2, 0.4],
            "ent_bonus_weight": [0.2],
            "num_bonus_weight": [0.15],
        }
    elif grid_size == "large":
        grid = {
            "lambda_contradict": [0.2, 0.4, 0.6, 0.8],
            "num_penalty_weight": [0.2, 0.3, 0.4, 0.5, 0.6],
            "ent_penalty_weight": [0.2, 0.3, 0.4],
            "ent_bonus_weight": [0.1, 0.2, 0.3],
            "num_bonus_weight": [0.10, 0.15, 0.20],
        }
    else:  # medium
        grid = {
            "lambda_contradict": [0.3, 0.5, 0.7],
            "num_penalty_weight": [0.3, 0.4, 0.5],
            "ent_penalty_weight": [0.25, 0.35],
            "ent_bonus_weight": [0.15, 0.25],
            "num_bonus_weight": [0.15],
        }

    # Build manifolds (C1, C2, C3, C4)
    manifolds = {
        "C1": items,
        "C2": _build_c2_entity_controlled(items, rng),
        "C3": _build_c3_paraphrase(items),
        "C4": _build_c4_realistic(items, rng),
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    best_worst = -1.0
    best_cfg: ESGConfig | None = None
    best_aurocs: dict[str, float] = {}

    erm_best_mean = -1.0
    erm_cfg: ESGConfig | None = None
    erm_aurocs: dict[str, float] = {}

    n_done = 0
    for combo in combos:
        cfg = ESGConfig(**dict(zip(keys, combo)))
        m_aurocs = {
            name: _eval_config_on_manifold(cfg, pairs)
            for name, pairs in manifolds.items()
        }
        worst = min(m_aurocs.values())
        mean = sum(m_aurocs.values()) / len(m_aurocs)

        if worst > best_worst:
            best_worst = worst
            best_cfg = cfg
            best_aurocs = m_aurocs

        if mean > erm_best_mean:
            erm_best_mean = mean
            erm_cfg = cfg
            erm_aurocs = m_aurocs

        n_done += 1

    assert best_cfg is not None and erm_cfg is not None
    return CalibrationResult(
        best_config=best_cfg,
        erm_config=erm_cfg,
        best_manifold_aurocs={k: round(v, 4) for k, v in best_aurocs.items()},
        erm_manifold_aurocs={k: round(v, 4) for k, v in erm_aurocs.items()},
        worst_auroc_best=round(best_worst, 4),
        worst_auroc_erm=round(min(erm_aurocs.values()), 4),
        mean_auroc_best=round(sum(best_aurocs.values()) / 4, 4),
        mean_auroc_erm=round(erm_best_mean, 4),
        n_configs_evaluated=n_done,
        n_items_per_manifold=len(items),
    )
