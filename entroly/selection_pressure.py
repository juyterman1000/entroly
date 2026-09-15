"""Selection Pressure Maintenance — Resonance Homeostasis for PRISM 5D.

PRISM has 5 weight dimensions:
  [w_recency, w_frequency, w_semantic, w_entropy] — content dimensions
  [w_resonance] — pairwise fragment interaction bonus

The first 4 are learned online by Dirichlet-REINFORCE in OnlinePrism.
The 5th (resonance) is set by archetypes and tuned by DreamingLoop offline.
This architectural split means resonance NEVER adapts in real-time — it's
blind to whether the content dimensions have collapsed.

Invention: use w_resonance as a HOMEOSTATIC REGULATOR for the other 4.

Biological analogy: hormones don't participate in metabolic pathways — they
monitor and adjust them.  w_resonance shouldn't be another weight in the
Dirichlet.  It should be the regulator that maintains selection health.

Control law:
  Let G(t) = Gini coefficient of [w_r, w_f, w_s, w_e] at time t
  w_resonance(t) = base + gain * max(0, G(t) - G_target)

  When content weights are healthy (G ~ 0.15), resonance stays at its
  archetype base.  When they start collapsing (G -> 0.7), resonance
  increases, forcing the selection to value fragment PAIRS, which
  diversifies the feedback, which re-balances the 4 content weights.

  This is a NEGATIVE FEEDBACK LOOP — the hallmark of stable control
  systems.  The paper ("Learning to Reason for Factuality", Meta FAIR
  2026) proves cross-dimensional checking prevents reward hacking.
  Resonance homeostasis implements that as a continuous control law.

Second invention: QUERY-CONDITIONED WEIGHT MODULATION.

  Different queries need different context composition.  A debugging
  query needs recent changes + callers/tests (boost recency + resonance).
  An architecture query needs broad, information-dense fragments (boost
  entropy + resonance).  The paper proves that different reasoning
  strategies need different optimization targets.

  Per-query modulation in log-space:
    w_eff_i = softmax(log(w_global_i) + sigma_i(query))

  sigma(query) is extracted from query features using simple heuristics.
  When sigma is zero, weights are unchanged.  The softmax ensures they
  still sum to 1.

Third mechanism: STRUCTURED REWARD.

  The paper's three structural constraints on the reward prevent the
  same failure modes in PRISM's feedback loop:
    1. Smoothed precision: F/(T+1)  — handles zero claims
    2. Log-discounted coverage: log(1+A)/log(1+S) — diminishing returns
    3. Binary recovery gate: any recovery caps the reward
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass

logger = logging.getLogger("entroly.selection_pressure")

PRISM_5D = ["w_recency", "w_frequency", "w_semantic", "w_entropy", "w_resonance"]
CONTENT_DIMS = ["w_recency", "w_frequency", "w_semantic", "w_entropy"]


# ═══════════════════════════════════════════════════════════════════════
# 1. Resonance Homeostasis
# ═══════════════════════════════════════════════════════════════════════


def weight_gini(weights: dict[str, float]) -> float:
    """Gini coefficient.  G=0 uniform, G->1 concentrated.  O(n^2), n=5."""
    vals = list(weights.values())
    n = len(vals)
    if n <= 1:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    s = sum(abs(vals[i] - vals[j]) for i in range(n) for j in range(n))
    return s / (2 * n * total)


def weight_entropy(weights: dict[str, float]) -> float:
    """Normalized Shannon entropy.  1.0=uniform, 0.0=concentrated."""
    vals = list(weights.values())
    n = len(vals)
    if n <= 1:
        return 1.0
    total = sum(vals)
    if total <= 0:
        return 1.0
    h = 0.0
    for v in vals:
        p = v / total
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(n)


class ResonanceHomeostasis:
    """Negative-feedback controller: adjusts w_resonance to maintain
    healthy diversity among the 4 content dimensions.

    The controller observes the Gini coefficient of the content weights.
    When Gini exceeds the target (one dimension dominates), it boosts
    w_resonance proportionally.  Higher resonance forces the selection
    to value fragment PAIRS, diversifying the feedback signal, which
    re-balances the content weights.

    The gain parameter k controls responsiveness:
      k=0.5 (default) — moderate.  At Gini=0.7 (severe collapse),
      resonance increases by k*(0.7-0.15) = 0.275 above base.
    """

    def __init__(
        self,
        *,
        base_resonance: float = 0.10,
        gini_target: float = 0.15,
        gain: float = 0.5,
        max_resonance: float = 0.35,
    ):
        self._base = base_resonance
        self._target = gini_target
        self._gain = gain
        self._max = max_resonance
        self._last_gini = 0.0
        self._last_resonance = base_resonance
        self._n_adjustments = 0

    def compute_resonance(self, content_weights: dict[str, float]) -> float:
        """Compute the homeostatic w_resonance for current content weights.

        Args:
            content_weights: dict with the 4 content dimensions.
                             Only the values are used; extra keys are ignored.

        Returns:
            Adjusted w_resonance in [base, max_resonance].
        """
        cw = {k: content_weights[k] for k in CONTENT_DIMS if k in content_weights}
        if not cw:
            return self._base

        g = weight_gini(cw)
        self._last_gini = g

        error = max(0.0, g - self._target)
        resonance = self._base + self._gain * error
        resonance = min(resonance, self._max)

        if resonance > self._base + 0.01:
            self._n_adjustments += 1
            logger.info(
                "Resonance homeostasis: Gini=%.3f > target=%.3f, "
                "w_resonance %.3f -> %.3f",
                g, self._target, self._base, resonance,
            )

        self._last_resonance = resonance
        return resonance

    def stats(self) -> dict:
        return {
            "base_resonance": self._base,
            "last_gini": round(self._last_gini, 4),
            "last_resonance": round(self._last_resonance, 4),
            "n_adjustments": self._n_adjustments,
            "gini_target": self._target,
            "gain": self._gain,
        }


# ═══════════════════════════════════════════════════════════════════════
# 2. Query-Conditioned Weight Modulation
# ═══════════════════════════════════════════════════════════════════════

_DEBUG_RE = re.compile(
    r"\b(bug|crash|error|fail|broke|wrong|fix|debug|trace|stack|exception|panic)\b",
    re.IGNORECASE,
)
_ARCH_RE = re.compile(
    r"\b(architect|system|flow|design|overview|explain|structure|how does .+ work)\b",
    re.IGNORECASE,
)
_REFACTOR_RE = re.compile(
    r"\b(refactor|rename|move|extract|clean|reorganize|split|merge)\b",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"\b(compare|differ|between|versus|vs|alternative)\b",
    re.IGNORECASE,
)
_RECENT_RE = re.compile(
    r"\b(recent|latest|just|last commit|just pushed|new|changed)\b",
    re.IGNORECASE,
)


def compute_query_modulation(query: str) -> dict[str, float]:
    """Extract per-dimension modulation sigma(q) from query text.

    Returns a dict of log-space adjustments.  Positive values amplify
    the corresponding weight; negative values suppress it.  Zero means
    no change.

    The modulation is additive in log-space, so it acts multiplicatively
    on the weights before softmax normalization.
    """
    sigma = {d: 0.0 for d in PRISM_5D}

    if _DEBUG_RE.search(query):
        sigma["w_recency"] += 0.4
        sigma["w_resonance"] += 0.5
        sigma["w_frequency"] -= 0.2

    if _ARCH_RE.search(query):
        sigma["w_entropy"] += 0.4
        sigma["w_resonance"] += 0.6
        sigma["w_semantic"] += 0.2

    if _REFACTOR_RE.search(query):
        sigma["w_frequency"] += 0.5
        sigma["w_semantic"] += 0.3

    if _COMPARE_RE.search(query):
        sigma["w_semantic"] += 0.4
        sigma["w_resonance"] += 0.4

    if _RECENT_RE.search(query):
        sigma["w_recency"] += 0.6

    return sigma


def modulate_weights(
    base_weights: dict[str, float],
    sigma: dict[str, float],
) -> dict[str, float]:
    """Apply log-space modulation to base weights.

    w_eff_i = softmax(log(w_base_i) + sigma_i)

    When sigma is all zeros, the output equals the input (up to
    floating-point precision).  The softmax ensures weights sum to 1.
    """
    dims = list(base_weights.keys())
    if not dims:
        return {}

    logits = []
    for d in dims:
        w = max(base_weights[d], 1e-8)
        logits.append(math.log(w) + sigma.get(d, 0.0))

    max_logit = max(logits)
    exp_logits = [math.exp(v - max_logit) for v in logits]
    total = sum(exp_logits)

    return {d: exp_logits[i] / total for i, d in enumerate(dims)}


# ═══════════════════════════════════════════════════════════════════════
# 3. Structured Reward (paper-inspired)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PressureSignals:
    """Input signals for the structured reward computation."""

    n_grounded: int = 0
    n_contradicted: int = 0
    n_unsupported: int = 0
    n_adequate_fragments: int = 0
    n_selected_fragments: int = 0
    n_distinct_sources: int = 0
    recovery_triggered: bool = False
    utilization: float = 0.5


@dataclass(frozen=True)
class PressureReward:
    """Decomposed reward with per-component visibility."""

    reward: float
    r_precision: float
    r_coverage: float
    r_utilization: float
    r_diversity: float
    gated: bool
    gate_reason: str = ""


def compute_pressure_reward(
    signals: PressureSignals,
    *,
    lambda_coverage: float = 0.3,
    mu_utilization: float = 0.2,
    nu_diversity: float = 0.15,
    gate_cap: float = 0.4,
) -> PressureReward:
    """Compute a structured reward with cross-checking constraints.

    Four components:

    1. Smoothed precision: n_grounded / (total_claims + 1)
    2. Log-discounted coverage: log(1+A) / log(1+S)
    3. Budget utilization (unchanged sweet spot: 70-95%)
    4. Source diversity: n_distinct_sources / max(n_selected, 1)
       — the resonance contribution attribution.  High diversity means
       resonance helped the selection.

    Binary recovery gate: any recovery caps reward at gate_cap.
    """
    total_claims = signals.n_grounded + signals.n_contradicted + signals.n_unsupported

    r_precision = signals.n_grounded / (total_claims + 1)
    if total_claims > 0:
        contradiction_rate = signals.n_contradicted / total_claims
        r_precision *= max(0.0, 1.0 - 2.0 * contradiction_rate)

    if signals.n_selected_fragments > 0:
        r_coverage = math.log(1 + signals.n_adequate_fragments) / math.log(
            1 + signals.n_selected_fragments
        )
    else:
        r_coverage = 0.0
    r_coverage = min(1.0, r_coverage)

    u = signals.utilization
    if u < 0.5:
        r_util = u / 0.5
    elif u <= 0.95:
        r_util = 1.0
    else:
        r_util = max(0.5, 1.0 - (u - 0.95) * 10)

    if signals.n_selected_fragments > 0:
        r_diversity = min(
            1.0, signals.n_distinct_sources / signals.n_selected_fragments
        )
    else:
        r_diversity = 0.0

    max_possible = 1.0 + lambda_coverage + mu_utilization + nu_diversity
    base = (
        r_precision
        + lambda_coverage * r_coverage
        + mu_utilization * r_util
        + nu_diversity * r_diversity
    )
    reward = base / max_possible

    gated = False
    gate_reason = ""
    if signals.recovery_triggered and reward > gate_cap:
        reward = gate_cap
        gated = True
        gate_reason = "recovery"

    reward = max(0.0, min(1.0, reward))

    return PressureReward(
        reward=reward,
        r_precision=r_precision,
        r_coverage=r_coverage,
        r_utilization=r_util,
        r_diversity=r_diversity,
        gated=gated,
        gate_reason=gate_reason,
    )


def estimate_contributions_5d(
    *,
    witness_score: float = 0.5,
    evidence_adequacy: float = 0.5,
    utilization: float = 0.5,
    n_recovered: int = 0,
    source_diversity: float = 0.5,
) -> dict[str, float]:
    """5D contribution estimates for REINFORCE gradient.

    Extends self_improving.estimate_contributions with w_resonance:
    high source diversity means resonance helped the selection.
    """
    c = {
        "w_recency": 0.20,
        "w_frequency": 0.20,
        "w_semantic": 0.20,
        "w_entropy": 0.20,
        "w_resonance": 0.20,
    }

    if evidence_adequacy > 0.6:
        c["w_semantic"] += 0.12
        c["w_entropy"] -= 0.04

    if 0.7 <= utilization <= 0.95:
        c["w_entropy"] += 0.08

    if n_recovered == 0:
        c["w_frequency"] += 0.08

    if witness_score > 0.7:
        c["w_recency"] += 0.08

    if source_diversity > 0.5:
        c["w_resonance"] += 0.15
    elif source_diversity < 0.2:
        c["w_resonance"] -= 0.10

    total = sum(c.values())
    return {k: max(0.0, v) / total for k, v in c.items()}


# ═══════════════════════════════════════════════════════════════════════
# 4. Collapse Detector
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CollapseEvent:
    """Record of a detected weight collapse."""

    observation_index: int
    content_gini: float
    entropy: float
    dominant_dim: str
    dominant_weight: float
    resonance_before: float
    resonance_after: float


class CollapseDetector:
    """Monitors PRISM 5D weights for selection collapse.

    Uses resonance homeostasis as the recovery mechanism instead of
    random Thompson noise.  When the 4 content dimensions collapse,
    the detector boosts w_resonance to restore diversity through
    the selection's pairwise interaction term.
    """

    def __init__(
        self,
        *,
        gini_threshold: float = 0.45,
        homeostasis: ResonanceHomeostasis | None = None,
        cooldown: int = 5,
    ):
        self._gini_threshold = gini_threshold
        self._homeostasis = homeostasis or ResonanceHomeostasis()
        self._cooldown = cooldown
        self._n_observed = 0
        self._n_collapses = 0
        self._last_collapse_at = -1000
        self._history: list[CollapseEvent] = []

    def check(
        self,
        weights: dict[str, float],
    ) -> tuple[float, CollapseEvent | None]:
        """Check weights and compute homeostatic resonance.

        Always returns the adjusted w_resonance.  Returns a CollapseEvent
        only when collapse is detected (Gini exceeds threshold).
        """
        self._n_observed += 1

        content_w = {k: weights.get(k, 0.25) for k in CONTENT_DIMS}
        g = weight_gini(content_w)

        old_res = weights.get("w_resonance", self._homeostasis._base)
        new_res = self._homeostasis.compute_resonance(content_w)

        if g < self._gini_threshold:
            return new_res, None

        if self._n_observed - self._last_collapse_at < self._cooldown:
            return new_res, None

        dominant_dim = max(content_w, key=lambda k: content_w[k])
        dominant_weight = content_w[dominant_dim]

        self._n_collapses += 1
        self._last_collapse_at = self._n_observed

        event = CollapseEvent(
            observation_index=self._n_observed,
            content_gini=g,
            entropy=weight_entropy(content_w),
            dominant_dim=dominant_dim,
            dominant_weight=dominant_weight,
            resonance_before=old_res,
            resonance_after=new_res,
        )
        self._history.append(event)

        logger.warning(
            "PRISM 5D collapse: content_gini=%.3f dominant=%s@%.3f "
            "resonance %.3f->%.3f",
            g, dominant_dim, dominant_weight, old_res, new_res,
        )

        return new_res, event

    def stats(self) -> dict:
        return {
            "n_observed": self._n_observed,
            "n_collapses": self._n_collapses,
            "collapse_rate": self._n_collapses / max(self._n_observed, 1),
            "homeostasis": self._homeostasis.stats(),
            "events": [
                {
                    "at": e.observation_index,
                    "gini": round(e.content_gini, 4),
                    "dominant": e.dominant_dim,
                    "resonance": f"{e.resonance_before:.3f}->{e.resonance_after:.3f}",
                }
                for e in self._history[-5:]
            ],
        }
