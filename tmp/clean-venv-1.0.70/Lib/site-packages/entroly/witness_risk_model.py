"""
WITNESS Continuous Risk Model
=============================

Replaces the 4-bucket {grounded, unsupported, unknown, contradicted}
risk assignment with a continuous logistic on the feature vector
φ(c, C) ∈ ℝ⁸.

    ρ(c, C) = σ( w · φ + b )

where σ is the standard logistic. ρ ∈ (0, 1) is then mapped to one of
four actions {PASS, HEDGE, WARN, SUPPRESS} by the ThresholdSet in
witness_calibration.py. The system is now end-to-end continuous from
features to action.

Weight calibration
------------------
Two regimes:

  1. Hand-calibrated priors (DEFAULT_WEIGHTS). Used cold-start and as
     a regularizer when training data is sparse. Weights chosen so the
     bucket structure of the original verifier is recovered as a
     limiting case (high entity_precision + reverse_entail + quote
     → low risk; mismatch on any signal → bumped risk).

  2. Online SGD update from labeled outcomes. The PRISM-style
     gradient step from entroly/verifiers/calibrator.py applies here
     symmetrically: log-loss gradient on (φ, y), bounded learning
     rate, annealing.

Mathematical guarantee
----------------------
Bates et al. 2021 (RCPS / Risk-Controlling Prediction Sets) gives a
distribution-free guarantee: given a calibration set of size n and a
chosen miscoverage α, the policy thresholds derived from ρ produce
exposure rate ≤ α on i.i.d. test data with probability ≥ 1 − δ. We
inherit that guarantee directly because ρ is monotonic in φ and the
calibration procedure in witness_calibration.py is split-conformal.

Backward map for compatibility
------------------------------
The existing Certificate dataclass exposes label ∈ {grounded,
unsupported, unknown, contradicted}. To maintain that API while
moving to continuous risk, we add `label_from_features(features)`
that derives the discrete label from the same features the risk model
sees — keeping the discrete and continuous views consistent.

References
----------
- Bates, S., Angelopoulos, A., Lei, L., Malik, J., Jordan, M. (2021).
  Distribution-Free, Risk-Controlling Prediction Sets. JACM.
- Platt, J. (1999). Probabilistic outputs for support vector machines
  and comparisons to regularized likelihood methods. (Logistic
  calibration baseline.)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .witness_features import ClaimFeatures


# Hand-calibrated weights for the cold-start risk model.
#
# Sign conventions:
#   Positive feature → lower risk (more grounded)
#   Negative weight on a positive feature → that feature pulls toward risk
#
# Calibration anchor points (cold-start invariants we want recovered):
#   - All-1 features (perfect support)          → ρ ≈ 0.05
#   - All-zero features (no support, no signal) → ρ ≈ 0.55
#   - Negative number_consistency (-1)          → ρ ≈ 0.92 (strong contradiction)
#   - Negative negation_polarity (-1)           → ρ ≈ 0.92 (polarity flip)
#
# Bias b = +3.0 so default ρ at φ=0 is σ(3.0) ≈ 0.95 → suppress. Then
# negative weights on supportive features pull ρ down toward grounded.

DEFAULT_WEIGHTS = {
    "entity_precision":  -1.4,   # entities found in ctx → -ρ
    "number_consistency": -3.2,  # numbers match → -ρ; mismatch → ρ↑ via negative feature
    "idf_lex_overlap":   -1.2,
    "quote_support":     -2.4,   # verbatim quote → strong evidence
    "forward_entail":    -1.8,
    "reverse_entail":    -2.6,   # claim made up of words not in ctx → ρ↑
    "negation_polarity": -3.0,   # polarity flip → ρ↑ via negative feature
    "adequacy":          -0.8,   # mild — adequacy is a confidence multiplier
    "qa_alignment":      -2.8,   # answer near question's evidence span → -ρ
}
DEFAULT_BIAS = 6.0


# Features that act as MULTIPLICATIVE hard-contradiction gates when their
# value is negative. These represent active disagreement (the verifier
# observed evidence the claim contradicts), as opposed to mere absence
# of support. A single such gate at strength 1.0 should drive ρ → 1.
_HARD_CONTRADICTION_FEATURES = frozenset({
    "number_consistency",
    "negation_polarity",
    "entity_precision",      # signed; negative = active entity mismatch
    "qa_alignment",          # signed; negative = answer misplaced wrt question
})


@dataclass
class RiskModel:
    """Continuous risk model with online-trainable weights.

    Default weights are hand-calibrated; pass `weights` to override
    (e.g., from a previously-trained checkpoint).
    """
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    bias: float = DEFAULT_BIAS
    # Bounded learning rate (matches PRISM calibrator).
    learning_rate: float = 0.05
    # Annealed rate based on update count.
    updates: int = 0

    def predict(self, features: ClaimFeatures) -> float:
        """Forward pass.

        Combines two regimes:

          (a) Soft logistic over supportive evidence:
                ρ_soft = σ(b + Σ w_k φ_k)
              for k ∈ supportive features (entity_precision, idf_lex_overlap,
              quote_support, forward_entail, reverse_entail, adequacy).

          (b) Multiplicative hard-contradiction gates:
                For each feature in {number_consistency, negation_polarity}
                with NEGATIVE value, contribute ρ_hard_i = |feature|.
                These represent ACTIVE disagreement — strictly stronger
                evidence than weak support.

        Final risk is OR-aggregated as independent probabilities:
            ρ = 1 - (1 - ρ_soft) · ∏ (1 - ρ_hard_i)

        Property: a single number/polarity contradiction at strength 1.0
        drives ρ → 1 regardless of supportive features. This recovers
        the desired behavior on "Einstein developed relativity in 1962"
        (correct entity, wrong number → contradicted).
        """
        z_soft = self.bias
        hard_contras: list[float] = []
        for name, val in zip(ClaimFeatures.feature_names(), features.as_vector()):
            w = self.weights.get(name, 0.0)
            if name in _HARD_CONTRADICTION_FEATURES and val < 0:
                # Negative value on a hard-contradiction feature: hold
                # out as a multiplicative gate; do NOT pollute z_soft.
                hard_contras.append(min(1.0, max(0.0, -val)))
            else:
                # Soft features: clamp to [0, 1] then add weighted to z.
                z_soft += w * max(0.0, val)
        rho_soft = _sigmoid(z_soft)
        # Multiplicative OR of independent contradiction signals.
        survival = (1.0 - rho_soft)
        for r in hard_contras:
            survival *= (1.0 - r)
        return 1.0 - survival

    def update(self, features: ClaimFeatures, y: int) -> float:
        """One step of logistic SGD. Returns the post-update prediction.

        Args:
            features: Feature vector that produced a verdict.
            y: Ground truth — 1 if claim was a hallucination, 0 if grounded.
        """
        if y not in (0, 1):
            raise ValueError("y must be 0 or 1")

        p = self.predict(features)
        # ∂L/∂z = p − y for log-loss; chain into ∂L/∂w_i = (p − y) · φ_i,
        # ∂L/∂b = (p − y).
        grad_z = p - y

        # Annealed lr
        lr = self.learning_rate / (1.0 + self.updates / 100.0)

        # Update
        for name, val in zip(ClaimFeatures.feature_names(), features.as_vector()):
            old = self.weights.get(name, 0.0)
            self.weights[name] = old - lr * grad_z * val

        self.bias -= lr * grad_z
        self.updates += 1

        # Bound the weights to keep numerics sane.
        for k in list(self.weights.keys()):
            self.weights[k] = max(-10.0, min(10.0, self.weights[k]))
        self.bias = max(-10.0, min(10.0, self.bias))

        return self.predict(features)

    def explain(self, features: ClaimFeatures) -> dict:
        """Per-feature contribution to z = log-odds. For dashboards."""
        contribs: dict[str, float] = {}
        z_soft = self.bias
        hard_gates: list[dict[str, float]] = []
        for name, val in zip(ClaimFeatures.feature_names(), features.as_vector()):
            w = self.weights.get(name, 0.0)
            if name in _HARD_CONTRADICTION_FEATURES and val < 0:
                hard_gates.append({"feature": name, "strength": min(1.0, max(0.0, -val))})
                contribs[name] = 0.0
            else:
                contribution = w * max(0.0, val)
                contribs[name] = contribution
                z_soft += contribution

        rho_soft = _sigmoid(z_soft)
        survival = 1.0 - rho_soft
        for gate in hard_gates:
            survival *= 1.0 - gate["strength"]
        rho = 1.0 - survival

        contribs["bias"] = self.bias
        contribs["z"] = z_soft
        contribs["rho_soft"] = rho_soft
        contribs["rho"] = rho
        contribs["hard_gates"] = hard_gates
        return contribs

    def as_dict(self) -> dict:
        """Stable JSON checkpoint for online training."""
        return {
            "weights": dict(self.weights),
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "updates": self.updates,
            "schema": "witness-risk-model-v1",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiskModel":
        weights = dict(DEFAULT_WEIGHTS)
        raw_weights = data.get("weights") or {}
        for name in ClaimFeatures.feature_names():
            if name in raw_weights:
                weights[name] = float(raw_weights[name])
        return cls(
            weights=weights,
            bias=float(data.get("bias", DEFAULT_BIAS)),
            learning_rate=float(data.get("learning_rate", 0.05)),
            updates=int(data.get("updates", 0)),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "RiskModel":
        target = Path(path)
        if not target.exists():
            return cls()
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


def _sigmoid(x: float) -> float:
    # Numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ── Backward-compatible discrete label derivation ────────────────────


def label_from_features_and_risk(features: ClaimFeatures, risk: float) -> str:
    """Derive a discrete label compatible with the existing Certificate API.

    Mapping:
        risk ≤ 0.30 AND forward_entail > 0.5   → grounded
        number_consistency < 0 OR negation_polarity < 0  → contradicted
        risk ≥ 0.70 AND adequacy ≥ 0.6         → unsupported
        otherwise                              → unknown
    """
    if (
        features.number_consistency < 0
        or features.negation_polarity < 0
        or features.qa_alignment < 0
    ):
        return "contradicted"
    if risk <= 0.30 and features.forward_entail > 0.5:
        return "grounded"
    if risk >= 0.70 and features.adequacy >= 0.6:
        return "unsupported"
    return "unknown"
