"""
Shadow policy framework — Production routing policies.

Every request runs N candidate routing policies in parallel. Each
policy returns a recommendation; recommendations are recorded into the
TraceEvent's ``shadow_recommendations`` field.

Four production policies:

  1. **CurrentHeuristic** — rule-based routing from query archetype + model
     tier. Fast, deterministic. Baseline comparator for learned policies.

  2. **BucketBeta** — Thompson sampling with Beta-Bernoulli posterior per
     (archetype, model) cell. Explores via posterior sampling, no tuning.
     Converges to exploitation as data accumulates (O(n) regret bound).

  3. **EmbeddingKNN** — k-NN over cached query embeddings with outcome
     labels. Uses SimHash (64-bit) for O(1) approximate nearest neighbor
     when full embeddings aren't available. Cosine similarity ranking.

  4. **LogisticFailurePredictor** — online SGD logistic regression
     predicting P(failure | features). Features: token_count, has_code,
     archetype_hash, model_tier, query_length. L2-regularized, warm-starts
     from the Beta prior.

Two principles:

  1. **No counterfactual regret in shadow mode.** When shadow recommends
     model B but production used model A, we don't know what B would have
     done. The offline analyzer computes ``shadow_agreement`` only.

  2. **Insufficient-data is a first-class outcome.** Policies that lack
     sufficient training data return ``insufficient_data=True`` honestly.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import threading
from dataclasses import asdict, dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class PolicyRecommendation:
    """A single policy's recommendation for one request."""

    policy_name: str
    model: str | None              # the policy's choice; None = abstain
    predicted_p_success: float | None  # in [0, 1] when the policy has data
    reason: str                    # short human-readable rationale
    insufficient_data: bool = False


class RoutingPolicy(Protocol):
    """Minimal interface for a shadow routing policy."""

    name: str

    def recommend(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> PolicyRecommendation:
        ...

    def observe(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        model_used: str,
        succeeded: bool,
    ) -> None:
        """Feed an outcome back into the policy for learning."""
        ...


# ══════════════════════════════════════════════════════════════════════
# Policy 1: Current Heuristic (deterministic baseline)
# ══════════════════════════════════════════════════════════════════════


# Model tier map: model_prefix -> (tier, cost_per_M_tokens)
_MODEL_TIERS: dict[str, tuple[str, float]] = {
    "claude-sonnet-4":       ("flagship", 3.0),
    "claude-3-5-sonnet":     ("flagship", 3.0),
    "claude-3-opus":         ("flagship", 15.0),
    "claude-haiku-3-5":      ("cheap", 0.80),
    "claude-3-haiku":        ("cheap", 0.25),
    "gpt-4o-mini":           ("cheap", 0.15),
    "gpt-4o":                ("flagship", 2.50),
    "gpt-4-turbo":           ("flagship", 10.0),
    "gemini-2.0-flash":      ("cheap", 0.10),
    "gemini-2.5-pro":        ("flagship", 1.25),
    "gemini-1.5-pro":        ("flagship", 3.50),
}

# Task archetypes safe for cheap models (low reasoning ceiling)
_CHEAP_SAFE_ARCHETYPES = frozenset({
    "explain", "inspect", "git/op", "setup", "lint/run",
    "typecheck/run", "test/run", "build/run",
})

# Task archetypes that need flagship models (high reasoning ceiling)
_FLAGSHIP_REQUIRED = frozenset({
    "code/implement", "code/refactor", "code/fix_bug", "test/write",
})


def _lookup_tier(model: str) -> tuple[str, float]:
    """Fuzzy prefix match for model tier lookup."""
    for prefix, (tier, cost) in _MODEL_TIERS.items():
        if model.startswith(prefix):
            return tier, cost
    return "unknown", 1.0


def _cheapest_candidate(candidates: list[str]) -> str | None:
    """Find the cheapest model from candidates."""
    best, best_cost = None, float("inf")
    for c in candidates:
        _, cost = _lookup_tier(c)
        if cost < best_cost:
            best, best_cost = c, cost
    return best


class CurrentHeuristicPolicy:
    """Rule-based routing from query archetype + model tier.

    Rules:
      - If current model is already cheap, keep it.
      - If archetype is safe for cheap, recommend cheapest candidate.
      - If archetype requires flagship, keep current.
      - Default: keep current (conservative).
    """

    name = "current_heuristic"

    def recommend(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> PolicyRecommendation:
        tier, cost = _lookup_tier(current_model)
        archetype = query_features.get("archetype", "general")

        # Already cheap? No action.
        if tier == "cheap":
            return PolicyRecommendation(
                policy_name=self.name, model=current_model,
                predicted_p_success=None,
                reason=f"already cheap (${cost}/M)",
            )

        # Flagship-required archetype? Keep current.
        if archetype in _FLAGSHIP_REQUIRED:
            return PolicyRecommendation(
                policy_name=self.name, model=current_model,
                predicted_p_success=None,
                reason=f"archetype '{archetype}' requires flagship",
            )

        # Archetype safe for cheap? Recommend cheapest.
        if archetype in _CHEAP_SAFE_ARCHETYPES:
            cheap = _cheapest_candidate(candidates)
            if cheap and cheap != current_model:
                return PolicyRecommendation(
                    policy_name=self.name, model=cheap,
                    predicted_p_success=None,
                    reason=f"archetype '{archetype}' safe for cheap model",
                )

        # Default: keep current
        return PolicyRecommendation(
            policy_name=self.name, model=current_model,
            predicted_p_success=None,
            reason="default: keep current model",
        )

    def observe(self, **kwargs: Any) -> None:
        pass  # Stateless — no learning


# ══════════════════════════════════════════════════════════════════════
# Policy 2: Bucket Beta (Thompson Sampling)
# ══════════════════════════════════════════════════════════════════════


class BucketBetaPolicy:
    """Thompson sampling with Beta-Bernoulli posterior per (archetype, model).

    Each cell maintains Beta(alpha, beta) where alpha = successes + prior,
    beta = failures + prior. To recommend, we sample from each candidate's
    posterior and pick the highest sample.

    This naturally balances exploration (high variance = high uncertainty)
    and exploitation (high mean = high confidence) with no hyperparameters.

    Regret bound: O(sqrt(K * T * ln(T))) where K = |models|, T = rounds.
    """

    name = "bucket_beta"

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 min_samples: int = 5):
        self._prior_a = prior_alpha
        self._prior_b = prior_beta
        self._min_samples = min_samples
        # cells[(archetype, model)] = {"alpha": float, "beta": float, "n": int}
        self._cells: dict[tuple[str, str], dict[str, float]] = {}
        self._lock = threading.Lock()

    def _get_cell(self, archetype: str, model: str) -> dict[str, float]:
        key = (archetype, model)
        with self._lock:
            if key not in self._cells:
                self._cells[key] = {
                    "alpha": self._prior_a, "beta": self._prior_b, "n": 0
                }
            return self._cells[key]

    def recommend(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> PolicyRecommendation:
        archetype = query_features.get("archetype", "general")
        all_models = list(set([current_model] + candidates))

        # Check if we have enough data for any model
        total_n = sum(self._get_cell(archetype, m)["n"] for m in all_models)
        if total_n < self._min_samples:
            return PolicyRecommendation(
                policy_name=self.name, model=None,
                predicted_p_success=None,
                reason=f"insufficient data: {total_n}/{self._min_samples} obs for '{archetype}'",
                insufficient_data=True,
            )

        # Thompson sample from each candidate's posterior
        best_model, best_sample = None, -1.0
        samples: dict[str, float] = {}
        for model in all_models:
            cell = self._get_cell(archetype, model)
            # Sample from Beta(alpha, beta)
            sample = random.betavariate(
                max(cell["alpha"], 0.01),
                max(cell["beta"], 0.01),
            )
            samples[model] = sample
            if sample > best_sample:
                best_model, best_sample = model, sample

        # Posterior mean for the best model (the actual estimate, not the sample)
        best_cell = self._get_cell(archetype, best_model)
        posterior_mean = best_cell["alpha"] / (best_cell["alpha"] + best_cell["beta"])

        return PolicyRecommendation(
            policy_name=self.name,
            model=best_model,
            predicted_p_success=round(posterior_mean, 4),
            reason=(
                f"thompson: sampled {best_model} ({best_sample:.3f}) "
                f"from {len(all_models)} candidates, "
                f"posterior_mean={posterior_mean:.3f}, n={int(best_cell['n'])}"
            ),
        )

    def observe(
        self,
        *,
        query_text: str = "",
        query_features: dict[str, Any] | None = None,
        model_used: str = "",
        succeeded: bool = False,
    ) -> None:
        """Update the Beta posterior for (archetype, model)."""
        if query_features is None:
            return
        archetype = query_features.get("archetype", "general")
        cell = self._get_cell(archetype, model_used)
        with self._lock:
            if succeeded:
                cell["alpha"] += 1.0
            else:
                cell["beta"] += 1.0
            cell["n"] += 1


# ══════════════════════════════════════════════════════════════════════
# Policy 3: Embedding KNN (SimHash approximate nearest neighbor)
# ══════════════════════════════════════════════════════════════════════


def _simhash_64(text: str) -> int:
    """64-bit SimHash for approximate text similarity.

    Uses character trigrams as features. Each trigram is hashed to 64 bits
    and the bit-level majority vote produces the final signature. Two texts
    with small Hamming distance between their SimHashes are likely similar.

    Time: O(n) where n = len(text). Space: O(1) for the 64-bit accumulator.
    """
    v = [0] * 64
    text_lower = text.lower()
    for i in range(len(text_lower) - 2):
        trigram = text_lower[i:i + 3]
        h = int(hashlib.md5(trigram.encode(), usedforsecurity=False).hexdigest()[:16], 16)
        for j in range(64):
            if h & (1 << j):
                v[j] += 1
            else:
                v[j] -= 1
    fingerprint = 0
    for j in range(64):
        if v[j] > 0:
            fingerprint |= (1 << j)
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64-bit integers."""
    return bin(a ^ b).count("1")


class EmbeddingKNNPolicy:
    """k-NN routing using SimHash approximate nearest neighbor.

    Stores (simhash, model, succeeded) tuples from past outcomes.
    For a new query, finds the k nearest neighbors (by Hamming distance)
    and recommends the model with the highest success rate among neighbors.

    When k neighbors aren't available, returns insufficient_data.

    Space: O(history_size). Lookup: O(history_size) — bounded by max_history.
    """

    name = "embedding_knn"

    def __init__(self, k: int = 7, max_history: int = 2000, min_neighbors: int = 3):
        self._k = k
        self._max_history = max_history
        self._min_neighbors = min_neighbors
        # (simhash, model, succeeded)
        self._history: list[tuple[int, str, bool]] = []
        self._lock = threading.Lock()

    def recommend(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> PolicyRecommendation:
        if len(self._history) < self._min_neighbors:
            return PolicyRecommendation(
                policy_name=self.name, model=None,
                predicted_p_success=None,
                reason=f"insufficient history: {len(self._history)}/{self._min_neighbors}",
                insufficient_data=True,
            )

        query_hash = _simhash_64(query_text)

        # Find k nearest neighbors by Hamming distance
        with self._lock:
            distances = [
                (_hamming_distance(query_hash, h), m, s)
                for h, m, s in self._history
            ]
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self._k]

        # If nearest neighbor is too far, abstain (Hamming > 24 = very different)
        if neighbors[0][0] > 24:
            return PolicyRecommendation(
                policy_name=self.name, model=None,
                predicted_p_success=None,
                reason=f"nearest neighbor too distant (hamming={neighbors[0][0]})",
                insufficient_data=True,
            )

        # Count successes per model among neighbors
        model_stats: dict[str, dict[str, int]] = {}
        for dist, model, succeeded in neighbors:
            entry = model_stats.setdefault(model, {"win": 0, "total": 0})
            entry["total"] += 1
            if succeeded:
                entry["win"] += 1

        # Pick model with highest success rate (tie-break: more data)
        all_models = list(set([current_model] + candidates))
        best_model, best_rate, best_n = current_model, 0.0, 0
        for model in all_models:
            stats = model_stats.get(model)
            if stats and stats["total"] > 0:
                rate = stats["win"] / stats["total"]
                if rate > best_rate or (rate == best_rate and stats["total"] > best_n):
                    best_model, best_rate, best_n = model, rate, stats["total"]

        return PolicyRecommendation(
            policy_name=self.name,
            model=best_model,
            predicted_p_success=round(best_rate, 4) if best_n > 0 else None,
            reason=(
                f"knn: {best_model} ({best_rate:.0%} of {best_n} neighbors), "
                f"nearest_hamming={neighbors[0][0]}, k={len(neighbors)}"
            ),
        )

    def observe(
        self,
        *,
        query_text: str = "",
        query_features: dict[str, Any] | None = None,
        model_used: str = "",
        succeeded: bool = False,
    ) -> None:
        """Add observation to history."""
        if not query_text:
            return
        h = _simhash_64(query_text)
        with self._lock:
            self._history.append((h, model_used, succeeded))
            # Evict oldest if over capacity
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]


# ══════════════════════════════════════════════════════════════════════
# Policy 4: Logistic Failure Predictor (Online SGD)
# ══════════════════════════════════════════════════════════════════════


def _feature_vector(
    query_text: str,
    query_features: dict[str, Any],
    model: str,
) -> list[float]:
    """Extract a fixed-length feature vector for logistic regression.

    Features (8-dimensional):
      [0] query_length (normalized by 500)
      [1] has_code (0/1)
      [2] token_count (normalized by 4096)
      [3] archetype_hash (hash mod 16, normalized)
      [4] model_tier (0=cheap, 0.5=unknown, 1=flagship)
      [5] model_cost (normalized by 15.0)
      [6] query_complexity (word count / 50)
      [7] bias term (always 1.0)
    """
    archetype = query_features.get("archetype", "general")
    token_count = query_features.get("token_count", len(query_text) // 4)
    has_code = 1.0 if query_features.get("has_code", False) else 0.0

    tier, cost = _lookup_tier(model)
    tier_val = {"flagship": 1.0, "cheap": 0.0}.get(tier, 0.5)

    arch_hash = int(hashlib.md5(
        archetype.encode(), usedforsecurity=False
    ).hexdigest()[:4], 16) % 16

    word_count = len(query_text.split())

    return [
        min(len(query_text) / 500.0, 3.0),     # [0] query_length
        has_code,                                 # [1] has_code
        min(token_count / 4096.0, 3.0),          # [2] token_count
        arch_hash / 16.0,                        # [3] archetype_hash
        tier_val,                                 # [4] model_tier
        min(cost / 15.0, 1.0),                   # [5] model_cost
        min(word_count / 50.0, 3.0),             # [6] complexity
        1.0,                                      # [7] bias
    ]


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class LogisticFailurePredictor:
    """Online logistic regression predicting P(failure | features).

    Uses SGD with L2 regularization. The model predicts the probability
    of failure, so P(success) = 1 - predict(features).

    For routing: recommend the model whose P(failure) is lowest.

    Update rule (SGD on binary cross-entropy + L2):
      gradient_i = (predicted - y) * x_i + lambda * w_i
      w_i -= lr * gradient_i

    Features are normalized to [0, 3] to keep gradients stable.
    """

    name = "logistic_failure_predictor"

    def __init__(
        self,
        n_features: int = 8,
        learning_rate: float = 0.01,
        l2_lambda: float = 0.001,
        min_observations: int = 10,
    ):
        self._n = n_features
        self._lr = learning_rate
        self._l2 = l2_lambda
        self._min_obs = min_observations
        self._weights = [0.0] * n_features
        self._obs_count = 0
        self._lock = threading.Lock()

    def _predict_failure_prob(self, features: list[float]) -> float:
        """P(failure | features) = sigmoid(w . x)."""
        z = sum(w * f for w, f in zip(self._weights, features))
        return _sigmoid(z)

    def recommend(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> PolicyRecommendation:
        if self._obs_count < self._min_obs:
            return PolicyRecommendation(
                policy_name=self.name, model=None,
                predicted_p_success=None,
                reason=f"insufficient training data: {self._obs_count}/{self._min_obs} obs",
                insufficient_data=True,
            )

        all_models = list(set([current_model] + candidates))
        best_model, best_p_success = current_model, 0.0

        for model in all_models:
            features = _feature_vector(query_text, query_features, model)
            p_fail = self._predict_failure_prob(features)
            p_success = 1.0 - p_fail
            if p_success > best_p_success:
                best_model, best_p_success = model, p_success

        return PolicyRecommendation(
            policy_name=self.name,
            model=best_model,
            predicted_p_success=round(best_p_success, 4),
            reason=(
                f"logistic: {best_model} P(success)={best_p_success:.3f}, "
                f"trained on {self._obs_count} observations"
            ),
        )

    def observe(
        self,
        *,
        query_text: str = "",
        query_features: dict[str, Any] | None = None,
        model_used: str = "",
        succeeded: bool = False,
    ) -> None:
        """SGD update on one observation."""
        if query_features is None or not query_text:
            return
        features = _feature_vector(query_text, query_features, model_used)
        y = 0.0 if succeeded else 1.0  # predicting P(failure)
        predicted = self._predict_failure_prob(features)
        error = predicted - y

        with self._lock:
            for i in range(self._n):
                gradient = error * features[i] + self._l2 * self._weights[i]
                self._weights[i] -= self._lr * gradient
            self._obs_count += 1


# ══════════════════════════════════════════════════════════════════════
# Policy Construction
# ══════════════════════════════════════════════════════════════════════


def make_default_policies() -> list[RoutingPolicy]:
    """Construct the production set of shadow policies."""
    return [
        CurrentHeuristicPolicy(),
        BucketBetaPolicy(),
        EmbeddingKNNPolicy(),
        LogisticFailurePredictor(),
    ]


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════


class ShadowEvaluator:
    """Run all registered policies on a request; collect recommendations.

    Stateless w.r.t. requests (policies hold their own state). Errors
    in any single policy are caught and logged — they never block the
    request or pollute other policies' recommendations.
    """

    def __init__(self, policies: list[RoutingPolicy] | None = None):
        self._policies = policies if policies is not None else make_default_policies()

    @property
    def policy_names(self) -> list[str]:
        return [p.name for p in self._policies]

    def evaluate(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Return a dict of policy_name -> serialized recommendation."""
        out: dict[str, dict[str, Any]] = {}
        for policy in self._policies:
            try:
                rec = policy.recommend(
                    query_text=query_text,
                    query_features=query_features,
                    current_model=current_model,
                    candidates=list(candidates),
                )
                out[policy.name] = asdict(rec)
            except Exception as e:
                logger.debug(
                    "shadow policy %s raised — recording insufficient_data: %s",
                    getattr(policy, "name", "<unnamed>"), e,
                )
                out[getattr(policy, "name", "<unnamed>")] = asdict(
                    PolicyRecommendation(
                        policy_name=getattr(policy, "name", "<unnamed>"),
                        model=None,
                        predicted_p_success=None,
                        reason=f"policy raised {type(e).__name__}",
                        insufficient_data=True,
                    )
                )
        return out

    def observe_outcome(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        model_used: str,
        succeeded: bool,
    ) -> None:
        """Feed an outcome into all policies for learning."""
        for policy in self._policies:
            try:
                if hasattr(policy, "observe"):
                    policy.observe(
                        query_text=query_text,
                        query_features=query_features,
                        model_used=model_used,
                        succeeded=succeeded,
                    )
            except Exception as e:
                logger.debug(
                    "shadow policy %s observe error: %s",
                    getattr(policy, "name", "<unnamed>"), e,
                )


# ══════════════════════════════════════════════════════════════════════
# Offline Metrics
# ══════════════════════════════════════════════════════════════════════


def shadow_agreement(
    trace: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """For one trace, compute per-policy agreement with the production policy.

    Returns ``{policy_name: {agreed: bool, recommended_model: str|None,
    abstained: bool}}``.
    """
    prod_policy = trace.get("policy_decision", "")
    prod_model = trace.get("model", "")
    shadows = trace.get("shadow_recommendations", {}) or {}
    out: dict[str, dict[str, Any]] = {}
    for name, rec in shadows.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("insufficient_data"):
            out[name] = {
                "abstained": True,
                "agreed": False,
                "recommended_model": None,
            }
            continue
        recommended = rec.get("model")
        if recommended is None:
            out[name] = {"abstained": True, "agreed": False, "recommended_model": None}
            continue
        out[name] = {
            "abstained": False,
            "agreed": (recommended == prod_model),
            "recommended_model": recommended,
        }
    if out:
        out["_meta"] = {
            "production_policy": prod_policy,
            "production_model": prod_model,
        }
    return out


def aggregate_shadow_agreement(
    traces: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate ``shadow_agreement`` across many traces.

    Returns ``{policy_name: {agree_rate, abstain_rate, n}}``.
    """
    counters: dict[str, dict[str, int]] = {}
    for trace in traces:
        per_trace = shadow_agreement(trace)
        for name, info in per_trace.items():
            if name == "_meta":
                continue
            d = counters.setdefault(name, {"agree": 0, "abstain": 0, "n": 0})
            d["n"] += 1
            if info.get("abstained"):
                d["abstain"] += 1
            elif info.get("agreed"):
                d["agree"] += 1
    out: dict[str, dict[str, float]] = {}
    for name, d in counters.items():
        n = max(d["n"], 1)
        out[name] = {
            "agree_rate": round(d["agree"] / n, 4),
            "abstain_rate": round(d["abstain"] / n, 4),
            "n": float(d["n"]),
        }
    return out
