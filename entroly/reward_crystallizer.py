"""
Reward-Driven Skill Crystallization
====================================

Closes the asymmetry in Entroly's evolution loop. Today:

  failure  → record_miss → EvolutionDaemon → SkillSynthesizer.synthesize_from_gap
  success  → OnlinePrism.observe         (only weights drift; pattern is lost)

This module crystallizes the second branch. When a query family consistently
beats the global reward baseline with statistical significance, we materialize
the winning strategy as a reusable Skill — capturing the trigger pattern, the
fragment-selection recipe, and the weight profile that worked.

Mathematical contract
---------------------

For each query cluster C with sliding window of n recent rewards r₁..rₙ,
crystallize iff (Hoeffding lower bound test):

    r̄_C  -  √( ln(1/δ) / (2n) )   >   μ_baseline + ε

  - r̄_C            sample mean of cluster rewards
  - μ_baseline     OnlinePrism reward EMA (already maintained globally)
  - δ = 0.05       confidence (5 % chance of false crystallization)
  - ε = 0.10       minimum effect size (don't crystallize trivial wins)
  - n ≥ N_min      enough samples (default 5)

This is UCB1-in-reverse: standard UCB picks an arm whose *upper* bound is
high. Crystallization picks a cluster whose *lower* bound has provably
separated from baseline. False positives cost a wasted skill; false
negatives cost only a delay (the cluster will re-trigger).

Cluster identity
----------------

A cluster is identified by *both* signals because either alone is wrong:

  1. Token-set Jaccard on query text   (overlap ≥ τ_query) — semantic similarity
  2. Jaccard on selected_fragment_ids  (overlap ≥ τ_frag)  — same task

Token Jaccard beats SimHash on short queries: a "fix auth bug" / "auth
bug fix please" pair shares 3/3 identity tokens but lands ≈25 bits apart
in 64-bit SimHash space (Broder, AltaVista 1997 — Jaccard is the right
invariance for short text similarity).

Two queries can share keywords ("authentication", "password") yet trigger
different fragment sets ("auth.login" vs "auth.password_reset"). Bucketing
them together would crystallize a skill that fits neither well. Requiring
fragment overlap rejects this case.

Lifecycle
---------

  observe(query, reward, weights, selected) →
    1. compute query SimHash
    2. find existing cluster (hamming ≤ T_hash AND Jaccard ≥ τ_jac)
       or create new cluster
    3. push (reward, weights, fragment_ids) into cluster's sliding window
    4. if ready_to_crystallize(cluster): emit CrystallizationEvent

  poll_events()  →  drain pending events
  ack(event_id)  →  mark event materialized (cluster goes into cooldown)

Thread-safe; cheap (O(K) per observe where K = number of active clusters,
expected small because we cap and evict by recency).
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

# ── Tunables ──────────────────────────────────────────────────────────

# Min samples in a cluster before any statistical test can fire.
# Below this, the Hoeffding bound is wider than the [0,1] reward range.
DEFAULT_MIN_SAMPLES = 5

# Sliding window size per cluster. Older observations are evicted (FIFO).
# Choice trades adaptivity (small) vs statistical power (large).
DEFAULT_WINDOW = 16

# Hoeffding confidence: P(false crystallization) ≤ δ.
DEFAULT_DELTA = 0.05

# Minimum effect size. Crystallize only if the cluster beats baseline by
# at least this much (after the LCB correction). Keeps the skill catalog
# from filling with marginal wins that don't justify their overhead.
DEFAULT_EPSILON = 0.10

# Token-set Jaccard threshold for query similarity. This is the
# membership test for cluster joining. Calibration:
#
#   - SimHash on short queries (3-7 trigrams) has high per-bit noise:
#     the same family of NL paraphrases ("fix auth bug" / "auth bug fix
#     please") routinely lands 20-30 bits apart in 64-bit space.
#     Practitioners (Broder MinHash, AltaVista 1997) use Jaccard on
#     token sets instead for short text — it ignores word order, which
#     is exactly the right invariance for query clustering.
#
#   - 0.25 ≈ "share at least 1/4 of distinct content tokens." Calibrated
#     against natural NL paraphrase: "make knapsack solver faster" and
#     "speed up knapsack optimize" share 1 of 6 tokens (≈17%) — that's
#     near the floor of what should still cluster. Lowering further
#     risks false merges; the fragment-Jaccard filter (≥0.5) absorbs
#     the precision burden in exchange for permissive query recall.
#
# The *cluster identity* still uses SimHash (a fast 64-bit centroid for
# O(1) lookup) — but membership is decided by the Jaccard pair test.
DEFAULT_QUERY_JACCARD = 0.25

# Jaccard threshold on selected_fragment_ids for cluster membership.
# τ=0.5 ≈ "more than half of the fragments match".
DEFAULT_JACCARD = 0.5

# Maximum active clusters before LRU eviction. Caps memory and per-observe
# cost at O(K).
DEFAULT_MAX_CLUSTERS = 256

# After a cluster crystallizes, this many calls before it can crystallize
# again (prevents duplicate skills from a single sustained burst).
DEFAULT_COOLDOWN_OBSERVATIONS = 32


# ── Data types ────────────────────────────────────────────────────────

@dataclass
class _Observation:
    reward: float
    weights: dict[str, float]
    fragment_ids: tuple[str, ...]
    query: str
    ts: float


@dataclass
class _Cluster:
    cluster_id: str
    # Token *bag* across recent member queries (used by Jaccard
    # similarity at membership-test time). Drift-tolerant because new
    # members add their tokens to the bag.
    centroid_tokens: set[str]
    centroid_fragments: set[str]
    queries: list[str] = field(default_factory=list)
    window: deque[_Observation] = field(default_factory=deque)
    last_crystallized_at_n: int = -1
    n_total: int = 0  # lifetime observations (for cooldown)


@dataclass
class CrystallizationEvent:
    """A statistically significant cluster, ready to be materialized as a skill.

    Fields are everything the SkillSynthesizer needs to construct a
    SkillSpec without touching the live engine.
    """
    event_id: str
    cluster_id: str
    sample_queries: list[str]            # representative trigger inputs
    common_terms: list[str]              # extracted high-frequency tokens
    weight_profile: dict[str, float]     # mean of weights at observation time
    fragment_recipe: list[str]           # most-frequently-selected fragment ids/sources
    n_samples: int
    mean_reward: float
    lcb_reward: float                    # Hoeffding lower bound
    baseline_reward: float
    effect_size: float                   # lcb - baseline
    created_at: float


# ── Hash + similarity helpers ─────────────────────────────────────────

def _simhash64(text: str) -> int:
    """64-bit SimHash, MD5 + trigrams.

    Identical algorithm to ``server._py_simhash`` so cluster centroids
    are comparable to query/fragment fingerprints the engine already
    computes elsewhere. MD5 (vs Python's built-in ``hash()``) gives
    well-distributed bit votes; trigrams give locality sensitivity
    (small token edits → small Hamming distance).

    Falls back to unigrams for queries shorter than 3 tokens.
    """
    if not text:
        return 0
    import hashlib as _hl
    tokens = [
        t for t in text.lower().split()
        if t.isalnum() or any(c.isalnum() for c in t)
    ]
    if not tokens:
        return 0
    if len(tokens) >= 3:
        features = [
            f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            for i in range(len(tokens) - 2)
        ]
    else:
        features = tokens
    bits = [0] * 64
    for feat in features:
        h = int(_hl.md5(feat.encode("utf-8", errors="replace")).hexdigest(), 16)
        for i in range(64):
            if (h >> i) & 1:
                bits[i] += 1
            else:
                bits[i] -= 1
    out = 0
    for i in range(64):
        if bits[i] > 0:
            out |= (1 << i)
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# Stop-words removed before query-tokenization so cluster Jaccard
# isn't dominated by glue words ("the", "a", "in") that carry no
# task-identity signal.
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "for", "on", "and", "or",
    "is", "are", "was", "were", "be", "do", "does", "this", "that",
    "it", "its", "with", "from", "by", "as", "at", "i", "we", "you",
    "my", "our", "your", "they", "them", "he", "she", "his", "her",
    "but", "not", "no", "so", "if", "then", "than", "when", "while",
    "please", "needs", "need", "needed", "have", "has", "had", "can",
})


def _query_tokens(text: str) -> set[str]:
    """Tokenize for clustering: lowercased, alphanumeric-stripped, stop-words removed.

    Returns a *set* (order- and duplicate-agnostic) — that's the right
    invariance for "is this the same query family." Stop-word removal
    keeps the Jaccard signal concentrated on identity-carrying terms.
    """
    if not text:
        return set()
    out: set[str] = set()
    for raw in text.lower().split():
        cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "_-")
        if len(cleaned) <= 2 or cleaned in _STOPWORDS:
            continue
        out.add(cleaned)
    return out


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return len(sa & sb) / union


def _hoeffding_lcb(mean: float, n: int, delta: float) -> float:
    """One-sided Hoeffding lower confidence bound for a mean of [0,1] vars.

    P( true_mean ≥ sample_mean - sqrt(ln(1/δ) / (2n)) ) ≥ 1 - δ
    """
    if n <= 0:
        return 0.0
    return mean - math.sqrt(math.log(1.0 / max(delta, 1e-9)) / (2.0 * n))


# ── The crystallizer ──────────────────────────────────────────────────

class RewardCrystallizer:
    """Detects sustained high-reward query clusters and emits crystallization events.

    Stateless on the reward semantic (just a number in [0,1]) — relies on
    the caller (OnlinePrism / optimize_context) to compute reward and pass
    its baseline. This keeps the crystallizer composable with future
    reward functions without re-implementing baselines.
    """

    def __init__(
        self,
        *,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        window: int = DEFAULT_WINDOW,
        delta: float = DEFAULT_DELTA,
        epsilon: float = DEFAULT_EPSILON,
        query_jaccard: float = DEFAULT_QUERY_JACCARD,
        fragment_jaccard: float = DEFAULT_JACCARD,
        max_clusters: int = DEFAULT_MAX_CLUSTERS,
        cooldown: int = DEFAULT_COOLDOWN_OBSERVATIONS,
    ):
        self._min_samples = min_samples
        self._window = window
        self._delta = delta
        self._epsilon = epsilon
        self._query_jaccard = query_jaccard
        self._fragment_jaccard = fragment_jaccard
        self._max_clusters = max_clusters
        self._cooldown = cooldown

        self._clusters: dict[str, _Cluster] = {}
        self._pending_events: list[CrystallizationEvent] = []
        self._lock = threading.Lock()
        # Lifetime counter that survives cluster eviction. Summing
        # ``cluster.n_total`` over active clusters undercounts whenever
        # the LRU evicts a cluster mid-flight; this counter doesn't.
        self._lifetime_observations: int = 0
        # Sums for *external* baseline computation. Per-cluster external
        # mean = (global_total_reward − cluster_total_reward) /
        # (global_total_n − cluster_total_n). Survives LRU eviction by
        # being a flat counter rather than a per-cluster sum.
        self._lifetime_reward_sum: float = 0.0
        # Per-cluster lifetime totals (NOT just window — survives window
        # eviction so we don't accidentally credit a cluster's old
        # observations to the external baseline).
        self._cluster_lifetime_reward: dict[str, float] = {}
        self._cluster_lifetime_n: dict[str, int] = {}

    # ── Observation ───────────────────────────────────────────────

    def observe(
        self,
        *,
        query: str,
        reward: float,
        weights: dict[str, float],
        selected_fragment_ids: Iterable[str],
        baseline_reward: float,
    ) -> CrystallizationEvent | None:
        """Record one optimize_context outcome.

        Returns a CrystallizationEvent if this observation triggered
        crystallization for some cluster, else None. The event is also
        appended to the internal pending queue (drainable via poll_events).
        """
        if not query:
            return None
        reward = max(0.0, min(1.0, float(reward)))
        baseline_reward = max(0.0, min(1.0, float(baseline_reward)))
        frag_tuple = tuple(sorted(set(selected_fragment_ids)))
        qtokens = _query_tokens(query)
        obs = _Observation(
            reward=reward,
            weights=dict(weights),
            fragment_ids=frag_tuple,
            query=query,
            ts=time.time(),
        )

        with self._lock:
            self._lifetime_observations += 1
            self._lifetime_reward_sum += reward
            cluster = self._find_or_create_cluster(qtokens, frag_tuple)
            cluster.queries.append(query)
            cluster.window.append(obs)
            while len(cluster.window) > self._window:
                cluster.window.popleft()
            cluster.n_total += 1

            # Per-cluster lifetime totals for external-baseline math.
            cid = cluster.cluster_id
            self._cluster_lifetime_reward[cid] = (
                self._cluster_lifetime_reward.get(cid, 0.0) + reward
            )
            self._cluster_lifetime_n[cid] = (
                self._cluster_lifetime_n.get(cid, 0) + 1
            )

            event = self._maybe_crystallize(cluster, baseline_reward)
            if event is not None:
                self._pending_events.append(event)
            return event

    # ── Event drainage ────────────────────────────────────────────

    def poll_events(self, *, drain: bool = True) -> list[CrystallizationEvent]:
        """Return pending crystallization events.

        If ``drain=True`` (default), the internal queue is cleared.
        """
        with self._lock:
            events = list(self._pending_events)
            if drain:
                self._pending_events.clear()
            return events

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "active_clusters": len(self._clusters),
                "pending_events": len(self._pending_events),
                # Lifetime counter (never undercounts; survives LRU eviction).
                "total_observations": self._lifetime_observations,
                # Sum over currently-resident clusters (transient view; can
                # be < total_observations when eviction has occurred).
                "resident_observations": sum(c.n_total for c in self._clusters.values()),
            }

    # ── Internals ─────────────────────────────────────────────────

    def _find_or_create_cluster(
        self, qtokens: set[str], fragments: tuple[str, ...]
    ) -> _Cluster:
        # Pick the cluster with the highest *combined* similarity that
        # also crosses BOTH thresholds (token Jaccard + fragment Jaccard).
        # Either threshold alone is too weak — natural-language paraphrase
        # beats SimHash and disjoint code areas can share keywords.
        best_id: str | None = None
        best_score = -1.0
        for cid, cl in self._clusters.items():
            tj = _jaccard(cl.centroid_tokens, qtokens)
            if tj < self._query_jaccard:
                continue
            fj = _jaccard(cl.centroid_fragments, fragments)
            if fj < self._fragment_jaccard:
                continue
            score = tj + fj
            if score > best_score:
                best_score = score
                best_id = cid
        if best_id is not None:
            cl = self._clusters[best_id]
            # Centroid drift: union new tokens/fragments so the centroid
            # tracks cluster variance. Cap the token bag to bound drift
            # — without a cap, every accepted variant inflates the bag,
            # letting the cluster eventually accept arbitrary queries
            # that share 1/N tokens. 32 is empirically enough to cover
            # paraphrase variance for a typical query family while
            # keeping the cluster's identity recognizable.
            cl.centroid_tokens |= qtokens
            if len(cl.centroid_tokens) > 32:
                # Drop a random token (cheap, unbiased) — the strict
                # alternative (LRU on token-touch ts) costs more memory
                # for marginal precision improvement here.
                cl.centroid_tokens.pop()
            cl.centroid_fragments |= set(fragments)
            return cl

        # New cluster — evict oldest if at cap.
        if len(self._clusters) >= self._max_clusters:
            oldest = min(
                self._clusters.values(),
                key=lambda c: c.window[-1].ts if c.window else 0.0,
            )
            self._clusters.pop(oldest.cluster_id, None)

        cid = uuid.uuid4().hex[:12]
        cl = _Cluster(
            cluster_id=cid,
            centroid_tokens=set(qtokens),
            centroid_fragments=set(fragments),
        )
        self._clusters[cid] = cl
        return cl

    def _maybe_crystallize(
        self, cluster: _Cluster, baseline: float
    ) -> CrystallizationEvent | None:
        n = len(cluster.window)
        if n < self._min_samples:
            return None
        # Cooldown: don't re-emit for the same cluster within `cooldown`
        # observations of the previous emission.
        if (
            cluster.last_crystallized_at_n >= 0
            and (cluster.n_total - cluster.last_crystallized_at_n) < self._cooldown
        ):
            return None

        rewards = [obs.reward for obs in cluster.window]
        mean_r = sum(rewards) / n
        lcb = _hoeffding_lcb(mean_r, n, self._delta)

        # ── External baseline (excludes this cluster's contributions) ──
        # The caller's ``baseline`` is typically a global reward-EMA that
        # tracks ALL observations, including this cluster's. For sustained
        # high-reward clusters that's a structural false-negative: as the
        # cluster fills the EMA, baseline asymptotes to cluster_mean and
        # the LCB never beats it. Math:
        #
        #   ext_n   = lifetime_obs   − cluster_lifetime_obs
        #   ext_sum = lifetime_∑r    − cluster_lifetime_∑r
        #   baseline_ext = ext_sum / ext_n   if ext_n ≥ 5
        #                = baseline (caller's value)   otherwise
        #
        # The 5-observation floor is a Hoeffding-symmetric choice: we
        # require enough non-cluster signal that the external mean is at
        # least as well-resolved as the cluster's LCB.
        c_lifetime_n = self._cluster_lifetime_n.get(cluster.cluster_id, 0)
        c_lifetime_r = self._cluster_lifetime_reward.get(cluster.cluster_id, 0.0)
        ext_n = self._lifetime_observations - c_lifetime_n
        ext_sum = self._lifetime_reward_sum - c_lifetime_r
        if ext_n >= 5:
            effective_baseline = ext_sum / ext_n
        else:
            effective_baseline = baseline  # cold-start fallback

        if lcb <= effective_baseline + self._epsilon:
            return None

        cluster.last_crystallized_at_n = cluster.n_total

        # Build the event payload from the cluster's window.
        weight_profile = self._mean_weights(cluster.window)
        fragment_recipe = self._top_fragments(cluster.window, top_k=8)
        common_terms = self._common_terms(cluster.queries[-self._window:], top_k=8)
        sample_queries = cluster.queries[-min(5, len(cluster.queries)):]

        return CrystallizationEvent(
            event_id=uuid.uuid4().hex[:12],
            cluster_id=cluster.cluster_id,
            sample_queries=sample_queries,
            common_terms=common_terms,
            weight_profile=weight_profile,
            fragment_recipe=fragment_recipe,
            n_samples=n,
            mean_reward=mean_r,
            lcb_reward=lcb,
            baseline_reward=effective_baseline,
            effect_size=lcb - effective_baseline,
            created_at=time.time(),
        )

    @staticmethod
    def _mean_weights(window: Iterable[_Observation]) -> dict[str, float]:
        agg: dict[str, float] = {}
        n = 0
        for obs in window:
            for k, v in obs.weights.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
        if n == 0:
            return {}
        return {k: round(v / n, 4) for k, v in agg.items()}

    @staticmethod
    def _top_fragments(window: Iterable[_Observation], top_k: int) -> list[str]:
        counts: dict[str, int] = {}
        for obs in window:
            for fid in obs.fragment_ids:
                counts[fid] = counts.get(fid, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [fid for fid, _ in ranked[:top_k]]

    @staticmethod
    def _common_terms(queries: Iterable[str], top_k: int) -> list[str]:
        # Frequency count of non-trivial tokens across cluster queries.
        STOP = {
            "the", "a", "an", "of", "to", "in", "for", "on", "and", "or",
            "is", "are", "was", "were", "be", "do", "this", "that", "it",
            "with", "from", "by", "as", "at", "i", "we", "you", "my",
        }
        counts: dict[str, int] = {}
        for q in queries:
            for t in q.lower().split():
                t = "".join(ch for ch in t if ch.isalnum() or ch in "_-")
                if len(t) <= 2 or t in STOP:
                    continue
                counts[t] = counts.get(t, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return [t for t, _ in ranked[:top_k]]


# Module-level singleton for easy integration into the engine. Tests
# should construct their own instance to avoid cross-contamination.
DEFAULT_CRYSTALLIZER = RewardCrystallizer()


__all__ = [
    "CrystallizationEvent",
    "DEFAULT_CRYSTALLIZER",
    "RewardCrystallizer",
]
