"""
Entroly MCP Server
========================

Thin MCP wrapper around the Rust EntrolyEngine.

All computation (knapsack, entropy, SimHash, dep graph, feedback loop,
context ordering) runs in Rust via PyO3. Python only handles:
  - MCP protocol (FastMCP tool registration + JSON-RPC)
  - Predictive pre-fetching (static analysis + co-access learning)
  - Checkpoint I/O (gzipped JSON serialization)

Architecture:
  MCP Client → JSON-RPC → Python (FastMCP) → PyO3 → Rust Engine → Results

Supported clients:
  - Cursor (add to .cursor/mcp.json)
  - Claude Code (claude mcp add)
  - Cline (add to mcp settings)
  - Any MCP-compatible client

Run:
    entroly        # Start as STDIO server
    python -m entroly.server   # Alternative
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .adaptive_pruner import EntrolyPruner, FragmentGuard
from .autotune import ComponentFeedbackBus, DreamingLoop, FeedbackJournal, TaskProfileOptimizer
from .cache_aligner import CacheAligner
from .belief_compiler import BeliefCompiler
from .change_listener import WorkspaceChangeListener
from .change_pipeline import ChangePipeline
from .checkpoint import CheckpointManager, ContextFragment
from .config import EntrolyConfig
from .epistemic_router import (
    EpistemicRouter,
)
from .evolution_daemon import EvolutionDaemon
from .evolution_logger import EvolutionLogger
from .flow_orchestrator import FlowOrchestrator
from .multimodal import ingest_diagram as _mm_diagram
from .multimodal import ingest_diff as _mm_diff
from .multimodal import ingest_voice as _mm_voice
from .prefetch import PrefetchEngine
from .provenance import build_provenance
from .proxy_transform import calibrated_token_count as _calibrated_token_count
from .query_refiner import QueryRefiner
from .repo_map import build_repo_map, render_repo_map_markdown
from .skill_engine import SkillEngine
from .value_tracker import ValueTracker, get_tracker
from .vault import (
    BeliefArtifact,
    VaultConfig,
    VaultManager,
)
from .verification_engine import VerificationEngine

# ── Rust engine import (preferred, 50-100× faster) ─────────────────
try:
    from entroly_core import EntrolyEngine as RustEngine
    from entroly_core import py_analyze_query, py_refine_heuristic
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    RustEngine = None  # type: ignore[assignment,misc]

    # Provide Python stubs for the Rust query helpers
    def py_analyze_query(query: str) -> dict:  # type: ignore[misc]
        """Pure-Python stub when entroly_core is not available."""
        terms = [w for w in query.lower().split() if len(w) > 2][:8]
        return {
            "vagueness_score": 0.5,
            "key_terms": terms,
            "needs_refinement": len(terms) < 3,
            "reason": "python_fallback",
        }

    def py_refine_heuristic(query: str, context: str) -> str:  # type: ignore[misc]
        """Pure-Python stub — returns query unchanged."""
        return query


# ══════════════════════════════════════════════════════════════════════
# Pure-Python fallback implementations (used when Rust engine unavailable)
# ══════════════════════════════════════════════════════════════════════


def _py_simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint from text.

    Uses trigrams (or unigrams for short text) hashed with MD5
    to build a locality-sensitive fingerprint.
    """
    import hashlib as _hl
    words = [w for w in text.lower().split() if w.isalnum() or any(c.isalnum() for c in w)]
    if not words:
        return 0

    # Build features: trigrams if >= 3 words, else unigrams
    features: list[str] = []
    if len(words) >= 3:
        for i in range(len(words) - 2):
            features.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    else:
        features = words

    bit_sums = [0] * 64
    for feat in features:
        h = int(_hl.md5(feat.encode("utf-8", errors="replace")).hexdigest(), 16)
        for i in range(64):
            if (h >> i) & 1:
                bit_sums[i] += 1
            else:
                bit_sums[i] -= 1

    fingerprint = 0
    for i in range(64):
        if bit_sums[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _py_hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit fingerprints."""
    return bin(a ^ b).count("1")


class _PyDedupIndex:
    """Pure-Python SimHash-based deduplication index (LSH banding)."""

    def __init__(self, hamming_threshold: int = 3):
        self._threshold = hamming_threshold
        self._fingerprints: dict[str, int] = {}
        self._bands: list[dict[int, list[str]]] = [{} for _ in range(4)]
        self._duplicates_detected = 0

    def insert(self, fragment_id: str, text: str) -> str | None:
        """Insert a fragment. Returns duplicate_id if near-duplicate found."""
        fp = _py_simhash(text)

        # Check bands for candidate matches
        for band_idx in range(4):
            band_hash = (fp >> (band_idx * 16)) & 0xFFFF
            candidates = self._bands[band_idx].get(band_hash, [])
            for cand_id in candidates:
                cand_fp = self._fingerprints.get(cand_id)
                if cand_fp is not None and _py_hamming_distance(fp, cand_fp) <= self._threshold:
                    self._duplicates_detected += 1
                    return cand_id

        # No duplicate — register this fragment
        self._fingerprints[fragment_id] = fp
        for band_idx in range(4):
            band_hash = (fp >> (band_idx * 16)) & 0xFFFF
            self._bands[band_idx].setdefault(band_hash, []).append(fragment_id)
        return None

    def remove(self, fragment_id: str) -> None:
        """Remove a fragment from the index."""
        fp = self._fingerprints.pop(fragment_id, None)
        if fp is None:
            return
        for band_idx in range(4):
            band_hash = (fp >> (band_idx * 16)) & 0xFFFF
            bucket = self._bands[band_idx].get(band_hash, [])
            if fragment_id in bucket:
                bucket.remove(fragment_id)

    def stats(self) -> dict:
        return {
            "total_fingerprints": len(self._fingerprints),
            "duplicates_detected": self._duplicates_detected,
        }


def _py_compute_information_score(
    text: str,
    global_token_counts: dict[str, int],
    total_tokens: int,
    other_fragments: list[str],
) -> float:
    """Compute information density score [0, 1] using Shannon entropy + redundancy."""
    import math as _m
    if not text:
        return 0.0

    # 1. Normalized Shannon entropy on bytes (40% weight)
    byte_counts: dict[int, int] = {}
    encoded = text.encode("utf-8", errors="replace")
    for b in encoded:
        byte_counts[b] = byte_counts.get(b, 0) + 1
    total_bytes = len(encoded)
    entropy = 0.0
    for count in byte_counts.values():
        p = count / total_bytes
        if p > 0:
            entropy -= p * _m.log2(p)
    normalized_entropy = min(entropy / 6.0, 1.0)

    # 2. Boilerplate penalty (30% weight)
    lines = text.strip().splitlines()
    boilerplate_lines = 0
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith(("import ", "from ")) or
                stripped in ("pass", "...", "}", ")", "]", '"""') or
                stripped.startswith("def __") or
                stripped in ("return None", "return self", "return True", "return False")):
            boilerplate_lines += 1
    bp_ratio = boilerplate_lines / max(len(lines), 1)
    bp_penalty = 1.0 - bp_ratio

    # 3. Cross-fragment redundancy (30% weight)
    uniqueness = 1.0
    if other_fragments:
        words = text.lower().split()
        if len(words) >= 3:
            text_trigrams = set()
            for i in range(len(words) - 2):
                text_trigrams.add(f"{words[i]} {words[i+1]} {words[i+2]}")
            if text_trigrams:
                other_trigrams: set = set()
                for other in other_fragments:
                    ow = other.lower().split()
                    for i in range(len(ow) - 2):
                        other_trigrams.add(f"{ow[i]} {ow[i+1]} {ow[i+2]}")
                overlap = len(text_trigrams & other_trigrams)
                redundancy = overlap / len(text_trigrams)
                uniqueness = 1.0 - min(redundancy, 1.0)

    score = 0.40 * normalized_entropy + 0.30 * bp_penalty + 0.30 * uniqueness
    return max(0.0, min(1.0, score))


def _py_compute_relevance(
    frag: ContextFragment,
    w_recency: float,
    w_frequency: float,
    w_semantic: float,
    w_entropy: float,
    feedback_multiplier: float = 1.0,
) -> float:
    """Compute weighted relevance score with softcap [0, 1]."""
    import math as _m
    total_weight = w_recency + w_frequency + w_semantic + w_entropy
    if total_weight == 0:
        return 0.0
    base = (
        w_recency * frag.recency_score
        + w_frequency * frag.frequency_score
        + w_semantic * frag.semantic_score
        + w_entropy * frag.entropy_score
    ) / total_weight
    raw = base * feedback_multiplier
    # Gemini-style logit softcap at 0.85
    cap = 0.85
    return cap * _m.tanh(raw / cap)


def _py_knapsack_optimize(
    fragments: list,
    token_budget: int,
    w_recency: float,
    w_frequency: float,
    w_semantic: float,
    w_entropy: float,
) -> tuple:
    """Pure-Python greedy knapsack optimizer.

    Returns (selected_fragments, stats_dict).
    """
    # Separate pinned fragments (always included)
    pinned = [f for f in fragments if f.is_pinned]
    candidates = [f for f in fragments if not f.is_pinned]

    pinned_tokens = sum(f.token_count for f in pinned)
    max(0, token_budget - pinned_tokens)

    # Score and sort candidates by relevance/token ratio (greedy)
    scored = []
    for frag in candidates:
        rel = _py_compute_relevance(frag, w_recency, w_frequency, w_semantic, w_entropy)
        if frag.token_count > 0:
            efficiency = rel / frag.token_count
        else:
            efficiency = rel
        scored.append((frag, rel, efficiency))
    scored.sort(key=lambda x: x[2], reverse=True)

    selected = list(pinned)
    used_tokens = pinned_tokens
    total_relevance = sum(
        _py_compute_relevance(f, w_recency, w_frequency, w_semantic, w_entropy) for f in pinned
    )

    for frag, rel, _ in scored:
        if used_tokens + frag.token_count <= token_budget:
            selected.append(frag)
            used_tokens += frag.token_count
            total_relevance += rel

    stats = {
        "total_tokens": used_tokens,
        "total_relevance": round(total_relevance, 4),
        "method": "greedy_python",
        "pinned_count": len(pinned),
        "candidate_count": len(candidates),
    }
    return selected, stats


def _py_apply_ebbinghaus_decay(
    fragments: list,
    current_turn: int,
    half_life: int,
) -> None:
    """Apply Ebbinghaus forgetting curve to fragment recency scores (in-place)."""
    import math as _m
    if half_life <= 0:
        return
    decay_rate = _m.log(2) / half_life
    for frag in fragments:
        dt = max(0, current_turn - frag.turn_last_accessed)
        frag.recency_score = _m.exp(-decay_rate * dt)

# Configure logging to stderr (MCP requires stdout for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [entroly] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("entroly")


class _WilsonFeedbackTracker:
    """
    Python-exact port of the Rust FeedbackTracker in guardrails.rs.

    Uses the Wilson score lower bound (the same formula Reddit uses for ranking)
    to produce a calibrated relevance multiplier for each fragment:

        p̂  = successes / (successes + failures)
        z  = 1.96   # 95% confidence interval
        lb = Wilson lower bound
        multiplier = 0.5 + lb × 1.5   → maps [0, 1] → [0.5, 2.0]

    > 1.0  fragment historically useful   (boosted)
    = 1.0  no data yet                    (neutral)
    < 1.0  fragment historically unhelpful (suppressed)

    Numerically identical to the Rust implementation — the Python fallback
    now has the same feedback quality as the Rust engine.
    """
    import math as _math

    _Z = 1.96  # 95% CI

    def __init__(self):
        self._success: dict[str, int] = {}
        self._failure: dict[str, int] = {}

    def record_success(self, fragment_ids: list[str]) -> None:
        for fid in fragment_ids:
            self._success[fid] = self._success.get(fid, 0) + 1

    def record_failure(self, fragment_ids: list[str]) -> None:
        for fid in fragment_ids:
            self._failure[fid] = self._failure.get(fid, 0) + 1

    def learned_value(self, fragment_id: str) -> float:
        import math
        s = self._success.get(fragment_id, 0)
        f = self._failure.get(fragment_id, 0)
        total = s + f
        if total == 0:
            return 1.0
        p = s / total
        z = self._Z
        denominator = 1.0 + z * z / total
        center = p + z * z / (2.0 * total)
        spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        lower_bound = (center - spread) / denominator
        raw = 0.5 + lower_bound * 1.5
        return max(0.5, min(2.0, raw))  # Clamp to documented [0.5, 2.0] range


class EntrolyEngine:
    """
    Orchestrates all subsystems. Delegates math to Rust when available.

    Rust handles: ingest, optimize, recall, stats, feedback, dep graph, ordering.
    Python handles: prefetch, checkpoint, MCP protocol.
    """

    def __init__(self, config: EntrolyConfig | None = None):
        self.config = config or EntrolyConfig()
        self._use_rust = _RUST_AVAILABLE

        if self._use_rust:
            self._rust = RustEngine(
                w_recency=self.config.weight_recency,
                w_frequency=self.config.weight_frequency,
                w_semantic=self.config.weight_semantic_sim,
                w_entropy=self.config.weight_entropy,
                decay_half_life=self.config.decay_half_life_turns,
                min_relevance=self.config.min_relevance_threshold,
            )
            logger.info("Using Rust engine (entroly_core)")
        else:
            # Python fallback state
            self._fragments: dict[str, Any] = {}
            self._current_turn: int = 0
            from collections import Counter
            self._global_token_counts: Counter = Counter()
            self._total_token_count: int = 0
            self._dedup = _PyDedupIndex(hamming_threshold=3)
            self._total_tokens_saved: int = 0
            self._total_optimizations: int = 0
            self._total_fragments_ingested: int = 0
            self._total_duplicates_caught: int = 0
            # Wilson-score feedback tracker — numerically identical to Rust FeedbackTracker
            self._wilson = _WilsonFeedbackTracker()
            logger.info("Using Python fallback engine (entroly_core not installed)")

        # Python-only subsystems
        self._prefetch = PrefetchEngine(co_access_window=5)
        self._checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            auto_interval=self.config.auto_checkpoint_interval,
        )
        # Query refinement: vague queries are expanded using in-memory file
        # context before context selection, reducing hallucination from wrong files.
        self._refiner = QueryRefiner()

        # ebbiforge CodeQualityGuard: scans ingested fragments for secrets/TODO/unsafe
        self._guard = FragmentGuard()
        # ebbiforge AdaptivePruner: RL weight learning on feedback
        self._pruner = EntrolyPruner()
        # Turn counter for provenance
        self._turn_counter: int = 0

        # Fix #5: Validate that the checkpoint directory is writable at startup.
        # Fail fast with a clear error rather than a cryptic gzip/PermissionError
        # during the first auto-checkpoint (which could happen mid-session).
        self._validate_checkpoint_dir()

        # ── Persistent Repo-Level Indexing ──
        # On startup, try to load a previous session's index for instant warm retrieval.
        # Index is stored at <checkpoint_dir>/index.json.gz (gzip-compressed JSON).
        self._index_path = str(Path(self.config.checkpoint_dir) / "index.json.gz")
        if self._use_rust:
            try:
                loaded = self._rust.load_index(self._index_path)
                if loaded:
                    n = self._rust.fragment_count()
                    logger.info(f"Loaded persistent index: {n} fragments from {self._index_path}")
                else:
                    logger.info("No persistent index found, starting fresh session")
            except AttributeError:
                # PyO3 EntrolyEngine may not expose load_index in this build
                logger.debug("Rust engine does not support load_index -- skipping persistent index")
            except OSError:
                # Index file doesn't exist yet — normal on first run
                logger.info("No persistent index found, starting fresh session")
            except Exception as e:
                logger.warning(f"Failed to load persistent index: {e}")

        # GC freeze at startup: Python's cyclic GC causes ~500ms stalls on large
        # heaps. Freeze all existing long-lived objects and disable automatic
        # collection. We manually collect every N tool calls in advance_turn()
        # to reclaim short-lived garbage without unpredictable pauses.
        self._gc_collect_interval = 50  # collect every 50 turns
        gc.collect()
        gc.freeze()
        gc.disable()

    def advance_turn(self) -> None:
        """Advance the turn counter and apply Ebbinghaus decay."""
        # Periodic GC amortization: frozen at init, collect every N turns
        if self._turn_counter > 0 and self._turn_counter % self._gc_collect_interval == 0:
            gc.collect()

        if self._use_rust:
            self._rust.advance_turn()
        else:
            self._current_turn += 1
            fragments = list(self._fragments.values())
            _py_apply_ebbinghaus_decay(
                fragments,
                self._current_turn,
                self.config.decay_half_life_turns,
            )
            to_evict = [
                fid for fid, f in self._fragments.items()
                if f.recency_score < self.config.min_relevance_threshold
                and not f.is_pinned
            ]
            for fid in to_evict:
                self._dedup.remove(fid)
                del self._fragments[fid]

    def ingest_fragment(
        self,
        content: str,
        source: str = "",
        token_count: int = 0,
        is_pinned: bool = False,
    ) -> dict[str, Any]:
        """Ingest a new context fragment."""
        # GC freeze: disable the garbage collector during the tight Python→Rust
        # dispatch to prevent unpredictable GC pauses. Manually collect after
        # returning to amortize the cost at a safe boundary.
        gc.disable()
        try:
            if self._use_rust:
                # Enforce max_fragments cap on Rust engine (Rust doesn't enforce it)
                if self._rust.fragment_count() >= self.config.max_fragments:
                    return {
                        "status": "rejected",
                        "reason": "max_fragments cap reached",
                        "max_fragments": self.config.max_fragments,
                    }
                result = self._rust.ingest(content, source, token_count, is_pinned)
                # result is a dict from PyO3
                if source:
                    self._prefetch.record_access(source, self._rust.get_turn())
                if self._checkpoint_mgr.should_auto_checkpoint():
                    self._auto_checkpoint()
                return dict(result)
            else:
                return self._ingest_python(content, source, token_count, is_pinned)
        finally:
            gc.enable()
            gc.collect()

    def optimize_context(
        self,
        token_budget: int = 0,
        query: str = "",
    ) -> dict[str, Any]:
        """Select the mathematically optimal subset of context fragments."""
        if token_budget <= 0:
            token_budget = self.config.default_token_budget

        # Query refinement: expand vague queries using in-memory file context.
        # This is the key fix for hallucination from incomplete context:
        # "fix the bug" → "bug fix in payments module (Python/Rust) involving
        # payment processing, error handling"
        refined_query = query
        refinement_info: dict[str, Any] = {}
        if query:
            fragment_summaries = []
            if self._use_rust:
                try:
                    recalled = list(self._rust.recall(query, 20))
                    fragment_summaries = [r.get("content", "") for r in recalled]
                except Exception:
                    pass
            # analyze() returns dict: vagueness_score, key_terms, needs_refinement, reason
            analysis_dict = self._refiner.analyze(query, fragment_summaries)
            # refine() returns the improved query string
            refined_query = self._refiner.refine(query, fragment_summaries)
            if analysis_dict.get("needs_refinement"):
                refinement_info = {
                    "original_query":    query,
                    "refined_query":     refined_query,
                    "vagueness_score":   analysis_dict["vagueness_score"],
                    "refinement_source": "rust_heuristic",
                    "key_terms":         analysis_dict.get("key_terms", []),
                }

        # Always capture query analysis (vagueness is needed by EGTC even
        # when the query doesn't trigger refinement)
        query_analysis = {
            "vagueness_score": analysis_dict["vagueness_score"],
            "key_terms": analysis_dict.get("key_terms", []),
        } if query and analysis_dict else {}

        # GC freeze: disable during hot Rust dispatch + final result assembly.
        gc.disable()
        try:
            if self._use_rust:
                result = self._rust.optimize(token_budget, refined_query)
                result = dict(result)
                # Normalize key: Rust returns "selected", Python uses "selected_fragments"
                if "selected" in result and "selected_fragments" not in result:
                    result["selected_fragments"] = result["selected"]
                if refinement_info:
                    result["query_refinement"] = refinement_info
                if query_analysis:
                    result["query_analysis"] = query_analysis
                return result
            else:
                result = self._optimize_python(token_budget, refined_query)
                if refinement_info:
                    result["query_refinement"] = refinement_info
                if query_analysis:
                    result["query_analysis"] = query_analysis
                return result
        finally:
            gc.enable()
            gc.collect()

    def recall_relevant(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic recall of relevant fragments."""
        if self._use_rust:
            result = self._rust.recall(query, top_k)
            return [dict(r) for r in result]
        else:
            return self._recall_python(query, top_k)

    def record_success(self, fragment_ids: list[str]) -> None:
        """Record that selected fragments led to a successful output."""
        if self._use_rust:
            self._rust.record_success(fragment_ids)
        else:
            self._wilson.record_success(fragment_ids)
        # Fix #2: Wire AdaptivePruner RL feedback on every record_success call.
        # apply_feedback(+1.0) boosts the learned weights for features that
        # were present when the fragment was selected and led to success.
        for fid in fragment_ids:
            self._pruner.apply_feedback(fid, 1.0)

    def record_failure(self, fragment_ids: list[str]) -> None:
        """Record that selected fragments led to a failed output."""
        if self._use_rust:
            self._rust.record_failure(fragment_ids)
        else:
            self._wilson.record_failure(fragment_ids)
        # Fix #2: Wire AdaptivePruner RL feedback on every record_failure call.
        # apply_feedback(-1.0) down-weights feature combinations that led to
        # unhelpful context selections.
        for fid in fragment_ids:
            self._pruner.apply_feedback(fid, -1.0)

    def record_reward(self, fragment_ids: list[str], reward: float) -> None:
        """Record a continuous reward signal for selected fragments.

        Unlike record_success/failure (binary), this allows graded feedback:
          reward > 0 → positive signal (boost fragment weight)
          reward < 0 → negative signal (suppress fragment weight)
          reward = 0 → neutral

        The Rust engine routes this to the EGSC cache's Thompson gate
        and hit predictor for continuous learning.
        """
        if self._use_rust:
            self._rust.record_reward(fragment_ids, reward)
        # RL pruner also gets the continuous signal
        for fid in fragment_ids:
            self._pruner.apply_feedback(fid, reward)

    def set_model(self, model_name: str) -> None:
        """Auto-configure cache cost model from model name.

        Covers 20+ models: OpenAI (gpt-4o, gpt-4, o1, o3), Anthropic
        (claude-3.5), Google (gemini), DeepSeek, Meta (llama), Mistral.
        Unknown models default to GPT-4o pricing ($0.000015/token).

        Example::

            engine.set_model("gpt-4o-mini")  # → $0.60/M tokens
            engine.set_model("claude-3-opus")  # → $75/M tokens
        """
        if self._use_rust:
            self._rust.set_model(model_name)

    def set_cache_cost_per_token(self, cost: float) -> None:
        """Set cost-per-token directly (power users only).

        Most developers should use set_model() instead.
        Default is already $0.000015 (GPT-4o output) — no config needed.
        """
        if self._use_rust:
            self._rust.set_cache_cost_per_token(cost)

    def cache_clear(self) -> None:
        """Clear all cached LLM responses.

        Useful when switching projects, after major refactors, or
        when cache correctness is suspect.
        """
        if self._use_rust:
            self._rust.cache_clear()

    def cache_len(self) -> int:
        """Return the number of entries in the response cache."""
        if self._use_rust:
            return self._rust.cache_len()
        return 0

    def cache_is_empty(self) -> bool:
        """Check if the response cache is empty."""
        if self._use_rust:
            return self._rust.cache_is_empty()
        return True

    def cache_hit_rate(self) -> float:
        """Return the cache hit rate (0.0 to 1.0).

        This is the primary observability metric for the EGSC cache.
        A healthy, warmed-up cache should show hit_rate > 0.3.
        """
        if self._use_rust:
            return self._rust.cache_hit_rate()
        return 0.0

    def prefetch_related(
        self,
        file_path: str,
        source_content: str = "",
        language: str = "python",
    ) -> list[dict[str, Any]]:
        """Predict and pre-load likely-needed context."""
        predictions = self._prefetch.predict(
            file_path, source_content, language
        )
        return [
            {
                "path": p.path,
                "reason": p.reason,
                "confidence": p.confidence,
            }
            for p in predictions
        ]

    def checkpoint(self, metadata: dict[str, Any] | None = None) -> str:
        """Manually create a checkpoint."""
        return self._auto_checkpoint(metadata)

    def resume(self) -> dict[str, Any]:
        """Resume from the latest checkpoint."""
        ckpt = self._checkpoint_mgr.load_latest()
        if ckpt is None:
            return {"status": "no_checkpoint_found"}

        if self._use_rust:
            # Try to restore from full engine state (preferred)
            engine_state = ckpt.metadata.get("engine_state") if ckpt.metadata else None
            if engine_state:
                self._rust.import_state(engine_state)
            else:
                # Fallback: re-create engine and re-ingest fragments
                self._rust = RustEngine(
                    w_recency=self.config.weight_recency,
                    w_frequency=self.config.weight_frequency,
                    w_semantic=self.config.weight_semantic_sim,
                    w_entropy=self.config.weight_entropy,
                    decay_half_life=self.config.decay_half_life_turns,
                    min_relevance=self.config.min_relevance_threshold,
                )
                for frag_data in ckpt.fragments:
                    self._rust.ingest(
                        frag_data["content"],
                        frag_data.get("source", ""),
                        frag_data.get("token_count", 0),
                        frag_data.get("is_pinned", False),
                    )
        else:
            self._fragments.clear()
            for frag in self._checkpoint_mgr.restore_fragments(ckpt):
                self._fragments[frag.fragment_id] = frag
            self._dedup = _PyDedupIndex(hamming_threshold=3)
            for fid, fp in ckpt.dedup_fingerprints.items():
                self._dedup._fingerprints[fid] = fp
            self._current_turn = ckpt.current_turn

        # Restore co-access patterns
        from collections import Counter
        for src, targets in ckpt.co_access_data.items():
            self._prefetch._co_access[src] = Counter(targets)

        return {
            "status": "resumed",
            "checkpoint_id": ckpt.checkpoint_id,
            "restored_fragments": len(ckpt.fragments),
            "restored_turn": ckpt.current_turn,
            "metadata": ckpt.metadata,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive session statistics."""
        if self._use_rust:
            rust_stats = dict(self._rust.stats())
            dep_stats = dict(self._rust.dep_graph_stats())
            rust_stats["dep_graph"] = dep_stats
            rust_stats["prefetch"] = self._prefetch.stats()
            rust_stats["checkpoint"] = self._checkpoint_mgr.stats()
            return rust_stats
        else:
            return self._stats_python()

    def explain_selection(self) -> dict[str, Any]:
        """Explain why each fragment was included or excluded."""
        if self._use_rust:
            result = self._rust.explain_selection()
            return dict(result)
        else:
            return {"error": "Explainability requires Rust engine"}

    def _auto_checkpoint(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create an auto-checkpoint."""
        if self._use_rust:
            co_access = {
                k: dict(v)
                for k, v in self._prefetch._co_access.items()
            }
            # Export full engine state (not empty fragments)
            engine_state = self._rust.export_state()
            # Auto-persist repo-level index alongside checkpoint
            try:
                self._rust.persist_index(self._index_path)
            except Exception as e:
                logger.debug(f"Failed to persist index: {e}")
            return self._checkpoint_mgr.save(
                fragments=[],
                dedup_fingerprints={},
                co_access_data=co_access,
                current_turn=self._rust.get_turn(),
                metadata={**(metadata or {}), "engine_state": engine_state},
                stats=self.get_stats(),
            )
        else:
            co_access = {
                k: dict(v)
                for k, v in self._prefetch._co_access.items()
            }
            return self._checkpoint_mgr.save(
                fragments=list(self._fragments.values()),
                dedup_fingerprints=dict(self._dedup._fingerprints),
                co_access_data=co_access,
                current_turn=self._current_turn,
                metadata=metadata,
                stats=self.get_stats(),
            )

    def _validate_checkpoint_dir(self) -> None:
        """Fix #5: Validate checkpoint directory is writable at startup."""
        ckpt_dir = self.config.checkpoint_dir
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            # Write a probe file to confirm the directory is actually writable
            probe = os.path.join(str(ckpt_dir), ".entroly_write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.unlink(probe)
        except OSError as e:
            raise RuntimeError(
                f"Entroly checkpoint directory '{ckpt_dir}' is not writable: {e}.\n"
                f"Set the ENTROLY_DIR env var or pass checkpoint_dir= to EntrolyConfig "
                f"to point to a writable location."
            ) from e

    # ── Python fallback implementations ──────────────────────────────

    def _ingest_python(self, content, source, token_count, is_pinned):
        """Python fallback for ingest (when Rust not available)."""
        import hashlib
        self._total_fragments_ingested += 1

        if token_count <= 0:
            token_count = _calibrated_token_count(content, source)

        # Fix #4 (Python fallback): enforce max_fragments cap
        if len(self._fragments) >= self.config.max_fragments:
            return {
                "status": "rejected",
                "reason": "max_fragments cap reached",
                "max_fragments": self.config.max_fragments,
            }

        frag_id = hashlib.sha256(
            f"{source}:{content[:200]}:{self._total_fragments_ingested}".encode()
        ).hexdigest()[:16]

        dup_id = self._dedup.insert(frag_id, content)
        if dup_id is not None:
            self._total_duplicates_caught += 1
            existing = self._fragments.get(dup_id)
            if existing:
                existing.access_count += 1
                existing.turn_last_accessed = self._current_turn
                max_freq = max(
                    f.access_count for f in self._fragments.values()
                ) or 1
                existing.frequency_score = min(
                    existing.access_count / max_freq, 1.0
                )
            return {
                "status": "duplicate",
                "duplicate_of": dup_id,
                "fragment_id": frag_id,
                "tokens_saved": token_count,
            }

        # Sample up to 50 fragments for entropy comparison (O(n) instead of O(n log n))
        import random as _rng
        all_frags = list(self._fragments.values())
        sample = _rng.sample(all_frags, min(50, len(all_frags))) if len(all_frags) > 50 else all_frags
        other_contents = [f.content for f in sample]
        entropy_score = _py_compute_information_score(
            content,
            global_token_counts=dict(self._global_token_counts),
            total_tokens=self._total_token_count,
            other_fragments=other_contents,
        )

        tokens = content.lower().split()
        self._global_token_counts.update(tokens)
        self._total_token_count += len(tokens)

        frag = ContextFragment(
            fragment_id=frag_id,
            content=content,
            token_count=token_count,
            source=source,
            recency_score=1.0,
            frequency_score=0.0,
            semantic_score=0.0,
            entropy_score=entropy_score,
            turn_created=self._current_turn,
            turn_last_accessed=self._current_turn,
            access_count=1,
            is_pinned=is_pinned,
            simhash=_py_simhash(content),
        )

        self._fragments[frag_id] = frag

        if source:
            self._prefetch.record_access(source, self._current_turn)

        if self._checkpoint_mgr.should_auto_checkpoint():
            self._auto_checkpoint()

        # Fix #3: extract skeleton for code files (Python fallback)
        has_skeleton = False
        skeleton_tc = None
        try:
            if _RUST_AVAILABLE:
                # entroly_core exposes extract_skeleton directly
                from entroly_core import extract_skeleton
                skel = extract_skeleton(content, source)
            else:
                from .skeleton import extract_skeleton  # type: ignore[import]
                skel = extract_skeleton(content, source)
            if skel:
                has_skeleton = True
                skeleton_tc = _calibrated_token_count(skel, source)
        except Exception:
            pass  # skeleton is best-effort; never block ingest

        result: dict[str, Any] = {
            "status": "ingested",
            "fragment_id": frag_id,
            "token_count": token_count,
            "entropy_score": round(entropy_score, 4),
            "total_fragments": len(self._fragments),
            "has_skeleton": has_skeleton,
        }
        if skeleton_tc is not None:
            result["skeleton_token_count"] = skeleton_tc
        return result

    def _optimize_python(self, token_budget, query):
        """Python fallback for optimize."""
        self._total_optimizations += 1

        if query:
            query_hash = _py_simhash(query)
            for frag in self._fragments.values():
                dist = _py_hamming_distance(query_hash, frag.simhash)
                frag.semantic_score = max(0.0, 1.0 - (dist / 64.0))

        # Apply Wilson feedback via the frequency dimension. Wilson learned_value
        # returns [0.5, 2.0] (neutral=1.0). We map this to [0, 1] and use it as
        # the frequency_score, so fragments with positive feedback history get
        # boosted in the knapsack and fragments with negative history get suppressed.
        for frag in self._fragments.values():
            wilson = self._wilson.learned_value(frag.fragment_id)
            frag.frequency_score = max(0.0, min((wilson - 0.5) / 1.5, 1.0))

        fragments = list(self._fragments.values())
        selected, stats = _py_knapsack_optimize(
            fragments,
            token_budget,
            w_recency=self.config.weight_recency,
            w_frequency=self.config.weight_frequency,
            w_semantic=self.config.weight_semantic_sim,
            w_entropy=self.config.weight_entropy,
        )

        total_available_tokens = sum(f.token_count for f in fragments)
        tokens_saved = total_available_tokens - stats["total_tokens"]
        self._total_tokens_saved += max(0, tokens_saved)

        for frag in selected:
            frag.turn_last_accessed = self._current_turn
            frag.access_count += 1

        return {
            "selected_fragments": [
                {
                    "id": f.fragment_id,
                    "source": f.source,
                    "token_count": f.token_count,
                    "relevance": round(
                        _py_compute_relevance(
                            f,
                            self.config.weight_recency,
                            self.config.weight_frequency,
                            self.config.weight_semantic_sim,
                            self.config.weight_entropy,
                        ), 4
                    ),
                    "content_preview": f.content[:100] + "..." if len(f.content) > 100 else f.content,
                }
                for f in selected
            ],
            "optimization_stats": stats,
            "tokens_saved_this_call": max(0, tokens_saved),
            "total_tokens_saved_session": self._total_tokens_saved,
        }

    def _recall_python(self, query, top_k):
        """Python fallback for recall."""
        if not self._fragments:
            return []

        query_hash = _py_simhash(query)

        scored = []
        for frag in self._fragments.values():
            dist = _py_hamming_distance(query_hash, frag.simhash)
            frag.semantic_score = max(0.0, 1.0 - (dist / 64.0))
            relevance = _py_compute_relevance(
                frag,
                self.config.weight_recency,
                self.config.weight_frequency,
                self.config.weight_semantic_sim,
                self.config.weight_entropy,
            )
            scored.append((frag, relevance))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "fragment_id": f.fragment_id,
                "source": f.source,
                "relevance": round(rel, 4),
                "entropy": round(f.entropy_score, 4),
                "content": f.content,
            }
            for f, rel in scored[:top_k]
        ]

    def _stats_python(self):
        """Python fallback for stats."""
        fragments = list(self._fragments.values())
        total_tokens = sum(f.token_count for f in fragments)
        avg_entropy = (
            sum(f.entropy_score for f in fragments) / len(fragments)
            if fragments else 0.0
        )

        return {
            "session": {
                "current_turn": self._current_turn,
                "total_fragments": len(fragments),
                "total_tokens_tracked": total_tokens,
                "avg_entropy_score": round(avg_entropy, 4),
                "pinned_fragments": sum(1 for f in fragments if f.is_pinned),
            },
            "savings": {
                "total_tokens_saved": self._total_tokens_saved,
                "total_duplicates_caught": self._total_duplicates_caught,
                "total_optimizations": self._total_optimizations,
                "total_fragments_ingested": self._total_fragments_ingested,
                "estimated_cost_saved_usd": round(
                    self._total_tokens_saved * 0.000003, 4
                ),
            },
            "dedup": self._dedup.stats(),
            "prefetch": self._prefetch.stats(),
            "checkpoint": self._checkpoint_mgr.stats(),
        }


# ══════════════════════════════════════════════════════════════════════
# MCP Server Definition
# ══════════════════════════════════════════════════════════════════════

def create_mcp_server():
    """
    Create the MCP server with all tools registered.

    Uses the FastMCP SDK for automatic tool schema generation
    from Python type hints and docstrings.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error(
            "MCP SDK not installed. Install with: pip install mcp"
        )
        raise

    mcp = FastMCP(
        "entroly",
        instructions=(
            "Information-theoretic context optimization for AI coding agents. "
            "Knapsack-optimal token budgeting, Shannon entropy scoring, "
            "SimHash deduplication, predictive pre-fetch, and checkpoint/resume."
        ),
    )

    # Shared engine instance — load autotuned weights if available
    _tuning_cfg = {}
    _tuning_path = Path(__file__).parent.parent / "bench" / "tuning_config.json"
    if _tuning_path.exists():
        try:
            _tuning_cfg = json.loads(_tuning_path.read_text())
            logger.info(f"Loaded autotuned config from {_tuning_path}")
        except Exception as e:
            logger.warning(f"Failed to load tuning_config.json: {e}")

    _config = EntrolyConfig(
        weight_recency=_tuning_cfg.get("weight_recency", 0.30),
        weight_frequency=_tuning_cfg.get("weight_frequency", 0.25),
        weight_semantic_sim=_tuning_cfg.get("weight_semantic_sim", 0.25),
        weight_entropy=_tuning_cfg.get("weight_entropy", 0.20),
        decay_half_life_turns=_tuning_cfg.get("decay_half_life_turns", 15),
        min_relevance_threshold=_tuning_cfg.get("min_relevance_threshold", 0.05),
    )
    engine = EntrolyEngine(config=_config)

    # Cross-session feedback journal + task-conditioned profiles
    _checkpoint_dir = os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly"))
    _feedback_journal = FeedbackJournal(_checkpoint_dir)
    _task_profiles = TaskProfileOptimizer(_feedback_journal)
    _task_profiles.optimize_all()  # warm from existing journal
    _last_opt_ctx = {}  # tracks last optimization for feedback attribution
    _vault_beliefs_loaded = False  # lazy: load vault beliefs on first optimize

    @mcp.tool()
    def remember_fragment(
        content: str,
        source: str = "",
        token_count: int = 0,
        is_pinned: bool = False,
    ) -> str:
        """Store a context fragment with automatic dedup and entropy scoring.

        Fragments are fingerprinted via SimHash for O(1) duplicate detection.
        Each fragment's information density is scored using Shannon entropy.
        Duplicates are automatically merged with salience boosting.

        Args:
            content: The text content to store (code, tool output, etc.)
            source: Origin label (e.g., 'file:utils.py', 'tool:grep')
            token_count: Token count (auto-estimated if 0)
            is_pinned: If True, always include in optimized context
        """
        # NOTE: turn is NOT advanced here — turns advance on optimize/recall
        result = engine.ingest_fragment(content, source, token_count, is_pinned)
        # CodeQualityGuard: scan for secrets, TODOs, unsafe blocks
        issues = engine._guard.scan(content, source)
        if issues:
            result["quality_issues"] = issues
        return json.dumps(result, indent=2)

    @mcp.tool()
    def optimize_context(
        token_budget: int = 128000,
        query: str = "",
    ) -> str:
        """Select the mathematically optimal context subset for a token budget.

        Uses 0/1 Knapsack dynamic programming to maximize relevance within
        the budget. Scores fragments on four dimensions: recency (Ebbinghaus
        decay), access frequency (spaced repetition), semantic similarity
        (SimHash), and information density (Shannon entropy).

        QUERY REFINEMENT: Vague queries like "fix the bug" or "add feature"
        are automatically expanded into precise master prompts using the files
        already in memory. This improves context selection accuracy and reduces
        hallucination from selecting wrong files. The response includes
        query_refinement.refined_query so you can see what drove selection.

        Output is ordered for optimal LLM attention: pinned/critical first,
        high-dependency foundation files early, then by relevance.

        This is the core tool — call it before sending context to the LLM.

        Args:
            token_budget: Maximum tokens allowed (default: 128K)
            query: Current query/task for semantic relevance scoring (can be vague)
        """
        nonlocal _last_opt_ctx
        engine._turn_counter += 1
        engine.advance_turn()  # One turn per optimization request

        # ── Signal activity to Evolution Daemon (gates dreaming) ──
        try:
            _evolution_daemon.record_activity()
        except Exception:
            pass

        # ── Vault Belief Bridge: lazy load on first optimize ──
        # Scan vault/beliefs/*.md, match to ingested fragments by basename,
        # and attach belief content so IOS can select at Belief resolution.
        # This gives 5-10x token savings: ~200-token belief REPLACES ~800-token code.
        nonlocal _vault_beliefs_loaded
        if not _vault_beliefs_loaded and engine._use_rust:
            vault_beliefs_dir = os.path.join(
                os.environ.get("ENTROLY_VAULT", os.path.join(
                    os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")),
                    "vault"
                )),
                "beliefs",
            )
            if os.path.isdir(vault_beliefs_dir) and hasattr(engine._rust, "load_vault_beliefs"):
                try:
                    n = engine._rust.load_vault_beliefs(vault_beliefs_dir)
                    if n > 0:
                        logger.info(f"Vault beliefs bridge: attached {n} beliefs to fragments")
                except Exception as e:
                    logger.debug(f"Vault belief loading failed: {e}")
            _vault_beliefs_loaded = True

        # Apply task-conditioned weights before optimization
        task_type, task_confidence = _task_profiles.apply_to_engine(engine, query)

        result = engine.optimize_context(token_budget, query)

        # ── Record savings to ValueTracker (funds evolution budget) ──
        tokens_saved = result.get("tokens_saved", 0)
        if tokens_saved > 0:
            try:
                _value_tracker.record(
                    tokens_saved=tokens_saved,
                    model=result.get("model", ""),
                    duplicates=result.get("duplicates_caught", 0),
                    optimized=True,
                )
            except Exception:
                pass  # Never fail the optimization for tracking

        # Capture optimization context for feedback attribution
        _last_opt_ctx = {
            "weights": {
                "w_r": _config.weight_recency, "w_f": _config.weight_frequency,
                "w_s": _config.weight_semantic_sim, "w_e": _config.weight_entropy,
            },
            "query": query, "token_budget": token_budget,
            "selected_count": result.get("selected_count", 0),
            "turn": engine._turn_counter,
            "task_type": task_type,
        }
        result["_task_profile"] = {"task_type": task_type, "confidence": task_confidence}

        # Build ContextProvenance (hallucination_risk, source_set, per-fragment risk)
        provenance = build_provenance(
            optimize_result=result,
            query=result.get("query", query),
            refined_query=result.get("query_refinement", {}).get("refined_query") if isinstance(result.get("query_refinement"), dict) else None,
            turn=engine._turn_counter,
            token_budget=token_budget,
            quality_scan_fn=engine._guard.scan if engine._guard.available else None,
        )
        result["provenance"] = provenance.to_dict()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def recall_relevant(
        query: str,
        top_k: int = 5,
    ) -> str:
        """Semantic recall of the most relevant stored fragments.

        Uses SimHash fingerprint distance + multi-dimensional scoring
        with feedback loop (fragments that previously led to successful
        outputs are boosted).

        Args:
            query: The search query
            top_k: Number of results to return
        """
        results = engine.recall_relevant(query, top_k)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def record_outcome(
        fragment_ids: str,
        success: bool = True,
    ) -> str:
        """Record whether selected fragments led to a successful output.

        This feeds the reinforcement learning loop: fragments that
        contribute to successful outputs get boosted in future selections,
        while unhelpful fragments get suppressed.

        Args:
            fragment_ids: Comma-separated fragment IDs
            success: True if output was good, False if bad
        """
        ids = [fid.strip() for fid in fragment_ids.split(",") if fid.strip()]
        if success:
            engine.record_success(ids)
        else:
            engine.record_failure(ids)

        # Log to cross-session feedback journal for autotune
        if _last_opt_ctx:
            _feedback_journal.log(
                weights=_last_opt_ctx.get("weights", {}),
                reward=1.0 if success else -1.0,
                selected_count=_last_opt_ctx.get("selected_count", 0),
                query=_last_opt_ctx.get("query", ""),
                token_budget=_last_opt_ctx.get("token_budget", 0),
                turn=_last_opt_ctx.get("turn", 0),
            )
            # Re-optimize task profiles periodically
            if _feedback_journal.count() % 5 == 0:
                _task_profiles.optimize_all()

        return json.dumps({
            "status": "recorded",
            "fragment_ids": ids,
            "outcome": "success" if success else "failure",
        }, indent=2)

    @mcp.tool()
    def explain_context() -> str:
        """Explain why each fragment was included or excluded in the last optimization.

        Shows per-fragment scoring breakdowns with all dimensions visible:
        recency, frequency, semantic, entropy, feedback multiplier,
        dependency boost, criticality, and composite score.

        Also shows context sufficiency (what % of referenced symbols
        have definitions included) and any exploration swaps.

        Call this after optimize_context to understand selection decisions.
        """
        result = engine.explain_selection()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def checkpoint_state(
        task_description: str = "",
        current_step: str = "",
    ) -> str:
        """Save current state to disk for crash recovery and session resume.

        Checkpoints include all fragments, dedup index, co-access patterns,
        and custom metadata. Stored as gzipped JSON (~50-200 KB).

        Args:
            task_description: What the agent is working on
            current_step: Where in the task it currently is
        """
        metadata = {}
        if task_description:
            metadata["task"] = task_description
        if current_step:
            metadata["step"] = current_step

        path = engine.checkpoint(metadata)
        return json.dumps({
            "status": "checkpoint_saved",
            "path": path,
        }, indent=2)

    @mcp.tool()
    def resume_state() -> str:
        """Resume from the latest checkpoint.

        Restores all context fragments, dedup index, co-access patterns,
        and custom metadata from the most recent checkpoint.
        """
        result = engine.resume()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def prefetch_related(
        file_path: str,
        source_content: str = "",
        language: str = "python",
    ) -> str:
        """Predict and pre-load context that will likely be needed next.

        Combines static analysis (imports, callees, test files) with
        learned co-access patterns to predict what the agent will need.

        Args:
            file_path: The file currently being accessed
            source_content: The source code content (for static analysis)
            language: Programming language (python, typescript, rust)
        """
        predictions = engine.prefetch_related(file_path, source_content, language)
        return json.dumps(predictions, indent=2)


    @mcp.tool()
    def get_stats() -> str:
        """Get comprehensive session statistics.

        Shows token savings, duplicate detection counts, entropy
        distribution, dependency graph stats, checkpoint status,
        and cost estimates.
        """
        stats = engine.get_stats()
        return json.dumps(stats, indent=2)

    @mcp.tool()
    def entroly_dashboard() -> str:
        """Show the real, live value Entroly is providing to YOUR session right now.

        Pulls from actual engine state — not synthetic data. Shows:
            Money saved: exact $ amounts from token optimization
            Performance: sub-millisecond selection speed vs API latency
            Bloat prevention: context compression ratio and memory footprint
            Selection quality: per-fragment scoring and context sufficiency
            Safety: duplicates caught, stale fragments filtered

        Call this anytime to see exactly what Entroly is doing for you.
        """
        stats = engine.get_stats()
        explanation = engine.explain_selection()

        # ── Real session metrics ──
        session = stats.get("session", {})
        savings = stats.get("savings", {})
        dep = stats.get("dep_graph", {})
        perf = stats.get("performance", {})
        mem = stats.get("memory", {})
        ctx_eff = stats.get("context_efficiency", {})
        checkpoint = stats.get("checkpoint", {})

        total_frags = session.get("total_fragments", 0)
        total_tokens = session.get("total_tokens_tracked", 0)
        current_turn = session.get("current_turn", 0)
        pinned = session.get("pinned", 0)

        tokens_saved = savings.get("total_tokens_saved", 0)
        dupes = savings.get("total_duplicates_caught", 0)
        total_opts = savings.get("total_optimizations", 0)
        total_ingested = savings.get("total_fragments_ingested", 0)

        # ── 💰 MONEY ──
        naive_cost = mem.get("naive_cost_per_call_usd", 0)
        optimized_cost = mem.get("optimized_cost_per_call_usd", 0)
        cost_saved_usd = savings.get("estimated_cost_saved_usd", 0)
        savings_pct = ((naive_cost - optimized_cost) / max(naive_cost, 1e-9)) * 100 if naive_cost > 0 else 0
        session_roi = naive_cost * total_opts - optimized_cost * total_opts

        # ── ⚡ PERFORMANCE ──
        avg_us = perf.get("avg_optimize_us", 0)
        peak_us = perf.get("peak_optimize_us", 0)
        avg_ms = avg_us / 1000
        # Typical API call is 500-3000ms; show the multiplier
        api_latency_ms = 2000  # typical GPT-4 API latency
        speedup = api_latency_ms / max(avg_ms, 0.001) if avg_ms > 0 else 0

        # ── 🧠 BLOAT PREVENTION ──
        compression = perf.get("context_compression", 1.0)
        bloat_prevented_pct = max(0, (1 - compression) * 100)
        mem_kb = mem.get("total_kb", 0)
        content_kb = mem.get("content_kb", 0)

        # ── 🎯 QUALITY ──
        info_efficiency = ctx_eff.get("context_efficiency", 0)
        dedup_rate = (dupes / max(total_ingested, 1)) * 100

        # ── Last optimization breakdown ──
        last_opt = None
        if not explanation.get("error"):
            included = [dict(f) for f in explanation.get("included", [])]
            excluded = [dict(f) for f in explanation.get("excluded", [])]
            sufficiency = explanation.get("sufficiency", 0)

            selected_summary = []
            for frag in included:
                scores = dict(frag.get("scores", {}))
                selected_summary.append({
                    "source": frag.get("source", ""),
                    "score": scores.get("composite", 0),
                    "top_signal": max(
                        [("recency", scores.get("recency", 0)),
                         ("semantic", scores.get("semantic", 0)),
                         ("entropy", scores.get("entropy", 0)),
                         ("frequency", scores.get("frequency", 0))],
                        key=lambda x: x[1]
                    )[0],
                    "reason": frag.get("reason", ""),
                })

            excluded_summary = []
            for frag in excluded[:5]:
                scores = dict(frag.get("scores", {}))
                excluded_summary.append({
                    "source": frag.get("source", ""),
                    "score": scores.get("composite", 0),
                    "reason": frag.get("reason", ""),
                })

            last_opt = {
                "context_sufficiency": f"{sufficiency:.0%}",
                "selected": len(included),
                "excluded": len(excluded),
                "fragments_selected": selected_summary,
                "fragments_excluded": excluded_summary,
            }

        dashboard = {
            "💰 money": {
                "tokens_saved_total": f"{tokens_saved:,}",
                "cost_saved_total_usd": f"${cost_saved_usd:.4f}",
                "cost_per_call_without_entroly": f"${naive_cost:.4f}",
                "cost_per_call_with_entroly": f"${optimized_cost:.4f}",
                "savings_pct": f"{savings_pct:.0f}%",
                "session_roi_usd": f"${session_roi:.4f}",
                "insight": (
                    f"Each optimize call costs ${optimized_cost:.4f} instead of ${naive_cost:.4f}. "
                    f"Over {total_opts} calls, that's ${session_roi:.4f} saved."
                    if total_opts > 0 else "Run optimize_context to see savings."
                ),
            },
            "⚡ performance": {
                "avg_optimize_latency": f"{avg_us:.0f}µs ({avg_ms:.2f}ms)",
                "peak_optimize_latency": f"{peak_us:.0f}µs",
                "vs_api_roundtrip": f"{speedup:.0f}x faster than a typical API call" if speedup > 0 else "N/A",
                "total_optimizations": total_opts,
                "insight": (
                    f"Context selection takes {avg_us:.0f}µs — that's {speedup:.0f}x faster "
                    f"than waiting for an API response."
                    if avg_us > 0 else "No optimizations run yet."
                ),
            },
            "🧠 bloat_prevention": {
                "total_tokens_in_memory": f"{total_tokens:,}",
                "context_compression": f"{compression:.2%}" if compression < 1 else "N/A (no optimize yet)",
                "bloat_filtered": f"{bloat_prevented_pct:.0f}% of context is noise that gets filtered",
                "duplicates_caught": f"{dupes} ({dedup_rate:.0f}% dedup rate)",
                "memory_footprint": f"{mem_kb} KB ({content_kb} KB content + {mem_kb - content_kb} KB metadata)",
                "insight": (
                    f"Entroly keeps {total_frags} fragments in {mem_kb} KB of memory. "
                    f"Without dedup, {dupes} duplicate fragments would bloat your context by "
                    f"~{dupes * (total_tokens // max(total_frags, 1)):,} extra tokens."
                    if total_frags > 0 else "Ingest some code to see memory stats."
                ),
            },
            "🎯 selection_quality": {
                "information_density": f"{info_efficiency:.4f} bits/token",
                "avg_entropy": f"{session.get('avg_entropy', 0):.4f}",
                "fragments_tracked": total_frags,
                "pinned_fragments": pinned,
                "dependency_edges": dep.get("edges", dep.get("total_edges", 0)),
                "turns_processed": current_turn,
                "insight": (
                    f"Entroly ranks {total_frags} fragments across {current_turn} turns. "
                    f"Information density: {info_efficiency:.4f} bits/token — higher = "
                    f"more valuable context per token spent."
                    if total_frags > 0 else "Ingest code to see quality metrics."
                ),
            },
            "🔒 safety": {
                "duplicates_blocked": dupes,
                "stale_fragments_deprioritized": "Ebbinghaus decay active (half-life: 15 turns)",
                "persistent_index": "active" if hasattr(engine, '_index_path') else "disabled",
                "checkpoints": checkpoint.get("total_checkpoints", 0),
            },
        }

        if last_opt:
            dashboard["📊 last_optimization"] = last_opt

        return json.dumps(dashboard, indent=2)


    @mcp.tool()
    def scan_for_vulnerabilities(content: str, source: str = "unknown") -> str:
        """Scan code content for security vulnerabilities (SAST analysis).

        Uses a 55-rule engine with taint-flow simulation and CVSS-inspired
        scoring. Detects hardcoded secrets, SQL injection, path traversal,
        command injection, insecure cryptography, unsafe deserialization,
        XSS, and authentication misconfigurations.

        Args:
            content: The source code to scan.
            source:  File path / identifier (used for language detection
                     and confidence scoring). E.g. "auth/login.py".

        Returns JSON with:
            - findings: [{rule_id, cwe, severity, line_number, description,
                          fix, confidence, taint_flow}]
            - risk_score: CVSS-inspired aggregate [0.0, 10.0]
            - critical_count, high_count, medium_count, low_count
            - top_fix: most impactful remediation action
        """
        if engine._use_rust:
            return _scan_via_rust_standalone(content, source)
        # Python fallback — basic pattern matching
        return _sast_python_fallback(content, source)

    def _scan_via_rust_standalone(content: str, source: str) -> str:
        """Use the module-level py_scan_content function from entroly_core."""
        try:
            from entroly_core import py_scan_content
            return py_scan_content(content, source)
        except Exception as e:
            return json.dumps({"error": str(e), "findings": [], "risk_score": 0.0})

    def _sast_python_fallback(content: str, source: str) -> str:
        """Minimal Python SAST fallback when Rust is unavailable."""
        findings = []
        lines = content.splitlines()
        SIMPLE_RULES = [
            ("SEC-001", "CWE-798", "Critical", "password", "=", "Hardcoded password"),
            ("SQL-001", "CWE-89",  "Critical", "execute(",  "%s", "SQL injection"),
            ("CMD-001", "CWE-78",  "Critical", "os.system(", None, "Command injection"),
            ("DESER-001", "CWE-502","Critical","pickle.loads(", None, "Unsafe deserialization"),
            ("CRYPTO-001","CWE-327","High",    "md5",        None, "Broken hash"),
        ]
        for i, line in enumerate(lines, 1):
            lower = line.lower()
            for rule_id, cwe, sev, pat, req, desc in SIMPLE_RULES:
                if pat in lower and (req is None or req in lower):
                    findings.append({
                        "rule_id": rule_id, "cwe": cwe, "severity": sev,
                        "line_number": i, "description": desc,
                        "line_content": line.strip(), "confidence": 0.7,
                        "taint_flow": False, "fix": "See OWASP for remediation guidance.",
                    })
        risk = min(10.0, sum(9.5 if f["severity"] == "Critical" else 6.5 for f in findings) * 0.25)
        return json.dumps({
            "source": source, "findings": findings, "risk_score": round(risk, 2),
            "critical_count": sum(1 for f in findings if f["severity"] == "Critical"),
            "high_count": sum(1 for f in findings if f["severity"] == "High"),
            "medium_count": 0, "low_count": 0,
        }, indent=2)

    @mcp.tool()
    def security_report() -> str:
        """Generate a session-wide security audit across all ingested fragments.

        Scans every fragment in the current session and returns an aggregated
        report showing: which fragments are most vulnerable, overall risk posture,
        finding distribution by category, and the single most important fix.

        Returns JSON with:
            - fragments_scanned, fragments_with_findings
            - critical_total, high_total, max_risk_score
            - most_vulnerable_fragment (fragment_id)
            - findings_by_category: {category: count}
            - vulnerable_fragments: sorted list by risk_score
        """
        if engine._use_rust:
            return engine._rust.security_report()
        # Python fallback: scan all fragments individually
        results = []
        for fid, frag in engine._fragments.items():
            raw = json.loads(_sast_python_fallback(frag.content, frag.source))
            if raw.get("findings"):
                results.append({"fragment_id": fid, "source": frag.source,
                                 "risk_score": raw["risk_score"],
                                 "finding_count": len(raw["findings"])})
        results.sort(key=lambda r: r["risk_score"], reverse=True)
        return json.dumps({
            "fragments_scanned": len(engine._fragments),
            "fragments_with_findings": len(results),
            "vulnerable_fragments": results,
        }, indent=2)

    @mcp.tool()
    def analyze_codebase_health() -> str:
        """Analyze the health of the ingested codebase.

        Runs 5 analysis passes over all fragments in the current session:
          1. Clone Detection — SimHash pairwise scan for Type-1/2/3 code clones
          2. Dead Symbol Analysis — defined but never referenced symbols
          3. God File Detection — files with > μ+2σ reverse dependencies
          4. Architecture Violation Detection — cross-layer imports
          5. Naming Convention Analysis — Python/Rust/React convention breaks

        Returns a JSON HealthReport with:
            - code_health_score [0–100] and health_grade (A/B/C/D/F)
            - Per-dimension scores: duplication, dead_code, coupling, arch, naming
            - clone_pairs, dead_symbols, god_files, arch_violations, naming_issues
            - summary (human-readable) and top_recommendation (most impactful action)
        """
        if engine._use_rust:
            return engine._rust.analyze_health()
        # Python fallback: basic clone detection only
        frags = list(engine._fragments.values())
        from .dedup import simhash as _simhash
        clone_pairs = []
        for i, a in enumerate(frags):
            for b in frags[i+1:]:
                if a.source == b.source:
                    continue
                ha = _simhash(a.content)
                hb = _simhash(b.content)
                dist = bin(ha ^ hb).count("1")
                if dist <= 8:
                    sim = round(1.0 - dist / 64.0, 4)
                    clone_pairs.append({"source_a": a.source, "source_b": b.source,
                                        "similarity": sim, "clone_type": "Type-1/2"})
        score = max(0.0, 100.0 - len(clone_pairs) * 5.0)
        return json.dumps({
            "fragment_count": len(frags),
            "clone_pairs": clone_pairs,
            "code_health_score": round(score, 1),
            "health_grade": "A" if score >= 90 else "B" if score >= 80 else "C",
            "summary": f"{len(frags)} fragments analyzed. {len(clone_pairs)} clone pairs found.",
        }, indent=2)

    @mcp.tool()
    def ingest_diagram(diagram_text: str, source: str, diagram_type: str = "auto") -> str:
        """Ingest an architecture or flow diagram into the context memory.

        Converts Mermaid, PlantUML, DOT/Graphviz, or informal diagram text into
        a structured semantic fragment capturing nodes, edges, and relationships.
        The result is stored as a normal context fragment and is retrievable
        by optimize_context and recall_relevant.

        Args:
            diagram_text: Raw diagram source (Mermaid/PlantUML/DOT/text description).
            source:       Identifier (e.g., 'arch_overview.mmd', 'db_schema.puml').
            diagram_type: 'mermaid', 'plantuml', 'dot', 'text', or 'auto' (default).

        Returns JSON with ingestion result (same as remember_fragment).
        """
        modal = _mm_diagram(diagram_text, source, diagram_type)
        data = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        data["modal_source_type"] = "diagram"
        data["diagram_type"] = diagram_type
        data["nodes_extracted"] = modal.metadata.get("node_count", 0)
        data["edges_extracted"] = modal.metadata.get("edge_count", 0)
        data["extraction_confidence"] = modal.confidence
        return json.dumps(data, indent=2)

    @mcp.tool()
    def ingest_voice(transcript: str, source: str) -> str:
        """Ingest a voice/meeting transcript into the context memory.

        Converts pre-transcribed text (from Whisper, AssemblyAI, etc.) into a
        structured fragment capturing decisions, action items, open questions,
        technical vocabulary, and key discussion excerpts.

        Args:
            transcript: The full transcript text.
            source:     Identifier (e.g., 'design_meeting_2026-03-07.txt').

        Returns JSON with ingestion result plus:
            - decisions, actions, open_questions (counts)
            - tech_terms_identified
        """
        modal = _mm_voice(transcript, source)
        data = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        data["modal_source_type"] = "voice"
        data["decisions_extracted"] = modal.metadata.get("decisions", 0)
        data["actions_extracted"] = modal.metadata.get("actions", 0)
        data["tech_terms"] = modal.metadata.get("tech_terms", 0)
        data["extraction_confidence"] = modal.confidence
        return json.dumps(data, indent=2)

    @mcp.tool()
    def ingest_diff(diff_text: str, source: str, commit_message: str = "") -> str:
        """Ingest a code diff/patch into the context memory.

        Converts a unified diff (git diff output) into a structured change summary:
        intent classification (bug-fix/feature/refactor), symbols changed,
        files modified, and line delta. Particularly useful for understanding
        recent changes and their architectural impact.

        Args:
            diff_text:      Raw unified diff text (git diff output).
            source:         Identifier (e.g., 'pr_42_auth_refactor.diff').
            commit_message: Optional commit message for better intent classification.

        Returns JSON with ingestion result plus:
            - intent: bug-fix/feature/refactor/test/security/performance
            - files_changed, added_lines, removed_lines
            - symbols_changed: functions/classes modified
        """
        modal = _mm_diff(diff_text, source, commit_message)
        data = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        data["modal_source_type"] = "diff"
        data["intent"] = modal.metadata.get("intent", "unknown")
        data["files_changed"] = modal.metadata.get("files_changed", 0)
        data["added_lines"] = modal.metadata.get("added_lines", 0)
        data["removed_lines"] = modal.metadata.get("removed_lines", 0)
        data["symbols_changed"] = modal.metadata.get("symbols_changed", [])
        return json.dumps(data, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # CogOps: Epistemic Router + Vault (ADDITIVE — existing tools untouched)
    # ══════════════════════════════════════════════════════════════════

    # Initialize the epistemic router and vault manager
    _vault_base = os.environ.get(
        "ENTROLY_VAULT",
        os.path.join(os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")), "vault"),
    )
    _vault_mgr = VaultManager(VaultConfig(base_path=_vault_base))
    _epistemic_router = EpistemicRouter(
        vault_path=_vault_base,
        miss_threshold=3,
        freshness_hours=24.0,
        min_confidence=0.6,
    )

    @mcp.tool()
    def epistemic_route(
        query: str,
        is_event: bool = False,
        event_type: str = "",
    ) -> str:
        """Route a query through the CogOps Epistemic Ingress Controller.

        Inspects 4 signals (intent, belief coverage, freshness, risk) and
        selects one of 5 canonical flows:

          ① Fast Answer:          Belief → Action (fresh, verified, low-risk)
          ② Verify Before Answer: Belief → Verification → Action (stale/risky)
          ③ Compile On Demand:    Truth → Belief → Verification → Action (no beliefs)
          ④ Change-Driven:        Event → Truth → Belief → ... (PR/commit/incident)
          ⑤ Self-Improvement:     Misses → Evolution → Belief (repeated failures)

        Call this BEFORE optimize_context to understand how the system should
        approach your query. Existing tools work exactly as before.

        Args:
            query: The user query or event description
            is_event: True if this is a change-driven event (PR, commit, etc.)
            event_type: Type of event (pr, commit, release, incident, scheduled)
        """
        decision = _epistemic_router.route(
            query=query,
            is_event=is_event,
            event_type=event_type or None,
        )
        return json.dumps(decision.to_dict(), indent=2)

    @mcp.tool()
    def vault_status() -> str:
        """Show the current state of the CogOps Knowledge Vault.

        Initializes the vault directory structure if needed, then returns
        a coverage index: total beliefs, verification status, confidence
        distribution, and routing statistics.

        The vault is the persistent Living Exocortex — the system's
        machine-auditable understanding of your codebase.
        """
        init_result = _vault_mgr.ensure_structure()
        coverage = _vault_mgr.coverage_index()
        routing_stats = _epistemic_router.stats()

        return json.dumps({
            "vault": init_result,
            "coverage": coverage,
            "routing": routing_stats,
        }, indent=2)

    @mcp.tool()
    def vault_write_belief(
        entity: str,
        title: str,
        body: str,
        confidence: float = 0.7,
        status: str = "inferred",
        sources: str = "",
        derived_from: str = "",
    ) -> str:
        """Write a belief artifact to the CogOps Knowledge Vault.

        Beliefs are durable system understanding — what Entroly thinks
        the codebase is. Each belief carries machine-auditable frontmatter:
        claim_id, entity, status, confidence, sources, last_checked.

        Args:
            entity: The system entity this belief is about (e.g., 'auth::token_rotation')
            title: Human-readable title
            body: The belief content (markdown)
            confidence: Machine-assigned confidence 0.0-1.0 (default: 0.7)
            status: observed|inferred|verified|stale|hypothesis (default: inferred)
            sources: Comma-separated source paths (e.g., 'src/auth.rs:142,src/token.rs:58')
            derived_from: Comma-separated component names that produced this belief
        """
        artifact = BeliefArtifact(
            entity=entity,
            title=title,
            body=body,
            confidence=confidence,
            status=status,
            sources=[s.strip() for s in sources.split(",") if s.strip()] if sources else [],
            derived_from=[d.strip() for d in derived_from.split(",") if d.strip()] if derived_from else [],
        )
        result = _vault_mgr.write_belief(artifact)
        result["artifact"] = artifact.to_dict()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def vault_query(
        entity: str = "",
        list_all: bool = False,
    ) -> str:
        """Query the CogOps Knowledge Vault for existing beliefs.

        Use this to check what the system already knows before compiling
        new understanding. Supports lookup by entity name or listing all.

        Args:
            entity: Entity name to look up (fuzzy match)
            list_all: If True, return all beliefs with frontmatter summary
        """
        if list_all:
            beliefs = _vault_mgr.list_beliefs()
            return json.dumps({"beliefs": beliefs, "total": len(beliefs)}, indent=2)

        if entity:
            result = _vault_mgr.read_belief(entity)
            if result:
                return json.dumps(result, indent=2)
            return json.dumps({"status": "not_found", "entity": entity}, indent=2)

        # Default: return coverage index
        return json.dumps(_vault_mgr.coverage_index(), indent=2)

    @mcp.tool()
    def vault_write_action(
        title: str,
        content: str,
        action_type: str = "report",
    ) -> str:
        """Write a task output or report to the CogOps Knowledge Vault.

        Action artifacts are developer-facing outputs: PR briefs, answers,
        architecture diagrams, slide decks, task reports. They live in
        actions/ and are timestamped for traceability.

        Args:
            title: Title of the output
            content: Full markdown content
            action_type: Type tag (report, pr_brief, answer, diagram, context_pack)
        """
        result = _vault_mgr.write_action(title, content, action_type)
        return json.dumps(result, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # CogOps Phase 2: Data Plane Engines (Rust preferred, Python fallback)
    #
    # Rust engine handles all heavy computation. Python fallback ensures
    # tools are always available for users without entroly_core installed.
    # WASM/JS users are unaffected — CogOps is Python/Rust only.
    #
    # Epistemic layers:
    #   Truth  → compile_beliefs (entity extraction, dependency resolution)
    #   Belief → vault_write_belief, vault_query (beliefs/, frontmatter)
    #   Verification → verify_beliefs, blast_radius (contradictions, staleness)
    #   Action → execute_flow, process_change, coverage_gaps (PR briefs, flows)
    #   Evolution → create_skill, manage_skills, refresh_beliefs (skills, promotion)
    # ══════════════════════════════════════════════════════════════════

    _source_dir = os.environ.get("ENTROLY_SOURCE", os.getcwd())

    try:
        from entroly_core import CogOpsEngine as _RustCogOps
        _cogops = _RustCogOps(_vault_base, miss_threshold=3, freshness_hours=24.0, min_confidence=0.5)
        _COGOPS_RUST = True
        logger.info("CogOps: Rust engine loaded")
    except ImportError:
        _cogops = None
        _COGOPS_RUST = False
        logger.info("CogOps: using Python fallback (entroly_core not installed)")

    # Python fallback engines — always initialized so tools work without Rust
    _py_compiler = BeliefCompiler(_vault_mgr)
    _py_verifier = VerificationEngine(_vault_mgr, freshness_hours=24.0, min_confidence=0.5)
    _py_change_pipe = ChangePipeline(_vault_mgr, _py_verifier)
    _py_skill_engine = SkillEngine(_vault_mgr)
    _py_evolution = EvolutionLogger(vault_path=_vault_base, gap_threshold=3)
    _py_orchestrator = FlowOrchestrator(
        vault=_vault_mgr,
        router=_epistemic_router,
        compiler=_py_compiler,
        verifier=_py_verifier,
        change_pipe=_py_change_pipe,
        evolution=_py_evolution,
        source_dir=_source_dir,
    )

    # ── Evolution Daemon: autonomous self-improvement (3 pillars) ──
    # Pillar 1: ValueTracker funds evolution via "tax on savings"
    # Pillar 2: StructuralSynthesizer creates tools at $0 (CPU-only)
    # Pillar 3: DreamingLoop optimizes weights during idle time
    _value_tracker = get_tracker()
    _cache_aligner = CacheAligner(similarity_threshold=0.90)

    # Universal self-improvement bus — every component logs metrics here
    _component_bus = ComponentFeedbackBus(_checkpoint_dir)

    _evolution_daemon = EvolutionDaemon(
        vault=_vault_mgr,
        evolution_logger=_py_evolution,
        value_tracker=_value_tracker,
        feedback_journal=_feedback_journal,
        rust_engine=engine._rust if engine._use_rust else None,
    )
    _evolution_daemon.start()  # non-blocking background thread
    logger.info("EvolutionDaemon: autonomous self-improvement started")

    # Wire ComponentFeedbackBus to all self-improving components
    _py_orchestrator._component_bus = _component_bus
    engine._prefetch.set_component_bus(_component_bus)

    _py_workspace_listener = WorkspaceChangeListener(
        vault=_vault_mgr,
        compiler=_py_compiler,
        verifier=_py_verifier,
        change_pipe=_py_change_pipe,
        project_dir=_source_dir,
    )

    @mcp.tool()
    def compile_beliefs(
        directory: str = "",
        max_files: int = 200,
    ) -> str:
        """Compile source code into belief artifacts (Truth → Belief pipeline).

        Scans a directory for source files (.py, .rs, .ts, .js), extracts
        code entities (classes, functions, structs, traits, imports),
        resolves cross-file dependencies, and writes belief artifacts to
        the vault with full frontmatter (claim_id, entity, status,
        confidence, sources, last_checked, derived_from).

        Args:
            directory: Path to scan. Defaults to the project root.
            max_files: Maximum files to process (default: 200)
        """
        target = directory or _source_dir
        if _COGOPS_RUST:
            return json.dumps(_cogops.compile_beliefs(target, max_files), indent=2)
        result = _py_compiler.compile_directory(target, max_files)
        return json.dumps({
            "status": "compiled", "files_processed": result.files_processed,
            "beliefs_written": result.beliefs_written,
            "entities_extracted": result.entities_extracted,
            "errors": result.errors[:10], "engine": "python",
        }, indent=2)

    @mcp.tool()
    def verify_beliefs() -> str:
        """Run a full verification pass on all beliefs in the vault.

        Checks for:
        - Staleness (beliefs past their freshness window)
        - Contradictions (conflicting claims about the same entity)
        - Confidence divergence between same-entity beliefs
        - Low confidence scores

        Writes verification artifacts to vault/verification/.
        """
        if _COGOPS_RUST:
            return json.dumps(_cogops.verify_beliefs(), indent=2)
        report = _py_verifier.full_verification_pass()
        return json.dumps({**report.to_dict(), "engine": "python"}, indent=2)

    @mcp.tool()
    def blast_radius(changed_files: str) -> str:
        """Analyze the blast radius of file changes on existing beliefs.

        Given a list of changed files, determines which beliefs need
        re-verification, which may be invalidated, and the overall risk
        level (low/medium/high).

        Args:
            changed_files: Comma-separated list of changed file paths
        """
        files = [f.strip() for f in changed_files.split(",") if f.strip()]
        if _COGOPS_RUST:
            return json.dumps(_cogops.blast_radius(files), indent=2)
        br = _py_verifier.blast_radius(files)
        return json.dumps({
            "affected_beliefs": br.affected_beliefs, "affected_entities": br.affected_entities,
            "risk_level": br.risk_level, "description": br.description, "engine": "python",
        }, indent=2)

    @mcp.tool()
    def process_change(
        diff_text: str,
        commit_message: str = "",
        pr_title: str = "",
    ) -> str:
        """Process a code change through the Change-Driven pipeline (Flow ④).

        Full pipeline: Diff → ChangeSet → Review → Blast Radius → Vault

        Classifies intent (bugfix/feature/refactor/test/security/performance),
        runs code review (hardcoded secrets, TODOs, broad exceptions, unsafe),
        computes belief impact, and returns a structured PR brief.

        Args:
            diff_text: Raw unified diff text (git diff output)
            commit_message: Optional commit message for intent classification
            pr_title: Optional PR title
        """
        if _COGOPS_RUST:
            return json.dumps(_cogops.process_change(diff_text, commit_message, pr_title), indent=2)
        brief = _py_change_pipe.process_diff(diff_text, commit_message, pr_title)
        return json.dumps({
            "title": brief.title, "summary": brief.summary, "risk_level": brief.risk_level,
            "intent": brief.changeset.intent,
            "files_modified": brief.changeset.files_modified,
            "lines_added": brief.changeset.lines_added, "lines_removed": brief.changeset.lines_removed,
            "findings_count": len(brief.findings), "engine": "python",
        }, indent=2)

    @mcp.tool()
    def execute_flow(
        query: str,
        diff_text: str = "",
        is_event: bool = False,
        event_type: str = "",
    ) -> str:
        """Execute a full canonical epistemic flow end-to-end.

        Routes the query through the Epistemic Ingress Controller (4 signals:
        intent, belief coverage, freshness, risk), then chains the appropriate
        pipeline steps automatically:

          ① Fast Answer:         Belief → Action
          ② Verify Before Answer: Belief → Verification → Action
          ③ Compile On Demand:   Truth → Belief → Verification → Action
          ④ Change-Driven:       Event → Truth → Belief → Verification → Action
          ⑤ Self-Improvement:    Misses → Verification → Evolution → Belief

        Args:
            query: The user query or event description
            diff_text: Raw diff for change-driven flows (Flow ④)
            is_event: True if this is a change-driven event
            event_type: Type of event (pr, commit, release, incident, scheduled)
        """
        if _COGOPS_RUST:
            return json.dumps(_cogops.execute_flow(query, diff_text, is_event, event_type), indent=2)
        flow_result = _py_orchestrator.execute(
            query=query, diff_text=diff_text, is_event=is_event, event_type=event_type,
        )
        result_dict = flow_result.to_dict()
        result_dict["engine"] = "python"
        return json.dumps(result_dict, indent=2)

    @mcp.tool()
    def create_skill(
        entity_key: str,
        failing_queries: str,
        intent: str = "",
    ) -> str:
        """Create a new skill from a capability gap (Evolution layer).

        When the system repeatedly fails on a topic, this generates a
        full skill package in vault/evolution/skills/<skill-id>/:
        - SKILL.md — procedure/SOP
        - tool.py — executable Python tool
        - metrics.json — fitness tracking
        - tests/test_cases.json — regression tests

        Args:
            entity_key: The entity this skill handles (e.g., 'protobuf_analysis')
            failing_queries: Pipe-separated list of failing queries
            intent: The intent class for this skill
        """
        queries = [q.strip() for q in failing_queries.split("|") if q.strip()]
        if _COGOPS_RUST:
            return json.dumps(_cogops.create_skill(entity_key, queries), indent=2)
        result = _py_skill_engine.create_skill(entity_key, queries, intent)
        result["engine"] = "python"
        return json.dumps(result, indent=2)

    @mcp.tool()
    def manage_skills(
        action: str = "list",
        skill_id: str = "",
    ) -> str:
        """Manage the CogOps skill lifecycle (Evolution layer).

        Actions:
        - list: Show all skills with status, fitness, and run counts
        - benchmark: Run test cases and compute fitness score (0.0-1.0)
        - promote: Promote (fitness >= 0.7) or prune (fitness <= 0.3)

        Args:
            action: list | benchmark | promote
            skill_id: Required for benchmark/promote actions
        """
        if action == "list":
            if _COGOPS_RUST:
                skills = _cogops.list_skills()
                return json.dumps({"skills": list(skills), "total": len(skills)}, indent=2)
            skills = _py_skill_engine.list_skills()
            return json.dumps({"skills": skills, "total": len(skills), "engine": "python"}, indent=2)

        if not skill_id:
            return json.dumps({"error": f"skill_id required for '{action}'"}, indent=2)

        if action == "benchmark":
            if _COGOPS_RUST:
                return json.dumps(_cogops.benchmark_skill(skill_id), indent=2)
            return json.dumps(_py_skill_engine.benchmark_skill(skill_id), indent=2)
        elif action == "promote":
            if _COGOPS_RUST:
                return json.dumps(_cogops.promote_skill(skill_id), indent=2)
            return json.dumps(_py_skill_engine.promote_or_prune(skill_id), indent=2)

        return json.dumps({"error": f"Unknown action '{action}'. Use: list, benchmark, promote"}, indent=2)

    @mcp.tool()
    def coverage_gaps(
        directory: str = "",
    ) -> str:
        """Find source files with no corresponding belief in the vault.

        Scans a directory for source files (.py, .rs, .ts, .js) and checks
        which ones have no belief artifact. Useful for identifying blind
        spots before running compile_beliefs.

        Args:
            directory: Path to scan. Defaults to the project root.
        """
        target = directory or _source_dir
        if _COGOPS_RUST:
            return json.dumps(_cogops.coverage_gaps(target), indent=2)
        gaps = _py_verifier.coverage_gaps(target)
        return json.dumps({
            "gaps": [{"file": g.file_path, "reason": g.reason, "suggested_entity": g.suggested_entity} for g in gaps],
            "total_gaps": len(gaps), "engine": "python",
        }, indent=2)

    @mcp.tool()
    def refresh_beliefs(
        changed_files: str,
    ) -> str:
        """Mark beliefs as stale after file changes (Flow ④ doc-refresh).

        Given changed files, finds related beliefs and marks their status
        as 'stale' so the next verify_beliefs pass will flag them for
        re-compilation.

        Args:
            changed_files: Comma-separated list of changed file paths
        """
        files = [f.strip() for f in changed_files.split(",") if f.strip()]
        if _COGOPS_RUST:
            return json.dumps(_cogops.refresh_beliefs(files), indent=2)
        result = _py_change_pipe.refresh_docs(files)
        result["engine"] = "python"
        return json.dumps(result, indent=2)

    @mcp.tool()
    def sync_workspace_changes(
        directory: str = "",
        force: bool = False,
        max_files: int = 100,
    ) -> str:
        """Synchronize workspace file changes into the belief and verification layers.

        Detects new, modified, and deleted source files, marks affected beliefs stale,
        recompiles changed files into fresh beliefs, runs a verification pass, and writes
        a sync report into actions/.
        """
        listener = _py_workspace_listener
        if directory:
            listener = WorkspaceChangeListener(
                vault=_vault_mgr,
                compiler=_py_compiler,
                verifier=_py_verifier,
                change_pipe=_py_change_pipe,
                project_dir=directory,
            )
        result = listener.scan_once(force=force, max_files=max_files)
        payload = result.to_dict()
        payload["engine"] = "python"
        return json.dumps(payload, indent=2)

    @mcp.tool()
    def repo_file_map(
        format: str = "markdown",
    ) -> str:
        """Return the canonical Entroly file map across the Python, Rust core, and WASM repos.

        Use this to understand ownership boundaries and where logic currently lives.
        Supported formats: markdown, json.
        """
        grouped = build_repo_map(Path(_source_dir).resolve().parents[0])
        if format.lower() == "json":
            serializable = {
                repo: [entry.__dict__ for entry in entries]
                for repo, entries in grouped.items()
            }
            return json.dumps(serializable, indent=2)
        return render_repo_map_markdown(grouped)

    @mcp.tool()
    def start_workspace_listener(
        directory: str = "",
        interval_s: int = 120,
        force_initial: bool = False,
        max_files: int = 100,
    ) -> str:
        """Start a background workspace listener that continuously feeds repo changes into CogOps.

        This is the long-running change-driven bridge from repo activity into Belief CI.
        """
        listener = _py_workspace_listener
        if directory:
            listener = WorkspaceChangeListener(
                vault=_vault_mgr,
                compiler=_py_compiler,
                verifier=_py_verifier,
                change_pipe=_py_change_pipe,
                project_dir=directory,
            )
        result = listener.start(interval_s=interval_s, max_files=max_files, force_initial=force_initial)
        result["engine"] = "python"
        return json.dumps(result, indent=2)

    @mcp.tool()
    def vault_search(
        query: str,
        top_k: int = 5,
    ) -> str:
        """Full-text search across all belief artifacts in the vault.

        Uses TF-IDF ranking with entity-name boosting (3x) to find the
        most relevant beliefs. Much cheaper than listing all beliefs —
        returns only the top matches with excerpts.

        Args:
            query: Natural language search query (e.g., "how does knapsack work?")
            top_k: Maximum number of results to return (default: 5)
        """
        if _COGOPS_RUST:
            results = _cogops.vault_search(query, top_k)
            return json.dumps({"query": query, "results": list(results), "total": len(results), "engine": "rust"}, indent=2)
        # Python fallback: simple substring match
        beliefs_dir = _vault_mgr.config.path / "beliefs"
        query_lower = query.lower()
        matches = []
        for md in sorted(beliefs_dir.rglob("*.md")):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                if query_lower in content.lower():
                    from .vault import _parse_frontmatter
                    fm = _parse_frontmatter(content) or {}
                    matches.append({
                        "entity": fm.get("entity", md.stem),
                        "confidence": float(fm.get("confidence", 0)),
                        "status": fm.get("status", "unknown"),
                    })
            except Exception:
                pass
        return json.dumps({"query": query, "results": matches[:top_k], "total": len(matches), "engine": "python"}, indent=2)

    @mcp.tool()
    def compile_docs(
        directory: str = "",
        max_files: int = 50,
    ) -> str:
        """Compile markdown documentation files into belief artifacts.

        Ingests project-level docs (README.md, ARCHITECTURE.md, docs/,
        CONTRIBUTING.md, etc.) into the vault as documentation beliefs
        with confidence 0.80 (human-authored > machine-inferred code beliefs).

        Args:
            directory: Project root to scan. Defaults to the project root.
            max_files: Maximum doc files to process (default: 50)
        """
        target = directory or _source_dir
        if _COGOPS_RUST:
            return json.dumps(_cogops.compile_docs(target, max_files), indent=2)
        # Python fallback: basic README ingest
        import pathlib
        root = pathlib.Path(target)
        compiled = 0
        entities = []
        for md in root.glob("*.md"):
            stem = md.stem.upper()
            if any(stem.startswith(p) for p in ["README", "ARCHITECTURE", "CONTRIBUTING", "CHANGELOG"]):
                entities.append(f"doc/{md.stem.lower()}")
                compiled += 1
        return json.dumps({"status": "compiled", "docs_found": compiled, "docs_compiled": compiled, "entities": entities, "engine": "python"}, indent=2)

    @mcp.tool()
    def export_training_data(
        output_path: str = "training_data.jsonl",
        format: str = "jsonl",
    ) -> str:
        """Export vault beliefs as JSONL training data for LLM finetuning.

        Generates instruction-following pairs from compiled beliefs:
        question about entity → belief body as answer. Filters out stale
        and low-confidence beliefs. Output is OpenAI-compatible JSONL.

        Uses PRISM scoring dimensions for quality-weighted sampling:
        only beliefs with confidence >= 0.5 and non-stale status are
        included in the training set.

        Args:
            output_path: Path to write JSONL file (default: training_data.jsonl)
            format: Output format, currently only 'jsonl' supported
        """
        if _COGOPS_RUST:
            return json.dumps(_cogops.export_training_data(output_path, format), indent=2)
        # Python fallback
        beliefs_dir = _vault_mgr.config.path / "beliefs"
        from .vault import _extract_body, _parse_frontmatter
        lines = []
        skipped = 0
        for md in sorted(beliefs_dir.rglob("*.md")):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                fm = _parse_frontmatter(content) or {}
                body = _extract_body(content)
                conf = float(fm.get("confidence", 0))
                status = fm.get("status", "")
                if conf < 0.5 or status == "stale":
                    skipped += 1
                    continue
                entity = fm.get("entity", md.stem)
                entry = json.dumps({"messages": [
                    {"role": "system", "content": f"You are an expert on the {entity} codebase."},
                    {"role": "user", "content": f"What does {entity} do?"},
                    {"role": "assistant", "content": body[:2000]},
                ]})
                lines.append(entry)
            except Exception:
                pass
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        return json.dumps({
            "status": "exported", "output_path": output_path, "format": format,
            "beliefs_used": len(lines), "beliefs_skipped": skipped,
            "training_pairs": len(lines), "engine": "python",
        }, indent=2)

    return mcp, engine



def _start_autotune_daemon(engine: EntrolyEngine) -> None:
    """
    Spawn the autotune loop as a daemon background thread.

    Dynamic tuning: weights are hot-reloaded into the running engine
    after each improvement round — no restart needed.

    Daemon threads die automatically when the MCP server exits — no cleanup
    needed. Runs at idle CPU priority so it never interferes with foreground
    tool calls.

    Controlled by tuning_config.json → autotuner.enabled (default: true).
    Set to false to disable background tuning.
    """
    import threading

    # Check if autotuning is enabled in tuning_config.json
    config_path = Path(__file__).parent.parent / "bench" / "tuning_config.json"
    enabled = True
    if config_path.exists():
        try:
            import json as _json
            cfg = _json.loads(config_path.read_text())
            enabled = cfg.get("autotuner", {}).get("enabled", True)
        except Exception:
            pass

    if not enabled:
        logger.info("Autotune: disabled via tuning_config.json")
        return

    # Lock protects engine weight updates from racing with optimize calls
    _weight_lock = threading.Lock()

    def _hot_reload_weights():
        """Read tuning_config.json and push weights into the live engine."""
        try:
            import json as _json
            cfg = _json.loads(config_path.read_text())
            w_r = cfg.get("weight_recency", 0.30)
            w_f = cfg.get("weight_frequency", 0.25)
            w_s = cfg.get("weight_semantic_sim", 0.25)
            w_e = cfg.get("weight_entropy", 0.20)
            if engine._use_rust:
                with _weight_lock:
                    engine._rust.set_weights(w_r, w_f, w_s, w_e)
                logger.info(
                    f"Autotune: hot-reloaded weights -> "
                    f"R={w_r:.2f} F={w_f:.2f} S={w_s:.2f} E={w_e:.2f}"
                )
            return True
        except Exception as e:
            logger.warning(f"Autotune: hot-reload failed: {e}")
            return False

    def _daemon_loop():
        import time
        # Lower this thread's OS scheduling priority (nice +10 on Linux)
        try:
            os.nice(10)
        except (AttributeError, OSError):
            pass  # Windows has no nice()

        try:
            from .autotune import CASES_PATH, run_autotune
            if not CASES_PATH.exists():
                logger.debug(
                    "Autotune: bench/cases.json not found at %s — "
                    "skipping benchmark-based autotune (pip install mode). "
                    "Cross-session RL feedback tuning still active.",
                    CASES_PATH,
                )
                return
            logger.info("Autotune: background self-tuning started (dynamic, low priority)")

            # Run in rounds of 10 iterations, hot-reload after each round
            while True:
                try:
                    run_autotune(iterations=10, bench_only=False)
                    _hot_reload_weights()
                except FileNotFoundError:
                    # bench/cases.json vanished (e.g. pip install mode) — stop silently
                    logger.debug("Autotune: bench/cases.json not found, stopping benchmark loop")
                    return
                except Exception as e:
                    logger.warning(f"Autotune round failed: {e}")
                time.sleep(30)  # 30s cooldown between rounds
        except Exception as e:
            logger.warning("Autotune: background thread exited: %s", e)

    t = threading.Thread(target=_daemon_loop, name="entroly-autotune", daemon=True)
    t.start()
    logger.info("Autotune: daemon thread launched (tid=%d)", t.ident or 0)


def main():
    """Entry point for the entroly MCP server."""
    engine_type = "Rust" if _RUST_AVAILABLE else "Python"
    try:
        from importlib.metadata import version as _pkg_version
        _version = _pkg_version("entroly")
    except Exception:
        _version = "0.6.5"
    logger.info(f"Starting Entroly MCP server v{_version} ({engine_type} engine)")
    mcp, engine = create_mcp_server()

    # Graceful shutdown: persist learned state on exit
    import atexit
    import signal

    def _shutdown_handler(*_args):
        logger.info("Shutdown signal received -- persisting state...")
        try:
            engine.checkpoint()
            logger.info("State persisted successfully")
        except Exception as e:
            logger.warning(f"Failed to persist state on shutdown: {e}")

    atexit.register(_shutdown_handler)
    try:
        signal.signal(signal.SIGTERM, lambda s, f: (_shutdown_handler(), sys.exit(0)))
    except (OSError, AttributeError):
        pass  # SIGTERM not available on Windows

    # Auto-index the project on startup (zero config)
    try:
        from entroly.auto_index import auto_index, start_incremental_watcher
        result = auto_index(engine)
        if result["status"] == "indexed":
            logger.info(
                f"Auto-indexed {result['files_indexed']} files "
                f"({result['total_tokens']:,} tokens) in {result['duration_s']}s"
            )
        # Start background watcher: re-scans for new/modified files every 120s
        start_incremental_watcher(engine)
    except Exception as e:
        logger.warning(f"Auto-index failed (non-fatal): {e}")

    # Start the autotune daemon in the background — zero config needed.
    # It reads/writes only tuning_config.json and runs at nice+10 priority.
    try:
        _start_autotune_daemon(engine)
    except Exception as e:
        logger.warning("Autotune: failed to start daemon: %s", e)

    # Multi-client support: SSE transport enables multiple IDE connections
    transport = os.environ.get("ENTROLY_MCP_TRANSPORT", "stdio")
    if "--sse" in sys.argv or transport == "sse":
        sse_port = int(os.environ.get("ENTROLY_MCP_PORT", "9379"))
        logger.info(f"MCP server running on SSE transport at port {sse_port}")
        logger.info("Multiple clients can connect simultaneously")
        # Set port on the FastMCP settings before running
        mcp.settings.port = sse_port
        try:
            mcp.run(transport="sse")
        except TypeError:
            # Older MCP SDK may not support transport kwarg
            logger.warning("SSE transport not supported by this MCP SDK version, falling back to stdio")
            mcp.run()
    else:
        mcp.run()


if __name__ == "__main__":
    main()
