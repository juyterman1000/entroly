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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EntrolyConfig
from .prefetch import PrefetchEngine
from .checkpoint import CheckpointManager
from .query_refiner import QueryRefiner
from .adaptive_pruner import EntrolyPruner, FragmentGuard
from .provenance import build_provenance, ContextProvenance
from .multimodal import ingest_image as _mm_image, ingest_diagram as _mm_diagram
from .multimodal import ingest_voice as _mm_voice, ingest_diff as _mm_diff
from .proxy_transform import calibrated_token_count as _calibrated_token_count
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

    def __init__(self, config: Optional[EntrolyConfig] = None):
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
            self._fragments: Dict[str, Any] = {}
            self._current_turn: int = 0
            from collections import Counter
            self._global_token_counts: Counter = Counter()
            self._total_token_count: int = 0
            self._dedup = DedupIndex(hamming_threshold=3)
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
            apply_ebbinghaus_decay(
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
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
        """Select the mathematically optimal subset of context fragments."""
        if token_budget <= 0:
            token_budget = self.config.default_token_budget

        # Query refinement: expand vague queries using in-memory file context.
        # This is the key fix for hallucination from incomplete context:
        # "fix the bug" → "bug fix in payments module (Python/Rust) involving
        # payment processing, error handling"
        refined_query = query
        refinement_info: Dict[str, Any] = {}
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
    ) -> List[Dict[str, Any]]:
        """Semantic recall of relevant fragments."""
        if self._use_rust:
            result = self._rust.recall(query, top_k)
            return [dict(r) for r in result]
        else:
            return self._recall_python(query, top_k)

    def record_success(self, fragment_ids: List[str]) -> None:
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

    def record_failure(self, fragment_ids: List[str]) -> None:
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


    def prefetch_related(
        self,
        file_path: str,
        source_content: str = "",
        language: str = "python",
    ) -> List[Dict[str, Any]]:
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

    def checkpoint(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Manually create a checkpoint."""
        return self._auto_checkpoint(metadata)

    def resume(self) -> Dict[str, Any]:
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
            self._dedup = DedupIndex(hamming_threshold=3)
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

    def get_stats(self) -> Dict[str, Any]:
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

    def explain_selection(self) -> Dict[str, Any]:
        """Explain why each fragment was included or excluded."""
        if self._use_rust:
            result = self._rust.explain_selection()
            return dict(result)
        else:
            return {"error": "Explainability requires Rust engine"}

    def _auto_checkpoint(
        self,
        metadata: Optional[Dict[str, Any]] = None,
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
        import os
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
        entropy_score = compute_information_score(
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
            simhash=simhash(content),
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

        result: Dict[str, Any] = {
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
            query_hash = simhash(query)
            from .dedup import hamming_distance
            for frag in self._fragments.values():
                dist = hamming_distance(query_hash, frag.simhash)
                frag.semantic_score = max(0.0, 1.0 - (dist / 64.0))

        # Apply Wilson feedback via the frequency dimension. Wilson learned_value
        # returns [0.5, 2.0] (neutral=1.0). We map this to [0, 1] and use it as
        # the frequency_score, so fragments with positive feedback history get
        # boosted in the knapsack and fragments with negative history get suppressed.
        for frag in self._fragments.values():
            wilson = self._wilson.learned_value(frag.fragment_id)
            frag.frequency_score = max(0.0, min((wilson - 0.5) / 1.5, 1.0))

        fragments = list(self._fragments.values())
        selected, stats = knapsack_optimize(
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
                        compute_relevance(
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

        query_hash = simhash(query)
        from .dedup import hamming_distance

        scored = []
        for frag in self._fragments.values():
            dist = hamming_distance(query_hash, frag.simhash)
            frag.semantic_score = max(0.0, 1.0 - (dist / 64.0))
            relevance = compute_relevance(
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
        engine._turn_counter += 1
        engine.advance_turn()  # One turn per optimization request
        result = engine.optimize_context(token_budget, query)
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
                "stale_fragments_deprioritized": f"Ebbinghaus decay active (half-life: 15 turns)",
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
            return engine._rust.scan_fragment.__func__(engine._rust, source) \
                if False else _scan_via_rust_standalone(content, source)
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
        result = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
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
        result = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
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
        result = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
        data["modal_source_type"] = "diff"
        data["intent"] = modal.metadata.get("intent", "unknown")
        data["files_changed"] = modal.metadata.get("files_changed", 0)
        data["added_lines"] = modal.metadata.get("added_lines", 0)
        data["removed_lines"] = modal.metadata.get("removed_lines", 0)
        data["symbols_changed"] = modal.metadata.get("symbols_changed", [])
        return json.dumps(data, indent=2)

    return mcp, engine



def _start_autotune_daemon(engine: "EntrolyEngine") -> None:
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
    import os
    from pathlib import Path

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
                    f"Autotune: hot-reloaded weights → "
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
            from .autotune import run_autotune
            logger.info("Autotune: background self-tuning started (dynamic, low priority)")

            # Run in rounds of 10 iterations, hot-reload after each round
            while True:
                try:
                    run_autotune(iterations=10, bench_only=False)
                    _hot_reload_weights()
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
        _version = "0.4.4"
    logger.info(f"Starting Entroly MCP server v{_version} ({engine_type} engine)")
    mcp, engine = create_mcp_server()

    # Graceful shutdown: persist learned state on exit
    import atexit
    import signal

    def _shutdown_handler(*_args):
        logger.info("Shutdown signal received — persisting state...")
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
        try:
            mcp.run(transport="sse", port=sse_port)
        except TypeError:
            # Older MCP SDK may not support transport kwarg
            logger.warning("SSE transport not supported by this MCP SDK version, falling back to stdio")
            mcp.run()
    else:
        mcp.run()


if __name__ == "__main__":
    main()
