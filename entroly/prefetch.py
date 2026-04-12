"""
Predictive Context Pre-fetcher
===============================

When an agent reads a code symbol, predict what it will need next
and pre-load it into the context cache — BEFORE it asks.

This is **CPU cache prefetching applied to LLM context windows**.

The Problem:
  An agent debugging function `process_payment()` will inevitably need:
    1. Callers of `process_payment()` — who triggers this?
    2. Callees from `process_payment()` — what does it depend on?
    3. Test file for `process_payment()` — how is it tested?
    4. Type definitions used — what are the data structures?

  Without pre-fetching, the agent makes 4 sequential tool calls,
  each adding latency and token cost. With pre-fetching, these are
  already in the context cache when the agent asks.

Heuristics:
  1. **Static call graph**: Extract function/method calls from source
  2. **Import graph**: Follow import statements to related modules
  3. **Naming conventions**: foo.py → test_foo.py, foo_test.py
  4. **Co-access patterns**: Track which files are accessed together
     across sessions (associative learning)

References:
  - CPU prefetch: Smith, J. "Sequential Program Prefetching" (1978)
  - Agentic Plan Caching (arXiv 2025) — reusing structured plans
  - Proximity (arXiv 2026) — LSH-bucketed pre-warming for caches
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PrefetchResult:
    """A predicted context fragment that might be needed next."""

    path: str
    """File path or symbol identifier."""

    reason: str
    """Why this was predicted (e.g., 'callee', 'test_file', 'co_access')."""

    confidence: float
    """Prediction confidence [0, 1]."""

    content: str | None = None
    """Pre-loaded content (if available)."""


# ── Static Analysis Patterns ───────────────────────────────────────────

# Python function/method calls
_PY_CALL_RE = re.compile(
    r"(?:self\.)?([a-zA-Z_]\w*)\s*\(", re.MULTILINE
)

# Python imports
_PY_IMPORT_RE = re.compile(
    r"(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE
)

# Python class inheritance
_PY_CLASS_RE = re.compile(
    r"class\s+\w+\s*\(\s*([\w.,\s]+)\s*\)\s*:", re.MULTILINE
)

# Rust use/mod statements
_RS_USE_RE = re.compile(
    r"(?:use\s+([\w:]+)|mod\s+(\w+))", re.MULTILINE
)

# TypeScript/JavaScript imports
_TS_IMPORT_RE = re.compile(
    r"(?:import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]|"
    r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))",
    re.MULTILINE,
)


def extract_callees(source: str, language: str = "python") -> list[str]:
    """
    Extract function/method names called from a source code fragment.

    Returns a list of callee names (not fully qualified — just the
    function name as it appears in source).
    """
    if language == "python":
        return _PY_CALL_RE.findall(source)
    return []


def extract_imports(source: str, language: str = "python") -> list[str]:
    """
    Extract import targets from a source code fragment.

    Returns module/path strings that could be resolved to files.
    """
    if language == "python":
        results = []
        for match in _PY_IMPORT_RE.finditer(source):
            mod = match.group(1) or match.group(2)
            if mod:
                results.append(mod)
        return results
    elif language in ("typescript", "javascript"):
        results = []
        for match in _TS_IMPORT_RE.finditer(source):
            path = match.group(1) or match.group(2)
            if path:
                results.append(path)
        return results
    elif language == "rust":
        results = []
        for match in _RS_USE_RE.finditer(source):
            mod = match.group(1) or match.group(2)
            if mod:
                results.append(mod)
        return results
    return []


def infer_test_files(file_path: str) -> list[str]:
    """
    Infer likely test file paths from a source file path.

    Heuristics:
      foo.py       → test_foo.py, foo_test.py, tests/test_foo.py
      utils/bar.py → tests/test_bar.py, utils/test_bar.py
      src/baz.rs   → tests/baz.rs, src/baz_test.rs
    """
    path = Path(file_path)
    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    candidates = [
        str(parent / f"test_{stem}{suffix}"),
        str(parent / f"{stem}_test{suffix}"),
        str(parent / "tests" / f"test_{stem}{suffix}"),
        str(parent.parent / "tests" / f"test_{stem}{suffix}"),
    ]

    # Rust-specific
    if suffix == ".rs":
        candidates.append(str(parent / "tests" / f"{stem}{suffix}"))

    return candidates


def module_to_file_candidates(
    module_path: str,
    base_dir: str = "",
    language: str = "python",
) -> list[str]:
    """
    Convert a module path (e.g., 'utils.helpers') to candidate file paths.

    Python: utils.helpers → utils/helpers.py, utils/helpers/__init__.py
    """
    if language == "python":
        parts = module_path.split(".")
        candidates = [
            os.path.join(base_dir, *parts) + ".py",
            os.path.join(base_dir, *parts, "__init__.py"),
        ]
        return candidates
    return []


class PrefetchEngine:
    """
    Predictive pre-fetcher that learns co-access patterns across
    sessions and combines them with static analysis for predictions.

    Two prediction strategies:
      1. **Static**: Parse imports, calls, and naming conventions
      2. **Learned**: Track which files are accessed together and
         predict based on historical co-access frequency

    The learned component uses a simple co-occurrence counter:
      When file A and file B are accessed within K turns of each other,
      increment co_access[A][B] and co_access[B][A].

    Confidence is computed as:
      - Static predictions: fixed confidence (0.7 for imports, 0.5 for tests)
      - Learned predictions: normalized co-access count
    """

    def __init__(self, co_access_window: int = 5):
        self.co_access_window = co_access_window

        # co_access[file_a][file_b] = count of times accessed together
        self._co_access: dict[str, Counter] = defaultdict(Counter)

        # Recent access history for learning
        self._recent_accesses: list[tuple[str, int]] = []  # (path, turn)

        # Self-improvement: hit rate tracking
        self._pending_predictions: dict[str, set[str]] = {}
        self._hits: int = 0
        self._total_predictions: int = 0
        self._component_bus: Any = None

    def record_access(self, file_path: str, turn: int) -> None:
        """
        Record that a file was accessed at a given turn.

        Updates co-access counts with all files accessed within
        the co-access window.
        """
        # Update co-access with recent files
        for prev_path, prev_turn in self._recent_accesses:
            if turn - prev_turn <= self.co_access_window and prev_path != file_path:
                self._co_access[file_path][prev_path] += 1
                self._co_access[prev_path][file_path] += 1

        self._recent_accesses.append((file_path, turn))

        # Prune old accesses (keep last 100)
        if len(self._recent_accesses) > 100:
            self._recent_accesses = self._recent_accesses[-100:]

    def predict(
        self,
        file_path: str,
        source_content: str,
        language: str = "python",
        max_results: int = 10,
    ) -> list[PrefetchResult]:
        """
        Predict what context fragments will be needed next, given
        that the agent just accessed `file_path` with `source_content`.

        Combines static analysis and learned co-access patterns.
        Results are sorted by confidence (highest first).
        """
        predictions: list[PrefetchResult] = []
        seen_paths: set[str] = set()

        # 1. Import graph (confidence: 0.7)
        imports = extract_imports(source_content, language)
        base_dir = str(Path(file_path).parent)
        for imp in imports:
            candidates = module_to_file_candidates(imp, base_dir, language)
            for candidate in candidates:
                if candidate not in seen_paths:
                    seen_paths.add(candidate)
                    predictions.append(PrefetchResult(
                        path=candidate,
                        reason="import",
                        confidence=0.70,
                    ))

        # 2. Test files (confidence: 0.5)
        test_candidates = infer_test_files(file_path)
        for tc in test_candidates:
            if tc not in seen_paths:
                seen_paths.add(tc)
                predictions.append(PrefetchResult(
                    path=tc,
                    reason="test_file",
                    confidence=0.50,
                ))

        # 3. Learned co-access patterns (confidence: normalized count)
        if file_path in self._co_access:
            co_counts = self._co_access[file_path]
            if co_counts:
                max_count = max(co_counts.values())
                for co_path, count in co_counts.most_common(max_results):
                    if co_path not in seen_paths:
                        seen_paths.add(co_path)
                        confidence = min(count / max(max_count, 1), 1.0) * 0.80
                        predictions.append(PrefetchResult(
                            path=co_path,
                            reason="co_access",
                            confidence=round(confidence, 2),
                        ))

        # Sort by confidence and limit results
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        # Track predictions for hit rate self-improvement
        self._pending_predictions[file_path] = {
            p.path for p in predictions[:max_results]
        }

        return predictions[:max_results]

    # ── Self-Improvement: Hit Rate Tracking & Adaptive Window ──────

    def record_actual_access(self, file_path: str) -> None:
        """Record that a file was actually accessed by the agent.

        Checks if this file was predicted by any pending prediction set.
        Computes hit rate and auto-adjusts co_access_window.
        """
        for source, predicted_paths in list(self._pending_predictions.items()):
            if file_path in predicted_paths:
                self._hits += 1
                predicted_paths.discard(file_path)
            # After enough accesses, record misses for remaining predictions
        self._total_predictions += 1

        # Self-tune every 20 accesses
        if self._total_predictions % 20 == 0 and self._total_predictions > 0:
            self._self_tune_window()

    def _self_tune_window(self) -> None:
        """Auto-adjust co_access_window based on hit rate.

        Formula: if hit_rate > 0.8 → widen window (capture more co-accesses)
                 if hit_rate < 0.4 → narrow window (reduce noise)
                 otherwise → stable

        This is a bandit-style exploration/exploitation trade-off
        on the prediction radius.
        """
        if self._total_predictions < 5:
            return

        hit_rate = self._hits / max(self._total_predictions, 1)

        if hit_rate > 0.80 and self.co_access_window < 15:
            self.co_access_window += 1
        elif hit_rate < 0.40 and self.co_access_window > 2:
            self.co_access_window -= 1

        # Log to component bus if available
        if self._component_bus:
            self._component_bus.log(
                component="prefetch",
                metric="hit_rate",
                value=hit_rate,
                params={"co_access_window": float(self.co_access_window)},
            )

    def set_component_bus(self, bus: Any) -> None:
        """Attach a ComponentFeedbackBus for persistent metric logging."""
        self._component_bus = bus

    def stats(self) -> dict:
        total_pairs = sum(
            len(targets) for targets in self._co_access.values()
        )
        hit_rate = self._hits / max(self._total_predictions, 1)
        return {
            "tracked_files": len(self._co_access),
            "co_access_pairs": total_pairs // 2,  # Undirected
            "recent_accesses": len(self._recent_accesses),
            "co_access_window": self.co_access_window,
            "prediction_hits": self._hits,
            "total_predictions": self._total_predictions,
            "hit_rate": round(hit_rate, 4),
        }
