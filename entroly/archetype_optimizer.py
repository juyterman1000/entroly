"""
Archetype Optimizer — Codebase-Adaptive Self-Evolution
======================================================

Instead of one global set of scoring weights, the DreamingLoop now
detects what *type* of codebase it's running inside and evolves
archetype-specific weight profiles.

This is the Python orchestration layer. The heavy math runs in Rust
(archetype.rs). This module handles:
  1. Collecting CodebaseStats from the file system
  2. Calling the Rust fingerprinter
  3. Managing the archetype→weight strategy table
  4. Integrating with the DreamingLoop for per-archetype optimization
  5. Persisting archetype state across sessions

Mathematical grounding:
  The strategy table is a mapping A: ℤ → ℝ⁷ from archetype IDs
  to 7-dimensional weight vectors. The DreamingLoop optimizes each
  independently:
    w_a(t+1) = optimize(w_a(t), benchmark(w_a(t)))  ∀a ∈ archetypes

  This is equivalent to multi-task optimization where each task
  (archetype) has its own loss landscape. The key insight: a React
  frontend and a Rust systems library occupy different regions of
  the loss surface, so they need different optima.

Usage:
    from entroly.archetype_optimizer import ArchetypeOptimizer

    optimizer = ArchetypeOptimizer(data_dir=".entroly")
    archetype_id = optimizer.detect_and_load()
    weights = optimizer.current_weights()
    # ... use weights for context optimization ...
    optimizer.record_feedback(reward=0.85)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly.archetype_optimizer")


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# File extensions → language bucket
_PYTHON_EXTS = {".py", ".pyi", ".pyx"}
_RUST_EXTS = {".rs"}
_JS_TS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
_TEST_PATTERNS = re.compile(
    r"(test[_s]?|spec[_s]?|__tests__|\.test\.|\.spec\.)", re.I
)
_FFI_MARKERS = re.compile(
    r"(#\[pyfunction\]|#\[pyclass\]|#\[wasm_bindgen\]|extern\s+\"C\"|ctypes\.|cffi|JNIEXPORT|napi::)",
)

# Strategy table persistence
_STRATEGY_FILE = "archetype_strategy.json"

# Default weight profile (matches autotune defaults)
# The 5 scoring weights correspond to PRISM 5D dimensions:
#   [0] Recency   — how recently was the fragment accessed?
#   [1] Frequency — how often is the fragment accessed?
#   [2] Semantic  — how similar is the fragment to the query?
#   [3] Entropy   — how information-dense is the fragment?
#   [4] Resonance — pairwise fragment interaction bonus (PRISM 5D)
DEFAULT_WEIGHTS = {
    "w_recency": 0.30,
    "w_frequency": 0.25,
    "w_semantic": 0.25,
    "w_entropy": 0.20,
    "w_resonance": 0.10,
    "decay_half_life": 15.0,
    "min_relevance": 0.05,
    "exploration_rate": 0.10,
}

WEIGHT_KEYS = list(DEFAULT_WEIGHTS.keys())


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CodebaseStats:
    """Raw statistics collected from scanning the codebase."""
    total_files: int = 0
    python_files: int = 0
    rust_files: int = 0
    js_ts_files: int = 0
    other_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0
    test_files: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    max_dep_depth: int = 0
    module_count: int = 0
    entropy_values: list[float] = field(default_factory=list)
    ffi_files: int = 0


@dataclass
class ArchetypeInfo:
    """Information about the detected archetype."""
    archetype_id: int
    label: str
    confidence: float
    weights: dict[str, float]
    sample_count: int


# ═══════════════════════════════════════════════════════════════════
# Codebase Scanner
# ═══════════════════════════════════════════════════════════════════

def scan_codebase(root: Path, max_files: int = 5000) -> CodebaseStats:
    """Scan a codebase directory and collect structural statistics.

    Fast scan: reads file extensions and first 100 lines of each file.
    Skips hidden dirs, node_modules, .git, __pycache__, etc.
    O(N) in number of files with bounded per-file work.
    """
    stats = CodebaseStats()
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        ".entroly", ".claude", "dist", "build", ".tox", ".mypy_cache",
        ".pytest_cache", "target",  # Rust target dir
    }

    files_scanned = 0
    top_level_dirs: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden and vendor directories
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not d.startswith(".")
        ]

        # Track top-level modules
        rel = os.path.relpath(dirpath, root)
        if os.sep not in rel and rel != ".":
            top_level_dirs.add(rel)

        for fname in filenames:
            if files_scanned >= max_files:
                break

            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()

            # Language classification
            if ext in _PYTHON_EXTS:
                stats.python_files += 1
            elif ext in _RUST_EXTS:
                stats.rust_files += 1
            elif ext in _JS_TS_EXTS:
                stats.js_ts_files += 1
            else:
                # Only count source-like files
                if ext in {".go", ".java", ".rb", ".php", ".c", ".cpp",
                           ".h", ".cs", ".swift", ".kt", ".scala",
                           ".toml", ".yaml", ".yml", ".json", ".md"}:
                    stats.other_files += 1
                else:
                    continue  # Skip binary/media files

            stats.total_files += 1

            # Test file detection
            if _TEST_PATTERNS.search(fpath):
                stats.test_files += 1

            # Quick content scan (first 100 lines for speed)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 100:
                            break
                        lines.append(line)

                    content = "".join(lines)
                    line_count = len(lines)
                    stats.total_lines += line_count

                    # Count structural elements
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith(("def ", "function ", "fn ",
                                               "func ", "public func")):
                            stats.total_functions += 1
                        elif stripped.startswith(("class ", "struct ",
                                                  "interface ", "trait ")):
                            stats.total_classes += 1
                        elif stripped.startswith(("import ", "from ",
                                                  "use ", "#include",
                                                  "require")):
                            stats.total_imports += 1

                    # FFI detection
                    if _FFI_MARKERS.search(content):
                        stats.ffi_files += 1

                    # Entropy approximation: compression ratio of content
                    if len(content) > 64:
                        try:
                            import zlib
                            compressed = zlib.compress(content.encode(), 1)
                            ratio = len(compressed) / len(content.encode())
                            # Scale to [0, 1]: 0.10 → 0.0, 0.80 → 1.0
                            entropy = max(0.0, min(1.0, (ratio - 0.10) / 0.70))
                            stats.entropy_values.append(entropy)
                        except Exception:
                            pass

            except (OSError, UnicodeDecodeError):
                continue

            files_scanned += 1

    stats.module_count = len(top_level_dirs)

    return stats


# ═══════════════════════════════════════════════════════════════════
# Archetype Optimizer
# ═══════════════════════════════════════════════════════════════════

class ArchetypeOptimizer:
    """Manages archetype detection and per-archetype weight optimization.

    Lifecycle:
      1. On startup: scan codebase → fingerprint → classify
      2. Load weight profile for detected archetype
      3. DreamingLoop optimizes weights for THIS archetype
      4. On shutdown: persist updated strategy table
    """

    def __init__(self, data_dir: str | Path, project_root: str | Path | None = None):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(project_root) if project_root else Path(".")
        self._strategy_path = self._data_dir / _STRATEGY_FILE

        # State
        self._strategy_table: dict[str, dict[str, Any]] = self._load_strategy()
        self._current_archetype: str | None = None
        self._current_weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        self._last_fingerprint: dict[str, float] | None = None
        self._rust_engine = None

        # Try to load the Rust archetype engine
        try:
            from entroly_core import ArchetypeEngine  # type: ignore
            self._rust_engine = ArchetypeEngine()
            logger.info("ArchetypeOptimizer: using Rust engine")
        except (ImportError, AttributeError):
            logger.info("ArchetypeOptimizer: using pure-Python fallback")

    def detect_and_load(self) -> ArchetypeInfo:
        """Scan the codebase, detect archetype, and load its weights.

        This is the main entry point, called once per session.
        Returns ArchetypeInfo with the detected archetype and weights.
        """
        # Step 1: Scan codebase
        stats = scan_codebase(self._project_root)

        # Step 2: Compute fingerprint
        fingerprint = self._compute_fingerprint(stats)
        self._last_fingerprint = fingerprint

        # Step 3: Classify into archetype
        archetype_label = self._classify(fingerprint, stats)
        self._current_archetype = archetype_label

        # Step 4: Load weights for this archetype
        if archetype_label in self._strategy_table:
            entry = self._strategy_table[archetype_label]
            self._current_weights = entry.get("weights", dict(DEFAULT_WEIGHTS))
            sample_count = entry.get("sample_count", 0)
            confidence = entry.get("confidence", 0.5)
            logger.info(
                "ArchetypeOptimizer: loaded archetype '%s' (samples=%d, conf=%.2f)",
                archetype_label, sample_count, confidence,
            )
        else:
            # New archetype: use seed weights based on classification
            self._current_weights = self._seed_weights(archetype_label).copy()
            self._strategy_table[archetype_label] = {
                "weights": self._current_weights.copy(),
                "sample_count": 0,
                "confidence": 0.5,
                "created_at": time.time(),
                "fingerprint": fingerprint,
            }
            self._save_strategy()
            logger.info(
                "ArchetypeOptimizer: new archetype '%s' initialized",
                archetype_label,
            )

        return ArchetypeInfo(
            archetype_id=hash(archetype_label) % 10000,
            label=archetype_label,
            confidence=self._strategy_table.get(archetype_label, {}).get("confidence", 0.5),
            weights=self._current_weights.copy(),
            sample_count=self._strategy_table.get(archetype_label, {}).get("sample_count", 0),
        )

    def current_weights(self) -> dict[str, float]:
        """Return the current weight profile for the active archetype."""
        return self._current_weights.copy()

    def current_archetype(self) -> str | None:
        """Return the label of the current archetype."""
        return self._current_archetype

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update the weights for the current archetype.

        Called by the DreamingLoop when it finds an improvement.
        Persists the update to disk.
        """
        if not self._current_archetype:
            return

        self._current_weights = new_weights.copy()

        entry = self._strategy_table.setdefault(self._current_archetype, {})
        entry["weights"] = new_weights.copy()
        entry["sample_count"] = entry.get("sample_count", 0) + 1
        entry["updated_at"] = time.time()

        # Update confidence based on sample count (more samples = more confident)
        samples = entry["sample_count"]
        entry["confidence"] = min(0.95, 0.5 + 0.05 * min(samples, 9))

        self._save_strategy()

        logger.debug(
            "ArchetypeOptimizer: updated weights for '%s' (sample %d)",
            self._current_archetype, samples,
        )

    def get_export_weights(self) -> dict[str, float]:
        """Export the 5 PRISM scoring weights (4D + resonance)."""
        return {
            "w_r": self._current_weights.get("w_recency", 0.30),
            "w_f": self._current_weights.get("w_frequency", 0.25),
            "w_s": self._current_weights.get("w_semantic", 0.25),
            "w_e": self._current_weights.get("w_entropy", 0.20),
            "w_res": self._current_weights.get("w_resonance", 0.10),
        }

    def stats(self) -> dict[str, Any]:
        """Return optimizer statistics for dashboard/monitoring."""
        return {
            "current_archetype": self._current_archetype,
            "total_archetypes": len(self._strategy_table),
            "strategy_table": {
                k: {
                    "sample_count": v.get("sample_count", 0),
                    "confidence": round(v.get("confidence", 0), 3),
                }
                for k, v in self._strategy_table.items()
            },
            "current_weights": self._current_weights,
            "fingerprint": self._last_fingerprint,
        }

    # ── Private Methods ──────────────────────────────────────────

    def _compute_fingerprint(self, stats: CodebaseStats) -> dict[str, float]:
        """Compute a normalized fingerprint dict from stats."""
        total = max(stats.total_files, 1)
        total_lines = max(stats.total_lines, 1)

        fp = {
            "lang_python": stats.python_files / total,
            "lang_rust": stats.rust_files / total,
            "lang_js_ts": stats.js_ts_files / total,
            "lang_other": stats.other_files / total,
            "avg_file_size": _log_norm(total_lines / total, 500),
            "func_density": _log_norm(
                stats.total_functions / max(total_lines / 100, 1), 20),
            "class_ratio": (
                stats.total_classes / max(stats.total_classes + stats.total_functions, 1)
            ),
            "import_density": _log_norm(stats.total_imports / total, 30),
            "test_ratio": stats.test_files / total,
            "module_count": _log_norm(stats.module_count, 50),
            "ffi_ratio": stats.ffi_files / total,
        }

        # Entropy stats
        if stats.entropy_values:
            n = len(stats.entropy_values)
            mean = sum(stats.entropy_values) / n
            var = sum((e - mean) ** 2 for e in stats.entropy_values) / n
            fp["entropy_mean"] = max(0, min(1, mean))
            fp["entropy_var"] = max(0, min(1, var))
        else:
            fp["entropy_mean"] = 0.5
            fp["entropy_var"] = 0.1

        return fp

    def _classify(self, fingerprint: dict[str, float], stats: CodebaseStats) -> str:
        """Classify a fingerprint into an archetype label.

        Uses a simple rule-based system that maps structural
        features to well-known codebase patterns. This is robust
        and interpretable — no fragile unsupervised clustering
        on small sample sizes.
        """
        # Primary language determines the major category
        lang_py = fingerprint.get("lang_python", 0)
        lang_rs = fingerprint.get("lang_rust", 0)
        lang_js = fingerprint.get("lang_js_ts", 0)
        test_ratio = fingerprint.get("test_ratio", 0)
        class_ratio = fingerprint.get("class_ratio", 0)
        ffi_ratio = fingerprint.get("ffi_ratio", 0)
        module_count = stats.module_count

        # Multi-language monorepo detection
        strong_langs = sum(1 for x in [lang_py, lang_rs, lang_js] if x > 0.15)
        if strong_langs >= 2 and module_count >= 5:
            if ffi_ratio > 0.05:
                return "polyglot_ffi_monorepo"
            return "fullstack_monorepo"

        # Single-language archetypes
        if lang_py > 0.5:
            if class_ratio > 0.3 and test_ratio > 0.15:
                return "python_enterprise_backend"
            elif fingerprint.get("import_density", 0) > 0.6:
                return "python_data_science"
            else:
                return "python_backend"

        if lang_rs > 0.5:
            if ffi_ratio > 0.1:
                return "rust_ffi_library"
            elif test_ratio > 0.2:
                return "rust_well_tested"
            else:
                return "rust_systems"

        if lang_js > 0.5:
            if class_ratio > 0.25:
                return "js_component_framework"
            else:
                return "js_frontend"

        # Fallback: use the dominant language with a generic label
        dominant = max(
            [("python", lang_py), ("rust", lang_rs), ("js", lang_js)],
            key=lambda x: x[1],
        )
        return f"{dominant[0]}_general"

    def _seed_weights(self, archetype_label: str) -> dict[str, float]:
        """Return seed weights for a given archetype.

        These are empirically derived starting points that give
        good initial performance. The DreamingLoop will optimize
        from here.

        Design rationale for w_resonance (PRISM 5D):
          - Rust systems: HIGH resonance (0.15) — tightly coupled code
            produces supermodular context (function + its trait impl
            together >> sum of parts)
          - JS frontend: LOW resonance (0.05) — components are more
            independent, pairwise interactions matter less
          - Monorepos: MEDIUM resonance (0.10) — mixed coupling
          - Data science: LOW resonance (0.06) — notebooks/scripts
            are often self-contained
        """
        seeds = {
            "python_backend": {
                "w_recency": 0.35, "w_frequency": 0.25,
                "w_semantic": 0.20, "w_entropy": 0.20,
                "w_resonance": 0.10,
                "decay_half_life": 15.0, "min_relevance": 0.05,
                "exploration_rate": 0.10,
            },
            "python_enterprise_backend": {
                "w_recency": 0.30, "w_frequency": 0.25,
                "w_semantic": 0.25, "w_entropy": 0.20,
                "w_resonance": 0.12,
                "decay_half_life": 20.0, "min_relevance": 0.04,
                "exploration_rate": 0.08,
            },
            "python_data_science": {
                "w_recency": 0.25, "w_frequency": 0.30,
                "w_semantic": 0.25, "w_entropy": 0.20,
                "w_resonance": 0.06,
                "decay_half_life": 12.0, "min_relevance": 0.06,
                "exploration_rate": 0.12,
            },
            "rust_systems": {
                "w_recency": 0.20, "w_frequency": 0.20,
                "w_semantic": 0.30, "w_entropy": 0.30,
                "w_resonance": 0.15,
                "decay_half_life": 20.0, "min_relevance": 0.04,
                "exploration_rate": 0.08,
            },
            "rust_ffi_library": {
                "w_recency": 0.20, "w_frequency": 0.20,
                "w_semantic": 0.35, "w_entropy": 0.25,
                "w_resonance": 0.18,
                "decay_half_life": 18.0, "min_relevance": 0.05,
                "exploration_rate": 0.10,
            },
            "rust_well_tested": {
                "w_recency": 0.25, "w_frequency": 0.20,
                "w_semantic": 0.25, "w_entropy": 0.30,
                "w_resonance": 0.14,
                "decay_half_life": 15.0, "min_relevance": 0.05,
                "exploration_rate": 0.08,
            },
            "js_frontend": {
                "w_recency": 0.40, "w_frequency": 0.20,
                "w_semantic": 0.25, "w_entropy": 0.15,
                "w_resonance": 0.05,
                "decay_half_life": 10.0, "min_relevance": 0.06,
                "exploration_rate": 0.12,
            },
            "js_component_framework": {
                "w_recency": 0.35, "w_frequency": 0.25,
                "w_semantic": 0.25, "w_entropy": 0.15,
                "w_resonance": 0.08,
                "decay_half_life": 12.0, "min_relevance": 0.05,
                "exploration_rate": 0.10,
            },
            "fullstack_monorepo": {
                "w_recency": 0.30, "w_frequency": 0.25,
                "w_semantic": 0.25, "w_entropy": 0.20,
                "w_resonance": 0.10,
                "decay_half_life": 20.0, "min_relevance": 0.04,
                "exploration_rate": 0.08,
            },
            "polyglot_ffi_monorepo": {
                "w_recency": 0.25, "w_frequency": 0.20,
                "w_semantic": 0.30, "w_entropy": 0.25,
                "w_resonance": 0.15,
                "decay_half_life": 18.0, "min_relevance": 0.05,
                "exploration_rate": 0.10,
            },
        }
        return seeds.get(archetype_label, dict(DEFAULT_WEIGHTS))

    def _load_strategy(self) -> dict[str, dict[str, Any]]:
        """Load the persisted strategy table from disk."""
        if self._strategy_path.exists():
            try:
                raw = self._strategy_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load archetype strategy: %s", e)
        return {}

    def _save_strategy(self) -> None:
        """Persist the strategy table to disk (atomic write)."""
        try:
            import tempfile
            content = json.dumps(self._strategy_table, indent=2)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._data_dir), suffix=".tmp", prefix="arch_"
            )
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                if os.name == "nt":
                    os.replace(tmp, str(self._strategy_path))
                else:
                    os.rename(tmp, str(self._strategy_path))
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except OSError as e:
            logger.debug("Failed to save archetype strategy: %s", e)


# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════



def _log_norm(value: float, max_ref: float) -> float:
    """Log-normalize to [0, 1]. f(x) = ln(1+x) / ln(1+max_ref)."""
    v = max(0.0, value)
    m = max(1.0, max_ref)
    return math.log(1.0 + v) / math.log(1.0 + m)
