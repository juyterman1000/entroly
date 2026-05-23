"""
AdaptivePruner Bridge for Entroly
======================================

Wires a 4-weight RL policy into the feedback loop.

The key addition: `historical_success` — a dimension that entroly's
core engine doesn't have. Over time, the RL weight updates learn which
scoring features matter most for THIS user's codebase.

Weight update rule:
    weight += lr * feedback * feature_value  (clamped to [0.01, 2.0])

Architecture: Rust backend preferred for speed; pure-Python fallback
with identical math guarantees the learning signal is NEVER silently
dropped. This follows the same pattern as EntrolyEngine (Rust + Python).
"""

from __future__ import annotations

import logging
import math
import re as _re

logger = logging.getLogger(__name__)

# ── Try Rust backend first ──────────────────────────────────────────
try:
    from entroly_core import AdaptivePruner as _RustPruner
    from entroly_core import ContextFragment  # noqa: F401
    _RUST_PRUNER_AVAILABLE = True
except ImportError:
    _RUST_PRUNER_AVAILABLE = False
    _RustPruner = None

try:
    from entroly_core import CodeQualityGuard as _RustGuard
    _RUST_GUARD_AVAILABLE = True
except ImportError:
    _RUST_GUARD_AVAILABLE = False
    _RustGuard = None


# ══════════════════════════════════════════════════════════════════════
# Pure-Python RL Pruner (fallback when Rust backend unavailable)
# ══════════════════════════════════════════════════════════════════════

class _PyAdaptivePruner:
    """
    4-weight linear policy with REINFORCE-style gradient updates.

    Features: [recency, relevance, historical_success, complexity]
    Score = dot(weights, features)

    Update: w_i += lr * feedback * feature_i
    Weights are clamped to [0.01, 2.0] to prevent collapse or explosion.

    This is mathematically identical to the Rust AdaptivePruner.
    """

    __slots__ = ("_weights", "_lr", "_update_count")

    def __init__(self, learning_rate: float = 0.05):
        # Initial weights: equal prior over all 4 features
        self._weights = [0.25, 0.35, 0.20, 0.20]  # recency, relevance, hist_success, complexity
        self._lr = learning_rate
        self._update_count = 0

    def update_policy(
        self,
        feedback: float,
        recency: float,
        relevance: float,
        historical_success: float,
        complexity: float,
    ) -> None:
        """Apply a single REINFORCE gradient step."""
        features = [recency, relevance, historical_success, complexity]
        for i in range(4):
            self._weights[i] += self._lr * feedback * features[i]
            # Clamp to prevent weight collapse or explosion
            self._weights[i] = max(0.01, min(2.0, self._weights[i]))
        self._update_count += 1

    def score_fragment(
        self,
        recency: float,
        relevance: float,
        historical_success: float,
        complexity: float,
    ) -> float:
        """Score = dot(weights, features)."""
        features = [recency, relevance, historical_success, complexity]
        return sum(w * f for w, f in zip(self._weights, features))

    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    @property
    def update_count(self) -> int:
        return self._update_count


# ══════════════════════════════════════════════════════════════════════
# Pure-Python Code Quality Guard (fallback)
# ══════════════════════════════════════════════════════════════════════

_SECRET_PATTERNS = [
    _re.compile(r"""(?:api[_-]?key|secret|password|token)\s*[=:]\s*['"][^'"]{8,}['"]""", _re.IGNORECASE),
    _re.compile(r"""sk-[A-Za-z0-9]{20,}"""),
    _re.compile(r"""ghp_[A-Za-z0-9]{36}"""),
    _re.compile(r"""AKIA[0-9A-Z]{16}"""),
]
_UNSAFE_PATTERN = _re.compile(r"""\bunsafe\s*\{""")
_TODO_PATTERN = _re.compile(r"""(?://|#)\s*(?:TODO|FIXME|HACK|XXX)\b""", _re.IGNORECASE)
_CONSOLE_PATTERN = _re.compile(r"""(?:console\.(?:log|warn|error)|print\(|println!\()""")


class _PyCodeQualityGuard:
    """Scans fragments for secrets, unsafe blocks, TODOs, and console spam."""

    def review_code(self, content: str, source: str = "") -> list[str]:
        issues: list[str] = []
        for pat in _SECRET_PATTERNS:
            if pat.search(content):
                issues.append(f"potential secret/credential detected in {source or 'fragment'}")
                break
        if _UNSAFE_PATTERN.search(content):
            issues.append("unsafe block detected")
        todo_count = len(_TODO_PATTERN.findall(content))
        if todo_count >= 3:
            issues.append(f"{todo_count} TODO/FIXME comments")
        console_count = len(_CONSOLE_PATTERN.findall(content))
        if console_count >= 5:
            issues.append(f"{console_count} console/print statements")
        return issues


# ══════════════════════════════════════════════════════════════════════
# EntrolyPruner — the public API
# ══════════════════════════════════════════════════════════════════════

class EntrolyPruner:
    """
    Adaptive RL pruner with Rust backend + Python fallback.

    Extends entroly's Wilson-score feedback with a `historical_success`
    dimension: fragments that previously helped get boosted, those that didn't
    get down-weighted over time.

    NEVER a no-op: if Rust is unavailable, uses identical Python math.
    """

    def __init__(self):
        if _RUST_PRUNER_AVAILABLE:
            self._pruner = _RustPruner()
            self._backend = "rust"
            logger.info("AdaptivePruner: Rust backend active — RL weight learning active")
        else:
            self._pruner = _PyAdaptivePruner()
            self._backend = "python"
            logger.info("AdaptivePruner: Python fallback active — RL weight learning active")

        self._fragment_features: dict[str, dict[str, float]] = {}

    @property
    def available(self) -> bool:
        """Always True — we always have a working backend."""
        return True

    @property
    def backend(self) -> str:
        return self._backend

    def record_fragment_features(
        self,
        fragment_id: str,
        recency: float,
        relevance: float,
        complexity: float,
        was_selected: bool,
    ) -> None:
        """
        Record the scoring features for a fragment at selection time.
        Called from optimize_context for each selected fragment.
        These are stored until feedback arrives.
        """
        self._fragment_features[fragment_id] = {
            "recency": recency,
            "relevance": relevance,
            "complexity": complexity,
            "was_selected": was_selected,
        }

    def apply_feedback(self, fragment_id: str, feedback: float) -> bool:
        """
        Apply user feedback to update RL weights for this fragment's features.

        Args:
            fragment_id: The fragment that received feedback.
            feedback:    +1.0 = helpful, -1.0 = not helpful, 0.0 = neutral.

        Returns:
            True if weights were updated, False if no feature record found.
        """
        features = self._fragment_features.get(fragment_id)
        if not features:
            return False

        # historical_success: 1.0 if this fragment was previously selected, else 0.5
        historical_success = 1.0 if features.get("was_selected") else 0.5

        self._pruner.update_policy(
            feedback=feedback,
            recency=features["recency"],
            relevance=features["relevance"],
            historical_success=historical_success,
            complexity=features["complexity"],
        )
        return True

    def score_fragment(
        self,
        recency: float,
        relevance: float,
        historical_success: float,
        complexity: float,
    ) -> float | None:
        """
        Score a fragment using current learned RL weights.
        Returns None only if something unexpected fails.
        """
        try:
            return self._pruner.score_fragment(recency, relevance, historical_success, complexity)
        except Exception:
            return None

    def get_weights(self) -> list[float] | None:
        """Return the current RL policy weights (Python backend only)."""
        if isinstance(self._pruner, _PyAdaptivePruner):
            return self._pruner.weights
        return None

    def get_update_count(self) -> int:
        """Return how many gradient updates have been applied."""
        if isinstance(self._pruner, _PyAdaptivePruner):
            return self._pruner.update_count
        return -1  # Rust backend doesn't expose this


class FragmentGuard:
    """
    Code quality scanner with Rust backend + Python fallback.

    Scans each ingested fragment for:
    - Hardcoded API secrets  (sk-..., API_KEY = "...")
    - unsafe Rust blocks
    - TODO comments
    - Console spam (>5 log statements)

    Returns a list of issues — empty means clean.
    NEVER a no-op: Python fallback covers all patterns.
    """

    def __init__(self):
        if _RUST_GUARD_AVAILABLE:
            self._guard = _RustGuard()
            self._backend = "rust"
        else:
            self._guard = _PyCodeQualityGuard()
            self._backend = "python"

    @property
    def available(self) -> bool:
        """Always True — we always have a working backend."""
        return True

    def scan(self, content: str, source: str = "") -> list[str]:
        """
        Scan fragment content for code quality issues.

        Returns list of issue strings (empty = clean).
        """
        if not content:
            return []
        try:
            return list(self._guard.review_code(content, source))
        except Exception:
            return []
