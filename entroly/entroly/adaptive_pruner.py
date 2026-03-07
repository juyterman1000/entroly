"""
AdaptivePruner Bridge for Entroly
======================================

Wires ebbiforge_core.AdaptivePruner into the feedback loop.

The key addition: `historical_success` — a dimension that entroly's
Rust engine doesn't have. Over time, the RL weight updates learn which
scoring features matter most for THIS user's codebase.

Weight update rule (from ebbiforge Rust source):
    weight += lr * feedback * feature_value  (clamped to [-1, 1])

Falls back to no-op if ebbiforge_core is not installed.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from ebbiforge_core import AdaptivePruner as _RustPruner, ContextFragment
    _PRUNER_AVAILABLE = True
except ImportError:
    _PRUNER_AVAILABLE = False
    _RustPruner = None

try:
    from ebbiforge_core import CodeQualityGuard as _RustGuard
    _GUARD_AVAILABLE = True
except ImportError:
    _GUARD_AVAILABLE = False
    _RustGuard = None


class EntrolyPruner:
    """
    Adaptive RL pruner backed by ebbiforge_core.AdaptivePruner.

    Extends entroly's Wilson-score feedback with a `historical_success`
    dimension: fragments that previously helped get boosted, those that didn't
    get down-weighted over time.

    Zero-config: if ebbiforge_core is unavailable, all methods are no-ops.
    """

    def __init__(self):
        self._pruner = _RustPruner() if _PRUNER_AVAILABLE else None
        self._fragment_features: Dict[str, Dict[str, float]] = {}
        if _PRUNER_AVAILABLE:
            logger.info("AdaptivePruner: ebbiforge_core available — RL weight learning active")
        else:
            logger.info("AdaptivePruner: ebbiforge_core not available — using static weights")

    @property
    def available(self) -> bool:
        return _PRUNER_AVAILABLE and self._pruner is not None

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
        if not self.available:
            return False

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
    ) -> Optional[float]:
        """
        Score a fragment using current learned RL weights.
        Returns None if pruner unavailable (use entroly's own scoring).
        """
        if not self.available:
            return None
        return self._pruner.score_fragment(recency, relevance, historical_success, complexity)


class FragmentGuard:
    """
    Code quality scanner backed by ebbiforge_core.CodeQualityGuard.

    Scans each ingested fragment for:
    - Hardcoded API secrets  (sk-..., API_KEY = "...")
    - unsafe Rust blocks
    - TODO comments
    - Console spam (>5 log statements)

    Returns a list of issues — empty means clean.
    Zero-config: no-op if ebbiforge_core unavailable.
    """

    def __init__(self):
        self._guard = _RustGuard() if _GUARD_AVAILABLE else None
        if _GUARD_AVAILABLE:
            logger.info("FragmentGuard: CodeQualityGuard active — scanning ingested fragments")

    @property
    def available(self) -> bool:
        return _GUARD_AVAILABLE and self._guard is not None

    def scan(self, content: str, source: str = "") -> list[str]:
        """
        Scan fragment content for code quality issues.

        Returns list of issue strings (empty = clean).
        """
        if not self.available or not content:
            return []
        try:
            return list(self._guard.review_code(content, source))
        except Exception:
            return []
