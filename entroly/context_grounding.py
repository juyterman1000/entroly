"""Context Grounding Gate — filter ungrounded fragments before injection.

Scores selected fragments against the project's SymbolManifest and demotes
those with high hallucination probability. Uses the existing GRAPHS verifier
(symbol_resolution.py) with a cached per-project manifest so the cost is
amortized across requests.

Fail-open: if the manifest cannot be built or verification errors, all
fragments pass through unchanged.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("entroly.context_grounding")

GROUNDING_THRESHOLD = 0.7
MANIFEST_TTL_SECONDS = 300


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Per-fragment grounding verdict."""

    fragment_id: str
    h_score: float
    grounded: bool
    n_unresolved: int


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Aggregate grounding outcome for a set of selected fragments."""

    total: int
    passed: int
    demoted: int
    results: tuple[GroundingResult, ...]


class ContextGroundingGate:
    """Cached grounding gate that filters fragments before injection."""

    def __init__(
        self,
        repo_root: str | None = None,
        threshold: float = GROUNDING_THRESHOLD,
    ) -> None:
        self._repo_root = repo_root or os.getcwd()
        self._threshold = threshold
        self._lock = threading.Lock()
        self._manifest: Any = None
        self._manifest_built_at: float = 0.0
        self._verifier: Any = None

    def _ensure_manifest(self) -> bool:
        now = time.monotonic()
        if (
            self._manifest is not None
            and (now - self._manifest_built_at) < MANIFEST_TTL_SECONDS
        ):
            return True

        with self._lock:
            if (
                self._manifest is not None
                and (time.monotonic() - self._manifest_built_at) < MANIFEST_TTL_SECONDS
            ):
                return True

            try:
                from .verifiers.symbol_resolution import (
                    SymbolManifest,
                    SymbolVerifier,
                )

                self._manifest = SymbolManifest.build_from_codebase(
                    self._repo_root
                )
                self._verifier = SymbolVerifier(self._manifest)
                self._manifest_built_at = time.monotonic()
                logger.debug(
                    "Grounding manifest built: %d symbols", self._manifest.size()
                )
                return True
            except Exception as exc:
                logger.debug("Grounding manifest build failed: %s", exc)
                return False

    def score_fragment(self, content: str) -> tuple[float, int]:
        """Return (h_score, n_unresolved) for a content string."""
        if self._verifier is None:
            return 0.0, 0
        try:
            result = self._verifier.verify(content)
            return result.h_score, result.n_unresolved
        except Exception:
            return 0.0, 0

    def filter_fragments(
        self,
        fragments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], GroundingReport]:
        """Score and filter fragments, demoting those above the threshold.

        Demoted fragments are not removed — they are moved to the end of the
        list so the knapsack can still include them if budget allows, but
        grounded fragments take priority.
        """
        if not fragments or not self._ensure_manifest():
            return fragments, GroundingReport(
                total=len(fragments),
                passed=len(fragments),
                demoted=0,
                results=(),
            )

        results: list[GroundingResult] = []
        grounded: list[dict[str, Any]] = []
        demoted: list[dict[str, Any]] = []

        for frag in fragments:
            content = frag.get("content", frag.get("text", ""))
            frag_id = frag.get("id", frag.get("fragment_id", ""))
            h_score, n_unresolved = self.score_fragment(content)
            is_grounded = h_score < self._threshold

            results.append(GroundingResult(
                fragment_id=frag_id,
                h_score=h_score,
                grounded=is_grounded,
                n_unresolved=n_unresolved,
            ))

            if is_grounded:
                grounded.append(frag)
            else:
                demoted.append(frag)
                logger.debug(
                    "Grounding demoted fragment %s: h_score=%.3f unresolved=%d",
                    frag_id, h_score, n_unresolved,
                )

        report = GroundingReport(
            total=len(fragments),
            passed=len(grounded),
            demoted=len(demoted),
            results=tuple(results),
        )

        return grounded + demoted, report
