"""
Fast-Path for Crystallized Skills
==================================

Closes the loop opened by :mod:`reward_crystallizer`: when a query matches
a previously crystallized skill, short-circuit the full optimization
pipeline and return the recipe directly.

Architecture
------------

  ``optimize_context(query, budget)``
       │
       ▼
  ``FastPathRouter.try_match(query, engine)``  ◀── new, this module
       │            │
       │ hit        │ miss / stale
       ▼            ▼
  recipe-based  full pipeline
  fragments     (knapsack + IOS + …)

Why this is correct
-------------------

A crystallized skill is, by construction, a query family for which the
Hoeffding lower bound on reward beats the global baseline by ≥ ε at
δ = 0.05. The recipe captured at crystallization (top-K most-frequently
selected fragment IDs across the cluster) is the empirical maximizer of
that reward signal. For matched queries, replaying the recipe is at least
as good as re-deriving it via the full pipeline — modulo two caveats:

  1. **Staleness.** Fragments referenced by the recipe may have been
     evicted, edited, or never loaded into this session. The router
     verifies presence and falls through if a configurable fraction is
     missing (default: > 50% missing → fall through, on the principle
     that a partial recipe is worse than a fresh selection).

  2. **Distributional drift.** The cluster's high-reward statistic was
     collected at the time of crystallization. If the underlying
     fragments have drifted (file edits, new neighbors), the recipe may
     no longer be optimal. We don't try to detect this — the
     ``crystallize_skill`` lifecycle is responsible for re-validating
     periodically, and the next dishonest cluster (low reward) will
     simply not crystallize a replacement.

Latency claim
-------------

Fast-path is O(K + skills) where K = recipe size and skills = number of
promoted skills loaded. Full path is O(N log N) over all fragments
(N typically 100s–1000s) plus knapsack DP. On real workloads with N=500,
fast-path is ~10–50× faster (the micro-bench is in :mod:`bench.fast_path_bench`).

Concurrency
-----------

Skill loading is cached behind an RLock; reload happens on
``invalidate_cache()`` (called by ``crystallize_skill`` to pick up new
skills) or on a TTL (default 60s) for resilience to vault edits made
out-of-band.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────

# Cache TTL for the loaded skill registry. Lower = picks up out-of-band
# vault edits faster; higher = lower per-call overhead. Crystallization
# events bypass the TTL via explicit cache invalidation, so this only
# matters for human/tool-driven skill edits.
DEFAULT_CACHE_TTL_S = 60.0

# If more than this fraction of recipe fragments are missing from the
# engine, fall through to the full pipeline. A partial recipe is usually
# worse than a fresh selection: knapsack would have picked replacements
# the recipe never saw.
DEFAULT_MAX_MISSING_FRACTION = 0.50

# Token budget enforcement: if the matched recipe's total tokens exceed
# the user's budget, we trim from the tail (lowest-rank recipe entries
# go first). Recipe is already in priority order from crystallizer.
DEFAULT_TRIM_FROM_TAIL = True


# ── Types ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _SkillEntry:
    """A loaded promoted skill, ready for matching."""
    skill_id: str
    cluster_id: str
    pattern: re.Pattern[str]
    fragment_recipe: tuple[str, ...]
    weight_profile: dict[str, float]
    fitness: float


@dataclass
class FastPathResult:
    """Outcome of a fast-path match. Conforms to a subset of the
    optimize_context return shape so the caller can splice it in.
    """
    skill_id: str
    cluster_id: str
    selected_fragments: list[dict[str, Any]]
    weight_profile: dict[str, float]
    fitness: float
    recipe_size: int
    matched_present: int
    matched_missing: int
    elapsed_ms: float


@dataclass
class _Cache:
    skills: list[_SkillEntry] = field(default_factory=list)
    loaded_at: float = 0.0


# ── Router ──────────────────────────────────────────────────────────

class FastPathRouter:
    """Match queries against promoted crystallized skills.

    Stateless w.r.t. the engine: takes a ``skill_lister`` (typically
    ``SkillEngine.list_skills``) and a ``fragment_lookup`` (typically
    ``engine.get_fragment``) at construction time. This keeps the
    router unit-testable without any of the broader engine or vault.

    Thread-safe.
    """

    def __init__(
        self,
        skill_lister: Callable[[], list[dict[str, Any]]],
        fragment_lookup: Callable[[str], dict[str, Any] | None],
        *,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
    ):
        self._lister = skill_lister
        self._lookup = fragment_lookup
        self._cache_ttl = cache_ttl_s
        self._max_missing = max_missing_fraction
        self._cache = _Cache()
        self._lock = threading.RLock()
        # Telemetry — read by the engine to surface in optimize_context.
        self._stats: dict[str, int] = {
            "attempts": 0,
            "hits": 0,
            "stale_misses": 0,
            "no_match": 0,
        }

    # ── Public API ──────────────────────────────────────────────

    def try_match(
        self,
        query: str,
        token_budget: int = 0,
    ) -> FastPathResult | None:
        """Return a FastPathResult if a promoted skill matches and is fresh.

        Returns None on any of:
          - empty query
          - no loaded promoted skills
          - no skill matches the query
          - matched skill's recipe is too stale (missing fragments)

        Never raises; failures degrade silently to None so the caller
        can fall through to the full pipeline.
        """
        if not query:
            return None
        with self._lock:
            self._stats["attempts"] += 1
        try:
            t0 = time.perf_counter()
            skills = self._get_skills()
            if not skills:
                with self._lock:
                    self._stats["no_match"] += 1
                return None

            # Best match: highest-fitness skill whose pattern fires.
            best: _SkillEntry | None = None
            for sk in skills:
                if sk.pattern.search(query):
                    if best is None or sk.fitness > best.fitness:
                        best = sk
            if best is None:
                with self._lock:
                    self._stats["no_match"] += 1
                return None

            present, missing, fragments = self._materialize_recipe(
                best.fragment_recipe, token_budget=token_budget
            )
            total = present + missing
            if total == 0 or (missing / total) > self._max_missing:
                with self._lock:
                    self._stats["stale_misses"] += 1
                logger.debug(
                    "fast-path stale: skill=%s present=%d missing=%d",
                    best.skill_id, present, missing,
                )
                return None

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._stats["hits"] += 1
            return FastPathResult(
                skill_id=best.skill_id,
                cluster_id=best.cluster_id,
                selected_fragments=fragments,
                weight_profile=best.weight_profile,
                fitness=best.fitness,
                recipe_size=len(best.fragment_recipe),
                matched_present=present,
                matched_missing=missing,
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            logger.debug("fast-path error (degrading to None): %s", e)
            return None

    def invalidate_cache(self) -> None:
        """Force reload of the skill registry on the next ``try_match``.

        Called by ``crystallize_skill`` when a new promoted skill is
        written, so the next request can use it immediately rather than
        waiting for the TTL.
        """
        with self._lock:
            self._cache = _Cache()

    def stats(self) -> dict[str, int | float]:
        """Read-only snapshot of router telemetry."""
        with self._lock:
            attempts = self._stats["attempts"] or 1
            return {
                **self._stats,
                "hit_rate": round(self._stats["hits"] / attempts, 4),
                "loaded_skills": len(self._cache.skills),
            }

    # ── Internals ───────────────────────────────────────────────

    def _get_skills(self) -> list[_SkillEntry]:
        with self._lock:
            now = time.time()
            if (now - self._cache.loaded_at) < self._cache_ttl and self._cache.skills:
                return self._cache.skills
            self._cache = _Cache(
                skills=self._load_skills(),
                loaded_at=now,
            )
            return self._cache.skills

    def _load_skills(self) -> list[_SkillEntry]:
        """Read the vault registry and parse promoted skills.

        Skills missing trigger or recipe metadata are silently skipped —
        the router is permissive on input shape so vault edits made by
        humans don't crash the hot path.
        """
        out: list[_SkillEntry] = []
        try:
            entries = self._lister() or []
        except Exception as e:
            logger.debug("skill_lister failed: %s", e)
            return out

        for info in entries:
            if not isinstance(info, dict):
                continue
            status = info.get("status", "")
            if status != "promoted":
                continue
            sid = str(info.get("skill_id", ""))
            cid = str(info.get("cluster_id", ""))
            metrics = info.get("metrics") or {}
            fitness = float(metrics.get("fitness_score", 0.0))

            # Recipe + pattern + weights live in the skill's tool.py
            # (faithfully written by SkillSynthesizer.synthesize_from_success).
            # Read them via execed module dict — same convention used by
            # auto-execute promoted skills at server.py:1330+.
            tool_path = info.get("path", "")
            recipe, pattern, weights = self._extract_from_tool(tool_path, sid)
            if pattern is None or not recipe:
                continue
            out.append(_SkillEntry(
                skill_id=sid,
                cluster_id=cid,
                pattern=pattern,
                fragment_recipe=tuple(recipe),
                weight_profile=weights,
                fitness=fitness,
            ))
        return out

    @staticmethod
    def _extract_from_tool(
        tool_path: str, skill_id: str
    ) -> tuple[list[str], re.Pattern[str] | None, dict[str, float]]:
        """Pull TRIGGER_PATTERN, FRAGMENT_RECIPE, WEIGHT_PROFILE from a skill's tool.py.

        We exec the module body in an isolated namespace. Skills are
        generated by us (no untrusted code path) but we still scope the
        exec with a minimal globals dict to keep blast radius small.
        """
        from pathlib import Path

        if not tool_path:
            return [], None, {}
        p = Path(tool_path) / "tool.py"
        if not p.exists():
            return [], None, {}
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            return [], None, {}

        ns: dict[str, Any] = {"__name__": f"entroly.skill.{skill_id}"}
        try:
            exec(compile(src, str(p), "exec"), ns)
        except Exception as e:
            logger.debug("skill %s tool.py exec failed: %s", skill_id, e)
            return [], None, {}

        recipe = ns.get("FRAGMENT_RECIPE") or []
        if not isinstance(recipe, list):
            recipe = []
        recipe = [str(x) for x in recipe if x]

        pattern = ns.get("TRIGGER_PATTERN")
        if not isinstance(pattern, re.Pattern):
            pattern = None

        weights = ns.get("WEIGHT_PROFILE") or {}
        if not isinstance(weights, dict):
            weights = {}
        else:
            weights = {str(k): float(v) for k, v in weights.items()
                       if isinstance(v, (int, float))}

        return recipe, pattern, weights

    def _materialize_recipe(
        self,
        recipe: tuple[str, ...],
        token_budget: int = 0,
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Look up each recipe entry in the engine.

        Returns (present_count, missing_count, fragments). Fragments
        match the optimize_context return shape (id, source, content,
        token_count, …) for a clean splice into the result dict.
        Trims from the tail when token_budget is positive.
        """
        present = 0
        missing = 0
        fragments: list[dict[str, Any]] = []
        used_tokens = 0
        for entry in recipe:
            f = None
            try:
                f = self._lookup(entry)
            except Exception:
                f = None
            if f is None:
                missing += 1
                continue
            present += 1
            tok = int(f.get("token_count", 0) or 0)
            if token_budget > 0 and used_tokens + tok > token_budget and fragments:
                # Recipe is in priority order; stop at first overflow once
                # we already have at least one fragment (don't return empty).
                break
            used_tokens += tok
            fragments.append(f)
        return present, missing, fragments


__all__ = [
    "DEFAULT_CACHE_TTL_S",
    "DEFAULT_MAX_MISSING_FRACTION",
    "FastPathResult",
    "FastPathRouter",
]
