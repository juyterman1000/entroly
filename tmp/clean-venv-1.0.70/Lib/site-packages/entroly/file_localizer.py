"""File-localization service — the one public entry point for
engine_s6 (deterministic edit-target rerank).

Single responsibility: given a query and either {path: content} files
or a list of selected fragments, return them re-ordered so the most
plausible edit targets surface first. Recall-safe by construction:
on any error the original order is returned untouched.

Validation
----------
The underlying rerank (`Tier0Localizer.rerank_edit_target`) was held-out
validated on SWE-bench Lite n=36 seed=7 — paired McNemar over
engine_s5: b=0 / c=5 at hit@10 (one-sided p=0.031, strict Pareto
improvement; zero regressions on @10 or @20). Manual review of three
doc/test/config gold targets in-sample showed no harm. See
`bench/_engine_s6_paired.py` and `bench/_engine_s6_ship_decision.py`.

Public surface
--------------
- `localize_files(files, query, *, k, base_ranked)`  → list[str]
  Pure file-path rerank when caller already has `{path: content}`.
- `localize_fragments(fragments, query, *, k)`       → list[dict]
  Higher-level: group by `source`, rerank, re-emit fragments in the
  new source order (within-source fragment order preserved). This is
  the surface the SDK / MCP / proxy call after their knapsack selection.

These functions never raise on user input; they fail-open to the input
order. The Tier0Localizer construction is lazy-imported so module import
remains cheap.
"""
from __future__ import annotations

from typing import Any

__all__ = ["localize_files", "localize_fragments"]


def localize_files(
    files: dict[str, str],
    query: str,
    *,
    k: int = 20,
    base_ranked: list[str] | None = None,
) -> list[str]:
    """Rank a `{path: content}` corpus by relevance to a query.

    Applies engine_s6 (deterministic edit-target prior on top of the
    engine_s5 fusion rerank). When `base_ranked` is provided it is used
    as the base order; otherwise the localizer's own fused ranking is
    computed. Returns at most `k` paths.

    Recall-safe: any internal error → returns `base_ranked` (or, if
    none, the original `files.keys()` order) unchanged.
    """
    if not files or not query:
        return (base_ranked or list(files))[:k]
    try:
        from .localization import Tier0Localizer
        if base_ranked is None:
            loc = Tier0Localizer(files)
            base_ranked = loc.rank(query, k=max(k, len(files)))
        else:
            loc = Tier0Localizer.for_edit_rerank(files)
            # Filter to entries actually present in the corpus so the
            # rerank operates on a clean candidate set.
            base_ranked = [p for p in base_ranked if p in files]
        return loc.rerank_edit_target(base_ranked, query, k=k)
    except Exception:  # noqa: BLE001 — fail-open to base order
        return (base_ranked or list(files))[:k]


def localize_fragments(
    fragments: list[dict[str, Any]],
    query: str,
    *,
    k: int | None = None,
) -> list[dict[str, Any]]:
    """Reorder a knapsack-selected fragment list by edit-target prior.

    Fragments are grouped by their `source` field (interpreted as a
    file path when it looks like one), the source order is rerun
    through engine_s6, then fragments are re-emitted in the new source
    order. Within-source fragment order is preserved.

    Sources without a content body, an empty `source`, or a `source`
    that does not look like a file path still participate in the
    rerank — engine_s6's structural classifier treats unknown shapes
    as "source" (cls=1) so they neither get promoted nor demoted
    relative to genuine source files. Recall-safe: any error returns
    the input list unchanged.
    """
    if not fragments or not query:
        return fragments
    try:
        # Group preserving first-occurrence order of each source so the
        # base_ranked we hand to engine_s6 mirrors the knapsack's own
        # ranking (which is meaningful — it already encodes relevance).
        groups: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        unkeyed: list[dict[str, Any]] = []
        for frag in fragments:
            src = (frag.get("source") or "").strip()
            if not src:
                unkeyed.append(frag)
                continue
            if src not in groups:
                groups[src] = []
                order.append(src)
            groups[src].append(frag)

        if len(order) < 2:
            return fragments      # nothing to reorder

        files_map: dict[str, str] = {}
        for src in order:
            files_map[src] = "\n".join(
                (f.get("content") or "") for f in groups[src]
            )

        kk = k if k is not None else max(20, len(order))
        from .localization import Tier0Localizer
        reranked = Tier0Localizer.for_edit_rerank(files_map).rerank_edit_target(
            order, query, k=kk,
        )

        # Emit in reranked order, then any remainder (sources missing
        # from the rerank window's tail), then unkeyed fragments at
        # the end — never lose a fragment.
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for src in reranked:
            if src in groups and src not in seen:
                out.extend(groups[src])
                seen.add(src)
        for src in order:
            if src not in seen:
                out.extend(groups[src])
                seen.add(src)
        out.extend(unkeyed)
        return out
    except Exception:  # noqa: BLE001 — fail-open to input order
        return fragments
