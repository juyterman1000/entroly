"""Query-Conditioned Compressive Retrieval (QCCR) — thin Python wrapper.

All ranking + selection logic is the Rust single source of truth in the
`entroly-qccr` crate (shared verbatim with the WASM/npm build), exposed here via
PyO3. Python owns only two things: config-driven weight overrides, and feeding
in the engine_s6 edit-target localizer reorder (a separate module) as a
preferred file order. Rust mandatory — no pure-Python fallback.

References: Robertson & Zaragoza (2009) BM25/BM25F; Carbonell & Goldstein (1998)
MMR. See entroly-qccr/src/lib.rs for the implementation.
"""
from __future__ import annotations

import json

# QCCR ranking/selection is the Rust SSOT (entroly-qccr crate via PyO3). Import is
# guarded so the package still *imports* on a base, engine-less `pip install
# entroly` (the pure-Python surface protects compress()/universal_compress). QCCR
# itself requires the engine; calling it without one raises a clear, actionable
# error rather than crashing at import time.
try:
    from entroly_core import py_qccr_expand_query as _rust_expand_query
    from entroly_core import py_qccr_rank_files as _rust_rank_files
    from entroly_core import py_qccr_select as _rust_select
    _HAS_RUST = True
except ImportError:  # pragma: no cover - covered by the pure-Python CI surface
    _HAS_RUST = False

    def _rust_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "QCCR (query-conditioned retrieval) requires the Entroly Rust engine. "
            "Install it with:  pip install 'entroly[full]'  "
            "(or run `maturin develop --release` in a source checkout)."
        )

    _rust_expand_query = _rust_rank_files = _rust_select = _rust_unavailable

_RANK_WEIGHTS_CACHE: dict[str, float] | None = None


def _load_rank_weights() -> dict[str, float]:
    """Per-repo ranking-weight OVERRIDES from the learned layer (tuning config).

    Weight DEFAULTS live in the Rust core; this returns only the overrides
    (possibly empty) that the archetype / PRISM / autotune pipeline wrote into
    tuning_config["qccr_rank"], to pass through to the Rust ranker. Cached.
    """
    global _RANK_WEIGHTS_CACHE
    if _RANK_WEIGHTS_CACHE is not None:
        return _RANK_WEIGHTS_CACHE
    weights: dict[str, float] = {}
    try:
        from .config import load_active_tuning_config
        active = load_active_tuning_config()
        if active is not None:
            _, cfg = active
            override = cfg.get("qccr_rank")
            if isinstance(override, dict):
                for k, v in override.items():
                    try:
                        weights[k] = float(v)
                    except (TypeError, ValueError):
                        continue
    except Exception:
        pass
    _RANK_WEIGHTS_CACHE = weights
    return weights


def _expanded_query_tokens(query: str) -> frozenset[str]:
    """Query expansion — Rust SSOT (kept for back-compat callers)."""
    return frozenset(_rust_expand_query(query))


def select(fragments: list[dict], token_budget: int, query: str = "") -> list[dict]:
    """Query-Conditioned Compressive Retrieval.

    Group fragments by file, rank (Rust BM25F + log-linear features), apply the
    engine_s6 edit-target localizer reorder, then extract the budget-bounded
    answer-relevant sentences per file — all in the Rust core. Returns synthetic
    per-file fragments. Empty query (or no expandable terms) ⇒ original
    fragments unchanged (no compression).
    """
    if not fragments:
        return []
    if not query:
        return fragments
    if not _rust_expand_query(query):
        return fragments

    overrides = _load_rank_weights()

    # engine_s6 edit-target rerank (a separate Python module) reorders the
    # ranked file list; the result is fed to Rust as the preferred file order so
    # the heavy ranking/selection stays a single Rust implementation.
    preferred: list[str] = []
    by_file: dict[str, list[dict]] = {}
    for raw in fragments:
        by_file.setdefault(raw.get("source", "") or "", []).append(raw)
    file_sources = list(by_file.keys())
    if len(file_sources) > 1:
        file_texts = [
            "\n".join((r.get("content") or "") for r in by_file[s]) for s in file_sources
        ]
        ranked = _rust_rank_files(file_sources, file_texts, query, overrides)
        if ranked and any(sc > 0 for _, sc in ranked):
            base_ranked = [file_sources[i] for i, _ in ranked]
            preferred = base_ranked
            try:
                from .file_localizer import localize_files
                files_map = dict(zip(file_sources, file_texts))
                preferred = localize_files(
                    files_map, query, k=len(base_ranked), base_ranked=base_ranked
                )
            except Exception:
                preferred = base_ranked

    slim = [
        {"source": r.get("source", "") or "", "content": r.get("content") or ""}
        for r in fragments
    ]
    out_json = _rust_select(
        json.dumps(slim),
        int(token_budget),
        query,
        json.dumps(overrides),
        json.dumps(preferred),
    )
    return json.loads(out_json)
