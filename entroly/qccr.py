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

from .native_status import QCCR_SYMBOLS, native_status, native_status_message

# QCCR ranking/selection is the Rust SSOT (entroly-qccr crate via PyO3). Import is
# guarded so the package still *imports* on a base, engine-less `pip install
# entroly` (the pure-Python surface protects compress()/universal_compress). QCCR
# itself requires the engine; calling it without one raises a clear, actionable
# error rather than crashing at import time.
_NATIVE_STATUS = native_status(QCCR_SYMBOLS)
if _NATIVE_STATUS.ok and _NATIVE_STATUS.module is not None:
    _rust_expand_query = _NATIVE_STATUS.module.py_qccr_expand_query
    _rust_rank_files = _NATIVE_STATUS.module.py_qccr_rank_files
    _rust_select = _NATIVE_STATUS.module.py_qccr_select
    _HAS_RUST = True
else:  # pragma: no cover - covered by the pure-Python CI surface
    _HAS_RUST = False

    def _rust_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            native_status_message(
                _NATIVE_STATUS,
                feature="QCCR (query-conditioned retrieval)",
            )
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


# Real-repo cost bound: cap the files that reach the per-file localizer and the
# Rust sentence-level selector, whose combined cost grows super-linearly in the
# file count (a query over an 858-file repo otherwise times out). The cap scales
# with the token budget — no budget can pack more files than this — and floors
# high enough that the validated small-corpus accuracy benchmarks, which pass at
# most a few dozen files, never reach it, so their selection stays byte-identical.
_PREFILTER_FILE_FLOOR = 128



def logical_source(source: str) -> str:
    """Storage identity -> retrieval identity.

    Oversized files are ingested as `file:path` plus `file:path#N` so each chunk
    keeps a unique storage identity (reconciliation requires exactly one
    fragment per source). Ranking must NOT see them as separate documents: a
    file split into 32 chunks gives every chunk a fraction of the query terms,
    so no chunk scores like the intact file, while small files keep all their
    terms in one document and win. Splitting also inflates the document count
    used for IDF. Grouping chunks back into one logical document restores
    parity between large and small files.
    """
    src = str(source or "")
    head, sep, tail = src.rpartition("#")
    return head if sep and tail.isdigit() else src


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
        by_file.setdefault(logical_source(raw.get("source", "")), []).append(raw)
    file_sources = list(by_file.keys())
    cap = max(_PREFILTER_FILE_FLOOR, int(token_budget) // 64)
    working_fragments = fragments
    if len(file_sources) > 1:
        file_texts = [
            "\n".join((r.get("content") or "") for r in by_file[s]) for s in file_sources
        ]
        # Max-passage scoring, not whole-document scoring.
        #
        # Regrouping a file's chunks is right for OUTPUT but wrong for RANKING:
        # it reintroduces the length penalty that chunking removed. cli.py as a
        # single document is ~65k tokens against an avgdl of ~2.7k, so BM25's
        # (1 - b + b*dl/avgdl) normalizer reaches ~18 and divides every term
        # frequency by it. Scoring each chunk on its own keeps dl close to avgdl
        # (length-neutral) while crediting a file with its BEST chunk, so a large
        # file competes on its most relevant passage instead of being penalised
        # for its bulk or diluted across it.
        #
        # This is why chunking and regrouping each failed alone: they trade the
        # same two failure modes back and forth. Measured here, both gave
        # 0.62/0.75/0.75 recall at 2k/8k/32k.
        chunk_keys: list[str] = []
        chunk_texts: list[str] = []
        for logical in file_sources:
            for part, raw in enumerate(by_file[logical]):
                chunk_keys.append(f"{logical}\x00{part}")
                chunk_texts.append(raw.get("content") or "")
        chunk_ranked = _rust_rank_files(chunk_keys, chunk_texts, query, overrides)
        best_by_file: dict[str, float] = {}
        for idx, score in chunk_ranked:
            if not 0 <= idx < len(chunk_keys):
                continue
            logical = chunk_keys[idx].split("\x00", 1)[0]
            if score > best_by_file.get(logical, float("-inf")):
                best_by_file[logical] = score
        ranked = [(i, best_by_file.get(src, 0.0)) for i, src in enumerate(file_sources)]
        if ranked and any(sc > 0 for _, sc in ranked):
            feedback_by_file = {
                source: sum(
                    float(r.get("feedback_multiplier", 1.0) or 1.0)
                    for r in by_file[source]
                ) / max(len(by_file[source]), 1)
                for source in file_sources
            }
            ranked.sort(
                key=lambda item: -item[1]
                * feedback_by_file[file_sources[item[0]]]
            )
            base_ranked = [file_sources[i] for i, _ in ranked]
            if len(base_ranked) > cap:
                # Keep the top-`cap` BM25F-ranked files; drop the long tail
                # before the super-linear localizer + sentence selector. The
                # tail scores ~0 and cannot fit the budget ahead of the head.
                base_ranked = base_ranked[:cap]
                kept = set(base_ranked)
                working_fragments = [
                    r for r in fragments
                    if logical_source(r.get("source", "")) in kept
                ]
                text_by_source = dict(zip(file_sources, file_texts))
                file_sources = base_ranked
                file_texts = [text_by_source[s] for s in base_ranked]
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
        {
            # Rust re-groups by source for per-file budgeting, so it must also
            # see the logical file, not the chunk.
            "source": logical_source(r.get("source", "")),
            "content": r.get("content") or "",
            "feedback_multiplier": float(
                r.get("feedback_multiplier", 1.0) or 1.0
            ),
        }
        for r in working_fragments
    ]
    out_json = _rust_select(
        json.dumps(slim),
        int(token_budget),
        query,
        json.dumps(overrides),
        json.dumps(preferred),
    )
    selected = json.loads(out_json)

    # QCCR emits one synthetic fragment per source file. Preserve the native
    # fragment IDs so callers can correlate compressed output with ingestion
    # receipts and feed outcomes back into the engine.
    source_fragment_ids: dict[str, list[str]] = {}
    for raw in fragments:
        source = logical_source(raw.get("source", ""))
        fragment_id = str(raw.get("fragment_id") or raw.get("id") or "")
        if fragment_id and fragment_id not in source_fragment_ids.setdefault(source, []):
            source_fragment_ids[source].append(fragment_id)
    for fragment in selected:
        origin_ids = source_fragment_ids.get(str(fragment.get("source") or ""), [])
        if origin_ids:
            fragment["source_fragment_ids"] = origin_ids
    return selected
