"""Measure whether retrieval returns a gold source within the token budget.

The credited reduction uses a full-corpus baseline for every query and credits
zero for a query with no gold source in its selection. Gold-source presence is a
retrieval diagnostic. Provider usage and answer quality require separate tests.

Ground truth maps each query to the SET of files that legitimately answer it.
A set, not one path: this repository carries parallel Rust and Python
implementations, so scoring against a single arbitrary file measures the
benchmark's assumption rather than the retriever. Every alternative is one a
maintainer can verify by reading it, so the gold set stays auditable.

The scoped excerpt check runs two separate prompts and tests whether selected
content contains the implementation details needed to answer them. It leaves
the original source-label score and exit status independent of that check.

    python benchmarks/evidence_retention.py [--budget 2000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import time

# (query, acceptable answer files) — see the module docstring on why this is a set.
GOLD: list[tuple[str, tuple[str, ...]]] = [
    ("where is the global checkpoint cap enforced",
     ("entroly/checkpoint.py",)),
    ("how does git file discovery avoid hanging the watcher",
     ("entroly/auto_index.py",)),
    ("where does the proxy inject compressed context into requests",
     ("entroly/proxy.py", "entroly-core/src/proxy.rs")),
    ("what does entroly doctor check and how does it report failures",
     ("entroly/cli.py",)),
    ("how is a byte range verified against a source snapshot",
     ("entroly/source_span.py",)),
    ("where are query conditioned fragments selected under a token budget",
     ("entroly/qccr.py", "entroly-qccr/src/lib.rs")),
    ("how are vault beliefs listed and read",
     ("entroly/vault.py",)),
    ("where is the air gap outbound guard installed",
     ("entroly/air_gap.py",)),
]

# (query, implementation path, required excerpts keyed by the behavior tested).
SCOPED_EVIDENCE: tuple[tuple[str, str, dict[str, str]], ...] = (
    (
        "how does the Python incremental watcher prevent git file discovery from hanging",
        "entroly/auto_index.py",
        {
            "temporary_file_capture": 'tempfile.TemporaryFile(mode="w+b") as capture',
            "stdout_redirected": "stdout=capture",
            "bounded_wait": "proc.wait(timeout=timeout)",
            "timeout_handling": "except subprocess.TimeoutExpired:",
            "process_tree_termination": "_terminate_process_tree(proc)",
        },
    ),
    (
        "how does a v2 context receipt verify a fragment byte range against a source snapshot",
        "entroly/source_span.py",
        {
            "source_digest_check": "digest(source_bytes) != self.source_digest",
            "byte_bounds_check": "0 <= self.byte_start <= self.byte_end <= len(source_bytes)",
            "fragment_digest_check": "digest(source_bytes[self.byte_start:self.byte_end]) == self.fragment_digest",
            "line_coordinates_check": "derive_lines(source_bytes, self.byte_start, self.byte_end) == (self.line_start, self.line_end)",
        },
    ),
)


def _logical(path: str) -> str:
    """Resolve a chunk of an oversized file back to the file itself.

    A large file is indexed as `path`, `path#1`, `path#2`, ... so retrieving
    `entroly/cli.py#16` IS retrieving entroly/cli.py. Comparing the raw source
    against the gold path scores every chunked file as a miss no matter how well
    it ranked -- and the only files large enough to be chunked here were exactly
    the three this benchmark reported as permanently unretrievable.
    """
    head, sep, tail = path.rpartition("#")
    return head if sep and tail.isdigit() else path


def _sources(selection) -> set[str]:
    return {_fragment_source(frag) for frag in selection or [] if isinstance(frag, dict)}


def _fragment_source(fragment: dict) -> str:
    return _logical(str(fragment.get("source") or "").removeprefix("file:"))


def _missing_evidence(selection: list[dict], source: str, required: dict[str, str]) -> list[str]:
    excerpts = []
    for fragment in selection:
        if not isinstance(fragment, dict) or _fragment_source(fragment) != source:
            continue
        content = fragment.get("content")
        if isinstance(content, str):
            excerpts.append(" ".join(content.split()))
    return [
        name for name, snippet in required.items()
        if not any(snippet in excerpt for excerpt in excerpts)
    ]


def _scoped_excerpt_report(engine, budget: int, fragments: list[dict]) -> dict:
    rows = []
    for query, source, required in SCOPED_EVIDENCE:
        result = engine.optimize_context(budget, query)
        selection = result.get("selected_fragments") or result.get("selected") or []
        missing = _missing_evidence(selection, source, required)
        rows.append({
            "query": query,
            "source": source,
            "source_retained": source in _sources(selection),
            "retained": not missing,
            "matched_evidence": [name for name in required if name not in missing],
            "missing_evidence": missing,
            "missing_from_index": _missing_evidence(fragments, source, required),
            "tokens_used": int(result.get("tokens_used") or result.get("total_tokens") or 0),
        })
    hits = sum(row["retained"] for row in rows)
    return {"hits": hits, "total": len(rows), "recall": hits / len(rows) if rows else None, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--json", dest="json_out", default="")
    ap.add_argument("--reindex", action="store_true",
                    help="re-ingest before measuring instead of trusting the "
                         "restored checkpoint (which may predate a fix)")
    args = ap.parse_args()

    os.environ.setdefault("ENTROLY_SOURCE", os.getcwd())
    from entroly.server import EntrolyConfig, EntrolyEngine

    engine = EntrolyEngine(config=EntrolyConfig())
    engine._ensure_index_loaded()
    fragments = list(engine._rust.export_fragments())
    if not fragments or args.reindex:
        # `_ensure_index_loaded` restores a checkpoint; it never ingests. Left
        # alone it will happily score whatever stale corpus is on disk.
        from entroly.auto_index import auto_index
        auto_index(engine)
        fragments = list(engine._rust.export_fragments())
    corpus_tokens = sum(int(f.get("token_count", 0) or 0) for f in fragments)
    if not fragments:
        raise SystemExit("index is empty and ingest produced nothing")

    # Self-contamination: this harness carries every query verbatim next to
    # its answer path, so indexing it hands the retriever the answer key. It was
    # measured being retrieved at rank 4 for a gold query on the strength of the
    # query text it holds. Any document that quotes a gold query is evicted
    # before scoring, and the eviction is reported rather than done silently --
    # a contaminated corpus that scores well is not evidence of anything.
    queries = [query for query, _ in GOLD] + [query for query, _, _ in SCOPED_EVIDENCE]
    contaminated: set[str] = set()
    for frag in fragments:
        text = str(frag.get("content") or "")
        if any(query in text for query in queries):
            contaminated.add(str(frag.get("source") or ""))
    gold_paths = {path for _, gold in GOLD for path in gold}
    contaminated = {
        source for source in contaminated
        if _logical(source.removeprefix("file:")) not in gold_paths
    }
    if contaminated:
        removed = engine.remove_sources(sorted(contaminated))
        print(f"  contamination : evicted {len(contaminated)} doc(s) quoting a "
              f"gold query ({removed.get('status', 'ok')})")
        for source in sorted(contaminated):
            print(f"                  - {source}")
        fragments = list(engine._rust.export_fragments())
        corpus_tokens = sum(int(f.get("token_count", 0) or 0) for f in fragments)

    # A retrieval benchmark whose gold answers are absent from the corpus is not
    # measuring retrieval — it is measuring ingest, and reporting the result as a
    # recall score. That happened here: cli.py (266 KB), proxy.py (255 KB) and
    # auto_index.py (65 KB) all exceeded the oversized-file cap and were dropped
    # at index time, so recall sat at 0.62/0.75 no matter what the ranker did.
    # Four structurally different ranking changes produced bit-identical scores
    # before anyone checked the corpus. Fail loudly instead.
    present = _sources(fragments)
    missing = sorted({path for _, gold in GOLD if present.isdisjoint(gold) for path in gold})
    if missing:
        raise SystemExit(
            "gold files absent from the index, so retrieval cannot be scored: "
            + ", ".join(missing)
            + "\nthis is an ingest defect, not a ranking result"
        )

    rows = []
    for query, gold in GOLD:
        start = time.perf_counter()
        result = engine.optimize_context(args.budget, query)
        latency = time.perf_counter() - start
        selection = result.get("selected_fragments") or result.get("selected") or []
        sources = _sources(selection)
        retained = any(g in sources for g in gold)
        used = int(result.get("tokens_used") or result.get("total_tokens") or 0)
        rows.append({
            "query": query,
            "gold": list(gold),
            "retained": retained,
            "tokens_used": used,
            "selected": len(selection),
            "selected_sources": sorted(sources),
            "selector": result.get("selector") or result.get("method"),
            "latency_s": round(latency, 3),
            "status": result.get("status", "ok"),
        })
        print(f"  {'HIT ' if retained else 'MISS'}  {used:>6} tok  "
              f"{latency:5.2f}s  {gold[0]:<34} {query[:44]}", flush=True)

    hits = [r for r in rows if r["retained"]]
    recall = len(hits) / len(rows)
    # Savings are credited ONLY on tasks whose evidence survived. A miss
    # contributes zero, never a "saving" — that is the accounting error this
    # benchmark exists to prevent.
    credited = sum(max(0, corpus_tokens - r["tokens_used"]) for r in hits)
    naive = corpus_tokens * len(rows)
    reduction = (credited / naive) if naive else 0.0
    mean_used = (sum(r["tokens_used"] for r in hits) / len(hits)) if hits else 0

    print(f"\n  corpus                : {corpus_tokens:,} tokens over {len(fragments)} fragments")
    print(f"  evidence retention    : {len(hits)}/{len(rows)}  (recall {recall:.2f})")
    print(f"  mean tokens on hits   : {mean_used:,.0f}")
    print(f"  credited reduction    : {reduction:.4%}  (full corpus per query; misses credited 0)")
    print(f"  max latency           : {max(r['latency_s'] for r in rows):.2f}s")
    verdict = (
        "ALL QUERIES RETAINED A GOLD SOURCE" if recall == 1.0
        else "PARTIAL — some queries missed every gold source"
        if hits else "NO GOLD SOURCES RETAINED"
    )
    print(f"  verdict               : {verdict}")

    excerpts = _scoped_excerpt_report(engine, args.budget, fragments)
    print(f"  scoped excerpts       : {excerpts['hits']}/{excerpts['total']} retained")
    for row in excerpts["rows"]:
        if row["missing_evidence"]:
            print(f"    {row['source']} missing " + ", ".join(row["missing_evidence"]))
        if row["missing_from_index"]:
            print("      absent from the index " + ", ".join(row["missing_from_index"]))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({
                "corpus_tokens": corpus_tokens,
                "fragments": len(fragments),
                "budget": args.budget,
                "recall": recall,
                "credited_reduction": reduction,
                "verdict": verdict,
                "rows": rows,
                "scoped_excerpt_retention": excerpts,
            }, fh, indent=2)
    return 0 if recall == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
