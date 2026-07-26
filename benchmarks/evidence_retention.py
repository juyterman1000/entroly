"""Does compression retain answer-critical evidence — or just withhold context?

The standard this measures:

    real token saving = fewer tokens AND the answer-critical evidence retained

A compressor that drops the answer is not saving tokens, it is losing the task.
So savings are only credited on tasks where the gold file survived selection;
recall and cost are reported separately and never averaged into one headline.

Ground truth maps each query to the SET of files that legitimately answer it.
A set, not one path: this repository carries parallel Rust and Python
implementations, so scoring against a single arbitrary file measures the
benchmark's assumption rather than the retriever. Every alternative is one a
maintainer can verify by reading it, so the gold set stays auditable.

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


def _sources(selection) -> set[str]:
    out = set()
    for frag in selection or []:
        if not isinstance(frag, dict):
            continue
        src = str(frag.get("source") or "")
        out.add(src.removeprefix("file:"))
    return out


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

    # A retrieval benchmark whose gold answers are absent from the corpus is not
    # measuring retrieval — it is measuring ingest, and reporting the result as a
    # recall score. That happened here: cli.py (266 KB), proxy.py (255 KB) and
    # auto_index.py (65 KB) all exceeded the oversized-file cap and were dropped
    # at index time, so recall sat at 0.62/0.75 no matter what the ranker did.
    # Four structurally different ranking changes produced bit-identical scores
    # before anyone checked the corpus. Fail loudly instead.
    present = {
        str(f.get("source") or "").removeprefix("file:") for f in fragments
    }
    def _indexed(path: str) -> bool:
        return any(p == path or p.startswith(path + "#") for p in present)

    missing = sorted({
        g for _, gold in GOLD for g in gold if not _indexed(g)
    } - {g for _, gold in GOLD for g in gold if any(_indexed(a) for a in gold)})
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
    naive = corpus_tokens * len(hits)
    reduction = (credited / naive) if naive else 0.0
    mean_used = (sum(r["tokens_used"] for r in hits) / len(hits)) if hits else 0

    print(f"\n  corpus                : {corpus_tokens:,} tokens over {len(fragments)} fragments")
    print(f"  evidence retention    : {len(hits)}/{len(rows)}  (recall {recall:.2f})")
    print(f"  mean tokens on hits   : {mean_used:,.0f}")
    print(f"  credited reduction    : {reduction:.4%}  (misses credited 0)")
    print(f"  max latency           : {max(r['latency_s'] for r in rows):.2f}s")
    verdict = (
        "USEFUL SAVING PROVEN" if recall == 1.0
        else "PARTIAL — evidence lost on some tasks; not a clean saving"
        if hits else "NOT PROVEN — no evidence retained"
    )
    print(f"  verdict               : {verdict}")

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
            }, fh, indent=2)
    return 0 if recall == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
