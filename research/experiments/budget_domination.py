"""Measure what fraction of the context budget the ranker actually controls.

Motivation: if force-pinned content consumes most of the budget, ranking
improvements cannot move task quality much, and algorithmic work on the
selector is misdirected. guardrails.rs:232 records a historical blowout
(90 pinned files = 167K tokens against an 8K budget) and the pin predicate has
since been narrowed to credential material. This re-measures on current main
rather than trusting that.

Isolation: each engine gets a fresh ENTROLY_DIR. Without it the engine lazily
loads this repository's own persisted index and the measurement silently
describes accumulated state instead of the corpus under test.

Run: python research/experiments/budget_domination.py
Writes research/ledger/budget_domination.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "ledger" / "budget_domination.json"

# Realistic corpus: this repository's own Python sources, which is the workload
# a coding agent actually faces.
CORPUS_GLOBS = ["entroly/*.py", "entroly/**/*.py"]
MAX_FILES = 120

QUERIES = [
    "how does knapsack selection handle pinned fragments within the token budget",
    "where is the BM25 relevance score computed for a query",
    "how are context receipts recovered from omitted chunks",
    "what happens when the native rust engine is unavailable",
    "how is a verification claim checked against supplied evidence",
]
BUDGETS = [2000, 8000, 32000]


def collect() -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for g in CORPUS_GLOBS:
        for p in ROOT.glob(g):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            try:
                t = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(t) > 200:
                seen[str(p.relative_to(ROOT)).replace("\\", "/")] = t
            if len(seen) >= MAX_FILES:
                break
    return sorted(seen.items())


def main() -> int:
    files = collect()
    print(f"corpus: {len(files)} files, {sum(len(c) for _, c in files):,} chars")

    rows = []
    for budget in BUDGETS:
        # One engine per budget, reused across queries. Rebuilding it per query
        # made this run past 10 minutes; ingestion dominates, and the budget is
        # the only construction-time variable.
        with tempfile.TemporaryDirectory() as d:
            os.environ["ENTROLY_DIR"] = d              # isolation, per above
            os.environ["ENTROLY_NO_SELF_HEAL"] = "1"
            from entroly.config import EntrolyConfig
            from entroly.server import EntrolyEngine

            cfg = EntrolyConfig(
                default_token_budget=budget,
                checkpoint_dir=d,
                auto_checkpoint_interval=99999,
            )
            eng = EntrolyEngine(config=cfg)
            for src, content in files:
                eng.ingest_fragment(content, source=src,
                                    token_count=max(1, len(content) // 4))

            frags = {f.get("fragment_id"): f for f in eng._rust.export_fragments()} \
                if getattr(eng, "_use_rust", False) else {}
            pinned_total = sum(
                int(f.get("token_count") or 0)
                for f in frags.values() if f.get("is_pinned")
            )
            n_pinned = sum(1 for f in frags.values() if f.get("is_pinned"))
            n_protected = sum(1 for f in frags.values() if f.get("is_protected"))

            for query in QUERIES:
                opt = eng.optimize_context(token_budget=budget, query=query)
                sel = opt.get("selected") or opt.get("fragments") or []
                sel_tokens = 0
                sel_pinned_tokens = 0
                for s in sel:
                    fid = (s.get("fragment_id") or s.get("id")) if isinstance(s, dict) else None
                    tk = int((s.get("token_count") if isinstance(s, dict) else 0) or 0)
                    if not tk and fid in frags:
                        tk = int(frags[fid].get("token_count") or 0)
                    sel_tokens += tk
                    if fid in frags and frags[fid].get("is_pinned"):
                        sel_pinned_tokens += tk

                ranker = sel_tokens - sel_pinned_tokens
                rows.append({
                    "budget": budget,
                    "query": query[:52],
                    "fragments_indexed": len(frags),
                    "pinned_fragments": n_pinned,
                        "protected_fragments": n_protected,
                    "pinned_tokens_in_corpus": pinned_total,
                    "selected_tokens": sel_tokens,
                    "selected_pinned_tokens": sel_pinned_tokens,
                    "ranker_controlled_tokens": ranker,
                    "pinned_share_of_budget": round(sel_pinned_tokens / budget, 4) if budget else None,
                    "ranker_share_of_budget": round(ranker / budget, 4) if budget else None,
                })
                print(f"  budget {budget:>6}  pinned {sel_pinned_tokens:>7} "
                      f"({sel_pinned_tokens/budget:>6.1%})  ranker {ranker:>7} "
                      f"({ranker/budget:>6.1%})  sel={sel_tokens:>7}  {query[:34]}")

    agg = {}
    for b in BUDGETS:
        sub = [r for r in rows if r["budget"] == b]
        if sub:
            agg[str(b)] = {
                "mean_pinned_share": round(sum(r["pinned_share_of_budget"] for r in sub)/len(sub), 4),
                "mean_ranker_share": round(sum(r["ranker_share_of_budget"] for r in sub)/len(sub), 4),
                "max_pinned_share": round(max(r["pinned_share_of_budget"] for r in sub), 4),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"baseline_commit": "e31f082f",
                               "corpus_files": len(files),
                               "by_budget": agg, "rows": rows}, indent=2), encoding="utf-8")
    print("\nby budget:", json.dumps(agg, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
