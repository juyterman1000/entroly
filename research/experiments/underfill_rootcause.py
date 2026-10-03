"""Root-cause the unused context budget (Stage 0.1).

Unused budget is foregone evidence. If the selector refuses to fill an allowed
window, allocation policy matters more than ranking quality.

Candidate mechanisms and how each is distinguished here:

  candidate cap      qccr.py:133 `cap = max(FLOOR, budget // 64)`, applied at
                     :200. Reported as `cap` vs `corpus_files` -- if cap >
                     corpus_files it cannot bind.
  corpus exhaustion  reported as corpus_tokens vs budget.
  per-file ceiling   QCCR emits one synthetic fragment per source file
                     (qccr.py:255), so selected_files x tokens/file bounds the
                     output independently of the budget. Reported directly.
  relevance floor    visible as files_selected plateauing while budget grows.
  token estimation   reported as the ratio of estimated to delivered tokens.

Run: python research/experiments/underfill_rootcause.py
Writes research/ledger/underfill.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "ledger" / "underfill.json"

BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]
QUERIES = [
    "how does knapsack selection handle pinned fragments within the token budget",
    "how are context receipts recovered from omitted chunks",
    "what happens when the native rust engine is unavailable",
]


def collect(max_files: int) -> list[tuple[str, str]]:
    out: dict[str, str] = {}
    for p in sorted(ROOT.glob("entroly/**/*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(t) > 200:
            out[str(p.relative_to(ROOT)).replace("\\", "/")] = t
        if len(out) >= max_files:
            break
    return sorted(out.items())


def main() -> int:
    files = collect(300)
    corpus_tokens = sum(max(1, len(c) // 4) for _, c in files)
    print(f"corpus: {len(files)} files ~{corpus_tokens:,} est tokens")

    # The cap formula and floor, read from source rather than assumed.
    from entroly import qccr as q
    floor = getattr(q, "_PREFILTER_FILE_FLOOR", None)
    print(f"_PREFILTER_FILE_FLOOR = {floor}")

    rows = []
    with tempfile.TemporaryDirectory() as d:
        os.environ["ENTROLY_DIR"] = d
        os.environ["ENTROLY_NO_SELF_HEAL"] = "1"
        from entroly.config import EntrolyConfig
        from entroly.server import EntrolyEngine

        eng = EntrolyEngine(config=EntrolyConfig(
            default_token_budget=8000, checkpoint_dir=d,
            auto_checkpoint_interval=99999))
        for src, content in files:
            eng.ingest_fragment(content, source=src,
                                token_count=max(1, len(content) // 4))

        for budget in BUDGETS:
            cap = max(floor or 0, budget // 64)
            for query in QUERIES:
                opt = eng.optimize_context(token_budget=budget, query=query)
                sel = opt.get("selected") or []
                toks = sum(int((s.get("token_count") or 0)) for s in sel
                           if isinstance(s, dict))
                srcs = {s.get("source") for s in sel if isinstance(s, dict)}
                rows.append({
                    "budget": budget,
                    "query": query[:46],
                    "cap_files": cap,
                    "corpus_files": len(files),
                    "cap_binds": cap < len(files),
                    "corpus_tokens": corpus_tokens,
                    "selected_items": len(sel),
                    "selected_files": len(srcs),
                    "selected_tokens": toks,
                    "fill_ratio": round(toks / budget, 4),
                    "tokens_per_file": round(toks / max(1, len(srcs)), 1),
                    "budget_utilization_reported": opt.get("budget_utilization"),
                    "selector": opt.get("selector"),
                    "method": opt.get("method"),
                })
                r = rows[-1]
                print(f"  b={budget:>6} fill={r['fill_ratio']:>6.1%} "
                      f"files={r['selected_files']:>4}/{len(files)} "
                      f"cap={cap:>5}{'*' if r['cap_binds'] else ' '} "
                      f"tok/file={r['tokens_per_file']:>7.1f} "
                      f"sel={toks:>6} {r['selector'] or r['method'] or ''}")

    agg = {}
    for b in BUDGETS:
        sub = [r for r in rows if r["budget"] == b]
        agg[str(b)] = {
            "mean_fill": round(sum(r["fill_ratio"] for r in sub)/len(sub), 4),
            "mean_files": round(sum(r["selected_files"] for r in sub)/len(sub), 1),
            "mean_tokens_per_file": round(sum(r["tokens_per_file"] for r in sub)/len(sub), 1),
            "cap_binds": sub[0]["cap_binds"],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"baseline_commit": "e31f082f",
                               "prefilter_file_floor": floor,
                               "by_budget": agg, "rows": rows}, indent=2),
                   encoding="utf-8")
    print("\n" + json.dumps(agg, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
