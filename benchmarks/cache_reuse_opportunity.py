#!/usr/bin/env python3
"""How much prompt-prefix cache reuse does selection leave on the table?

Providers cache by *prefix*: a request sharing a leading run of tokens with an
earlier one bills that run at a large discount. `entroly/cache_aligner.py`
matches on a SHA-256 of the whole context, so it is all-or-nothing -- any change
in selection is a total miss and everything is billed fresh. No selector
considers cache state at all (`qccr.py` contains zero references to it).

This measures the gap between what is billed today and what could be billed if
selection preserved a prefix, using two quantities per consecutive query pair:

  shared_prefix   longest common prefix of the two rendered contexts
  shared_content  fragments present in BOTH selections

`shared_content` is the interesting one. Where the same fragments are selected
but emitted in a different order, a cache-aware *ordering* recovers a long
common prefix at zero cost in relevance -- the fragments are identical, only
their sequence changed. That is a free win if it is large, and evidence the
mechanism is not worth building if it is small.

No model and no API key: cached-vs-fresh token counts are arithmetic once the
prefix is known. That is the point -- this validates a cost claim without
needing an answer-quality judgement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.graph_lane_quality import (
    REPO,
    Task,
    _pool_for,
    _read,
    _tokens,
    _tracked_python_files,
)


def _render(fragments: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"### {str(f.get('source', '')).removeprefix('file:')}\n{f.get('content', '')}"
        for f in fragments
    )


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def run(limit: int, budget: int, seed: int) -> dict[str, Any]:
    from entroly.qccr import select as qccr_select

    payload = json.loads(
        (REPO / "benchmarks" / "results" / "graph_lane_tasks.json").read_text(encoding="utf-8")
    )
    tasks = [Task(**t) for t in payload["tasks"]][:limit]
    corpus = sorted(
        {str(p.relative_to(REPO)).replace("\\", "/") for p in _tracked_python_files()}
    )

    rendered: list[tuple[str, str, set[str]]] = []
    for task in tasks:
        pool = _pool_for(task, corpus, 48, seed)
        texts = {rel: _read(rel) for rel in pool}
        frags = [
            {"id": f"file:{rel}", "source": f"file:{rel}", "content": texts[rel],
             "token_count": _tokens(texts[rel]), "relevance": 0.5}
            for rel in pool
        ]
        picked = qccr_select(frags, token_budget=budget, query=task.query)
        sources = {str(f.get("source", "")) for f in picked}
        rendered.append((task.query, _render(picked), sources))

    rows: list[dict[str, Any]] = []
    for (q_a, text_a, src_a), (q_b, text_b, src_b) in zip(rendered, rendered[1:]):
        shared_prefix_chars = _common_prefix_len(text_a, text_b)
        overlap = src_a & src_b
        rows.append({
            "query_a": q_a[:60],
            "query_b": q_b[:60],
            "tokens_b": _tokens(text_b),
            "shared_prefix_tokens": shared_prefix_chars // 4,
            "fragments_a": len(src_a),
            "fragments_b": len(src_b),
            "fragments_shared": len(overlap),
            "content_overlap_frac": len(overlap) / max(len(src_b), 1),
        })

    return {
        "pinned_ref": payload["pinned_ref"],
        "budget": budget,
        "pairs": len(rows),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "cache_reuse_opportunity.json")
    args = ap.parse_args()

    payload = run(args.limit, args.budget, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload["rows"]
    if not rows:
        print("no pairs")
        return 0
    pre = [r["shared_prefix_tokens"] / max(r["tokens_b"], 1) for r in rows]
    ovl = [r["content_overlap_frac"] for r in rows]
    pre.sort()
    ovl.sort()

    def q(v: list[float], p: float) -> float:
        return v[min(len(v) - 1, int(p * len(v)))]

    print(f"pairs {payload['pairs']}  budget {payload['budget']}")
    print(f"\n  {'metric':<34}{'median':>9}{'p90':>9}{'max':>9}")
    print(f"  {'shared PREFIX (today, billed)':<34}"
          f"{q(pre,0.5):>8.1%}{q(pre,0.9):>9.1%}{pre[-1]:>9.1%}")
    print(f"  {'shared CONTENT (reorderable)':<34}"
          f"{q(ovl,0.5):>8.1%}{q(ovl,0.9):>9.1%}{ovl[-1]:>9.1%}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
