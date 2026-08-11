#!/usr/bin/env python3
"""Is context an index rather than a document?

Five measurements in this programme point the same way once read together:

  * qccr (shallow coverage of many files) beat hierarchical selection (deep
    coverage of few) 76.7% vs 3.3% on evidence delivery;
  * the codec gains came from holding IDENTIFIERS out verbatim, at which point
    compression and preservation stopped trading off;
  * cross-fragment redundancy is only 16-24%, so fragments carry distinct
    information rather than restating each other;
  * sound abstraction is 93-99% decidable at signature level and 6.5% at effect
    level;
  * the proxy defect destroyed identifiers and lost evidence catastrophically
    *at the same compression ratio* as codecs that kept everything.

The unifying reading: a token's value is its power as a **join key** between the
query and the corpus, not its semantic content. Identifiers survive because they
are join keys. Shallow-and-wide wins because it maximises join keys per token.
Signatures are decidable because they *are* the index; effects are elaboration.

    THESIS. Optimise addressability, not comprehension. The model does not need
    to read the corpus; it needs enough of an index to know what to ask for.

Addressability is **query-agnostic** in a way sufficiency is not -- an index
serves every query, a summary serves the query it was written for. That is a
route around the information-bottleneck impossibility recorded in section 5,
which applies to minimal *sufficient* statistics.

This test is deliberately model-free. For each mined task the gold evidence is
the callee's parameter names, which live in its signature. The question is
whether a pure index -- every file's path and top-level signatures, no bodies at
all -- carries that evidence, and at what token cost, against qccr's
budget-bounded span selection over the same pool.

A pure index that matches span selection at a fraction of the tokens supports
the thesis. One that misses the evidence refutes it, and the excluded level-1
map was correctly excluded.
"""

from __future__ import annotations

import argparse
import ast
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
from entroly.native_status import native_status


def index_of(text: str, rel: str) -> str:
    """Path plus every top-level signature. No bodies, no docstrings."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return f"### {rel}\n"

    lines = [f"### {rel}"]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            if node.args.vararg:
                args.append("*" + node.args.vararg.arg)
            if node.args.kwarg:
                args.append("**" + node.args.kwarg.arg)
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            lines.append(f"{kind} {node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in sub.args.args if a.arg not in {"self", "cls"}]
                    lines.append(f"    def {sub.name}({', '.join(args)})")
    return "\n".join(lines)


def run(limit: int, budget: int, seed: int) -> dict[str, Any]:
    from benchmarks.answer_correctness_bridge import build_probes
    from entroly.qccr import select as qccr_select

    payload = json.loads(
        (REPO / "benchmarks" / "results" / "graph_lane_tasks.json").read_text(encoding="utf-8")
    )
    probes = build_probes([Task(**t) for t in payload["tasks"]], limit)
    corpus = sorted(
        {str(p.relative_to(REPO)).replace("\\", "/") for p in _tracked_python_files()}
    )

    rows: list[dict[str, Any]] = []
    for probe in probes:
        task = probe.task
        pool = _pool_for(task, corpus, 48, seed)
        texts = {rel: _read(rel) for rel in pool}
        gold = list(probe.params)

        # Arm A: index over the WHOLE pool -- every file, signatures only.
        index_text = "\n\n".join(index_of(texts[rel], rel) for rel in sorted(pool))

        # Arm B: qccr span selection under the budget.
        frags = [
            {"id": f"file:{rel}", "source": f"file:{rel}", "content": texts[rel],
             "token_count": _tokens(texts[rel]), "relevance": 0.5}
            for rel in pool
        ]
        picked = qccr_select(frags, token_budget=budget, query=task.query)
        qccr_text = "\n\n".join(str(f.get("content", "")) for f in picked)

        # Arm C: the whole pool verbatim -- the ceiling, and the cost of it.
        raw_text = "\n\n".join(texts[rel] for rel in sorted(pool))

        def carries(blob: str) -> bool:
            """Does the evidence appear, as a signature rather than by luck?

            Requires the callee name AND every gold parameter name, so a stray
            occurrence of a common parameter word cannot score a hit.
            """
            return task.callee in blob and all(p in blob for p in gold)

        rows.append({
            "caller": task.caller,
            "callee": task.callee,
            "gold_params": gold,
            "index_tokens": _tokens(index_text),
            "qccr_tokens": _tokens(qccr_text),
            "raw_tokens": _tokens(raw_text),
            "index_carries": carries(index_text),
            "qccr_carries": carries(qccr_text),
            "raw_carries": carries(raw_text),
        })

    # qccr delegates span selection to the Rust core, so the selector that ran
    # is part of the result, not a footnote. A stale artifact once recorded
    # qccr at 9/12 here; the native engine scores 12/12 on the same budget,
    # corpus and pairs, and without this field there was no way to tell from
    # the JSON which selector produced the number.
    status = native_status()
    engine = {
        "native_engine_active": status.ok,
        "entroly_core_version": status.version,
        "entroly_core_version_ok": status.version_ok,
    }
    return {"pinned_ref": payload["pinned_ref"], "budget": budget,
            "probes": len(rows), "engine": engine, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "addressability.json")
    args = ap.parse_args()

    payload = run(args.limit, args.budget, args.seed)
    rows = payload["rows"]
    if not rows:
        print("no probes")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n = len(rows)
    print(f"\n  probes {n}   pool 48 files   qccr budget {payload['budget']}\n")
    print(f"  {'arm':<26}{'carries evidence':>18}{'median tokens':>15}")
    for label, hit_key, tok_key in (
        ("index (signatures only)", "index_carries", "index_tokens"),
        ("qccr (span selection)", "qccr_carries", "qccr_tokens"),
        ("raw pool (ceiling)", "raw_carries", "raw_tokens"),
    ):
        hits = sum(1 for r in rows if r[hit_key])
        toks = sorted(r[tok_key] for r in rows)
        print(f"  {label:<26}{hits:>8}/{n:<9}{toks[len(toks)//2]:>15,}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
