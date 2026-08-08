#!/usr/bin/env python3
"""Where does an index stop working?

Section 10 measured that a signature index beats span selection at matched
budget when the answer lives in a signature. That result is real and it is also
structurally favourable: parameter names ARE the index. The interesting question
is the one it cannot answer.

An index must lose when the evidence lives in a function BODY. Standard practice
in coding agents -- aider's repo map, SigMap, jCodeMunch -- assumes a signature
map is the right default, but none of the surveyed work characterises where that
assumption fails. This measures the boundary directly, with two task families
run through identical arms:

    signature-resident   gold = the callee's parameter names   (index favoured)
    body-resident        gold = the exception a function raises (index cannot see it)

The body-resident family is mined so the answer is provably absent from the
signature: the function raises exactly one named exception class, and that name
appears nowhere in the query text. A signature index physically cannot carry it,
so if the index arm scores above chance here, the metric is leaking rather than
the index working.

Reporting both families together is the point. A mechanism that wins one regime
and loses the other is not better or worse -- it is a component with a domain,
and knowing the domain is what makes it usable.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.addressability import index_of
from benchmarks.graph_lane_quality import (
    REPO,
    Task,
    _read,
    _tokens,
    _tracked_python_files,
)


@dataclass(frozen=True)
class BodyTask:
    func: str
    file: str
    exception: str
    query: str


def mine_body_tasks(limit: int, seed: int) -> list[BodyTask]:
    """Functions whose body raises exactly one named exception class."""
    out: list[BodyTask] = []
    for path in _tracked_python_files():
        rel = str(path.relative_to(REPO)).replace("\\", "/")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = (ast.get_docstring(node) or "").strip()
            if not doc:
                continue
            raised: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Raise) and sub.exc is not None:
                    exc = sub.exc
                    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                        raised.add(exc.func.id)
                    elif isinstance(exc, ast.Name):
                        raised.add(exc.id)
            if len(raised) != 1:
                continue
            exception = raised.pop()
            query = f"{node.name} {doc.splitlines()[0].strip()}"
            # The answer must not be readable from the query itself.
            if exception.lower() in query.lower():
                continue
            # Nor from the signature, which is what the index would carry.
            if exception in index_of(path.read_text(encoding="utf-8", errors="replace"), rel):
                continue
            out.append(BodyTask(node.name, rel, exception, query))
    rng = random.Random(seed)
    rng.shuffle(out)
    seen: set[str] = set()
    picked: list[BodyTask] = []
    for task in out:
        if task.file in seen:
            continue
        seen.add(task.file)
        picked.append(task)
        if len(picked) >= limit:
            break
    return picked


def _pool(target: str, corpus: list[str], size: int, seed: str) -> list[str]:
    others = [rel for rel in corpus if rel != target]
    random.Random(seed).shuffle(others)
    return sorted([target] + others[: size - 1])


def run(limit: int, budget: int, seed: int) -> dict[str, Any]:
    from entroly.qccr import select as qccr_select

    corpus = sorted(
        {str(p.relative_to(REPO)).replace("\\", "/") for p in _tracked_python_files()}
    )
    tasks = mine_body_tasks(limit, seed)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        pool = _pool(task.file, corpus, 48, f"{seed}:{task.func}")
        texts = {rel: _read(rel) for rel in pool}

        index_text = "\n\n".join(index_of(texts[rel], rel) for rel in sorted(pool))
        frags = [
            {"id": f"file:{rel}", "source": f"file:{rel}", "content": texts[rel],
             "token_count": _tokens(texts[rel]), "relevance": 0.5}
            for rel in pool
        ]
        picked = qccr_select(frags, token_budget=budget, query=task.query)
        qccr_text = "\n\n".join(str(f.get("content", "")) for f in picked)
        raw_text = "\n\n".join(texts[rel] for rel in sorted(pool))

        def carries(blob: str) -> bool:
            return task.func in blob and task.exception in blob

        rows.append({
            "func": task.func,
            "file": task.file,
            "exception": task.exception,
            "index_tokens": _tokens(index_text),
            "qccr_tokens": _tokens(qccr_text),
            "raw_tokens": _tokens(raw_text),
            "index_carries": carries(index_text),
            "qccr_carries": carries(qccr_text),
            "raw_carries": carries(raw_text),
        })
    return {"budget": budget, "probes": len(rows), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "regime_boundary.json")
    args = ap.parse_args()

    payload = run(args.limit, args.budget, args.seed)
    rows = payload["rows"]
    if not rows:
        print("no body-resident tasks mined")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n = len(rows)
    print(f"\n  BODY-RESIDENT regime: gold = the exception a function raises")
    print(f"  probes {n}   pool 48 files   qccr budget {payload['budget']}\n")
    print(f"  {'arm':<26}{'carries evidence':>18}{'median tokens':>15}")
    for label, hit, tok in (
        ("index (signatures only)", "index_carries", "index_tokens"),
        ("qccr (span selection)", "qccr_carries", "qccr_tokens"),
        ("raw pool (ceiling)", "raw_carries", "raw_tokens"),
    ):
        hits = sum(1 for r in rows if r[hit])
        toks = sorted(r[tok] for r in rows)
        print(f"  {label:<26}{hits:>8}/{n:<9}{toks[len(toks)//2]:>15,}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
