#!/usr/bin/env python3
"""Does graph-aware selection beat chunk ranking on dependency-sensitive tasks?

Protocol is fixed in GRAPH_LANE_PREREGISTRATION.md and was written before this
was run. Registers open question Q-B from BREAKTHROUGH_RESEARCH.md.

A task is a caller symbol `S` in file `A` whose body calls `T`, defined in a
different file `B`. The query names only `S` (and its docstring first line), so
`B` is reachable only by following the call edge. Indirect recall -- did the
selector deliver `B`? -- is the primary metric.

Subcommands:
    mine    build and validate the task set, write JSON
    run     execute the arms against a mined task set
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MAX_BYTES = 400 * 1024


@dataclass(frozen=True)
class Task:
    caller: str          # S
    caller_file: str     # A  (direct gold)
    callee: str          # T
    callee_file: str     # B  (indirect gold)
    query: str
    caller_lineno: int


def _pinned_ref() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    files = []
    for line in out.stdout.splitlines():
        path = REPO / line
        # Test and benchmark files describe behaviour rather than implement it;
        # including them lets a lexical ranker win on prose about the symbol.
        # Matched on any path segment: `entroly-core/tests/` and `bench/` both
        # leaked past a prefix-only filter.
        parts = set(line.split("/"))
        if parts & {"tests", "test", "benchmarks", "bench"}:
            continue
        try:
            if path.is_file() and path.stat().st_size <= MAX_BYTES:
                files.append(path)
        except OSError:
            continue
    return files


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return None


def _definition_map(trees: dict[str, ast.Module]) -> dict[str, list[str]]:
    """symbol name -> list of files defining it at top level."""
    defs: dict[str, list[str]] = {}
    for rel, tree in trees.items():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.setdefault(node.name, []).append(rel)
    return defs


def _called_names(fn: ast.AST) -> set[str]:
    """Names called as bare functions in `fn`.

    Deliberately excludes `ast.Attribute` calls. Collecting `.attr` matched any
    module-level function sharing a method's name, which fabricated dependency
    edges: `path.write_text(...)` resolved to a `write_text` defined in
    `context_receipts/store.py`, gold evidence for a call that never happened.
    A bare `Name` call is the only form an import can be verified against.
    """
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def _imported_names(tree: ast.Module) -> set[str]:
    """Names bound into this module by `from X import T` / `import T`.

    A callee is credited only if the caller's file actually imports it. Without
    this the benchmark counts coincidental name equality as a dependency edge.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _query_for(fn: ast.AST, name: str) -> str:
    doc = ast.get_docstring(fn) or ""
    first = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    return f"{name} {first}".strip()


def mine(limit: int, seed: int) -> dict[str, Any]:
    files = _tracked_python_files()
    trees: dict[str, ast.Module] = {}
    for path in files:
        tree = _parse(path)
        if tree is not None:
            trees[str(path.relative_to(REPO)).replace("\\", "/")] = tree

    defs = _definition_map(trees)
    # A symbol defined in more than one file cannot be resolved unambiguously
    # by AST alone, so it cannot serve as gold evidence.
    unique = {name: paths[0] for name, paths in defs.items() if len(paths) == 1}

    tasks: list[Task] = []
    for rel, tree in trees.items():
        imported = _imported_names(tree)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            caller = node.name
            if unique.get(caller) != rel:
                continue  # caller must itself be unambiguous
            if not (ast.get_docstring(node) or "").strip():
                continue  # a bare name is too weak a query to be informative
            query = _query_for(node, caller)
            for callee in sorted(_called_names(node)):
                target = unique.get(callee)
                if target is None or target == rel:
                    continue
                if callee not in imported:
                    continue  # the caller must actually import the callee
                if callee.lower() in query.lower():
                    continue  # callee must not be nameable from the query
                tasks.append(Task(
                    caller=caller,
                    caller_file=rel,
                    callee=callee,
                    callee_file=target,
                    query=query,
                    caller_lineno=node.lineno,
                ))

    rng = random.Random(seed)
    rng.shuffle(tasks)
    # One task per (caller_file, callee_file) pair keeps a single hub module
    # from dominating the set.
    seen: set[tuple[str, str]] = set()
    per_caller: dict[str, int] = {}
    picked: list[Task] = []
    for task in tasks:
        key = (task.caller_file, task.callee_file)
        if key in seen:
            continue
        # cli.py alone supplied 4 of the first 8 picks under pair-dedup only;
        # one hub file must not define the result.
        if per_caller.get(task.caller_file, 0) >= 3:
            continue
        seen.add(key)
        per_caller[task.caller_file] = per_caller.get(task.caller_file, 0) + 1
        picked.append(task)
        if len(picked) >= limit:
            break

    return {
        "pinned_ref": _pinned_ref(),
        "seed": seed,
        "corpus_files": len(trees),
        "unique_symbols": len(unique),
        "candidate_tasks": len(tasks),
        "tasks": [asdict(t) for t in picked],
    }


def _tokens(text: str) -> int:
    """One estimator used by every arm, so budgets are genuinely matched."""
    return max(1, len(text) // 4)


def _pool_for(task: Task, corpus: list[str], size: int, seed: int) -> list[str]:
    """Candidate files for one task: the two gold files plus distractors."""
    gold = [task.caller_file, task.callee_file]
    others = [rel for rel in corpus if rel not in gold]
    rng = random.Random(f"{seed}:{task.caller}:{task.callee}")
    rng.shuffle(others)
    pool = gold + others[: max(0, size - len(gold))]
    pool.sort()  # stable order, independent of which files are gold
    return pool


def _read(rel: str) -> str:
    try:
        return (REPO / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _bm25_rank(pool: list[str], texts: dict[str, str], query: str) -> list[str]:
    """Plain BM25 over whole files. Transparent, deterministic, no engine."""
    import math
    from collections import Counter

    k1, b = 1.5, 0.75

    def toks(s: str) -> list[str]:
        return [t for t in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(t) > 2]

    docs = {rel: Counter(toks(texts[rel])) for rel in pool}
    lens = {rel: sum(c.values()) for rel, c in docs.items()}
    avgdl = (sum(lens.values()) / len(lens)) if lens else 1.0
    n = len(pool)
    scores: dict[str, float] = {}
    qt = toks(query)
    for rel in pool:
        score = 0.0
        for term in qt:
            df = sum(1 for r in pool if docs[r].get(term))
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = docs[rel].get(term, 0)
            denom = tf + k1 * (1 - b + b * lens[rel] / (avgdl or 1.0))
            score += idf * (tf * (k1 + 1) / denom) if denom else 0.0
        scores[rel] = score
    return sorted(pool, key=lambda r: (-scores[r], r))


def _fill(ranked: list[str], texts: dict[str, str], budget: int) -> set[str]:
    """Take files in rank order until the budget is exhausted."""
    out: set[str] = set()
    used = 0
    for rel in ranked:
        cost = _tokens(texts[rel])
        if used + cost > budget:
            continue
        out.add(rel)
        used += cost
    return out


def _delivered_sources(fragments: Any) -> set[str]:
    """Normalise any arm's output to the set of repo-relative files it covers."""
    out: set[str] = set()
    if not fragments:
        return out
    for frag in fragments:
        src = ""
        if isinstance(frag, dict):
            src = str(frag.get("source") or frag.get("path") or "")
        else:
            src = str(getattr(frag, "source", "") or "")
        src = src.removeprefix("file:").replace("\\", "/")
        if src:
            out.add(src)
    return out


def _arm_hcc(pool: list[str], texts: dict[str, str], query: str, budget: int) -> set[str]:
    from benchmarks.engine_isolation import isolated_engine_dir, assert_engine_isolated

    with isolated_engine_dir():
        assert_engine_isolated()
        from entroly_core import EntrolyEngine  # noqa: PLC0415

        engine = EntrolyEngine()
        for rel in pool:
            engine.ingest(texts[rel], f"file:{rel}", _tokens(texts[rel]), False)
        result = engine.hierarchical_compress(budget, query, None)
        if not isinstance(result, dict):
            return set()
        full = _delivered_sources(result.get("level3_fragments"))
        skeleton = _delivered_sources(result.get("level2_fragments"))
        # Level 1 is a one-line-per-file map: it names every file but carries no
        # content, so counting it as delivery would score a table of contents as
        # evidence.
        #
        # Level 2 IS counted, but its resolution is recorded separately. A
        # skeleton carries signatures without bodies, so "delivered" is not the
        # same claim at level 2 and level 3, and collapsing them would hide the
        # difference. HCC charges itself the skeleton cost (380 tokens for a
        # 2541-token file), so the budget comparison remains matched.
        return full | skeleton, full, skeleton


def _arm_qccr(pool: list[str], texts: dict[str, str], query: str, budget: int) -> set[str]:
    from entroly.qccr import select as qccr_select

    frags = [
        {"id": f"file:{rel}", "source": f"file:{rel}",
         "content": texts[rel], "token_count": _tokens(texts[rel]), "relevance": 0.5}
        for rel in pool
    ]
    return _delivered_sources(qccr_select(frags, token_budget=budget, query=query))


def run(task_path: Path, budgets: list[int], pool_size: int,
        limit: int | None, seed: int) -> dict[str, Any]:
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    tasks = [Task(**t) for t in payload["tasks"]]
    if limit:
        tasks = tasks[:limit]
    corpus = sorted({str(p.relative_to(REPO)).replace("\\", "/") for p in _tracked_python_files()})

    arms = ["null", "random", "raw_truncated", "raw_full", "bm25", "qccr", "hcc"]
    records: list[dict[str, Any]] = []

    for task in tasks:
        pool = _pool_for(task, corpus, pool_size, seed)
        texts = {rel: _read(rel) for rel in pool}
        for budget in budgets:
            for arm in arms:
                try:
                    if arm == "null":
                        got: set[str] = set()
                    elif arm == "raw_full":
                        got = set(pool)
                    elif arm == "random":
                        order = list(pool)
                        random.Random(f"{seed}:{arm}:{task.caller}").shuffle(order)
                        got = _fill(order, texts, budget)
                    elif arm == "raw_truncated":
                        got = _fill(sorted(pool), texts, budget)
                    elif arm == "bm25":
                        got = _fill(_bm25_rank(pool, texts, task.query), texts, budget)
                    elif arm == "qccr":
                        got = _arm_qccr(pool, texts, task.query, budget)
                    else:
                        got, hcc_full, hcc_skel = _arm_hcc(pool, texts, task.query, budget)
                    error = ""
                except Exception as exc:  # noqa: BLE001
                    got, error = set(), f"{type(exc).__name__}: {exc}"
                    hcc_full = hcc_skel = set()
                if arm != "hcc":
                    hcc_full, hcc_skel = got, set()
                records.append({
                    "caller": task.caller, "callee": task.callee,
                    "arm": arm, "budget": budget,
                    "direct_hit": task.caller_file in got,
                    "indirect_hit": task.callee_file in got,
                    "indirect_full": task.callee_file in hcc_full,
                    "indirect_skeleton": task.callee_file in hcc_skel,
                    "delivered": len(got),
                    "error": error,
                })

    return {
        "pinned_ref": payload["pinned_ref"],
        "seed": seed,
        "pool_size": pool_size,
        "tasks": len(tasks),
        "budgets": budgets,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="build and validate the task set")
    m.add_argument("--limit", type=int, default=60)
    m.add_argument("--seed", type=int, default=20260807)
    m.add_argument("--out", type=Path, default=REPO / "benchmarks" / "results" / "graph_lane_tasks.json")
    r = sub.add_parser("run", help="execute the arms")
    r.add_argument("--tasks", type=Path, default=REPO / "benchmarks" / "results" / "graph_lane_tasks.json")
    r.add_argument("--budgets", type=int, nargs="+", default=[2000, 8000])
    r.add_argument("--pool-size", type=int, default=48)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--seed", type=int, default=20260807)
    r.add_argument("--out", type=Path, default=REPO / "benchmarks" / "results" / "graph_lane_results.json")
    args = parser.parse_args()

    if args.cmd == "run":
        payload = run(args.tasks, args.budgets, args.pool_size, args.limit, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        errors = [r for r in payload["records"] if r["error"]]
        print(f"tasks {payload['tasks']}  records {len(payload['records'])}  errors {len(errors)}")
        if errors:
            print("first error:", errors[0]["arm"], errors[0]["error"][:160])
        print(f"-> {args.out}")
        return 0

    if args.cmd == "mine":
        payload = mine(args.limit, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"corpus_files      {payload['corpus_files']}")
        print(f"unique_symbols    {payload['unique_symbols']}")
        print(f"candidate_tasks   {payload['candidate_tasks']}")
        print(f"picked            {len(payload['tasks'])}")
        print(f"pinned_ref        {payload['pinned_ref'][:12]}")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
