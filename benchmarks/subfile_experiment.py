"""Minimum sub-file provenance experiment: file vs line-window vs syntax-block.

Per the preregistration §"Minimum experiment": on a few ContextBench tasks,
compare three selection granularities over the SAME candidate files (BM25 top-K),
measuring file recall, line recall/precision/F1, token cost, exact-offset
verification rate, and selection reproducibility.

Success criteria (not a preregistered precision target — establishing the range):
  * 100% of selected spans independently verifiable
  * 100% deterministic span ordering (two runs identical)
  * material line-precision improvement from finer granularity
  * no meaningful file-recall regression

    python benchmarks/subfile_experiment.py <tasks.json> <checkout_root> [--n 5] [--budget 8000] [--topk 40]
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import sys
import time

# Applied around `main` rather than at module import. This variable is
# process-global and `tests/test_contextbench_runner_safety.py` imports this
# module, so setting it at import time leaked the raised cap into every test
# that ran afterwards in the same session. `main` imports entroly lazily, so
# this is still set before any entroly import on the script path.
_SIZE_CAP_VARS = ("ENTROLY_MAX_SOURCE_FILE_BYTES",)


def _with_raised_size_caps(fn):
    """Raise the caps for the duration of the call, then restore them."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        previous = {name: os.environ.get(name) for name in _SIZE_CAP_VARS}
        for name in _SIZE_CAP_VARS:
            os.environ.setdefault(name, "500000")
        try:
            return fn(*args, **kwargs)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    return wrapper


def _read_py_files(
    root: str,
    cap: int = 500_000,
) -> tuple[list[tuple[str, bytes]], list[str]]:
    files: list[tuple[str, bytes]] = []
    oversized: list[str] = []
    skip = {".git", "node_modules", "__pycache__", ".tox", "build", "dist"}
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for name in names:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > cap:
                    oversized.append(os.path.relpath(full, root).replace("\\", "/"))
                    continue
                with open(full, "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                raise RuntimeError(f"cannot read candidate file {full}: {exc}") from exc
            rel = os.path.relpath(full, root).replace("\\", "/")
            files.append((rel, data))
    files.sort(key=lambda item: item[0])
    oversized.sort()
    return files, oversized


def _spans_digest(spans) -> str:
    blob = json.dumps([(s.source_path, s.byte_start, s.byte_end) for s in spans], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


@_with_raised_size_caps
def main(tasks_json: str, co_root: str, n: int, budget: int, topk: int) -> int:
    from benchmarks.contextbench_determinism_tax import file_score, line_score, parse_gold
    from benchmarks.contextbench_pilot import _download, _extract_stripped
    from benchmarks.subfile_modes import (
        bm25_scores,
        est_tokens,
        file_units,
        block_units,
        rank_and_select,
        spans_to_lines,
        verify_rate,
        window_units,
    )

    if n <= 0 or budget <= 0 or topk <= 0:
        raise ValueError("n, budget, and topk must be positive")
    with open(tasks_json, encoding="utf-8") as task_file:
        available_tasks = json.load(task_file)
    if not isinstance(available_tasks, list):
        raise ValueError("tasks file must contain a list")
    tasks = available_tasks[:n]
    if len(tasks) != n:
        raise ValueError(f"requested {n} tasks but only {len(tasks)} are available")
    os.makedirs(co_root, exist_ok=True)
    modes = {"file": file_units, "line_window": window_units, "syntax_block": block_units}
    agg: dict[str, list] = {m: [] for m in modes}
    errors: list[dict[str, str]] = []

    for i, task in enumerate(tasks):
        dest = os.path.join(co_root, f"s{i}")
        try:
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
            nfiles = _extract_stripped(_download(task["repo_url"], task["base_commit"]), dest)
            if nfiles <= 0:
                raise RuntimeError("checkout archive produced no files")
            all_files, oversized = _read_py_files(dest)
            if not all_files:
                raise RuntimeError("checkout contains no eligible Python files")
            gold = parse_gold(task["gold_context"])
            if not gold:
                raise ValueError("task has empty or malformed gold context")
            q = task["problem_statement"]
            # Coarse-to-fine: BM25 top-K candidate files, shared across modes.
            fscores = bm25_scores([src.decode("utf-8", "replace") for _, src in all_files], q)
            order = sorted(range(len(all_files)), key=lambda j: (-fscores[j], all_files[j][0]))
            candidates = [all_files[j] for j in order[:topk]]
            commit = task["base_commit"]

            line = f"  s{i} {task['instance_id'][:34]} [{len(all_files)}f]:"
            for mode, units_fn in modes.items():
                t0 = time.time()
                units = units_fn(candidates)
                spans = rank_and_select(candidates, units, q, budget, source_commit=commit)
                spans2 = rank_and_select(candidates, list(reversed(units)), q, budget, source_commit=commit)
                pred = spans_to_lines(spans)
                fs, ls = file_score(pred, gold), line_score(pred, gold)
                rec = {
                    "file_recall": fs.recall, "file_f1": fs.f1,
                    "line_recall": ls.recall, "line_prec": ls.precision, "line_f1": ls.f1,
                    "tokens": sum(est_tokens(s.byte_len()) for s in spans),
                    "verify": verify_rate(spans, candidates),
                    "reproducible": _spans_digest(spans) == _spans_digest(spans2),
                    "span_count": len(spans),
                    "lat_s": round(time.time() - t0, 2),
                    "candidate_files": len(all_files),
                    "oversized_files": len(oversized),
                }
                if rec["tokens"] > budget:
                    raise RuntimeError(
                        f"{mode} selected {rec['tokens']} tokens under budget {budget}"
                    )
                agg[mode].append(rec)
                line += f"  {mode}:fF1={fs.f1:.2f}/lF1={ls.f1:.3f}/lP={ls.precision:.3f}"
            print(line, flush=True)
        except Exception as exc:
            errors.append(
                {
                    "instance_id": str(task.get("instance_id", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  s{i} ERROR {type(exc).__name__}: {exc}", flush=True)
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def avg(mode, key):
        vals = [r[key] for r in agg[mode]]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    print("\n=== sub-file mode comparison (mean over tasks) ===")
    hdr = f"  {'mode':12s} {'file_rec':>8s} {'file_F1':>7s} {'line_rec':>8s} {'line_prec':>9s} {'line_F1':>7s} {'tokens':>7s} {'verify':>6s} {'repro':>5s}"
    print(hdr)
    for mode in modes:
        print(f"  {mode:12s} {avg(mode,'file_recall'):8} {avg(mode,'file_f1'):7} "
              f"{avg(mode,'line_recall'):8} {avg(mode,'line_prec'):9} {avg(mode,'line_f1'):7} "
              f"{avg(mode,'tokens'):7} {avg(mode,'verify'):6} "
              f"{all(r['reproducible'] for r in agg[mode])!s:>5}")

    complete = not errors and all(len(agg[mode]) == n for mode in modes)
    all_verify = complete and all(r["verify"] == 1.0 for m in modes for r in agg[m])
    all_repro = complete and all(r["reproducible"] for m in modes for r in agg[m])
    file_ok = complete and all(
        avg(mode, "file_recall") >= avg("file", "file_recall") - 0.05
        for mode in ("line_window", "syntax_block")
    )
    gains = {
        mode: avg(mode, "line_prec") - avg("file", "line_prec")
        for mode in ("line_window", "syntax_block")
    }
    precision_ok = complete and all(gain > 0 for gain in gains.values())
    nonempty = complete and all(
        record.get("span_count", 0) > 0
        for mode in modes
        for record in agg[mode]
    )
    print("\n=== success criteria ===")
    print(f"  100% spans verifiable:        {all_verify}")
    print(f"  100% deterministic ordering:  {all_repro}")
    print(f"  line-precision gain (window-file): {gains['line_window']:+.3f}")
    print(f"  line-precision gain (block-file):  {gains['syntax_block']:+.3f}")
    print(f"  no file-recall regression:    {file_ok}")
    print(f"  every mode selected evidence: {nonempty}")
    print(f"  all tasks completed:          {complete}")

    os.makedirs("benchmarks/results", exist_ok=True)
    passed = (
        complete
        and all_verify
        and all_repro
        and precision_ok
        and file_ok
        and nonempty
    )
    result = {
        "schema_version": "entroly.contextbench.subfile.v2",
        "valid": complete,
        "passed": passed,
        "protocol": {
            "budget": budget,
            "tasks_requested": n,
            "topk": topk,
            "strict_budget": True,
        },
        "criteria": {
            "complete": complete,
            "all_spans_verifiable": all_verify,
            "all_ordering_reproducible": all_repro,
            "line_precision_improved": precision_ok,
            "no_file_recall_regression": file_ok,
            "nonempty_selection": nonempty,
        },
        "errors": errors,
        "modes": agg,
    }
    with open(
        "benchmarks/results/subfile_experiment.json", "w", encoding="utf-8"
    ) as result_file:
        json.dump(result, result_file, indent=2)
        result_file.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--n"), 5)
    budget = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--budget"), 8000)
    topk = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--topk"), 40)
    raise SystemExit(main(args[0], args[1], n, budget, topk))
