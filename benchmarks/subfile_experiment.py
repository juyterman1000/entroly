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

import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault("ENTROLY_MAX_SOURCE_FILE_BYTES", "500000")


def _read_py_files(root: str, cap: int = 500_000) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    skip = {".git", "node_modules", "__pycache__", ".tox", "build", "dist"}
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for name in names:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > cap:
                    continue
                with open(full, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            rel = os.path.relpath(full, root).replace("\\", "/")
            files.append((rel, data))
    return files


def _spans_digest(spans) -> str:
    blob = json.dumps([(s.source_path, s.byte_start, s.byte_end) for s in spans], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


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

    tasks = json.load(open(tasks_json, encoding="utf-8"))[:n]
    os.makedirs(co_root, exist_ok=True)
    modes = {"file": file_units, "line_window": window_units, "syntax_block": block_units}
    agg: dict[str, list] = {m: [] for m in modes}

    for i, task in enumerate(tasks):
        dest = os.path.join(co_root, f"s{i}")
        try:
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
            _extract_stripped(_download(task["repo_url"], task["base_commit"]), dest)
            all_files = _read_py_files(dest)
            gold = parse_gold(task["gold_context"])
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
                    "lat_s": round(time.time() - t0, 2),
                }
                agg[mode].append(rec)
                line += f"  {mode}:fF1={fs.f1:.2f}/lF1={ls.f1:.3f}/lP={ls.precision:.3f}"
            print(line, flush=True)
        except Exception as exc:
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

    all_verify = all(r["verify"] == 1.0 for m in modes for r in agg[m])
    all_repro = all(r["reproducible"] for m in modes for r in agg[m])
    file_ok = avg("syntax_block", "file_recall") >= avg("file", "file_recall") - 0.05
    lp_gain = avg("syntax_block", "line_prec") - avg("file", "line_prec")
    print("\n=== success criteria ===")
    print(f"  100% spans verifiable:        {all_verify}")
    print(f"  100% deterministic ordering:  {all_repro}")
    print(f"  line-precision gain (block-file): {lp_gain:+.3f}")
    print(f"  no file-recall regression:    {file_ok}")

    os.makedirs("benchmarks/results", exist_ok=True)
    json.dump({m: agg[m] for m in modes}, open("benchmarks/results/subfile_experiment.json", "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--n"), 5)
    budget = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--budget"), 8000)
    topk = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--topk"), 40)
    raise SystemExit(main(args[0], args[1], n, budget, topk))
