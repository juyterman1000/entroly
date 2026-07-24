"""ContextBench 25-task deterministic floor: Entroly vs BM25 (see prereg v2).

File-level is the primary metric (both systems select whole files, so this
isolates ranking/determinism from granularity). Line-level is reported but
flagged granularity-limited. Paired bootstrap CI over the same tasks; early stop
after 10 tasks if Entroly trails BM25 by >0.15 file-F1 with a CI clearly < 0.

    python benchmarks/contextbench_floor.py <tasks.json> <checkout_root> [--budget N] [--n 25]

Checkouts are downloaded to an external short-path root and deleted per task.
"""
from __future__ import annotations

import os

# Raise the source-size cap BEFORE any entroly import so gold-bearing large files
# (e.g. astropy table.py = 147 KB) are indexable — a v2 protocol requirement.
os.environ.setdefault("ENTROLY_MAX_SOURCE_FILE_BYTES", "500000")
os.environ.setdefault("ENTROLY_MAX_FILE_BYTES", "500000")

import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402


def recall_at_k(ranked: list[str], gold_files: set[str], ks=(1, 5, 10)) -> dict[int, float]:
    return {k: (len(set(ranked[:k]) & gold_files) / len(gold_files) if gold_files else 0.0) for k in ks}


def paired_bootstrap(deltas: list[float], iters: int = 10000, seed: int = 42):
    if not deltas:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (statistics.mean(deltas), means[int(0.025 * iters)], means[int(0.975 * iters)])


def _tokens(records) -> int:
    return sum(r.token_cost for r in records if r.mapped)


def main(tasks_json: str, co_root: str, budget: int, n: int) -> int:
    from benchmarks.contextbench_determinism_tax import (
        bm25_rank_files,
        bm25_select,
        build_engine_for_repo,
        entroly_select,
        evidence_drop,
        file_score,
        line_score,
        parse_gold,
    )
    from benchmarks.contextbench_pilot import _download, _extract_stripped
    from benchmarks.contextbench_span_adapter import canonical_path, to_spans

    tasks = json.load(open(tasks_json, encoding="utf-8"))[:n]
    os.makedirs(co_root, exist_ok=True)
    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        proc = None

    rows: list[dict] = []
    deltas: list[float] = []
    stopped = None
    for i, task in enumerate(tasks):
        dest = os.path.join(co_root, f"f{i}")
        repo = task["repo_url"].rstrip("/").removesuffix(".git").split("/")[-1]
        row: dict = {"instance_id": task["instance_id"], "repo": repo}
        try:
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
            _extract_stripped(_download(task["repo_url"], task["base_commit"]), dest)
            engine = build_engine_for_repo(dest)
            gold = parse_gold(task["gold_context"])
            gold_files = set(gold)
            q = task["problem_statement"]

            t0 = time.time()
            e_rec = entroly_select(engine, dest, q, budget)
            e_lat = time.time() - t0
            e_pred = to_spans(e_rec)
            e_rank = list(dict.fromkeys(r.path for r in e_rec))

            t0 = time.time()
            b_rec = bm25_select(engine, dest, q, budget)
            b_lat = time.time() - t0
            b_pred = to_spans(b_rec)
            b_ranked, _ = bm25_rank_files(engine, q)
            b_rank = list(dict.fromkeys(canonical_path(s) for s, sc, _, _ in b_ranked if sc > 0))

            e_f, e_l = file_score(e_pred, gold), line_score(e_pred, gold)
            b_f, b_l = file_score(b_pred, gold), line_score(b_pred, gold)
            row["entroly"] = {
                "file": (round(e_f.recall, 3), round(e_f.precision, 3), round(e_f.f1, 3)),
                "line_f1": round(e_l.f1, 3), "drop": round(evidence_drop(e_pred, gold), 3),
                "rk": recall_at_k(e_rank, gold_files), "tokens": _tokens(e_rec),
                "n_sel": len(e_rec), "lat_s": round(e_lat, 2),
            }
            row["bm25"] = {
                "file": (round(b_f.recall, 3), round(b_f.precision, 3), round(b_f.f1, 3)),
                "line_f1": round(b_l.f1, 3), "drop": round(evidence_drop(b_pred, gold), 3),
                "rk": recall_at_k(b_rank, gold_files), "tokens": _tokens(b_rec),
                "n_sel": len(b_rec), "lat_s": round(b_lat, 2),
            }
            row["delta_file_f1"] = round(e_f.f1 - b_f.f1, 3)
            if proc is not None:
                row["rss_mb"] = round(proc.memory_info().rss / (1024 * 1024))
            deltas.append(e_f.f1 - b_f.f1)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(dest, ignore_errors=True)
        rows.append(row)
        if row.get("error"):
            print(f"  f{i} {row['instance_id'][:40]} [{repo}]: ERROR {row['error']}", flush=True)
        else:
            print(f"  f{i} {row['instance_id'][:40]} [{repo}]: "
                  f"Efile_f1={row['entroly']['file'][2]} Bfile_f1={row['bm25']['file'][2]} "
                  f"d={row['delta_file_f1']:+.3f}  E_r@1={row['entroly']['rk'][1]:.2f} B_r@1={row['bm25']['rk'][1]:.2f}",
                  flush=True)

        if len(deltas) >= 10 and stopped is None:
            mean, lo, hi = paired_bootstrap(deltas)
            if mean < -0.15 and hi < 0:
                stopped = {"after": len(deltas), "mean": mean, "ci": [lo, hi]}
                print(f"\n  EARLY STOP after {len(deltas)}: Entroly trails BM25 by {mean:+.3f} file-F1, CI [{lo:+.3f},{hi:+.3f}]", flush=True)
                break

    ok = [r for r in rows if not r.get("error")]
    mean, lo, hi = paired_bootstrap(deltas)

    def _avg(sys_key, path):
        vals = [_dig(r[sys_key], path) for r in ok]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "budget": budget, "tasks_run": len(ok), "tasks_total": len(tasks), "stopped": stopped,
        "delta_file_f1_mean": round(mean, 3), "delta_file_f1_ci95": [round(lo, 3), round(hi, 3)],
        "entroly": {"file_recall": _avg("entroly", "file.0"), "file_prec": _avg("entroly", "file.1"),
                     "file_f1": _avg("entroly", "file.2"), "line_f1": _avg("entroly", "line_f1"),
                     "r@1": _avg("entroly", "rk.1"), "r@5": _avg("entroly", "rk.5"), "r@10": _avg("entroly", "rk.10"),
                     "drop": _avg("entroly", "drop"), "tokens": _avg("entroly", "tokens"), "lat_s": _avg("entroly", "lat_s")},
        "bm25": {"file_recall": _avg("bm25", "file.0"), "file_prec": _avg("bm25", "file.1"),
                  "file_f1": _avg("bm25", "file.2"), "line_f1": _avg("bm25", "line_f1"),
                  "r@1": _avg("bm25", "rk.1"), "r@5": _avg("bm25", "rk.5"), "r@10": _avg("bm25", "rk.10"),
                  "drop": _avg("bm25", "drop"), "tokens": _avg("bm25", "tokens"), "lat_s": _avg("bm25", "lat_s")},
    }
    verdict = ("Entroly > BM25 (CI>0)" if lo > 0 else
               "BM25 > Entroly (CI<0)" if hi < 0 else "inconclusive (CI spans 0)")
    summary["verdict_F2"] = verdict

    os.makedirs("benchmarks/results", exist_ok=True)
    out = {"summary": summary, "per_task": rows}
    json.dump(out, open("benchmarks/results/contextbench_determinism_floor.json", "w", encoding="utf-8"), indent=2)

    print("\n=== 25-task deterministic floor (file-level primary; line-level granularity-limited) ===")
    print(f"  tasks: {len(ok)}/{len(tasks)}  budget={budget}")
    for k in ("file_recall", "file_prec", "file_f1", "r@1", "r@5", "r@10", "line_f1", "drop", "tokens", "lat_s"):
        print(f"  {k:11s}  Entroly={summary['entroly'][k]:<8} BM25={summary['bm25'][k]}")
    print(f"  delta file-F1 (Entroly-BM25): {summary['delta_file_f1_mean']:+.3f}  CI95 {summary['delta_file_f1_ci95']}")
    print(f"  F2 verdict: {verdict}")
    print("  -> results/contextbench_determinism_floor.json")
    return 0


def _dig(d: dict, path: str):
    cur = d
    for part in path.split("."):
        cur = cur[int(part)] if part.isdigit() else cur[part]
    return cur


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--budget"), 8000)
    n = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--n"), 25)
    raise SystemExit(main(args[0], args[1], budget, n))
