"""ContextBench deterministic pilot — harness/adapter trust check (not conclusions).

For a handful of tasks: download the pinned tree, ingest, run Entroly selection
under two fresh-process/seed/thread conditions, map to exact line spans, score
against gold, and check end-to-end reproducibility. Checkouts are downloaded to
an external short-path root (Windows MAX_PATH) and deleted per task; nothing here
is committed to Git except this script.

    # prepare tasks JSON first (streamed from HuggingFace), then:
    python benchmarks/contextbench_pilot.py <tasks.json> <checkout_root> [--budget N] [--max-bytes N]
    # worker (internal): --worker <checkout> <task.json> <budget>
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request


def _extract_stripped(data: bytes, dest: str) -> int:
    """Extract a GitHub tarball, stripping the top-level `repo-<sha>/` component."""
    os.makedirs(dest, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            m.name = parts[1]
            try:
                tf.extract(m, dest)
                n += m.isfile()
            except (FileNotFoundError, OSError):
                pass  # skip paths the OS rejects (long path / illegal char); fail-open on corpus
    return n


def _download(repo_url: str, sha: str) -> bytes:
    owner_repo = repo_url.rstrip("/").removesuffix(".git").split("github.com/")[-1]
    url = f"https://github.com/{owner_repo}/archive/{sha}.tar.gz"
    return urllib.request.urlopen(url, timeout=300).read()


def _spans_digest(spans: dict[str, list[list[int]]]) -> str:
    blob = json.dumps(sorted((p, sorted(iv)) for p, iv in spans.items()), separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _spans_to_lines(spans: dict[str, list[list[int]]]) -> dict[str, set[int]]:
    return {p: {n for s, e in iv for n in range(s, e + 1)} for p, iv in spans.items()}


# ── worker: one selection in an isolated process ───────────────────────────────

def _run_worker(checkout: str, task_json: str, budget: int) -> int:
    from benchmarks.contextbench_determinism_tax import build_engine_for_repo, entroly_select
    from benchmarks.contextbench_span_adapter import SelectedSpan

    task = json.load(open(task_json, encoding="utf-8"))
    try:
        engine = build_engine_for_repo(checkout)
        records: list[SelectedSpan] = entroly_select(engine, checkout, task["problem_statement"], budget)
    except Exception as exc:  # report, never crash the orchestrator silently
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 0
    spans = {r.path: [list(iv) for iv in r.intervals()] for r in records if r.mapped and r.lines}
    out = {
        "spans": spans,
        "digest": _spans_digest(spans),
        "n_selected": len(records),
        "n_unmapped": sum(1 for r in records if not r.mapped),
        "unmapped_reasons": sorted({r.reason for r in records if not r.mapped}),
        "tokens": sum(r.token_cost for r in records),
    }
    print(json.dumps(out))
    return 0


def _worker_env(base: dict, *, seed: str, threads: str, max_bytes: int) -> dict:
    env = dict(base)
    env["PYTHONHASHSEED"] = seed
    env["RAYON_NUM_THREADS"] = threads
    env["OMP_NUM_THREADS"] = threads
    env["ENTROLY_MAX_SOURCE_FILE_BYTES"] = str(max_bytes)
    env["ENTROLY_MAX_FILE_BYTES"] = str(max_bytes)
    return env


def _invoke_worker(checkout: str, task_json: str, budget: int, env: dict) -> dict:
    out = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", checkout, task_json, str(budget)],
        capture_output=True, text=True, env=env, cwd=os.getcwd(),
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"error": f"no output; stderr tail: {out.stderr[-300:]}"}
    return json.loads(lines[-1])


# ── orchestrator ───────────────────────────────────────────────────────────────

def main(tasks_json: str, co_root: str, budget: int, max_bytes: int) -> int:
    from benchmarks.contextbench_determinism_tax import (
        evidence_drop,
        file_score,
        line_score,
        parse_gold,
    )

    tasks = json.load(open(tasks_json, encoding="utf-8"))
    os.makedirs(co_root, exist_ok=True)
    results = []
    for i, task in enumerate(tasks):
        dest = os.path.join(co_root, f"t{i}")
        task_path = os.path.join(co_root, f"task{i}.json")
        json.dump(task, open(task_path, "w", encoding="utf-8"))
        rec: dict = {"instance_id": task["instance_id"]}
        try:
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
            t0 = time.time()
            nfiles = _extract_stripped(_download(task["repo_url"], task["base_commit"]), dest)
            rec["files"] = nfiles
            rec["fetch_s"] = round(time.time() - t0, 1)

            base = dict(os.environ)
            a = _invoke_worker(dest, task_path, budget, _worker_env(base, seed="0", threads="1", max_bytes=max_bytes))
            b = _invoke_worker(dest, task_path, budget, _worker_env(base, seed="random", threads="4", max_bytes=max_bytes))
            rec["error"] = a.get("error") or b.get("error")
            if not rec["error"]:
                rec["reproducible"] = (a["digest"] == b["digest"])
                rec["n_selected"] = a["n_selected"]
                rec["n_unmapped"] = a["n_unmapped"]
                rec["unmapped_reasons"] = a["unmapped_reasons"]
                gold = parse_gold(task["gold_context"])
                pred = _spans_to_lines(a["spans"])
                fs, ls = file_score(pred, gold), line_score(pred, gold)
                rec["file"] = {"recall": round(fs.recall, 3), "precision": round(fs.precision, 3), "f1": round(fs.f1, 3)}
                rec["line"] = {"recall": round(ls.recall, 3), "precision": round(ls.precision, 3), "f1": round(ls.f1, 3)}
                rec["evidence_drop"] = round(evidence_drop(pred, gold), 3)
        finally:
            shutil.rmtree(dest, ignore_errors=True)
            if os.path.exists(task_path):
                os.remove(task_path)
        results.append(rec)
        print(f"  t{i} {rec['instance_id'][:52]}: "
              + (rec["error"] if rec.get("error") else
                 f"repro={rec['reproducible']} file_f1={rec['file']['f1']} line_f1={rec['line']['f1']} "
                 f"sel={rec['n_selected']} unmapped={rec['n_unmapped']}"), flush=True)

    ok = [r for r in results if not r.get("error")]
    print("\n=== pilot acceptance ===")
    print(f"  executed end-to-end:   {len(ok)}/{len(results)}")
    print(f"  100% reproducible:     {all(r.get('reproducible') for r in ok)} ({sum(bool(r.get('reproducible')) for r in ok)}/{len(ok)})")
    print(f"  zero unmapped spans:   {all(r.get('n_unmapped') == 0 for r in ok)}")
    print(f"  no metric exceptions:  {len(ok) == len(results)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        raise SystemExit(_run_worker(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--budget"), 8000)
    max_bytes = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--max-bytes"), 500_000)
    raise SystemExit(main(args[0], args[1], budget, max_bytes))
