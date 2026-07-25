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
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from urllib.parse import urlparse

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_REPO_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAX_ARCHIVE_BYTES = 1_000_000_000
_MAX_EXTRACTED_BYTES = 4_000_000_000
_MAX_ARCHIVE_MEMBERS = 250_000


def _extract_stripped(data: bytes, dest: str) -> int:
    """Safely extract a GitHub tarball after stripping its top-level directory.

    Archive paths, links, devices, duplicate targets, and containment escapes
    are rejected. A partially rejected corpus is never treated as benchmark
    input because that would silently change the evaluated snapshot.
    """
    root = os.path.realpath(dest)
    os.makedirs(root, exist_ok=True)
    n = 0
    seen: set[str] = set()
    seen_targets: set[str] = set()
    archive_root: str | None = None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = tf.getmembers()
        if len(members) > _MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"archive contains too many members: {len(members)}")
        expanded_bytes = sum(member.size for member in members if member.isfile())
        if expanded_bytes > _MAX_EXTRACTED_BYTES:
            raise ValueError(
                f"archive expands to {expanded_bytes} bytes, above {_MAX_EXTRACTED_BYTES}"
            )
        for member in members:
            if "\\" in member.name:
                raise ValueError(f"unsafe archive path separator: {member.name!r}")
            archive_path = PurePosixPath(member.name)
            if archive_path.is_absolute() or ".." in archive_path.parts:
                raise ValueError(f"unsafe archive member: {member.name!r}")
            parts = archive_path.parts
            if len(parts) < 2:
                continue
            if archive_root is None:
                archive_root = parts[0]
            elif parts[0] != archive_root:
                raise ValueError(
                    "archive contains multiple top-level roots; refusing mixed snapshot"
                )
            relative = PurePosixPath(*parts[1:])
            if (
                relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError(f"unsafe archive member: {member.name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"unsupported archive member type: {member.name!r}")

            relative_text = relative.as_posix()
            if relative_text in seen:
                raise ValueError(f"duplicate archive member: {relative_text!r}")
            seen.add(relative_text)
            target = os.path.realpath(os.path.join(root, *relative.parts))
            normalized_target = os.path.normcase(target)
            if normalized_target in seen_targets:
                raise ValueError(
                    f"archive members collide on this filesystem: {relative_text!r}"
                )
            seen_targets.add(normalized_target)
            try:
                if os.path.commonpath([root, target]) != root:
                    raise ValueError(f"archive member escapes destination: {member.name!r}")
            except ValueError as exc:
                raise ValueError(f"unsafe archive member: {member.name!r}") from exc

            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                raise ValueError(f"archive file has no readable payload: {member.name!r}")
            with source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)
            n += 1
    if n == 0:
        raise ValueError("archive contained no regular files")
    return n


def _download(repo_url: str, sha: str) -> bytes:
    parsed = urlparse(repo_url)
    parts = [part for part in parsed.path.rstrip("/").removesuffix(".git").split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.hostname != "github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
        or len(parts) != 2
        or not all(_REPO_PART_RE.fullmatch(part) for part in parts)
    ):
        raise ValueError(f"repo_url must be a canonical https://github.com/owner/repo URL: {repo_url!r}")
    if not _COMMIT_RE.fullmatch(sha):
        raise ValueError("base_commit must be a full 40-character hexadecimal SHA")
    owner_repo = "/".join(parts)
    url = f"https://github.com/{owner_repo}/archive/{sha}.tar.gz"
    with urllib.request.urlopen(url, timeout=300) as response:
        final_url = urlparse(response.geturl())
        if (
            final_url.scheme != "https"
            or final_url.hostname not in {"github.com", "codeload.github.com"}
        ):
            raise ValueError(
                f"archive download redirected to an untrusted host: {response.geturl()!r}"
            )
        length = response.headers.get("Content-Length")
        if length and int(length) > _MAX_ARCHIVE_BYTES:
            raise ValueError(f"archive exceeds {_MAX_ARCHIVE_BYTES} byte limit")
        data = response.read(_MAX_ARCHIVE_BYTES + 1)
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise ValueError(f"archive exceeds {_MAX_ARCHIVE_BYTES} byte limit")
    return data


def _spans_digest(spans: dict[str, list[list[int]]]) -> str:
    blob = json.dumps(sorted((p, sorted(iv)) for p, iv in spans.items()), separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _spans_to_lines(spans: dict[str, list[list[int]]]) -> dict[str, set[int]]:
    return {p: {n for s, e in iv for n in range(s, e + 1)} for p, iv in spans.items()}


def _lines_to_intervals(lines: set[int]) -> list[list[int]]:
    if not lines:
        return []
    ordered = sorted(lines)
    intervals: list[list[int]] = []
    start = previous = ordered[0]
    for line in ordered[1:]:
        if line == previous + 1:
            previous = line
            continue
        intervals.append([start, previous])
        start = previous = line
    intervals.append([start, previous])
    return intervals


def _records_to_spans(records) -> dict[str, list[list[int]]]:
    """Aggregate every mapped record; never overwrite same-file evidence."""
    by_path: dict[str, set[int]] = {}
    for record in records:
        if record.mapped and record.lines:
            by_path.setdefault(record.path, set()).update(record.lines)
    return {
        path: _lines_to_intervals(lines)
        for path, lines in sorted(by_path.items())
    }


def _selection_digest(records) -> str:
    """Hash the full ordered attribution contract, not only its line-set union."""
    contract = [
        {
            "path": record.path,
            "rank": record.rank,
            "score": record.score,
            "token_cost": record.token_cost,
            "intervals": [list(interval) for interval in record.intervals()],
            "mapped": record.mapped,
            "reason": record.reason,
            "mapped_blocks": record.mapped_blocks,
            "unmapped_blocks": record.unmapped_blocks,
            "unmapped_lines": record.unmapped_lines,
        }
        for record in records
    ]
    blob = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── worker: one selection in an isolated process ───────────────────────────────

def _run_worker(checkout: str, task_json: str, budget: int) -> int:
    from benchmarks.contextbench_determinism_tax import build_engine_for_repo, entroly_select
    from benchmarks.contextbench_span_adapter import SelectedSpan

    with open(task_json, encoding="utf-8") as task_file:
        task = json.load(task_file)
    try:
        engine = build_engine_for_repo(checkout)
        records: list[SelectedSpan] = entroly_select(engine, checkout, task["problem_statement"], budget)
    except Exception as exc:  # report, never crash the orchestrator silently
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    spans = _records_to_spans(records)
    index_receipt = getattr(engine, "_contextbench_index_receipt", {})
    out = {
        "spans": spans,
        "spans_digest": _spans_digest(spans),
        "digest": _selection_digest(records),
        "n_selected": len(records),
        "n_unmapped": sum(
            1 for r in records if not r.mapped or r.unmapped_blocks > 0
        ),
        "n_unmapped_blocks": sum(r.unmapped_blocks for r in records),
        "unmapped_lines": sum(r.unmapped_lines for r in records),
        "unmapped_reasons": sorted({r.reason for r in records if r.reason}),
        "tokens": sum(r.token_cost for r in records),
        "index": {
            "files_indexed": index_receipt.get("files_indexed"),
            "skipped_too_large": index_receipt.get("skipped_too_large"),
            "skipped_unreadable": index_receipt.get("skipped_unreadable"),
        },
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
    try:
        out = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__), "--worker", checkout, task_json, str(budget)],
            capture_output=True, text=True, env=env, cwd=os.getcwd(), timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return {"error": "worker timed out after 1800 seconds"}
    lines = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"error": f"no output; stderr tail: {out.stderr[-300:]}"}
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {"error": f"invalid worker JSON: {exc}"}
    if out.returncode != 0:
        return {
            "error": payload.get("error")
            or f"worker exited {out.returncode}; stderr tail: {out.stderr[-300:]}"
        }
    return payload


# ── orchestrator ───────────────────────────────────────────────────────────────

def main(tasks_json: str, co_root: str, budget: int, max_bytes: int) -> int:
    from benchmarks.contextbench_determinism_tax import (
        evidence_drop,
        file_score,
        line_score,
        parse_gold,
    )

    if budget <= 0 or max_bytes <= 0:
        raise ValueError("budget and max_bytes must be positive")
    with open(tasks_json, encoding="utf-8") as task_file:
        tasks = json.load(task_file)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks file must be a non-empty list")
    os.makedirs(co_root, exist_ok=True)
    results = []
    for i, task in enumerate(tasks):
        dest = os.path.join(co_root, f"t{i}")
        task_path = os.path.join(co_root, f"task{i}.json")
        with open(task_path, "w", encoding="utf-8") as task_file:
            json.dump(task, task_file)
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
                rec["n_unmapped_blocks"] = a["n_unmapped_blocks"]
                rec["unmapped_lines"] = a["unmapped_lines"]
                rec["unmapped_reasons"] = a["unmapped_reasons"]
                rec["tokens"] = a["tokens"]
                rec["index"] = a["index"]
                gold = parse_gold(task["gold_context"])
                if not gold:
                    raise ValueError("task has empty or malformed gold context")
                pred = _spans_to_lines(a["spans"])
                fs, ls = file_score(pred, gold), line_score(pred, gold)
                rec["file"] = {"recall": round(fs.recall, 3), "precision": round(fs.precision, 3), "f1": round(fs.f1, 3)}
                rec["line"] = {"recall": round(ls.recall, 3), "precision": round(ls.precision, 3), "f1": round(ls.f1, 3)}
                rec["evidence_drop"] = round(
                    evidence_drop(pred, gold, unmapped_lines=a["unmapped_lines"]), 3
                )
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
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
    complete = len(ok) == len(results)
    reproducible = complete and all(r.get("reproducible") for r in ok)
    mapped = complete and all(
        r.get("n_unmapped") == 0
        and r.get("n_unmapped_blocks") == 0
        and r.get("unmapped_lines") == 0
        for r in ok
    )
    within_budget = complete and all(r.get("tokens", budget + 1) <= budget for r in ok)
    print("\n=== pilot acceptance ===")
    print(f"  executed end-to-end:   {len(ok)}/{len(results)}")
    print(f"  100% reproducible:     {reproducible} ({sum(bool(r.get('reproducible')) for r in ok)}/{len(ok)})")
    print(f"  zero unmapped spans:   {mapped}")
    print(f"  strict token budget:   {within_budget}")
    print(f"  no metric exceptions:  {complete}")
    return 0 if complete and reproducible and mapped and within_budget else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        raise SystemExit(_run_worker(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--budget"), 8000)
    max_bytes = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--max-bytes"), 500_000)
    raise SystemExit(main(args[0], args[1], budget, max_bytes))
