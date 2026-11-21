#!/usr/bin/env python3
"""Execute frozen real-history development tasks; never silently backfill them.

Candidate source is committed *before* model calls. Real repository oracles are
revalidated at the exact checked-out head, and mismatches fail the campaign.
Outputs are timestamped CI artifacts, not a 400-task production study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from benchmarks.agentic_task_miner import discover, validate
from benchmarks.real_commit_pilot import run


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE,
    ).strip()


def prepare(
    manifest: dict, repo: Path, outdir: Path, *, evaluation_version: int = 1
) -> tuple[Path, Path]:
    if evaluation_version not in (1, 2):
        raise ValueError("evaluation version must be 1 or 2")
    if manifest.get("schema") != "entroly.real-commit-pilot-plan.v1":
        raise ValueError("unknown protocol plan")
    shas = manifest.get("candidate_shas")
    if not isinstance(shas, list) or not shas or len(shas) != len(set(shas)):
        raise ValueError("candidate list must be frozen, nonempty, and unique")
    baseline = manifest.get("source_baseline")
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", baseline, "HEAD"],
                   check=True, stdout=subprocess.DEVNULL)
    candidates = {x.sha: x for x in discover(repo, max_commits=700)}
    missing = set(shas) - set(candidates)
    if missing:
        raise RuntimeError(f"frozen candidates no longer discoverable: {sorted(missing)}")
    tasks = []
    for sha in shas:
        task = validate(repo, candidates[sha], timeout=manifest["test_timeout_s"])
        if task is None:
            raise RuntimeError(f"frozen candidate oracle cannot be revalidated: {sha}")
        tasks.append(task)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks_path = outdir / "frozen-validated-tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in tasks),
        encoding="utf-8",
    )
    protocol = {k: manifest[k] for k in (
        "candidate_shas", "model", "model_digest", "seed", "max_corpus_bytes",
        "selection_budget", "max_output_tokens", "context_window",
        "model_timeout_s", "test_timeout_s",
    )}
    protocol.update(
        schema=f"entroly.real-commit-pilot.v{evaluation_version}",
        source_sha=git(repo, "rev-parse", "HEAD"),
        tasks_sha256=hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        baseline=baseline,
    )
    protocol_path = outdir / "effective-frozen-protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return protocol_path, tasks_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--evaluation-version", type=int, choices=(1, 2), default=1,
                        help="2 uses task-conditioned selection, a null control, and rotating arm order")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.outdir.resolve()
    if output.is_relative_to(repo):
        raise ValueError("results containing model output cannot be saved inside repo")
    if os.environ.get("ENTROLY_REAL_COMMIT_SANDBOX") != "1":
        raise RuntimeError("real repository model trials require sandboxed pytest")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol, tasks = prepare(manifest, repo, output, evaluation_version=args.evaluation_version)
    result = run(protocol, tasks, repo, output / "real-commit-model-results.json",
                 base_url="http://127.0.0.1:11434")
    rows = result["rows"]
    modeled = [x for x in rows if x.get("arm") in ("raw", "selected", "raw_repeat", "null")
               and isinstance(x.get("input_tokens"), int)]
    outcomes = {}
    for row in rows:
        key = row.get("outcome", "not_started")
        outcomes[key] = outcomes.get(key, 0) + 1
    stats = {
        "schema": "entroly.real-history-evidence-summary.v1",
        "source_sha": result["source_sha"],
        "candidate_count": len(manifest["candidate_shas"]),
        "modeled_attempt_count": len(modeled),
        "model_reported_input_tokens": sum(x["input_tokens"] for x in modeled),
        "model_reported_output_tokens": sum(x["output_tokens"] for x in modeled if isinstance(x.get("output_tokens"), int)),
        "outcome_counts": outcomes,
        "paired_outcomes": result["summary"],
        "baseline_comparison": (
            "raw vs selected vs raw-repeat vs null"
            if args.evaluation_version == 2 else "raw vs selected vs raw-repeat, not closed loop"
        ),
        "quality_claim_allowed": False,
        "invoice_reconciliation": "NOT_EXECUTED",
        "cross_repository_holdout": "NOT_EXECUTED",
    }
    (output / "real-commit-summary.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(stats, sort_keys=True), flush=True)
    if result["status"] != "complete" or not modeled:
        raise RuntimeError("no real model result: cannot claim dogfood completion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
