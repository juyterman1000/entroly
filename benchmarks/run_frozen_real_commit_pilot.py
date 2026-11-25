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
from benchmarks.real_commit_pilot import _local_model_digest, run
from benchmarks.pilot_runtime_identity import runtime_identity
from entroly.native_status import native_status


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE,
    ).strip()


def _check_output_files(outdir: Path) -> None:
    for name in (
        "effective-frozen-protocol.json", "frozen-validated-tasks.jsonl",
        "real-commit-model-results.json", "real-commit-summary.json",
    ):
        path = outdir / name
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite an existing pilot file: {path}")


def _manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare(
    manifest: dict, repo: Path, outdir: Path, *, evaluation_version: int = 1
) -> tuple[Path, Path]:
    if evaluation_version not in (1, 2):
        raise ValueError("evaluation version must be 1 or 2")
    _check_output_files(outdir)
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
    with tasks_path.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in tasks))
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
    if evaluation_version == 2:
        protocol["manifest_sha256"] = _manifest_sha256(manifest)
    protocol_path = outdir / "effective-frozen-protocol.json"
    with protocol_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol_path, tasks_path


def prepare_continuation(
    manifest: dict, repo: Path, outdir: Path, continue_from: Path, *,
    evaluation_version: int = 1,
) -> tuple[Path, Path]:
    _check_output_files(outdir)
    if evaluation_version != 2:
        raise ValueError("continuation requires explicit evaluation version 2")
    parent = json.loads(continue_from.read_text(encoding="utf-8"))
    protocol_path = continue_from.parent / "effective-frozen-protocol.json"
    tasks_path = continue_from.parent / "frozen-validated-tasks.jsonl"
    protocol_bytes = protocol_path.read_bytes()
    tasks_bytes = tasks_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if (protocol.get("schema") != "entroly.real-commit-pilot.v2"
            or parent.get("schema") != "entroly.real-commit-pilot-result.v2"):
        raise ValueError("continuation requires schema v2 frozen protocol and parent artifact")
    if protocol.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("manifest differs from frozen protocol")
    if protocol.get("baseline") != manifest.get("source_baseline"):
        raise ValueError("baseline differs from frozen protocol")
    if protocol.get("source_sha") != git(repo, "rev-parse", "HEAD"):
        raise ValueError("runner source checkout differs from frozen protocol")
    if hashlib.sha256(protocol_bytes).hexdigest() != parent.get("protocol_sha256"):
        raise ValueError("frozen protocol differs from parent artifact")
    tasks_digest = hashlib.sha256(tasks_bytes).hexdigest()
    if (tasks_digest != parent.get("tasks_sha256")
            or tasks_digest != protocol.get("tasks_sha256")):
        raise ValueError("frozen tasks differ from parent artifact or protocol")
    outdir.mkdir(parents=True, exist_ok=True)
    protocol_copy = outdir / protocol_path.name
    tasks_copy = outdir / tasks_path.name
    with protocol_copy.open("xb") as stream:
        stream.write(protocol_bytes)
    with tasks_copy.open("xb") as stream:
        stream.write(tasks_bytes)
    return protocol_copy, tasks_copy


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _preflight(manifest: dict, repo: Path) -> None:
    """Reject unavailable execution prerequisites before expensive task mining."""
    engine = native_status()
    if not engine.ok:
        raise RuntimeError("pilot requires the current native engine")
    endpoint = "http://127.0.0.1:11434"
    if _local_model_digest(endpoint, manifest["model"]) != manifest["model_digest"]:
        raise ValueError("local model digest differs from frozen manifest")
    runtime_identity(repo, engine, endpoint)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--evaluation-version", type=int, choices=(1, 2), default=1,
                        help="2 uses task-conditioned selection, a null control, and rotating arm order")
    parser.add_argument("--max-model-calls", type=_positive_int,
                        help="bound arm attempts in this invocation (requires --evaluation-version 2)")
    parser.add_argument("--continue-from", type=Path,
                        help="continue a parent artifact into a new output directory (requires --evaluation-version 2)")
    args = parser.parse_args()
    if (args.max_model_calls is not None or args.continue_from is not None) and args.evaluation_version != 2:
        parser.error("--max-model-calls and --continue-from require explicit --evaluation-version 2")
    repo = args.repo.resolve()
    output = args.outdir.resolve()
    if output.is_relative_to(repo):
        raise ValueError("results containing model output cannot be saved inside repo")
    _check_output_files(output)
    if os.environ.get("ENTROLY_REAL_COMMIT_SANDBOX") != "1":
        raise RuntimeError("real repository model trials require sandboxed pytest")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.max_model_calls is not None or args.continue_from is not None:
        _preflight(manifest, repo)
    run_options = {}
    if args.continue_from is not None:
        protocol, tasks = prepare_continuation(
            manifest, repo, output, args.continue_from,
            evaluation_version=args.evaluation_version,
        )
        run_options["continue_from"] = args.continue_from
    else:
        protocol, tasks = prepare(manifest, repo, output, evaluation_version=args.evaluation_version)
    if args.max_model_calls is not None:
        run_options["max_model_calls"] = args.max_model_calls
    result = run(protocol, tasks, repo, output / "real-commit-model-results.json",
                 base_url="http://127.0.0.1:11434", **run_options)
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
        "status": result["status"],
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
    if "continuation" in result:
        stats.update({key: result["continuation"][key] for key in (
            "new_arm_attempts", "total_arm_attempts",
        )})
    with (output / "real-commit-summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, sort_keys=True), flush=True)
    if result["status"] == "call_limit_reached":
        return 0
    if result["status"] != "complete" or not modeled:
        raise RuntimeError("no real model result: cannot claim dogfood completion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
