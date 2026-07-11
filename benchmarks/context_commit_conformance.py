"""Falsification benchmark for the Entroly Context Commit contract."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from entroly.context_commit import (
    create_context_commit,
    replay_context,
    verify_context_commit,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "context_commit_conformance.json"


def _case(case_id: int) -> tuple[list[tuple[str, str]], str]:
    documents: list[tuple[str, str]] = []
    target_doc = case_id % 6
    target_window = 15 + (case_id % 17)
    target_retries = 2 + (case_id % 5)
    for doc_id in range(6):
        lines = [f"# Service {doc_id} case {case_id}"]
        for fact_id in range(12):
            value = (case_id + 3) * (doc_id + 5) + fact_id
            lines.append(
                f"service_{doc_id}_fact_{fact_id} = {value}  # deterministic fixture"
            )
        if doc_id == target_doc:
            lines.extend(
                [
                    f"DEPLOYMENT_WINDOW_MINUTES = {target_window}",
                    f"PAYMENT_RETRY_LIMIT = {target_retries}",
                    "FAIL_CLOSED_ON_MISSING_EVIDENCE = True",
                ]
            )
        documents.append((f"services/service_{doc_id}.py", "\n".join(lines) + "\n"))
    query = (
        f"For service {target_doc}, what are DEPLOYMENT_WINDOW_MINUTES and "
        "PAYMENT_RETRY_LIMIT?"
    )
    return documents, query


def _tampered_variants(commit: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []

    selected = copy.deepcopy(commit)
    selected_items = selected["receipt"]["selected_context"]
    if selected_items:
        selected_items[0]["text"] += "\nTAMPERED = True"
    else:
        selected["receipt"]["token_budget"] += 1
    variants.append(selected)

    recovery = copy.deepcopy(commit)
    recovery_chunk = next(iter(recovery["recovery_bundle"]["chunks"].values()))
    recovery_chunk["text"] += "\nTAMPERED = True"
    variants.append(recovery)

    receipt = copy.deepcopy(commit)
    receipt["receipt"]["token_budget"] += 1
    variants.append(receipt)

    engine = copy.deepcopy(commit)
    engine["engine"]["implementation"] = "tampered"
    variants.append(engine)

    lineage = copy.deepcopy(commit)
    lineage["parent_commit_id"] = "not-a-context-commit"
    variants.append(lineage)

    identity = copy.deepcopy(commit)
    identity["commit_id"] += "0"
    variants.append(identity)
    return variants


def _mode_benchmark(cases: int, *, prefer_rust: bool) -> dict[str, Any]:
    deterministic = 0
    valid = 0
    replay_exact = 0
    omitted = 0
    recovered = 0
    tamper_trials = 0
    tamper_detected = 0
    artifact_sizes: list[int] = []
    elapsed_ms: list[float] = []
    actual_engine = ""

    for case_id in range(cases):
        documents, query = _case(case_id)
        started = time.perf_counter()
        first = create_context_commit(
            documents,
            query=query,
            token_budget=120,
            chunk_tokens=80,
            overlap_tokens=12,
            prefer_rust=prefer_rust,
        )
        elapsed_ms.append((time.perf_counter() - started) * 1000)
        second = create_context_commit(
            reversed(documents),
            query=query,
            token_budget=120,
            chunk_tokens=80,
            overlap_tokens=12,
            prefer_rust=prefer_rust,
        )
        actual_engine = str(first["engine"]["implementation"])
        verification = verify_context_commit(first)
        valid += int(verification.valid)
        deterministic += int(first["commit_id"] == second["commit_id"])
        replay_exact += int(replay_context(first) == first["receipt"]["selected_context"])
        omitted += verification.omitted_chunks
        recovered += verification.recovered_omitted_chunks
        artifact_sizes.append(
            len(json.dumps(first, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        )

        for tampered in _tampered_variants(first):
            tamper_trials += 1
            tamper_detected += int(not verify_context_commit(tampered).valid)

    return {
        "requested_engine": "rust" if prefer_rust else "python",
        "actual_engine": actual_engine,
        "cases": cases,
        "valid_commits": valid,
        "deterministic_replays": deterministic,
        "exact_context_replays": replay_exact,
        "omitted_chunks": omitted,
        "exact_omission_recoveries": recovered,
        "tamper_trials": tamper_trials,
        "tamper_detections": tamper_detected,
        "median_create_ms": round(statistics.median(elapsed_ms), 3),
        "median_artifact_bytes": int(statistics.median(artifact_sizes)),
    }


def run_benchmark(cases: int = 64) -> dict[str, Any]:
    modes = [_mode_benchmark(cases, prefer_rust=False)]
    rust = _mode_benchmark(cases, prefer_rust=True)
    if rust["actual_engine"] == "rust":
        modes.append(rust)

    total_cases = sum(mode["cases"] for mode in modes)
    total_omitted = sum(mode["omitted_chunks"] for mode in modes)
    total_tamper = sum(mode["tamper_trials"] for mode in modes)
    aggregate = {
        "engine_modes": len(modes),
        "cases": total_cases,
        "valid_commit_rate": sum(mode["valid_commits"] for mode in modes)
        / max(1, total_cases),
        "deterministic_replay_rate": sum(
            mode["deterministic_replays"] for mode in modes
        )
        / max(1, total_cases),
        "exact_context_replay_rate": sum(
            mode["exact_context_replays"] for mode in modes
        )
        / max(1, total_cases),
        "exact_omission_recovery_rate": sum(
            mode["exact_omission_recoveries"] for mode in modes
        )
        / max(1, total_omitted),
        "tamper_detection_rate": sum(
            mode["tamper_detections"] for mode in modes
        )
        / max(1, total_tamper),
        "omitted_chunks_verified": total_omitted,
        "tamper_trials": total_tamper,
    }
    return {
        "benchmark": "entroly-context-commit-conformance",
        "schema_version": "entroly.benchmark.context-commit.v1",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "configuration": {
            "cases_per_engine": cases,
            "token_budget": 120,
            "chunk_tokens": 80,
            "overlap_tokens": 12,
            "tamper_mutations_per_case": 6,
        },
        "aggregate": aggregate,
        "modes": modes,
        "caveats": [
            "Synthetic deterministic code fixtures; no LLM or network calls.",
            "Measures artifact integrity, replay, and recovery, not answer quality.",
            "Does not claim Python and Rust select identical context.",
            "Content addressing detects mutation; signer identity requires optional attestation APIs.",
            "Recovery bundles contain source text and must follow the user's data-retention policy.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    result = run_benchmark(args.cases)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
