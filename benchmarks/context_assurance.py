"""Frozen local assurance/continuity fixture. Never calls a model provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from entroly.context_assurance import (
    AssuranceScope,
    HardObligation,
    audit_engine_selection,
    audit_receipt,
    recover_context_evidence,
)
from entroly.context_commit import create_context_commit
from entroly.context_receipts import ingest_documents, select_from_index
from entroly.context_receipts.models import stable_hash
from entroly.context_receipts.receipts import build_receipt
from entroly.context_receipts.models import ContextIndex
from entroly.decision_evaluation import (
    DecisionScope,
    JsonFieldProjection,
    context_regret,
    continuity_debt,
    observe_decisions,
)
from entroly.tokens import count_tokens

PROTOCOL_PATH = Path(__file__).with_name("context_assurance_protocol.json")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROTOCOL_PATH.parent.parent, text=True
    ).strip()


def _decision(context: str, target: str) -> dict:
    # A declared local rule policy, not a simulated model answer. Execute an
    # actual assertion command and project both the chosen target and its exit.
    match = re.search(r"command_route = '([^']+)'", context)
    chosen = match.group(1) if match else "missing_route"
    command = "import sys; sys.exit(0 if sys.argv[1] == sys.argv[2] else 1)"
    result = subprocess.run(
        [sys.executable, "-c", command, chosen, target],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return {"target": chosen, "exit_code": result.returncode}


def _percentiles(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        f"p{p}": ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]
        for p in (50, 95, 99)
    }


def run_benchmark(*, performance_samples: int | None = None) -> dict:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    sha = _git("rev-parse", "HEAD")
    projection = JsonFieldProjection(
        protocol["projection"], tuple(protocol["decision_fields"])
    )
    documents = [
        ("events.txt", "runtime logs tracing " * 70),
        ("routing.txt", protocol["fact"]),
    ]
    workload_hash = stable_hash(documents)
    index = ingest_documents(documents, prefer_rust=False)
    target_id = next(
        c["chunk_id"] for c in index["chunks"] if c["source_path"] == "routing.txt"
    )
    full_text = "\n\n".join(text for _, text in documents)
    static_text = full_text
    while count_tokens(static_text) > protocol["budget"]:
        static_text = static_text[:-1]
    all_rows = []
    summaries = []
    false_passes = 0
    inexact_recoveries = 0
    for turns in protocol["turn_counts"]:
        artifact_scope = AssuranceScope(
            "fixture:command-route", f"turns:{turns}", "agent:local"
        )
        commit = create_context_commit(
            documents,
            query=protocol["selection_query"],
            token_budget=protocol["budget"],
            prefer_rust=False,
            assurance_scope=artifact_scope,
        )
        receipt = commit["receipt"]
        selected_text = "\n\n".join(c["text"] for c in receipt["selected_context"])
        selected_ids = {c["chunk_id"] for c in receipt["selected_context"]}
        assert target_id not in selected_ids, (
            "the frozen delayed-fact fixture was not omitted"
        )
        recovered = recover_context_evidence(commit, target_id, scope=artifact_scope)
        inexact_recoveries += int(recovered["text"] != protocol["fact"])
        obligation = HardObligation("command_route", (target_id,))
        for strategy in protocol["context_strategies"]:
            scope = DecisionScope(
                "local-rule-policy-not-LLM",
                "command-route.v1",
                json.dumps({"temperature": None, "seed": 7}),
                protocol["task_family"],
                protocol["evaluator"],
                projection.protocol_id,
                "o200k_base",
                f"{strategy}/budget:{protocol['budget']}/turns:{turns}",
                workload_hash,
                sha,
            )
            rows = []
            cumulative_debt = 0.0
            for turn in range(1, turns + 1):
                relevant = (
                    turn > protocol["required_fact_delay_turns"] and turn % 10 == 0
                )
                if not relevant:
                    continue
                start = time.perf_counter()
                context = {
                    "full": full_text,
                    "static_prefix": static_text,
                    "receipt_selection": selected_text,
                    "receipt_exact_recovery": selected_text
                    + "\n\n"
                    + recovered["text"],
                }[strategy]
                query = protocol["query"]
                if strategy == "static_prefix":
                    cert = audit_engine_selection(
                        [{"content": context}],
                        query=query,
                        token_budget=protocol["budget"],
                    )
                else:
                    candidate = {**receipt, "query": query}
                    if strategy == "full":
                        candidate["selected_context"] = list(index["chunks"])
                        candidate["token_budget"] = count_tokens(
                            "\n\n".join(c["text"] for c in index["chunks"])
                        )
                    elif strategy == "receipt_exact_recovery":
                        candidate["selected_context"] = [
                            *receipt["selected_context"],
                            next(
                                c for c in index["chunks"] if c["chunk_id"] == target_id
                            ),
                        ]
                        candidate["token_budget"] = count_tokens(context)
                    cert = audit_receipt(index, candidate, obligations=[obligation])
                verified = (
                    [target_id]
                    if strategy in {"full", "receipt_exact_recovery"}
                    else []
                )
                debt = continuity_debt(
                    previously_omitted=[target_id],
                    newly_required=[target_id],
                    visible_before_decision=verified,
                    verified_recovered=verified,
                )
                cumulative_debt += debt["missed_context_debt"]
                full_output = _decision(full_text, protocol["target"])
                reduced_output = _decision(context, protocol["target"])
                repeat_output = _decision(full_text, protocol["target"])
                loss = lambda decision, truth: float(  # noqa: E731
                    decision["exit_code"] != 0 or decision["target"] != truth
                )
                observation = observe_decisions(
                    scope=scope,
                    trial_id=f"turn:{turn}",
                    task_id="delayed-command-route",
                    query=query,
                    full_output=full_output,
                    selected_output=reduced_output,
                    full_repeat_output=repeat_output,
                    projection=projection,
                    full_context_tokens=count_tokens(full_text),
                    selected_tokens=count_tokens(context),
                    recovered_tokens=count_tokens(recovered["text"])
                    if strategy == "receipt_exact_recovery"
                    else 0,
                    budget=candidate["token_budget"]
                    if strategy != "static_prefix"
                    else protocol["budget"],
                    assurance_verdict=cert["verdict"],
                    task_loss=loss,
                    truth=protocol["target"],
                    latency_ms=(time.perf_counter() - start) * 1000,
                    seeds=protocol["seeds"],
                )
                observation.update(
                    turn=turn,
                    strategy=strategy,
                    continuity=debt,
                    cumulative_missed_context_debt=cumulative_debt,
                    comparison="full/recovery use extra context; selection/prefix use fixed budget",
                )
                observation["observation_id"] = stable_hash(
                    {k: v for k, v in observation.items() if k != "observation_id"}
                )
                false_passes += int(
                    cert["verdict"] == "structurally_valid_risk_unmeasured"
                    and reduced_output["target"] != protocol["target"]
                )
                rows.append(observation)
            report = context_regret(rows)
            report.update(
                turns=turns,
                strategy=strategy,
                cumulative_missed_context_debt=cumulative_debt,
            )
            summaries.append(report)
            all_rows.extend(rows)

    samples = (
        performance_samples
        if performance_samples is not None
        else protocol["performance_samples"]
    )
    if type(samples) is not int or samples < 1:
        raise ValueError("performance_samples must be a positive integer")
    perf_docs = [
        (f"unit{i}.txt", f"runtime logging dimension evidence value {i}")
        for i in range(protocol["performance_unit_count"])
    ]
    perf_index = ingest_documents(perf_docs, prefer_rust=False)
    py_index = ContextIndex.from_dict(perf_index)
    select_ms, audit_ms, recover_ms, certificate_bytes, certificate_tokens = (
        [],
        [],
        [],
        [],
        [],
    )
    perf_scope = AssuranceScope("fixture:performance", "sample", "local")
    perf_commit = create_context_commit(
        perf_docs,
        query="runtime",
        token_budget=20,
        assurance_scope=perf_scope,
        prefer_rust=False,
    )
    perf_selected = {c["chunk_id"] for c in perf_commit["receipt"]["selected_context"]}
    perf_omitted = next(
        cid
        for cid in perf_commit["recovery_bundle"]["chunks"]
        if cid not in perf_selected
    )
    for _ in range(samples):
        start = time.perf_counter()
        candidate = build_receipt(py_index, query="runtime", token_budget=20).to_dict()
        select_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        cert = audit_receipt(perf_index, candidate)
        audit_ms.append((time.perf_counter() - start) * 1000)
        certificate_bytes.append(
            len(json.dumps(cert, separators=(",", ":")).encode("utf-8"))
        )
        certificate_tokens.append(count_tokens(json.dumps(cert, separators=(",", ":"))))
        start = time.perf_counter()
        recover_context_evidence(perf_commit, perf_omitted, scope=perf_scope)
        recover_ms.append((time.perf_counter() - start) * 1000)
    return {
        "schema": protocol["schema"],
        "protocol_sha256": protocol_hash,
        "git_sha": sha,
        "branch": _git("branch", "--show-current"),
        "dirty_worktree": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_workload_sha256": workload_hash,
        "claim_scope": protocol["claim_scope"],
        "limitations": protocol["limitations"],
        "false_passes_declared_obligations": false_passes,
        "inexact_recoveries": inexact_recoveries,
        "risk_calibration": "unavailable; dependent fixture observations",
        "summaries": summaries,
        "observations": all_rows,
        "performance": {
            "samples": samples,
            "selection_ms": _percentiles(select_ms),
            "certificate_and_joint_audit_ms": _percentiles(audit_ms),
            "exact_recovery_ms": _percentiles(recover_ms),
            "certificate_bytes": _percentiles(certificate_bytes),
            "certificate_tokens": _percentiles(certificate_tokens),
            "risk_lookup_ms": None,
            "recovery_planning_ms": None,
            "runtime_resolution_ms": None,
        },
        "provider_observed_usage": None,
        "provider_cost": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "git_sha",
                    "protocol_sha256",
                    "performance",
                    "false_passes_declared_obligations",
                    "inexact_recoveries",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
