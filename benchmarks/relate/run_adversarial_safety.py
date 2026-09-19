"""Adversarial expansion of the omission safety benchmark.

Targets specific failure modes of dimension clustering:
- Same-dimension independent constraints
- High-dimensional threshold scaling
- False dimension separation
- Constraint within summary context
- Single-dimension both-necessary

Usage:
    python benchmarks/relate/run_adversarial_safety.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from entroly.relate.types import EvidenceCandidate, RelationVector
from entroly.relate.constraints import compile_query_contract
from entroly.relate.omission import verify_omission_safety
from entroly.relate.joint_omission import (
    verify_omission_with_dimensions,
    verify_joint_omission_safety,
)


DATA_PATH = pathlib.Path(__file__).with_name("data") / "omission_safety_adversarial_v1.json"


def _make_candidate(frag: dict) -> EvidenceCandidate:
    return EvidenceCandidate(
        candidate_id=frag["id"],
        text=frag["text"],
        token_cost=frag["token_cost"],
        deterministic_score=frag.get("deterministic_score", 0.0),
        relation=RelationVector(),
        recoverable_ref=frag.get("recoverable_ref"),
    )


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def run_single_case(case: dict) -> list[dict]:
    verdicts: list[dict] = []
    task = case["task"]
    contract = compile_query_contract(task)

    if case.get("joint_omission"):
        fragments = {f["id"]: _make_candidate(f) for f in case["fragments"]}
        all_candidates = list(fragments.values())
        all_evidence = tuple(all_candidates)
        for ev in case["evaluations"]:
            omit_ids = ev.get("omit_ids", [ev["omit_id"]] if "omit_id" in ev else [])
            retain_ids = ev["retain_ids"]
            retained = tuple(fragments[rid] for rid in retain_ids)
            gt_safe = ev["ground_truth_safe"]

            if len(omit_ids) == 1:
                witness = verify_omission_with_dimensions(
                    fragments[omit_ids[0]], retained, contract,
                    all_evidence=all_evidence,
                )
                said_safe = witness.safe_to_omit
                reasons = list(witness.reasons)
            else:
                jw = verify_joint_omission_safety(
                    [fragments[oid] for oid in omit_ids],
                    all_candidates,
                    contract,
                )
                said_safe = jw.safe_to_omit
                reasons = list(jw.reasons)

            correct = said_safe == gt_safe
            verdicts.append({
                "case_id": f"{case['id']}:{'|'.join(omit_ids)}",
                "category": case["category"],
                "correct": correct,
                "ground_truth_safe": gt_safe,
                "witness_said_safe": said_safe,
                "false_negative": said_safe and not gt_safe,
                "false_positive": not said_safe and gt_safe,
                "trap": case["trap"],
                "witness_reasons": reasons,
            })
    else:
        omit = _make_candidate(case["omit"])
        retained = tuple(_make_candidate(f) for f in case["retain"])
        all_evidence = (omit,) + retained
        gt_safe = case["ground_truth_safe"]
        witness = verify_omission_with_dimensions(
            omit, retained, contract, all_evidence=all_evidence,
        )
        said_safe = witness.safe_to_omit
        correct = said_safe == gt_safe
        verdicts.append({
            "case_id": case["id"],
            "category": case["category"],
            "correct": correct,
            "ground_truth_safe": gt_safe,
            "witness_said_safe": said_safe,
            "false_negative": said_safe and not gt_safe,
            "false_positive": not said_safe and gt_safe,
            "trap": case.get("trap", ""),
            "witness_reasons": list(witness.reasons),
        })
    return verdicts


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"dataset not found: {DATA_PATH}")

    payload = DATA_PATH.read_bytes()
    dataset = json.loads(payload)
    sha256 = hashlib.sha256(payload).hexdigest()

    all_verdicts: list[dict] = []
    for case in dataset["cases"]:
        all_verdicts.extend(run_single_case(case))

    total = len(all_verdicts)
    correct = sum(1 for v in all_verdicts if v["correct"])
    false_neg = sum(1 for v in all_verdicts if v["false_negative"])
    false_pos = sum(1 for v in all_verdicts if v["false_positive"])
    unsafe_cases = sum(1 for v in all_verdicts if not v["ground_truth_safe"])
    safe_cases = sum(1 for v in all_verdicts if v["ground_truth_safe"])

    result = {
        "provenance": {
            "dataset_sha256": sha256,
            "dataset_version": dataset["version"],
            "git_head": _git_head(),
            "backend": "info_residual_with_dimensions",
        },
        "summary": {
            "total_evaluations": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0,
            "false_negatives": false_neg,
            "false_negative_rate": round(false_neg / unsafe_cases, 4) if unsafe_cases else 0,
            "false_positives": false_pos,
            "false_positive_rate": round(false_pos / safe_cases, 4) if safe_cases else 0,
        },
        "verdicts": all_verdicts,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
