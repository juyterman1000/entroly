"""Directional containment analysis.

The global compression residual does not separate safe from unsafe: a
lexically near-identical *contradiction* compresses as well as a genuine
paraphrase.  Compression measures recoverability of form, not preservation
of meaning.

This script tests a narrower and better-posed hypothesis.  "Safe to omit"
means the retained set carries strictly more information than the omitted
fragment -- directional containment, not similarity.  The signature is
asymmetry:

    CCR(F | R)  low   and  CCR(R | F)  high     =>  R subsumes F

Measured only on the population where it would actually be consulted:
omissions that hard structural checks have already cleared, where the sole
remaining objection is soft lexical obligation matching.

Usage:
    python benchmarks/relate/analyze_subsumption.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from entroly.relate.compression_residual import certify_recoverable, conditional_residual
from entroly.relate.constraints import compile_query_contract
from entroly.relate.joint_omission import _SOFT_PREFIX
from entroly.relate.omission import verify_omission_safety
from entroly.relate.types import EvidenceCandidate, RelationVector

DATA_DIR = pathlib.Path(__file__).with_name("data")
DATASETS = (
    DATA_DIR / "omission_safety_v1.json",
    DATA_DIR / "omission_safety_adversarial_v1.json",
)


def _cand(frag: dict) -> EvidenceCandidate:
    return EvidenceCandidate(
        candidate_id=frag["id"],
        text=frag["text"],
        token_cost=frag["token_cost"],
        deterministic_score=frag.get("deterministic_score", 0.0),
        relation=RelationVector(),
        recoverable_ref=frag.get("recoverable_ref"),
    )


def _evaluations(dataset: dict):
    for case in dataset["cases"]:
        task = case["task"]
        trap = case.get("trap", "")
        if case.get("joint_omission"):
            frags = {f["id"]: _cand(f) for f in case["fragments"]}
            all_ev = tuple(frags.values())
            for ev in case["evaluations"]:
                omit_ids = ev.get("omit_ids", [ev["omit_id"]] if "omit_id" in ev else [])
                if len(omit_ids) != 1:
                    continue
                retained = tuple(frags[r] for r in ev["retain_ids"])
                yield (
                    f"{case['id']}:{omit_ids[0]}", task, frags[omit_ids[0]],
                    retained, all_ev, ev["ground_truth_safe"], trap,
                )
        else:
            omit = _cand(case["omit"])
            retained = tuple(_cand(f) for f in case["retain"])
            yield (
                case["id"], task, omit, retained, (omit,) + retained,
                case["ground_truth_safe"], trap,
            )


def main() -> None:
    soft_only = []
    for path in DATASETS:
        dataset = json.loads(path.read_text(encoding="utf-8"))
        for case_id, task, omit, retained, all_ev, gt_safe, trap in _evaluations(dataset):
            contract = compile_query_contract(task)
            # Calibration population is defined by the BASE witness, so this
            # analysis stays reproducible after an override path goes live.
            w = verify_omission_safety(omit, retained, contract)
            if w.safe_to_omit:
                continue
            if not all(r.startswith(_SOFT_PREFIX) for r in w.reasons):
                continue

            retained_text = "\n".join(r.text for r in retained)
            fwd = conditional_residual(omit.text, retained_text)
            rev = conditional_residual(retained_text, omit.text)
            cert = certify_recoverable(omit.text, retained_text)
            soft_only.append({
                "case_id": case_id,
                "trap": trap,
                "ground_truth_safe": gt_safe,
                "ccr_fragment_given_retained": round(fwd, 4),
                "ccr_retained_given_fragment": round(rev, 4),
                "containment_gap": round(rev - fwd, 4),
                "ensemble_residual": round(cert.residual, 4),
            })

    print("=== SOFT-ONLY POPULATION (where an override would be consulted) ===")
    print(f"{'gap':>8} {'fwd':>7} {'rev':>7} {'safe?':>6}  case")
    print("-" * 78)
    for r in sorted(soft_only, key=lambda x: -x["containment_gap"]):
        print(
            f"{r['containment_gap']:>8.4f} "
            f"{r['ccr_fragment_given_retained']:>7.4f} "
            f"{r['ccr_retained_given_fragment']:>7.4f} "
            f"{str(r['ground_truth_safe']):>6}  {r['case_id'][:46]}"
        )

    safe_gaps = [r["containment_gap"] for r in soft_only if r["ground_truth_safe"]]
    unsafe_gaps = [r["containment_gap"] for r in soft_only if not r["ground_truth_safe"]]
    print()
    print(f"safe gaps  : {sorted(safe_gaps, reverse=True)}")
    print(f"unsafe gaps: {sorted(unsafe_gaps, reverse=True)}")
    if safe_gaps and unsafe_gaps:
        separable = min(safe_gaps) > max(unsafe_gaps)
        print(f"separable by containment gap: {separable}")
        print(f"  min safe gap   = {min(safe_gaps):.4f}")
        print(f"  max unsafe gap = {max(unsafe_gaps):.4f}")
        if separable:
            print(f"  margin         = {min(safe_gaps) - max(unsafe_gaps):.4f}")
    print()
    print(f"population size: {len(soft_only)} "
          f"({len(safe_gaps)} safe, {len(unsafe_gaps)} unsafe)")


if __name__ == "__main__":
    main()
