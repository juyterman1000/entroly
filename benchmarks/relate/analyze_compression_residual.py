"""Falsification analysis for the conditional compression residual.

Answers four questions with measurements, not assertions:

1. Is the residual genuinely asymmetric on the subsumption case that
   defeated the structural witness?
2. Do safe and unsafe omissions separate by residual across both frozen
   datasets, and by what margin?
3. Is the residual ALONE sufficient as a safety witness?  (Prediction: no,
   it cannot see constraints.)
4. Which threshold, if any, is defensible -- and what does it cost?

Usage:
    python benchmarks/relate/analyze_compression_residual.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from entroly.relate.compression_residual import (
    COMPRESSORS,
    asymmetry,
    certify_recoverable,
    conditional_residual,
)

DATA_DIR = pathlib.Path(__file__).with_name("data")
FROZEN = DATA_DIR / "omission_safety_v1.json"
ADVERSARIAL = DATA_DIR / "omission_safety_adversarial_v1.json"


def _iter_evaluations(dataset: dict):
    """Yield (case_id, task, omitted_text, retained_text, gt_safe, trap)."""
    for case in dataset["cases"]:
        task = case["task"]
        trap = case.get("trap", "")
        if case.get("joint_omission"):
            frags = {f["id"]: f["text"] for f in case["fragments"]}
            for ev in case["evaluations"]:
                omit_ids = ev.get("omit_ids", [ev["omit_id"]] if "omit_id" in ev else [])
                retain_ids = ev["retain_ids"]
                omitted = "\n".join(frags[i] for i in omit_ids)
                retained = "\n".join(frags[i] for i in retain_ids)
                yield (
                    f"{case['id']}:{'|'.join(omit_ids)}",
                    task, omitted, retained, ev["ground_truth_safe"], trap,
                )
        else:
            omitted = case["omit"]["text"]
            retained = "\n".join(f["text"] for f in case["retain"])
            yield (
                case["id"], task, omitted, retained,
                case["ground_truth_safe"], trap,
            )


def question_1_asymmetry() -> dict:
    """The subsumption pair that the structural witness got wrong."""
    short = "Prometheus scrapes metrics every 15 seconds with 30-day retention."
    long = (
        "Application metrics are collected by Prometheus at 15-second intervals "
        "and stored for 30 days with downsampling at 5-minute resolution after 7 days."
    )
    out = {"short_text": short, "long_text": long, "by_compressor": {}}
    for name in COMPRESSORS:
        a, b = asymmetry(short, long, compressor=name)
        out["by_compressor"][name] = {
            "ccr_short_given_long": round(a, 4),
            "ccr_long_given_short": round(b, 4),
            "gap": round(b - a, 4),
        }
    return out


def question_2_separation() -> dict:
    """Residual distributions by ground-truth label across both datasets."""
    rows = []
    for path, tag in ((FROZEN, "frozen"), (ADVERSARIAL, "adversarial")):
        dataset = json.loads(path.read_text(encoding="utf-8"))
        for case_id, task, omitted, retained, gt_safe, trap in _iter_evaluations(dataset):
            cert = certify_recoverable(omitted, retained)
            rows.append({
                "dataset": tag,
                "case_id": case_id,
                "trap": trap,
                "ground_truth_safe": gt_safe,
                "residual": round(cert.residual, 4),
                "deciding": cert.deciding_compressor,
                "zlib": round(conditional_residual(omitted, retained, compressor="zlib"), 4),
            })

    safe = sorted(r["residual"] for r in rows if r["ground_truth_safe"])
    unsafe = sorted(r["residual"] for r in rows if not r["ground_truth_safe"])
    overlap = [u for u in unsafe if safe and u <= max(safe)]
    return {
        "rows": rows,
        "safe_residuals": safe,
        "unsafe_residuals": unsafe,
        "safe_max": max(safe) if safe else None,
        "unsafe_min": min(unsafe) if unsafe else None,
        "separable": bool(safe and unsafe and max(safe) < min(unsafe)),
        "unsafe_below_safe_max": len(overlap),
    }


def question_3_ccr_alone(rows: list[dict], threshold: float) -> dict:
    """Is the residual alone a sufficient safety witness?"""
    tp = fp = tn = fn = 0
    failures = []
    for r in rows:
        said_safe = r["residual"] <= threshold
        gt = r["ground_truth_safe"]
        if said_safe and gt:
            tp += 1
        elif said_safe and not gt:
            fn += 1
            failures.append({
                "case_id": r["case_id"], "trap": r["trap"],
                "residual": r["residual"],
            })
        elif not said_safe and gt:
            fp += 1
        else:
            tn += 1
    unsafe_total = fn + tn
    safe_total = tp + fp
    return {
        "threshold": threshold,
        "accuracy": round((tp + tn) / len(rows), 4) if rows else 0,
        "false_negatives": fn,
        "false_negative_rate": round(fn / unsafe_total, 4) if unsafe_total else 0,
        "false_positives": fp,
        "false_positive_rate": round(fp / safe_total, 4) if safe_total else 0,
        "unsafe_omissions_approved": failures,
    }


def main() -> None:
    report: dict = {}

    report["q1_asymmetry"] = question_1_asymmetry()
    sep = question_2_separation()
    report["q2_separation"] = {
        k: v for k, v in sep.items() if k != "rows"
    }

    # Sweep thresholds rather than assuming one.
    sweep = []
    for t in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        sweep.append(question_3_ccr_alone(sep["rows"], t))
    report["q3_ccr_alone_sweep"] = sweep

    best = min(sweep, key=lambda s: (s["false_negative_rate"], s["false_positive_rate"]))
    report["q4_best_threshold_for_ccr_alone"] = best

    report["per_case"] = sep["rows"]

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
