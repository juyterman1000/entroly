"""Re-run WITNESS benchmarks through the Python graduated-policy path.

This is the validation harness for the v3 changes:
  - Dialogue fallback claim extraction
  - 4-action graduated policy (pass/hedge/warn/suppress)
  - Per-profile risk-graduated default thresholds

Comparison with the old binary policy (run_witness_benchmarks.py) is
documented in the README under "Witness v3 benchmark deltas".

We bypass the Rust fast-path by setting force_python=True so the new
Python code path is what's measured. Same metrics, same dataset, same
sample counts — only the verifier internals differ.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entroly.witness import WitnessAnalyzer  # noqa: E402


def load_halueval(config: str, n: int = 200) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("pminervini/HaluEval", config, split="data")
    samples: list[dict] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        label = str(row.get("hallucination", "")).strip().lower()
        is_hallucinated = label == "yes"

        if "knowledge" in row and "answer" in row:
            context = str(row.get("knowledge", ""))
            question = str(row.get("question", ""))
            output = str(row.get("answer", ""))
            if context and output:
                samples.append({
                    "context": f"{context}\n\nQuestion: {question}" if question else context,
                    "output": output,
                    "is_hallucinated": is_hallucinated,
                })
        elif "dialogue_history" in row:
            knowledge = str(row.get("knowledge", ""))
            history = str(row.get("dialogue_history", ""))
            output = str(row.get("response", row.get("right_response", "")))
            context = f"{knowledge}\n\nDialogue history:\n{history}" if knowledge else history
            if context and output:
                samples.append({"context": context, "output": output, "is_hallucinated": is_hallucinated})
        elif "document" in row:
            context = str(row.get("document", ""))
            output = str(row.get("summary", row.get("right_summary", "")))
            if context and output:
                samples.append({"context": context, "output": output, "is_hallucinated": is_hallucinated})
    return samples


def evaluate(analyzer: WitnessAnalyzer, samples: list[dict], name: str, threshold: float = 0.35) -> dict:
    tp = fp = fn = tn = 0
    stp = sfp = sfn = stn = 0
    total_ms = 0.0

    for sample in samples:
        result, rewrite = analyzer.analyze_and_rewrite(
            sample["context"], sample["output"], mode="strict"
        )
        total_ms += result.latency_ms

        actual = sample["is_hallucinated"]
        detected = result.summary_score < threshold
        suppressed = rewrite.suppressed_count > 0

        if detected and actual:
            tp += 1
        elif detected and not actual:
            fp += 1
        elif not detected and actual:
            fn += 1
        else:
            tn += 1

        if suppressed and actual:
            stp += 1
        elif suppressed and not actual:
            sfp += 1
        elif not suppressed and actual:
            sfn += 1
        else:
            stn += 1

    n_eval = tp + fp + fn + tn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(n_eval, 1)

    sprecision = stp / max(stp + sfp, 1)
    srecall = stp / max(stp + sfn, 1)
    sf1 = 2 * sprecision * srecall / max(sprecision + srecall, 1e-9)
    exposure_rate = sfn / max(stp + sfn, 1)
    retention = stn / max(stn + sfp, 1)

    return {
        "benchmark": name,
        "n_samples": n_eval,
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "suppression_f1": round(sf1, 4),
        "unsupported_exposure_rate": round(exposure_rate, 4),
        "supported_retention_rate": round(retention, 4),
        "avg_ms": round(total_ms / max(n_eval, 1), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "stp": stp, "sfp": sfp, "sfn": sfn, "stn": stn,
    }


def main() -> None:
    n_per = 200
    configs = [
        ("qa_samples", "HaluEval-QA", "benchmark_qa", n_per),
        ("dialogue_samples", "HaluEval-Dialogue", "dialogue", n_per),
        ("summarization_samples", "HaluEval-Summarization", "summary", n_per),
    ]

    print("=" * 84)
    print("  WITNESS v3 Benchmark — Python path with graduated policy")
    print("=" * 84)

    datasets: dict[str, tuple[list[dict], str]] = {}
    for config, name, profile, n in configs:
        print(f"  Loading {name}...", flush=True)
        samples = load_halueval(config, n)
        datasets[name] = (samples, profile)
        n_h = sum(1 for s in samples if s["is_hallucinated"])
        print(f"    {len(samples)} samples ({n_h} hallucinated, {len(samples) - n_h} grounded)")

    results: list[dict] = []
    print()
    print("=" * 84)
    print("  MODE: python-graduated (force_python=True, use_nli=False)")
    print("=" * 84)
    for name, (samples, profile) in datasets.items():
        analyzer = WitnessAnalyzer(use_nli=False, force_python=True, profile=profile)
        result = evaluate(analyzer, samples, f"{name} (python-graduated)")
        results.append(result)
        print(
            f"  {name:<28s} "
            f"F1={result['f1']:.3f} "
            f"Acc={result['accuracy']:.3f} "
            f"SuppF1={result['suppression_f1']:.3f} "
            f"Expose={result['unsupported_exposure_rate']:.3f} "
            f"Retain={result['supported_retention_rate']:.3f} "
            f"({result['avg_ms']:.1f}ms)"
        )

    print()
    print("  Side-by-side vs old binary policy (from prior run):")
    print("  " + "-" * 80)
    print(f"  {'Benchmark':<28s} {'Metric':<14s} {'Old':>8s} {'New':>8s} {'Delta':>10s}")
    print("  " + "-" * 80)
    # Hardcode the previously-shipped numbers from the user's report.
    old = {
        "HaluEval-QA":            {"f1": 0.524, "supp_f1": 0.646, "exp": 0.021, "ret": 0.010},
        "HaluEval-Dialogue":      {"f1": 0.060, "supp_f1": 0.000, "exp": 1.000, "ret": 1.000},
        "HaluEval-Summarization": {"f1": 0.040, "supp_f1": 0.269, "exp": 0.835, "ret": 0.942},
    }
    for r in results:
        name = r["benchmark"].split(" (")[0]
        o = old.get(name)
        if not o:
            continue
        for metric_label, old_v, new_v in [
            ("F1",          o["f1"],      r["f1"]),
            ("SuppF1",      o["supp_f1"], r["suppression_f1"]),
            ("Exposure",    o["exp"],     r["unsupported_exposure_rate"]),
            ("Retention",   o["ret"],     r["supported_retention_rate"]),
        ]:
            delta = new_v - old_v
            arrow = "+" if delta > 0 else ""
            print(f"  {name:<28s} {metric_label:<14s} {old_v:>8.3f} {new_v:>8.3f} {arrow}{delta:>9.3f}")
        print("  " + "-" * 80)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "witness_python_path.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved to: {out_file}")


if __name__ == "__main__":
    main()
