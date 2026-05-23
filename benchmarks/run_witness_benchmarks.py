"""Run WITNESS against HaluEval slices.

The detector metrics are still reported for comparability, but the product
metric is suppression: whether an unsupported, contradicted, or unknown claim
would be blocked/hedged before it reaches the user.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entroly.witness import WitnessAnalyzer  # noqa: E402


def load_halueval(config: str, n: int = 200) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("pminervini/HaluEval", config, split="data")
    print(f"        Columns: {ds.column_names}")

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
    examples: list[dict] = []

    for i, sample in enumerate(samples):
        result, rewrite = analyzer.analyze_and_rewrite(sample["context"], sample["output"], mode="strict")
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

        if i < 5:
            examples.append({
                "output_snippet": sample["output"][:80],
                "actual": "hallucinated" if actual else "grounded",
                "detected": "hallucinated" if detected else "grounded",
                "suppressed": suppressed,
                "warned": rewrite.warned_count > 0,
                "summary_score": round(result.summary_score, 3),
                "labels": [cert.label for cert in result.certificates],
            })

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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "suppression_precision": round(sprecision, 4),
        "suppression_recall": round(srecall, 4),
        "suppression_f1": round(sf1, 4),
        "unsupported_exposure_rate": round(exposure_rate, 4),
        "supported_retention_rate": round(retention, 4),
        "avg_ms": round(total_ms / max(n_eval, 1), 1),
        "examples": examples,
    }


def main() -> None:
    configs = [
        ("qa_samples", "HaluEval-QA", 200),
        ("dialogue_samples", "HaluEval-Dialogue", 200),
        ("summarization_samples", "HaluEval-Summarization", 200),
    ]
    datasets: dict[str, list[dict]] = {}

    print("=" * 78)
    print("  WITNESS Benchmark Suite")
    print("  Proof-carrying hallucination suppression")
    print("=" * 78)
    for config, name, n in configs:
        print(f"  Loading {name}...")
        samples = load_halueval(config, n)
        datasets[name] = samples
        n_h = sum(1 for sample in samples if sample["is_hallucinated"])
        print(f"    {len(samples)} samples ({n_h} hallucinated, {len(samples) - n_h} grounded)")

    results: list[dict] = []
    for mode_name, use_nli, limit in [
        ("string/local", False, None),
        ("nli", True, 50),
    ]:
        print()
        print("=" * 78)
        print(f"  MODE: {mode_name}")
        print("=" * 78)
        for name, samples in datasets.items():
            subset = samples[:limit] if limit else samples
            profile = "summary" if "Summarization" in name else "dialogue" if "Dialogue" in name else "benchmark_qa"
            analyzer = WitnessAnalyzer(use_nli=use_nli, profile=profile)
            result = evaluate(analyzer, subset, f"{name} ({mode_name})")
            results.append(result)
            print(
                f"  {name:<28s} "
                f"P={result['precision']:.3f} R={result['recall']:.3f} "
                f"F1={result['f1']:.3f} Acc={result['accuracy']:.3f} "
                f"SuppF1={result['suppression_f1']:.3f} "
                f"Expose={result['unsupported_exposure_rate']:.3f} "
                f"Retain={result['supported_retention_rate']:.3f} "
                f"({result['avg_ms']:.1f}ms)"
            )

    print()
    print("=" * 78)
    print("  FULL RESULTS")
    print("=" * 78)
    print(f"  {'Benchmark':<37s} {'N':>4s} {'F1':>6s} {'Acc':>6s} {'Supp':>6s} {'Exp':>6s} {'Ret':>6s} {'ms':>7s}")
    print("  " + "-" * 82)
    for result in results:
        print(
            f"  {result['benchmark']:<37s} {result['n_samples']:>4d} "
            f"{result['f1']:>6.3f} {result['accuracy']:>6.3f} "
            f"{result['suppression_f1']:>6.3f} "
            f"{result['unsupported_exposure_rate']:>6.3f} "
            f"{result['supported_retention_rate']:>6.3f} "
            f"{result['avg_ms']:>7.1f}"
        )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "witness_benchmarks.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved to: {out_file}")


if __name__ == "__main__":
    main()
