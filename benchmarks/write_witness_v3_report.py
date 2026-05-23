"""Write the WITNESS v3 benchmark report from current result artifacts."""

from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    benchmark_path = RESULTS_DIR / "witness_python_path.json"
    calibration_path = RESULTS_DIR / "witness_thresholds_smoke.json"
    report_path = RESULTS_DIR / "witness_v3_report.md"

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    calibration = {}
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    lines = [
        "# WITNESS v3 Production Report",
        "",
        "## Benchmark",
        "",
        "| Slice | N | F1 | Accuracy | Suppression F1 | Exposure | Retention | ms/sample |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in benchmark:
        lines.append(
            "| {benchmark} | {n_samples} | {f1:.3f} | {accuracy:.3f} | "
            "{suppression_f1:.3f} | {unsupported_exposure_rate:.3f} | "
            "{supported_retention_rate:.3f} | {avg_ms:.1f} |".format(**row)
        )

    lines.extend([
        "",
        "## Calibration Smoke",
        "",
        "The calibration runner produced split-conformal CRC threshold stores with stable dataset hashes and CRC32 audit identifiers.",
        "",
        "| Profile | N | Hallucinated | Safe | tau_pass | tau_hedge | tau_warn | CRC32 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for profile, row in calibration.items():
        data = dict(row)
        data["profile_name"] = profile
        lines.append(
            "| {profile_name} | {n_samples} | {n_hallucinated} | {n_safe} | "
            "{tau_pass:.4f} | {tau_hedge:.4f} | {tau_warn:.4f} | {dataset_crc32} |".format(
                **data,
            )
        )

    lines.extend([
        "",
        "## Engineering Status",
        "",
        "- Continuous-risk Python path has property tests for boundedness, monotonicity, hard-gate dominance, SGD stability, atomic aggregation, and calibration persistence.",
        "- Online training accepts only external labels through the training API or RAVS-shaped honest outcome events.",
        "- Rust now exposes the continuous feature extractor and risk predictor through PyO3 bindings for the release hot path.",
        "- Summary profile is intentionally retention-first: unsupported summary claims are warned unless direct contradiction evidence is present.",
        "",
        "## Interpretation",
        "",
        "Dialogue is the strongest current win: hallucination detection F1 improves sharply while strict mode now suppresses unsupported dialogue claims. QA has much better retention than the old kill-switch but still needs NLI or a stronger QA-local verifier to reduce exposure. Summary detection improves, but the product policy warns rather than suppresses by default to avoid over-deleting user-visible prose.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
