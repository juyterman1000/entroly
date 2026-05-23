"""Calibrate WITNESS thresholds on HaluEval with conformal risk control.

This runner is intentionally separate from the benchmark report: it creates
policy thresholds from labeled calibration data and writes an auditable JSON
store that production can load later. It does not tune on the test report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.run_witness_python_path import load_halueval  # noqa: E402
from entroly.witness import WitnessAnalyzer  # noqa: E402
from entroly.witness_calibration import (  # noqa: E402
    CalibrationSample,
    CalibrationStore,
    calibrate,
)


HALUEVAL_PROFILES: tuple[tuple[str, str, str], ...] = (
    ("qa_samples", "benchmark_qa", "HaluEval-QA"),
    ("dialogue_samples", "dialogue", "HaluEval-Dialogue"),
    ("summarization_samples", "summary", "HaluEval-Summarization"),
)


def risk_for_sample(analyzer: WitnessAnalyzer, sample: dict[str, Any]) -> float:
    result = analyzer.analyze(sample["context"], sample["output"])
    if not result.certificates:
        return 0.0
    return max(float(cert.risk) for cert in result.certificates)


def collect_calibration_samples(
    *,
    profile: str,
    config: str,
    n: int,
    force_python: bool = True,
    use_nli: bool = False,
) -> list[CalibrationSample]:
    samples = load_halueval(config, n)
    analyzer = WitnessAnalyzer(use_nli=use_nli, force_python=force_python, profile=profile)
    out: list[CalibrationSample] = []
    for sample in samples:
        rho = risk_for_sample(analyzer, sample)
        out.append(CalibrationSample(
            rho=rho,
            y=1 if sample["is_hallucinated"] else 0,
            profile=profile,
        ))
    return out


def run_calibration(
    *,
    n: int,
    out_path: Path,
    alpha_pass: float,
    alpha_hedge: float,
    alpha_warn: float,
    use_nli: bool = False,
) -> list[dict[str, Any]]:
    store = CalibrationStore(out_path)
    rows: list[dict[str, Any]] = []
    for config, profile, label in HALUEVAL_PROFILES:
        samples = collect_calibration_samples(
            profile=profile,
            config=config,
            n=n,
            use_nli=use_nli,
        )
        result = calibrate(
            samples,
            profile=profile,
            alpha_pass=alpha_pass,
            alpha_hedge=alpha_hedge,
            alpha_warn=alpha_warn,
        )
        store.update(result)
        row = {
            "benchmark": label,
            "profile": profile,
            "n": result.n_samples,
            "hallucinated": result.n_hallucinated,
            "safe": result.n_safe,
            "tau_pass": round(result.tau_pass, 6),
            "tau_hedge": round(result.tau_hedge, 6),
            "tau_warn": round(result.tau_warn, 6),
            "empirical_exposure_at_warn": result.empirical_exposure_at_warn,
            "empirical_retention_at_pass": result.empirical_retention_at_pass,
            "dataset_hash": result.dataset_hash,
            "dataset_crc32": result.dataset_crc32,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Samples per HaluEval slice")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/witness_thresholds.json"),
        help="Calibration store path",
    )
    parser.add_argument("--alpha-pass", type=float, default=0.80)
    parser.add_argument("--alpha-hedge", type=float, default=0.50)
    parser.add_argument("--alpha-warn", type=float, default=0.20)
    parser.add_argument("--use-nli", action="store_true", help="Use OpenAI-backed NLI during scoring")
    args = parser.parse_args()

    rows = run_calibration(
        n=args.n,
        out_path=args.out,
        alpha_pass=args.alpha_pass,
        alpha_hedge=args.alpha_hedge,
        alpha_warn=args.alpha_warn,
        use_nli=args.use_nli,
    )
    print("WITNESS threshold calibration (split-conformal CRC)")
    print(f"store: {args.out}")
    print(f"{'benchmark':<24s} {'profile':<12s} {'n':>5s} {'tau_pass':>9s} {'tau_hedge':>10s} {'tau_warn':>9s} {'crc32':>10s}")
    for row in rows:
        print(
            f"{row['benchmark']:<24s} {row['profile']:<12s} {row['n']:>5d} "
            f"{row['tau_pass']:>9.4f} {row['tau_hedge']:>10.4f} {row['tau_warn']:>9.4f} "
            f"{row['dataset_crc32']:>10s}"
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
