"""Measure WITNESS on the SHIPPED (Rust) path — force_python=False.

`run_witness_python_path.py` measures the Python implementation. Native-
engine users (the proxy/MCP default) hit the Rust implementation
instead. After porting the QA binding + question-residual gates into
`entroly-core/src/witness.rs`, this script reports the exposure /
retention the *shipped* path actually delivers — which, given known
Python↔Rust divergence (best-sentence selection, stopword sets,
tokenizer), may differ from the Python numbers. Reported honestly, not
assumed equal.

Reuses the exact metric/loader code from run_witness_python_path so the
only variable is the runtime.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entroly.witness import WitnessAnalyzer, _rust_witness_analyze  # noqa: E402
from benchmarks.run_witness_python_path import evaluate, load_halueval  # noqa: E402


def main() -> None:
    if _rust_witness_analyze is None:
        print("[skip] entroly_core not installed — Rust path unavailable. "
              "Run `maturin develop --release -m entroly-core/Cargo.toml`.")
        return

    configs = [
        ("qa_samples", "HaluEval-QA", "benchmark_qa", 200),
        ("dialogue_samples", "HaluEval-Dialogue", "dialogue", 200),
        ("summarization_samples", "HaluEval-Summarization", "summary", 200),
    ]
    print("=" * 84)
    print("  WITNESS — SHIPPED Rust path (force_python=False, use_nli=False)")
    print("=" * 84)
    results = []
    for config, name, profile, n in configs:
        samples = load_halueval(config, n)
        analyzer = WitnessAnalyzer(
            use_nli=False, force_python=False, profile=profile
        )
        r = evaluate(analyzer, samples, f"{name} (rust-shipped)")
        results.append(r)
        print(
            f"  {name:<28s} F1={r['f1']:.3f} Acc={r['accuracy']:.3f} "
            f"SuppF1={r['suppression_f1']:.3f} "
            f"Expose={r['unsupported_exposure_rate']:.3f} "
            f"Retain={r['supported_retention_rate']:.3f} "
            f"({r['avg_ms']:.1f}ms)"
        )

    print()
    print("  Reference (Python path, prior run): "
          "QA Expose≈0.340 Retain≈0.738")
    print("  Any QA gap here = real Python↔Rust divergence; the parity "
          "test (tests/test_witness_parity.py) gates the decision-level "
          "cases.")

    out = Path(__file__).parent / "results" / "witness_rust_path.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
