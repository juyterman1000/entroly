"""
HaluEval-Summarization Falsification Probe (C1-C4).

Per EICV_PREREGISTRATION.md:
  - Target: F1 ≥ 0.55 (NLI baselines); AUROC anchor ≥ 0.75
  - Falsification: survives iff min(C1..C4) ≥ 0.72

Item construction (mirrors benchmarks/run_witness_benchmarks.py:49-53):
  For each document: right_summary vs hallucinated_summary, document as context.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks._falsification_common import (  # noqa: E402
    FalsItem, build_records, dataset_hash, run_probe,
)

SEED = 42
N_ITEMS = 400
TARGET_AUROC = 0.75
RESULTS_DIR = _THIS / "results"
OUT_PATH = RESULTS_DIR / "halueval_summarization_falsification.json"
CACHE_PATH = RESULTS_DIR / "halueval_summarization_falsification_cache.json"


def load_items() -> list[FalsItem]:
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    items: list[FalsItem] = []
    for i, row in enumerate(ds):
        document = str(row.get("document", "") or "")
        right = str(row.get("right_summary", row.get("summary", "")) or "")
        halu = str(row.get("hallucinated_summary", "") or "")
        if not (document and right and halu):
            continue
        items.append(FalsItem(
            context=document, query="",
            right=right, halu=halu,
            item_id=f"hes_{i:04d}",
        ))
    rng = random.Random(SEED)
    rng.shuffle(items)
    return items[:N_ITEMS]


def main() -> int:
    print("=" * 74)
    print("  HaluEval-Summarization Falsification Probe (C1-C4)")
    print("=" * 74)
    items = load_items()
    n = len(items)
    print(f"  Loaded {n} items")
    if n == 0:
        return 1
    ds_hash = dataset_hash(items)
    print(f"  Dataset hash: {ds_hash}")

    backend = "deterministic"
    print(f"  Backend: {backend}")
    t0 = time.perf_counter()
    records = build_records(items, backend=backend, seed=SEED, cache_path=CACHE_PATH)
    print(f"  Built {len(records)} records ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    result = run_probe("halueval_summarization", records, profile="summary",
                       target=TARGET_AUROC, tolerance=0.03)
    print(f"  Scored in {time.perf_counter() - t0:.1f}s")

    result["dataset_hash"] = ds_hash
    result["preregistered_target"] = TARGET_AUROC
    result["preregistration_doc"] = "benchmarks/EICV_PREREGISTRATION.md"

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print(f"  Saved: {OUT_PATH}")
    print()
    print(f"  {'condition':<34}{'fusion':>8}{'G-only':>8}{'WIT':>7}")
    print("  " + "-" * 56)
    for c in result["conditions"]:
        print(f"  {c['name']:<34}{c['fusion']:>8.4f}"
              f"{c['g_only']:>8.4f}{c['witness_only']:>7.4f}")
    print()
    print(f"  min(C1-C4)             = {result['min_fusion_auroc_c1_c4']:.4f}")
    print(f"  artifact_drop C1->C4   = {result['artifact_drop_c1_c4']:+.4f}")
    print(f"  target - tolerance     = {result['survives_threshold']:.4f}")
    print(f"  survives_falsification = {result['survives_falsification']}")
    print(f"  artifact_detected      = {result['artifact_detected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
