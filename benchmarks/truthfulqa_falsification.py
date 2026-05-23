"""
TruthfulQA Falsification Probe
================================

Runs the C1-C4 falsification protocol (benchmarks/_falsification_common.py)
against TruthfulQA. Mirrors the protocol that fusion4_falsification.py applied
to HaluEval-QA, extended to this benchmark.

Per EICV_PREREGISTRATION.md:
  - Target: AUROC ≥ 0.78 (LLM-judge baselines)
  - Survives iff min(C1..C4) ≥ 0.75

Item construction (mirroring benchmarks/truthfulqa_benchmark.py):
  For each TruthfulQA question:
    context = best_answer
    right   = a correct_answer
    halu    = an incorrect_answer (sampled deterministically)

Output: benchmarks/results/truthfulqa_falsification.json
        benchmarks/results/truthfulqa_falsification_cache.json (variants)
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
N_ITEMS = 400          # match fusion4_falsification.py default
TARGET_AUROC = 0.78    # from EICV_PREREGISTRATION.md §2.3
RESULTS_DIR = _THIS / "results"
OUT_PATH = RESULTS_DIR / "truthfulqa_falsification.json"
CACHE_PATH = RESULTS_DIR / "truthfulqa_falsification_cache.json"


def load_items() -> list[FalsItem]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    rng = random.Random(SEED)
    items: list[FalsItem] = []
    for i, row in enumerate(ds):
        question = str(row.get("question", "")).strip()
        best = str(row.get("best_answer", "")).strip()
        correct_list = row.get("correct_answers") or []
        incorrect_list = row.get("incorrect_answers") or []
        if not (best and correct_list and incorrect_list and question):
            continue
        # Deterministic selection: hash-based pick to be reproducible
        ca = sorted(s for s in correct_list if isinstance(s, str) and s.strip())
        ia = sorted(s for s in incorrect_list if isinstance(s, str) and s.strip())
        if not ca or not ia:
            continue
        right = ca[rng.randrange(len(ca))]
        halu = ia[rng.randrange(len(ia))]
        items.append(FalsItem(
            context=best,
            query=question,
            right=right,
            halu=halu,
            item_id=f"tqa_{i:04d}",
        ))
    rng.shuffle(items)
    return items[:N_ITEMS]


def main() -> int:
    print("=" * 74)
    print("  TruthfulQA Falsification Probe (C1-C4)")
    print("=" * 74)
    items = load_items()
    n = len(items)
    print(f"  Loaded {n} items (TruthfulQA generation split)")
    if n == 0:
        print("  ERROR: no items loaded")
        return 1
    ds_hash = dataset_hash(items)
    print(f"  Dataset hash: {ds_hash}")

    # Deterministic backend by default — reproducible, no API needed.
    # Switch to "gpt-4o-mini" once API key is rotated and budget approved.
    backend = "deterministic"
    print(f"  Backend: {backend}")
    print(f"  Building 4-variant records...", flush=True)
    t0 = time.perf_counter()
    records = build_records(
        items, backend=backend, seed=SEED, cache_path=CACHE_PATH,
    )
    print(f"  Built {len(records)} records ({time.perf_counter() - t0:.1f}s)")

    print(f"  Scoring C1-C4 fusion AUROC (n={len(records)})...", flush=True)
    t0 = time.perf_counter()
    result = run_probe(
        "truthfulqa", records, profile="benchmark_qa",
        target=TARGET_AUROC, tolerance=0.03,
    )
    print(f"  Scored in {time.perf_counter() - t0:.1f}s")

    result["dataset_hash"] = ds_hash
    result["preregistered_target"] = TARGET_AUROC
    result["preregistration_doc"] = "benchmarks/EICV_PREREGISTRATION.md"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"  min(C1-C4)            = {result['min_fusion_auroc_c1_c4']:.4f}")
    print(f"  artifact_drop C1->C4  = {result['artifact_drop_c1_c4']:+.4f}")
    print(f"  target - tolerance    = {result['survives_threshold']:.4f}")
    print(f"  survives_falsification = {result['survives_falsification']}")
    print(f"  artifact_detected      = {result['artifact_detected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
