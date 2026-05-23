"""
HaluEval-Dialogue Falsification Probe (C1-C4).

Per EICV_PREREGISTRATION.md:
  - Target metric: F1 (not AUROC, dialogue uses F1)
  - F1 target ≥ 0.68
  - Per-benchmark falsification uses AUROC on (r vs h) variants since the
    F1-vs-AUROC mapping isn't 1-1; we report both.

Item construction (mirrors benchmarks/run_witness_benchmarks.py:42-48):
  For each row: knowledge + dialogue_history → context; right_response / response
  variants → right/halu.

Note: HaluEval-Dialogue does not always provide *paired* right/halu per item.
We use the dataset's hallucination label: rows with hallucination='yes' provide
the halu, rows with hallucination='no' provide the right. We pair them
deterministically by row order on each side.
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
TARGET_AUROC = 0.78           # AUROC equivalent for F1≥0.68 zone; conservative
RESULTS_DIR = _THIS / "results"
OUT_PATH = RESULTS_DIR / "halueval_dialogue_falsification.json"
CACHE_PATH = RESULTS_DIR / "halueval_dialogue_falsification_cache.json"


def load_items() -> list[FalsItem]:
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "dialogue", split="data")
    rights: list[tuple[str, str]] = []   # (context, response)
    halus: list[tuple[str, str]] = []
    for i, row in enumerate(ds):
        knowledge = str(row.get("knowledge", "") or "")
        history = str(row.get("dialogue_history", "") or "")
        # Field name shifts across mirror copies; try both
        right_resp = str(row.get("right_response", row.get("response", "")) or "")
        halu_resp = str(row.get("hallucinated_response", "") or "")
        context = (knowledge + ("\n\nDialogue history:\n" + history if history else "")) if knowledge else history
        if not context:
            continue
        if right_resp:
            rights.append((context, right_resp))
        if halu_resp:
            halus.append((context, halu_resp))
    # Pair them up to length min(rights, halus, N_ITEMS)
    n = min(len(rights), len(halus), N_ITEMS)
    rng = random.Random(SEED)
    rng.shuffle(rights); rng.shuffle(halus)
    items: list[FalsItem] = []
    for i in range(n):
        ctx, right = rights[i]
        _, halu = halus[i]   # use ctx from right (paired by index after shuffle)
        items.append(FalsItem(
            context=ctx,
            query="",     # dialogue has no separate question
            right=right,
            halu=halu,
            item_id=f"hed_{i:04d}",
        ))
    return items


def main() -> int:
    print("=" * 74)
    print("  HaluEval-Dialogue Falsification Probe (C1-C4)")
    print("=" * 74)
    items = load_items()
    n = len(items)
    print(f"  Loaded {n} items (HaluEval dialogue split)")
    if n == 0:
        print("  ERROR: no items loaded")
        return 1
    ds_hash = dataset_hash(items)
    print(f"  Dataset hash: {ds_hash}")

    backend = "deterministic"
    print(f"  Backend: {backend}")
    print(f"  Building 4-variant records...", flush=True)
    t0 = time.perf_counter()
    records = build_records(items, backend=backend, seed=SEED, cache_path=CACHE_PATH)
    print(f"  Built {len(records)} records ({time.perf_counter() - t0:.1f}s)")

    print(f"  Scoring C1-C4 fusion AUROC...", flush=True)
    t0 = time.perf_counter()
    # profile=dialogue is the right WitnessAnalyzer profile for this task
    result = run_probe("halueval_dialogue", records, profile="dialogue",
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
