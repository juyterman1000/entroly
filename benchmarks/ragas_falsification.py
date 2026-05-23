"""
RAGAS-style Falsification Probe (C1-C4) on SQuAD v2.

Per EICV_PREREGISTRATION.md:
  - Target: AUROC ≥ 0.85 (existing benchmark reports 0.90/0.91; conservative target)
  - Survives iff min(C1..C4) ≥ 0.82

Item construction (mirrors benchmarks/ragas_faithfulness_benchmark.py:102-115):
  For each SQuAD v2 row:
    context = passage
    query   = question
    right   = gold_answer
    halu    = a random answer from a *different* passage (cross-context hallucination)

This is the same hallucination construction as the existing RAGAS benchmark,
so the C1 number here is comparable to the existing benchmarks/results/
ragas_faithfulness_benchmark.json (AUROC 0.9002 / 0.9115).
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
TARGET_AUROC = 0.85
RESULTS_DIR = _THIS / "results"
OUT_PATH = RESULTS_DIR / "ragas_falsification.json"
CACHE_PATH = RESULTS_DIR / "ragas_falsification_cache.json"


def load_items() -> list[FalsItem]:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(SEED)
    pool = []
    for row in ds:
        ctx = str(row.get("context", "") or "")
        question = str(row.get("question", "") or "")
        answers = row.get("answers", {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        if ctx and question and texts and len(texts[0]) > 10:
            pool.append((ctx, question, texts[0]))
    rng.shuffle(pool)
    pool = pool[: max(N_ITEMS * 2, 2 * N_ITEMS)]  # keep enough for cross-pairing

    items: list[FalsItem] = []
    n = len(pool)
    for i, (ctx, q, gold) in enumerate(pool[:N_ITEMS]):
        # Halu = a different passage's gold answer
        j = (i + rng.randint(1, n - 1)) % n
        wrong = pool[j][2]
        items.append(FalsItem(
            context=ctx,
            query=q,
            right=gold,
            halu=wrong,
            item_id=f"ragas_{i:04d}",
        ))
    return items


def main() -> int:
    print("=" * 74)
    print("  RAGAS-style Falsification Probe (C1-C4) on SQuAD v2")
    print("=" * 74)
    items = load_items()
    n = len(items)
    print(f"  Loaded {n} items (SQuAD v2 validation)")
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
    result = run_probe("ragas", records, profile="rag",
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
