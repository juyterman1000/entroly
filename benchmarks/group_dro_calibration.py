"""
Group-DRO Calibration Run — EICV Phase 2
==========================================

Runs the group-DRO hyperparameter search on a held-out validation set,
selecting θ* that maximises worst-manifold AUROC.

Compares to ERM (average-best) baseline to quantify DRO's value.
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

from entroly.adversarial_calibration import group_dro_search

SEED = 42
N_ITEMS = 150
RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_squad_pairs() -> list[tuple[str, str, str]]:
    """Returns (context, right, hallucinated) triples from SQuAD v2."""
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(SEED)
    pool = []
    for row in ds:
        ctx = str(row.get("context", "") or "")
        ans = row.get("answers", {})
        texts = ans.get("text", []) if isinstance(ans, dict) else []
        if ctx and texts and len(texts[0]) > 5:
            pool.append((ctx, texts[0]))
    rng.shuffle(pool)
    n = len(pool)
    items = []
    for i, (ctx, right) in enumerate(pool[:N_ITEMS]):
        j = (i + rng.randint(1, n - 1)) % n
        items.append((ctx, right, pool[j][1]))
    return items


def load_halueval_pairs() -> list[tuple[str, str, str]]:
    """Returns (knowledge, right_answer, hallucinated_answer) from HaluEval-QA."""
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    rng = random.Random(SEED)
    items = []
    for row in ds:
        know = str(row.get("knowledge", "") or "")
        right = str(row.get("right_answer", "") or "")
        halu = str(row.get("hallucinated_answer", "") or "")
        if know and right and halu:
            items.append((know, right, halu))
    rng.shuffle(items)
    return items[:N_ITEMS]


def main() -> int:
    print("=" * 74)
    print("  Group-DRO Calibration -- EICV Phase 2")
    print("=" * 74)

    print(f"\n  Loading HaluEval-QA ({N_ITEMS} items) — known artifact dataset")
    items = load_halueval_pairs()
    print(f"  Loaded {len(items)} (context, right, halu) triples")
    print(f"  Running medium-grid DRO search (108 configs x 4 manifolds = {108*4} runs)...")

    t0 = time.perf_counter()
    result = group_dro_search(items, grid_size="medium", seed=SEED)
    elapsed = time.perf_counter() - t0

    print(f"  Search complete in {elapsed:.1f}s\n")

    print("  Group-DRO Best Config (worst-case optimised):")
    for k, v in result.best_config.as_dict().items():
        print(f"    {k:<22}= {v}")
    print(f"\n  Per-manifold AUROC under best_config:")
    for m, auc in result.best_manifold_aurocs.items():
        print(f"    {m}: {auc:.4f}")
    print(f"    worst-case = {result.worst_auroc_best:.4f}")
    print(f"    mean       = {result.mean_auroc_best:.4f}")

    print("\n  ERM Baseline (average-best config):")
    for k, v in result.erm_config.as_dict().items():
        print(f"    {k:<22}= {v}")
    print(f"\n  Per-manifold AUROC under erm_config:")
    for m, auc in result.erm_manifold_aurocs.items():
        print(f"    {m}: {auc:.4f}")
    print(f"    worst-case = {result.worst_auroc_erm:.4f}")
    print(f"    mean       = {result.mean_auroc_erm:.4f}")

    dro_gain = result.worst_auroc_best - result.worst_auroc_erm
    print(f"\n  Worst-case improvement (DRO over ERM): {dro_gain:+.4f}")

    out = {
        "schema": "dro-calibration-v1",
        "preregistration_doc": "benchmarks/EICV_PREREGISTRATION.md",
        "dataset": "halueval_qa data (N=150)",
        "elapsed_s": round(elapsed, 2),
        **result.as_dict(),
    }
    out_path = RESULTS_DIR / "dro_calibration.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
