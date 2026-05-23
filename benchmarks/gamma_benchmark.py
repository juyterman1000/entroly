"""
Falsification Gradient Gamma Benchmark -- EICV Layer 4 (Phase 1C)
===================================================================

Computes Gamma(x) = mean_{m in M} |T(G)(x) - T(G)(x + delta_m)|
over the 5 perturbation manifolds on two datasets.

Interpretation:
  - M1 entity / M2 numeric / M5 retrieval: SEMANTIC manifolds.
    High Gamma here is expected (detector is sensitive to meaningful changes).
  - M3 compose / M4 temporal: STYLISTIC manifolds.
    Low Gamma here is REQUIRED (detector should not fire on style).

Pre-registered target (EICV_PREREGISTRATION.md section 3.4):
  - Stylistic manifolds (M3, M4) mean_delta < 0.10
    (detector is robust to clause reordering and tense changes)
  - Semantic manifolds (M1, M2, M5) mean_delta > 0.15
    (detector is sensitive to factual changes -- too low would be under-fitting)
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

from entroly.counterfactual import compute_gamma
from entroly.esg import compute_tension

SEED = 42
N_ITEMS = 300
RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Targets per manifold type
STYLISTIC_TARGET_MAX = 0.10    # M3 compose, M4 temporal
SEMANTIC_TARGET_MIN  = 0.15   # M1 entity, M2 numeric, M5 retrieval


def load_squad_grounded(n: int = N_ITEMS) -> list[tuple[str, str]]:
    """Grounded items only (label=0) from SQuAD v2 validation."""
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(SEED)
    items = []
    for row in ds:
        ctx = str(row.get("context", "") or "")
        ans = row.get("answers", {})
        texts = ans.get("text", []) if isinstance(ans, dict) else []
        if ctx and texts and len(texts[0]) > 5:
            items.append((ctx, texts[0]))
    rng.shuffle(items)
    return items[:n]


def load_halueval_all(n: int = N_ITEMS) -> list[tuple[str, str]]:
    """All items (right + hallucinated) from HaluEval-QA."""
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    rng = random.Random(SEED)
    items = []
    for row in ds:
        knowledge = str(row.get("knowledge", "") or "")
        right = str(row.get("right_answer", "") or "")
        halu  = str(row.get("hallucinated_answer", "") or "")
        if knowledge and right and halu:
            items.append((knowledge, right))
            items.append((knowledge, halu))
    rng.shuffle(items)
    return items[:n]


def _check_manifold(
    manifold_name: str,
    mean_delta: float,
    kind: str,
) -> tuple[bool, str]:
    """Returns (passes, description)."""
    if kind == "stylistic":
        passes = mean_delta <= STYLISTIC_TARGET_MAX
        target_str = f"<= {STYLISTIC_TARGET_MAX}"
    else:  # semantic
        passes = mean_delta >= SEMANTIC_TARGET_MIN
        target_str = f">= {SEMANTIC_TARGET_MIN}"
    status = "PASS" if passes else "FAIL"
    return passes, f"{manifold_name}: delta={mean_delta:.4f} {status} (target {target_str})"


def run_dataset(name: str, items: list[tuple[str, str]]) -> dict:
    print(f"  Computing Gamma on {name} ({len(items)} items)...", flush=True)
    t0 = time.perf_counter()
    result = compute_gamma(items, scorer=compute_tension, seed=SEED)
    elapsed = time.perf_counter() - t0

    STYLISTIC  = {"compose", "temporal"}
    SEMANTIC   = {"entity", "numeric", "retrieval"}

    manifold_verdicts = {}
    all_pass = True
    print(f"    Gamma (aggregate) = {result.gamma:.4f}")
    for m in result.manifolds:
        kind = "stylistic" if m.manifold in STYLISTIC else "semantic"
        passes, desc = _check_manifold(m.manifold, m.mean_delta, kind)
        if not passes:
            all_pass = False
        symbol = "[OK]" if passes else "[!!]"
        print(f"    {symbol} {desc}  (n_perturb={m.n_perturbable})")
        manifold_verdicts[m.manifold] = {
            "mean_delta": m.mean_delta,
            "fraction_changed": m.fraction_changed,
            "n_perturbable": m.n_perturbable,
            "kind": kind,
            "passes": passes,
        }

    print(f"    elapsed = {elapsed:.1f}s")
    return {
        "dataset": name,
        "n_items": len(items),
        "gamma": result.gamma,
        "manifolds": manifold_verdicts,
        "all_manifolds_pass": all_pass,
        "elapsed_s": round(elapsed, 2),
    }


def main() -> int:
    print("=" * 74)
    print("  Gamma Benchmark -- EICV Layer 4 (Phase 1C)")
    print("=" * 74)

    results = []
    overall_pass = True

    # A: SQuAD grounded (to check false-positive brittleness)
    print("\n  [A] SQuAD v2 -- GROUNDED items only")
    try:
        items_a = load_squad_grounded()
        res_a = run_dataset("squad_grounded", items_a)
        results.append(res_a)
        if not res_a["all_manifolds_pass"]:
            overall_pass = False
    except Exception as e:
        print(f"    ERROR: {e}")
        overall_pass = False

    # B: HaluEval-QA (mixed grounded + hallucinated)
    print("\n  [B] HaluEval-QA -- mixed items")
    try:
        items_b = load_halueval_all()
        res_b = run_dataset("halueval_mixed", items_b)
        results.append(res_b)
        if not res_b["all_manifolds_pass"]:
            overall_pass = False
    except Exception as e:
        print(f"    ERROR: {e}")
        overall_pass = False

    # Save
    out = {
        "schema": "gamma-benchmark-v1",
        "scorer": "ESG_T(G)",
        "stylistic_target_max": STYLISTIC_TARGET_MAX,
        "semantic_target_min": SEMANTIC_TARGET_MIN,
        "preregistration_doc": "benchmarks/EICV_PREREGISTRATION.md",
        "results": results,
        "overall_passes": overall_pass,
    }
    out_path = RESULTS_DIR / "gamma_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    print(f"  Overall: {'PASSES' if overall_pass else 'NEEDS IMPROVEMENT'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
