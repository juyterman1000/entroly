"""
RNR* Benchmark — EICV Layer 3 (Phase 1B)
==========================================

Computes I(Ŷ; S) — the Retrieval Necessity Ratio — for ESG T(G) on
three datasets spanning different hallucination patterns:

  A: SQuAD v2        — cross-passage wrong answers (strong retrieval signal)
  B: HaluEval-QA     — entity-swap within-domain (moderate retrieval signal)
  C: FEVER binary    — Wikipedia claims (SUPPORTS vs REFUTES)

High RNR* indicates the detector is genuinely retrieval-grounded.
Low RNR* (< 0.05 nats) indicates the detector fires on claim-internal
cues regardless of evidence — a known artifact failure mode.

Pre-registered target (EICV_PREREGISTRATION.md §3.3):
  I(Ŷ; S) > 0.10 nats for all datasets
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

from entroly.esg import compute_tension  # noqa: E402
from entroly.rnr import compute_rnr  # noqa: E402

SEED = 42
N_ITEMS = 400
TARGET_MI = 0.10   # nats
RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_squad() -> list[tuple[str, str, int]]:
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
    items: list[tuple[str, str, int]] = []
    for i, (ctx, right) in enumerate(pool[:N_ITEMS]):
        items.append((ctx, right, 0))                          # SUPPORTS
        j = (i + rng.randint(1, n - 1)) % n
        items.append((ctx, pool[j][1], 1))                     # REFUTES (cross-passage)
    return items


def load_halueval_qa() -> list[tuple[str, str, int]]:
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    rng = random.Random(SEED)
    items: list[tuple[str, str, int]] = []
    for row in ds:
        knowledge = str(row.get("knowledge", "") or "")
        right = str(row.get("right_answer", "") or "")
        halu  = str(row.get("hallucinated_answer", "") or "")
        if knowledge and right and halu:
            items.append((knowledge, right, 0))
            items.append((knowledge, halu, 1))
    rng.shuffle(items)
    return items[:N_ITEMS * 2]


def load_fever_binary() -> list[tuple[str, str, int]] | None:
    from datasets import load_dataset
    try:
        ds = load_dataset("copenlu/fever_gold_evidence", split="validation")
    except Exception:
        return None
    rng = random.Random(SEED)
    label_map = {"SUPPORTS": 0, "REFUTES": 1,
                 "NOT ENOUGH INFO": None, "NOT_ENOUGH_INFO": None}
    items: list[tuple[str, str, int]] = []
    for row in ds:
        claim = str(row.get("claim", "") or "")
        ev = row.get("evidence") or row.get("evidence_text") or ""
        if isinstance(ev, list):
            parts = []
            for x in ev:
                if isinstance(x, dict):
                    parts.append(str(x.get("sentence", x.get("text", ""))))
                else:
                    parts.append(str(x))
            ev_text = "\n".join(p for p in parts if p)
        else:
            ev_text = str(ev)
        label_raw = str(row.get("label", "") or "").upper().strip()
        label = label_map.get(label_raw)
        if claim and ev_text and label is not None:
            items.append((ev_text, claim, label))
    rng.shuffle(items)
    return items[:N_ITEMS * 2]


def run_dataset(
    name: str,
    items: list[tuple[str, str, int]],
) -> dict:
    print(f"  Computing RNR* on {name} ({len(items)} items)...", flush=True)
    t0 = time.perf_counter()
    result = compute_rnr(items, scorer=compute_tension, n_shuffles=3, seed=SEED)
    elapsed = time.perf_counter() - t0
    passes = result.mutual_information >= TARGET_MI
    print(f"    I(Y;S)        = {result.mutual_information:.4f} nats  "
          f"({'PASS' if passes else 'FAIL'}, target>{TARGET_MI})")
    print(f"    AUROC_real    = {result.auroc_real:.4f}")
    print(f"    AUROC_null    = {result.auroc_null:.4f}")
    print(f"    AUROC_gap     = {result.auroc_gap:.4f}")
    print(f"    Necessity     = {result.necessity_fraction:.4f}")
    print(f"    elapsed       = {elapsed:.1f}s")
    return {
        "dataset": name,
        "n_items": len(items),
        "passes_target": passes,
        "target_mi": TARGET_MI,
        **result.as_dict(),
        "elapsed_s": round(elapsed, 2),
    }


def main() -> int:
    print("=" * 74)
    print("  RNR* Benchmark -- EICV Layer 3 (Phase 1B)")
    print("=" * 74)

    results = []
    all_pass = True

    # A: SQuAD v2
    print("\n  [A] SQuAD v2")
    try:
        items_a = load_squad()
        res_a = run_dataset("squad_v2", items_a)
        results.append(res_a)
        if not res_a["passes_target"]:
            all_pass = False
    except Exception as e:
        print(f"    ERROR: {e}")
        all_pass = False

    # B: HaluEval-QA
    print("\n  [B] HaluEval-QA")
    try:
        items_b = load_halueval_qa()
        res_b = run_dataset("halueval_qa", items_b)
        results.append(res_b)
        if not res_b["passes_target"]:
            all_pass = False
    except Exception as e:
        print(f"    ERROR: {e}")
        all_pass = False

    # C: FEVER binary
    print("\n  [C] FEVER binary (SUPPORTS vs REFUTES)")
    try:
        items_c = load_fever_binary()
        if items_c:
            res_c = run_dataset("fever", items_c)
            results.append(res_c)
            if not res_c["passes_target"]:
                all_pass = False
        else:
            print("    FEVER not available — skipping")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Save
    out = {
        "schema": "rnr-benchmark-v1",
        "target_mi_nats": TARGET_MI,
        "preregistration_doc": "benchmarks/EICV_PREREGISTRATION.md",
        "scorer": "ESG_T(G)",
        "results": results,
        "overall_passes": all_pass,
    }
    out_path = RESULTS_DIR / "rnr_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    print(f"  Overall: {'PASSES' if all_pass else 'FAILS'} RNR* target (>{TARGET_MI} nats)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
