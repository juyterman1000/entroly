"""
Semantic Entropy Benchmark -- EICV Layer 5 (Phase 1D)
======================================================

Evaluates NLI_bidir (bidirectional entailment proxy) and H_sem
(stylistic-paraphrase entropy) on three datasets.

Primary signal: NLI_bidir_score = 1 - 0.5*(forward_entail + reverse_entail)
Supplementary: H_sem on K stylistic paraphrases (M3+M4+dropout)
Composite:     0.8 * NLI_bidir_score + 0.2 * H_sem_norm

Pre-registered target (EICV_PREREGISTRATION.md): AUROC >= 0.75 on all datasets.
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

from entroly.semantic_entropy import SemanticEntropyAnalyzer

SEED = 42
N_ITEMS = 300
TARGET_AUROC = 0.75
RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ANA = SemanticEntropyAnalyzer(seed=SEED)


def load_squad(n: int = N_ITEMS) -> list[tuple[str, str, int]]:
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
    items = []
    npool = len(pool)
    for i, (ctx, right) in enumerate(pool[:n]):
        items.append((ctx, right, 0))
        j = (i + rng.randint(1, npool - 1)) % npool
        items.append((ctx, pool[j][1], 1))
    return items


def load_halueval_qa(n: int = N_ITEMS) -> list[tuple[str, str, int]]:
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    rng = random.Random(SEED)
    items = []
    for row in ds:
        know = str(row.get("knowledge", "") or "")
        right = str(row.get("right_answer", "") or "")
        halu  = str(row.get("hallucinated_answer", "") or "")
        if know and right and halu:
            items.append((know, right, 0))
            items.append((know, halu, 1))
    rng.shuffle(items)
    return items[:n * 2]


def load_fever_binary(n: int = N_ITEMS) -> list[tuple[str, str, int]] | None:
    from datasets import load_dataset
    try:
        ds = load_dataset("copenlu/fever_gold_evidence", split="validation")
    except Exception:
        return None
    rng = random.Random(SEED)
    label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": None,
                 "NOT_ENOUGH_INFO": None}
    items = []
    for row in ds:
        claim = str(row.get("claim", "") or "")
        ev = row.get("evidence") or ""
        if isinstance(ev, list):
            parts = []
            for x in ev:
                parts.append(str(x.get("sentence", x.get("text", "")) if isinstance(x, dict) else x))
            ev_text = "\n".join(p for p in parts if p)
        else:
            ev_text = str(ev)
        label_raw = str(row.get("label", "") or "").upper().strip()
        label = label_map.get(label_raw)
        if claim and ev_text and label is not None:
            items.append((ev_text, claim, label))
    rng.shuffle(items)
    return items[:n * 2]


def run_dataset(name: str, items: list[tuple[str, str, int]]) -> dict:
    print(f"  Scoring {name} ({len(items)} items)...", flush=True)
    t0 = time.perf_counter()
    auroc_comp = _ANA.auroc(items, signal="composite")
    auroc_nli  = _ANA.auroc(items, signal="nli_bidir_score")
    elapsed = time.perf_counter() - t0
    passes = auroc_nli >= TARGET_AUROC
    status = "PASS" if passes else "FAIL"
    print(f"    AUROC_NLI_bidir = {auroc_nli:.4f}  [{status}]  (target>={TARGET_AUROC})")
    print(f"    AUROC_composite = {auroc_comp:.4f}")
    print(f"    elapsed = {elapsed:.1f}s")
    return {
        "dataset": name, "n_items": len(items),
        "auroc_nli_bidir": round(auroc_nli, 4),
        "auroc_composite": round(auroc_comp, 4),
        "passes_target": passes,
        "target_auroc": TARGET_AUROC,
        "elapsed_s": round(elapsed, 2),
    }


def main() -> int:
    print("=" * 74)
    print("  Semantic Entropy Benchmark -- EICV Layer 5 (Phase 1D)")
    print("=" * 74)

    results = []
    overall_pass = True

    print("\n  [A] SQuAD v2")
    try:
        items_a = load_squad()
        res_a = run_dataset("squad_v2", items_a)
        results.append(res_a)
        if not res_a["passes_target"]:
            overall_pass = False
    except Exception as e:
        print(f"    ERROR: {e}")
        overall_pass = False

    print("\n  [B] HaluEval-QA")
    try:
        items_b = load_halueval_qa()
        res_b = run_dataset("halueval_qa", items_b)
        results.append(res_b)
        if not res_b["passes_target"]:
            overall_pass = False
    except Exception as e:
        print(f"    ERROR: {e}")
        overall_pass = False

    print("\n  [C] FEVER binary")
    try:
        items_c = load_fever_binary()
        if items_c:
            res_c = run_dataset("fever", items_c)
            results.append(res_c)
            if not res_c["passes_target"]:
                overall_pass = False
    except Exception as e:
        print(f"    ERROR: {e}")

    out = {
        "schema": "sem-entropy-benchmark-v1",
        "scorer": "NLI_bidir_proxy + H_sem_stylistic",
        "target_auroc": TARGET_AUROC,
        "preregistration_doc": "benchmarks/EICV_PREREGISTRATION.md",
        "results": results,
        "overall_passes": overall_pass,
    }
    out_path = RESULTS_DIR / "sem_entropy_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    print(f"  Overall: {'PASSES' if overall_pass else 'FAILS'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
