"""
Phi-weight search via group-DRO.

Searches Phi-integration weights to maximise the WORST-dataset AUROC
across {SQuAD, HaluEval, FEVER}. This is the cross-domain robustness
question: which weighting gives a usable Phi on ALL three benchmarks?
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

import itertools
import math

from entroly.esg import ESGAnalyzer
from entroly.semantic_entropy import SemanticEntropyAnalyzer

SEED = 42
N_PER = 150
RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n0 = sum(1 for _, y in pairs if y == 0)
    n1 = sum(1 for _, y in pairs if y == 1)
    if n0 == 0 or n1 == 0:
        return 0.5
    rs = sum(r for r, (_, y) in enumerate(pairs, 1) if y == 1)
    return (rs - n1 * (n1 + 1) / 2) / (n0 * n1)


def load_squad():
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
    n = len(pool)
    for i, (ctx, right) in enumerate(pool[:N_PER]):
        j = (i + rng.randint(1, n - 1)) % n
        items += [(ctx, right, 0), (ctx, pool[j][1], 1)]
    return items


def load_halueval():
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    rng = random.Random(SEED)
    rows = []
    for row in ds:
        k = str(row.get("knowledge", "") or "")
        r = str(row.get("right_answer", "") or "")
        h = str(row.get("hallucinated_answer", "") or "")
        if k and r and h:
            rows.append((k, r, h))
    rng.shuffle(rows)
    items = []
    for k, r, h in rows[:N_PER]:
        items += [(k, r, 0), (k, h, 1)]
    return items


def load_fever():
    from datasets import load_dataset
    ds = load_dataset("copenlu/fever_gold_evidence", split="validation")
    label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": None,
                 "NOT_ENOUGH_INFO": None}
    rng = random.Random(SEED)
    items = []
    for row in ds:
        claim = str(row.get("claim", "") or "")
        ev_field = row.get("evidence") or ""
        if isinstance(ev_field, list):
            parts = []
            for x in ev_field:
                if isinstance(x, dict):
                    parts.append(str(x.get("sentence", x.get("text", ""))))
                else:
                    parts.append(str(x))
            ev = "\n".join(p for p in parts if p)
        else:
            ev = str(ev_field)
        label = label_map.get(str(row.get("label", "") or "").upper().strip())
        if claim and ev and label is not None:
            items.append((ev, claim, label))
    rng.shuffle(items)
    return items[:2 * N_PER]


def main() -> int:
    print("=" * 74)
    print("  Phi-weight DRO search across {SQuAD, HaluEval, FEVER}")
    print("=" * 74)

    # Load all three datasets
    print("\n  Loading SQuAD...")
    squad = load_squad()
    print("  Loading HaluEval-QA...")
    halu  = load_halueval()
    print("  Loading FEVER...")
    fever = load_fever()
    print(f"\n  N: SQuAD={len(squad)}  HaluEval={len(halu)}  FEVER={len(fever)}")

    # Precompute per-item per-signal scores
    esg = ESGAnalyzer()
    sem = SemanticEntropyAnalyzer()

    def _score(items, name):
        print(f"\n  Scoring {name}...", flush=True)
        t0 = time.perf_counter()
        out = []
        for ev, cl, lab in items:
            er = esg.score(ev, cl)
            sr = sem.analyze(ev, cl)
            out.append({
                "esg":      er.tension,
                "nli":      sr.nli_bidir_score,
                "hsem":     sr.h_sem_norm,
                "label":    lab,
            })
        print(f"    done in {time.perf_counter()-t0:.1f}s")
        return out

    s_squad = _score(squad,  "SQuAD")
    s_halu  = _score(halu,   "HaluEval-QA")
    s_fever = _score(fever,  "FEVER")

    def auroc_at(weights, data):
        w_esg, w_nli, w_hsem = weights
        scores = []
        labels = []
        for r in data:
            phi_neg = w_esg * r["esg"] + w_nli * r["nli"] + w_hsem * r["hsem"]
            scores.append(phi_neg)
            labels.append(r["label"])
        return _auroc(scores, labels)

    # Grid of weight triples that sum to 1.0 (within 1e-9)
    grid_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    combos = []
    for a, b in itertools.product(grid_vals, repeat=2):
        c = 1.0 - a - b
        if -1e-9 <= c <= 1.0 + 1e-9:
            combos.append((round(a, 1), round(b, 1), round(max(0.0, c), 1)))

    print(f"\n  Evaluating {len(combos)} weight combinations...")
    t0 = time.perf_counter()
    best_dro = (-1.0, None)   # worst-case best
    best_avg = (-1.0, None)   # average best
    results = []

    for w in combos:
        au_squad = auroc_at(w, s_squad)
        au_halu  = auroc_at(w, s_halu)
        au_fever = auroc_at(w, s_fever)
        worst = min(au_squad, au_halu, au_fever)
        avg = (au_squad + au_halu + au_fever) / 3
        results.append({
            "weights": {"esg": w[0], "nli": w[1], "hsem": w[2]},
            "squad": round(au_squad, 4),
            "halueval": round(au_halu, 4),
            "fever": round(au_fever, 4),
            "worst": round(worst, 4),
            "mean": round(avg, 4),
        })
        if worst > best_dro[0]:
            best_dro = (worst, w, au_squad, au_halu, au_fever)
        if avg > best_avg[0]:
            best_avg = (avg, w, au_squad, au_halu, au_fever)

    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    print()
    print("  TOP 5 by worst-case (DRO):")
    by_worst = sorted(results, key=lambda r: -r["worst"])
    for r in by_worst[:5]:
        print(f"    w={r['weights']}  squad={r['squad']:.4f}  "
              f"halu={r['halueval']:.4f}  fever={r['fever']:.4f}  "
              f"worst={r['worst']:.4f}  mean={r['mean']:.4f}")

    print()
    print("  TOP 5 by mean (ERM):")
    by_mean = sorted(results, key=lambda r: -r["mean"])
    for r in by_mean[:5]:
        print(f"    w={r['weights']}  squad={r['squad']:.4f}  "
              f"halu={r['halueval']:.4f}  fever={r['fever']:.4f}  "
              f"worst={r['worst']:.4f}  mean={r['mean']:.4f}")

    print()
    print(f"  DRO winner: weights = {best_dro[1]}")
    print(f"    SQuAD={best_dro[2]:.4f}  HaluEval={best_dro[3]:.4f}  FEVER={best_dro[4]:.4f}")
    print(f"  ERM winner: weights = {best_avg[1]}")
    print(f"    SQuAD={best_avg[2]:.4f}  HaluEval={best_avg[3]:.4f}  FEVER={best_avg[4]:.4f}")

    out = {
        "schema": "phi-weight-search-v1",
        "n_per_dataset": N_PER,
        "grid_size": len(combos),
        "best_dro": {
            "weights": dict(zip(("esg", "nli", "hsem"), best_dro[1])),
            "squad_auroc": round(best_dro[2], 4),
            "halueval_auroc": round(best_dro[3], 4),
            "fever_auroc": round(best_dro[4], 4),
            "worst_auroc": round(best_dro[0], 4),
        },
        "best_erm": {
            "weights": dict(zip(("esg", "nli", "hsem"), best_avg[1])),
            "squad_auroc": round(best_avg[2], 4),
            "halueval_auroc": round(best_avg[3], 4),
            "fever_auroc": round(best_avg[4], 4),
            "mean_auroc": round(best_avg[0], 4),
        },
        "all_results": results,
    }
    (RESULTS_DIR / "phi_weight_search.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {RESULTS_DIR / 'phi_weight_search.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
