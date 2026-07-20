"""
EICV End-to-End Benchmark
==========================

Runs the full EICV pipeline (all 6 layers integrated into Φ) through the
C1-C4 falsification protocol on two datasets:

  A: SQuAD v2 cross-passage
  B: HaluEval-QA entity-swap

For each condition, reports:
  - AUROC of Φ as a hallucination score (using 1 - Φ since Φ = grounded)
  - decision distribution at α = 0.05 (supported / abstain / hallucinated)
  - mean Φ separation between right/halu pairs
  - latency per verification
  - worst-case bound: min AUROC across C1-C4

This is the FINAL test: EICV must survive falsification AND produce
auditable certificates.
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

from entroly.eicv import EICVAnalyzer  # noqa: E402

SEED = 42
N_ITEMS = 200
RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _auroc(scores, labels):
    from entroly.metrics import tie_corrected_auroc
    return tie_corrected_auroc(scores, labels)


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
    for i, (ctx, right) in enumerate(pool[:N_ITEMS]):
        j = (i + rng.randint(1, n - 1)) % n
        items.append((ctx, right, pool[j][1]))
    # Calibration set: separate grounded pairs
    cal_pairs = pool[N_ITEMS:N_ITEMS + 200]
    return items, cal_pairs


def load_halueval():
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
    cal_pairs = [(ev, r) for ev, r, _ in items[N_ITEMS:N_ITEMS + 200]]
    return items[:N_ITEMS], cal_pairs


def _entity_shuffle(text, rng):
    import re
    ents = list({m.group() for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", text)})
    if len(ents) < 2:
        return text
    mapping = dict(zip(ents, rng.sample(ents, len(ents))))
    out = text
    for k, v in mapping.items():
        if k != v:
            out = out.replace(k, v)
    return out


def _paraphrase(text):
    swaps = {"is": "was", "are": "were", "has": "had", "have": "had",
             "shows": "demonstrates", "found": "discovered"}
    return " ".join(swaps.get(w.lower(), w) for w in text.split())


def run_condition(name, items, ana, rng):
    """Score one C-condition. Returns (auroc, decisions, latencies, phi_gap)."""
    scores = []
    labels = []
    decisions = {"supported": 0, "abstain": 0, "hallucinated": 0}
    latencies = []
    phi_right_sum = 0.0
    phi_halu_sum = 0.0

    for ctx, right, halu in items:
        if name == "C1":
            r, h = right, halu
        elif name == "C2":
            r = _entity_shuffle(right, rng)
            h = _entity_shuffle(halu, rng)
        elif name == "C3":
            r = _paraphrase(right)
            h = halu
        elif name == "C4":
            r = right
            h = _entity_shuffle(halu, rng)
        else:
            raise ValueError(name)

        c_r = ana.verify(ctx, r)
        c_h = ana.verify(ctx, h)
        scores += [1.0 - c_r.phi, 1.0 - c_h.phi]
        labels += [0, 1]
        decisions[c_r.decision] = decisions.get(c_r.decision, 0) + 1
        decisions[c_h.decision] = decisions.get(c_h.decision, 0) + 1
        latencies += [c_r.elapsed_ms, c_h.elapsed_ms]
        phi_right_sum += c_r.phi
        phi_halu_sum  += c_h.phi

    n = len(items)
    return {
        "auroc": _auroc(scores, labels),
        "decisions": decisions,
        "mean_latency_ms": sum(latencies) / max(len(latencies), 1),
        "phi_right_mean": phi_right_sum / n,
        "phi_halu_mean":  phi_halu_sum / n,
        "phi_gap":        (phi_right_sum - phi_halu_sum) / n,
    }


def run_dataset(name, items, cal_pairs):
    print(f"\n  Dataset: {name}  ({len(items)} items, {len(cal_pairs)} calibration pairs)")
    ana = EICVAnalyzer()

    print("    Fitting calibrators...", flush=True)
    t0 = time.perf_counter()
    ana.fit_calibrators(cal_pairs)
    print(f"    Calibrated in {time.perf_counter()-t0:.1f}s (n={ana.calibration.n_calibration})")

    rng = random.Random(SEED)
    results = {}
    for c in ("C1", "C2", "C3", "C4"):
        print(f"    Scoring {c}...", flush=True)
        t1 = time.perf_counter()
        res = run_condition(c, items, ana, rng)
        elapsed = time.perf_counter() - t1
        print(f"      AUROC={res['auroc']:.4f}  Phi_gap={res['phi_gap']:+.4f}  "
              f"latency={res['mean_latency_ms']:.2f}ms  elapsed={elapsed:.1f}s")
        print(f"      decisions: {res['decisions']}")
        results[c] = res

    # Aggregate
    aurocs = [results[c]["auroc"] for c in ("C1", "C2", "C3", "C4")]
    min_auc = min(aurocs)
    max_auc = max(aurocs)
    print(f"\n    SUMMARY for {name}:")
    print(f"      min(C1-C4) AUROC = {min_auc:.4f}")
    print(f"      max(C1-C4) AUROC = {max_auc:.4f}")
    print(f"      C1->C4 drop      = {aurocs[0] - aurocs[3]:+.4f}")

    return {
        "dataset": name,
        "n_items": len(items),
        "conditions": {c: {
            "auroc": round(results[c]["auroc"], 4),
            "phi_gap": round(results[c]["phi_gap"], 4),
            "phi_right_mean": round(results[c]["phi_right_mean"], 4),
            "phi_halu_mean":  round(results[c]["phi_halu_mean"], 4),
            "mean_latency_ms": round(results[c]["mean_latency_ms"], 2),
            "decisions": results[c]["decisions"],
        } for c in ("C1", "C2", "C3", "C4")},
        "min_auroc_c1_c4": round(min_auc, 4),
        "max_auroc_c1_c4": round(max_auc, 4),
        "c1_c4_drop":      round(aurocs[0] - aurocs[3], 4),
        "artifact_detected": (aurocs[0] - aurocs[3]) > 0.05,
        "survives_falsification": min_auc >= 0.74,    # 0.77 - 0.03 tolerance
    }


def main() -> int:
    print("=" * 74)
    print("  EICV End-to-End Benchmark — Phase 4 Integration")
    print("=" * 74)

    overall = {}

    print("\n  Loading SQuAD v2...")
    squad_items, squad_cal = load_squad()
    overall["squad_v2"] = run_dataset("squad_v2", squad_items, squad_cal)

    print("\n  Loading HaluEval-QA...")
    halu_items, halu_cal = load_halueval()
    overall["halueval_qa"] = run_dataset("halueval_qa", halu_items, halu_cal)

    out = {
        "schema": "eicv-endtoend-v1",
        "preregistration_doc": "benchmarks/EICV_PREREGISTRATION.md",
        "n_items": N_ITEMS,
        "results": overall,
        "all_survive": all(r["survives_falsification"] for r in overall.values()),
    }
    out_path = RESULTS_DIR / "eicv_endtoend.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    print(f"\n  Overall: {'PASSES' if out['all_survive'] else 'FAILS'} EICV end-to-end falsification")
    return 0 if out["all_survive"] else 1


if __name__ == "__main__":
    sys.exit(main())
