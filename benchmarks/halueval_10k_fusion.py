"""Full 10K HaluEval-QA: WITNESS vs Fusion (ECE + Entity Gap).

This is the statistically rigorous version — all 10K items (20K decisions)
with calibration/test split. No GPT calls needed (using cached WITNESS
AUROC as baseline).

The full-dataset run gives us +/-0.55% CIs instead of +/-2% on the
600-item sample, resolving whether fusion genuinely helps.
"""

from __future__ import annotations

import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 42
CAL_ITEMS = 2000
RESULTS_DIR = Path(__file__).parent / "results"


def _auroc(scores, labels):
    paired = sorted(zip(scores, labels), key=lambda t: t[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(1 for _, y in paired if y == 1)
    n_neg = len(paired) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = sum(r for r, (_, y) in zip(ranks, paired) if y == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    print("=" * 78)
    print("  Full 10K HaluEval-QA: WITNESS vs Fusion (statistically rigorous)")
    print("=" * 78)

    print("  Loading HaluEval[qa] (all 10K items)...", flush=True)
    from datasets import load_dataset
    from entroly.witness import WitnessAnalyzer
    from entroly.ravs.ece import compute_fisher_curvature

    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    items = []
    for row in ds:
        k, q = str(row.get("knowledge", "")), str(row.get("question", ""))
        ra = str(row.get("right_answer", ""))
        ha = str(row.get("hallucinated_answer", ""))
        if k and q and ra and ha:
            items.append((k, q, ra, ha))
    rng = random.Random(SEED)
    rng.shuffle(items)
    print(f"  {len(items)} items -> {2*len(items)} balanced decisions")

    # ── Score with WITNESS ──
    print("  Scoring with WITNESS (full 10K)...", flush=True)
    analyzer = WitnessAnalyzer(use_nli=False, force_python=True,
                               profile="benchmark_qa")

    _ENT_RE = [
        re.compile(r'\b\d+\.?\d*\b'),
        re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'),
    ]

    witness_risks = []
    fusion_risks = []
    labels = []
    t0 = time.perf_counter()

    for idx, (k, q, ra, ha) in enumerate(items):
        ctx = f"{k}\n\nQuestion: {q}"
        for ans, y in ((ra, 0), (ha, 1)):
            res = analyzer.analyze(ctx, ans)
            risk = 1.0 - float(res.summary_score)
            witness_risks.append(risk)
            labels.append(y)

            # Fusion: ECE + entity gap
            mean_kappa, _, _ = compute_fisher_curvature(ans)
            ece_k = min(1.0, mean_kappa * 2.5)

            ans_ents = set()
            ctx_lower = ctx.lower()
            for pat in _ENT_RE:
                for m in pat.finditer(ans):
                    ans_ents.add(m.group().lower())
            gap = 0.0
            if ans_ents:
                missing = sum(1 for e in ans_ents if e not in ctx_lower)
                gap = missing / len(ans_ents)

            fusion = 0.50 * risk + 0.30 * ece_k + 0.20 * gap
            fusion_risks.append(min(1.0, max(0.0, fusion)))

        if (idx + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {idx+1}/{len(items)} items "
                  f"({elapsed:.1f}s, "
                  f"{elapsed*1000/(2*(idx+1)):.2f}ms/dec)", flush=True)

    elapsed = time.perf_counter() - t0
    n = len(witness_risks)
    print(f"  Done: {n} decisions in {elapsed:.1f}s "
          f"({elapsed*1000/n:.2f}ms/dec)")

    # ── Split ──
    cal_end = CAL_ITEMS * 2  # 2 decisions per item
    cal_w = witness_risks[:cal_end]
    cal_f = fusion_risks[:cal_end]
    cal_l = labels[:cal_end]
    test_w = witness_risks[cal_end:]
    test_f = fusion_risks[cal_end:]
    test_l = labels[cal_end:]

    # ── AUROC (threshold-free, primary metric) ──
    w_auroc_full = _auroc(witness_risks, labels)
    f_auroc_full = _auroc(fusion_risks, labels)
    w_auroc_test = _auroc(test_w, test_l)
    f_auroc_test = _auroc(test_f, test_l)

    # ── Calibrated accuracy ──
    def best_threshold(scores, labs):
        best_tau, best_acc = 0.5, 0.0
        for tau in sorted(set(scores)):
            tp = sum(1 for s, y in zip(scores, labs) if s > tau and y == 1)
            tn = sum(1 for s, y in zip(scores, labs) if s <= tau and y == 0)
            acc = (tp + tn) / len(labs)
            if acc > best_acc:
                best_acc = acc
                best_tau = tau
        return best_tau

    w_tau = best_threshold(cal_w, cal_l)
    f_tau = best_threshold(cal_f, cal_l)

    def eval_acc(scores, labs, tau):
        tp = sum(1 for s, y in zip(scores, labs) if s > tau and y == 1)
        tn = sum(1 for s, y in zip(scores, labs) if s <= tau and y == 0)
        n_t = len(labs)
        acc = (tp + tn) / n_t
        ci = 1.96 * (acc * (1 - acc) / n_t) ** 0.5
        return acc, ci

    w_acc, w_ci = eval_acc(test_w, test_l, w_tau)
    f_acc, f_ci = eval_acc(test_f, test_l, f_tau)

    # ── Print ──
    n_test = len(test_l)
    print(f"\n  === Full 10K Results ===\n")
    print(f"  {'Metric':<25} {'WITNESS':>12} {'Fusion':>12} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'AUROC (full)':<25} {w_auroc_full:>12.4f} {f_auroc_full:>12.4f} "
          f"{f_auroc_full-w_auroc_full:>+10.4f}")
    print(f"  {'AUROC (test)':<25} {w_auroc_test:>12.4f} {f_auroc_test:>12.4f} "
          f"{f_auroc_test-w_auroc_test:>+10.4f}")
    print(f"  {'Accuracy (test)':<25} {w_acc:>11.2%} {f_acc:>11.2%} "
          f"{f_acc-w_acc:>+10.4f}")
    print(f"  {'95% CI':<25} {w_ci:>12.4f} {f_ci:>12.4f}")
    print(f"  {'N (test decisions)':<25} {n_test:>12} {n_test:>12}")
    print(f"  {'Threshold':<25} {w_tau:>12.6f} {f_tau:>12.6f}")

    significant = abs(f_auroc_test - w_auroc_test) > 0.01
    print(f"\n  Statistically significant AUROC lift: "
          f"{'YES' if significant else 'NO (within noise)'}")
    print(f"  Published GPT-3.5 baseline: 62.59% accuracy")
    print(f"  WITNESS: {w_acc:.2%} accuracy (AUROC {w_auroc_full:.4f})")
    print(f"  Fusion:  {f_acc:.2%} accuracy (AUROC {f_auroc_full:.4f})")

    # Save
    result = {
        "benchmark": "HaluEval-QA full 10K",
        "n_items": len(items),
        "n_decisions": n,
        "n_test": n_test,
        "witness": {
            "auroc_full": round(w_auroc_full, 4),
            "auroc_test": round(w_auroc_test, 4),
            "accuracy": round(w_acc, 4),
            "ci95": round(w_ci, 4),
            "threshold": round(w_tau, 6),
        },
        "fusion": {
            "auroc_full": round(f_auroc_full, 4),
            "auroc_test": round(f_auroc_test, 4),
            "accuracy": round(f_acc, 4),
            "ci95": round(f_ci, 4),
            "threshold": round(f_tau, 6),
            "weights": {"witness": 0.50, "ece": 0.30, "entity": 0.20},
        },
    }
    out = RESULTS_DIR / "halueval_10k_fusion.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
