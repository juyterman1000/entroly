"""4-Signal Fusion Benchmark: WITNESS + ECE + Entity + Spectral.

The hypothesis: adding spectral consistency (EigenScore-inspired SVD of
entity cross-similarity) provides an ORTHOGONAL fourth signal that
improves AUROC beyond the 3-signal fusion (0.8431 on full 10K).

Tested on full 10K HaluEval-QA with honest calibration/test split.
"""

from __future__ import annotations

import json
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
    print("  4-Signal Fusion: WITNESS + ECE + Entity + Spectral (Full 10K)")
    print("=" * 78)

    from datasets import load_dataset
    from entroly.witness import WitnessAnalyzer
    from entroly.ravs.ece import compute_fisher_curvature
    from entroly.ravs.spectral import compute_spectral_consistency

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
    print(f"  {len(items)} items -> {2*len(items)} decisions")

    analyzer = WitnessAnalyzer(use_nli=False, force_python=True,
                               profile="benchmark_qa")

    _ENT_RE = [
        re.compile(r'\b\d+\.?\d*\b'),
        re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'),
    ]

    witness_risks = []
    fusion3_risks = []  # 3-signal (baseline)
    fusion4_risks = []  # 4-signal (+ spectral)
    spectral_scores = []
    labels = []

    t0 = time.perf_counter()
    for idx, (k, q, ra, ha) in enumerate(items):
        ctx = f"{k}\n\nQuestion: {q}"
        for ans, y in ((ra, 0), (ha, 1)):
            # Signal 1: WITNESS
            res = analyzer.analyze(ctx, ans)
            risk = 1.0 - float(res.summary_score)
            witness_risks.append(risk)
            labels.append(y)

            # Signal 2: ECE
            mean_kappa, _, _ = compute_fisher_curvature(ans)
            ece_k = min(1.0, mean_kappa * 2.5)

            # Signal 3: Entity gap
            ans_ents = set()
            ctx_lower = ctx.lower()
            for pat in _ENT_RE:
                for m in pat.finditer(ans):
                    ans_ents.add(m.group().lower())
            gap = 0.0
            if ans_ents:
                missing = sum(1 for e in ans_ents if e not in ctx_lower)
                gap = missing / len(ans_ents)

            # Signal 4: Spectral consistency
            spec = compute_spectral_consistency(ctx, ans)
            spec_risk = 1.0 - spec.score  # higher risk = lower consistency
            spectral_scores.append(spec.score)

            # 3-signal fusion (baseline)
            f3 = 0.50 * risk + 0.30 * ece_k + 0.20 * gap
            fusion3_risks.append(min(1.0, max(0.0, f3)))

            # 4-signal fusion
            f4 = 0.40 * risk + 0.25 * ece_k + 0.15 * gap + 0.20 * spec_risk
            fusion4_risks.append(min(1.0, max(0.0, f4)))

        if (idx + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {idx+1}/{len(items)} "
                  f"({elapsed:.1f}s, {elapsed*1000/(2*(idx+1)):.2f}ms/dec)",
                  flush=True)

    elapsed = time.perf_counter() - t0
    n = len(labels)
    print(f"  Done: {n} decisions in {elapsed:.1f}s "
          f"({elapsed*1000/n:.2f}ms/dec)")

    # Split
    cal_end = CAL_ITEMS * 2
    test_w = witness_risks[cal_end:]
    test_f3 = fusion3_risks[cal_end:]
    test_f4 = fusion4_risks[cal_end:]
    test_sp = [1.0 - s for s in spectral_scores[cal_end:]]  # risk
    test_l = labels[cal_end:]

    # AUROC for each signal
    w_auroc = _auroc(test_w, test_l)
    f3_auroc = _auroc(test_f3, test_l)
    f4_auroc = _auroc(test_f4, test_l)
    sp_auroc = _auroc(test_sp, test_l)
    ece_risks = [min(1.0, compute_fisher_curvature(ans)[0] * 2.5)
                 for ans in [""] * 0]  # skip standalone — too slow

    # Full AUROC
    w_auroc_full = _auroc(witness_risks, labels)
    f3_auroc_full = _auroc(fusion3_risks, labels)
    f4_auroc_full = _auroc(fusion4_risks, labels)
    sp_auroc_full = _auroc([1.0 - s for s in spectral_scores], labels)

    # Accuracy at calibrated threshold
    cal_w = witness_risks[:cal_end]
    cal_f3 = fusion3_risks[:cal_end]
    cal_f4 = fusion4_risks[:cal_end]
    cal_l = labels[:cal_end]

    def best_tau(scores, labs):
        best_t, best_a = 0.5, 0.0
        for t in sorted(set(scores)):
            tp = sum(1 for s, y in zip(scores, labs) if s > t and y == 1)
            tn = sum(1 for s, y in zip(scores, labs) if s <= t and y == 0)
            a = (tp + tn) / len(labs)
            if a > best_a:
                best_a, best_t = a, t
        return best_t

    w_tau = best_tau(cal_w, cal_l)
    f3_tau = best_tau(cal_f3, cal_l)
    f4_tau = best_tau(cal_f4, cal_l)

    def acc(scores, labs, tau):
        tp = sum(1 for s, y in zip(scores, labs) if s > tau and y == 1)
        tn = sum(1 for s, y in zip(scores, labs) if s <= tau and y == 0)
        return (tp + tn) / len(labs)

    w_acc = acc(test_w, test_l, w_tau)
    f3_acc = acc(test_f3, test_l, f3_tau)
    f4_acc = acc(test_f4, test_l, f4_tau)

    n_test = len(test_l)
    ci = lambda a: 1.96 * (a * (1 - a) / n_test) ** 0.5

    print(f"\n  === Results (test split, n={n_test}) ===\n")
    print(f"  {'System':<30} {'AUROC(test)':>12} {'AUROC(full)':>12} "
          f"{'Accuracy':>10} {'CI95':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
    print(f"  {'WITNESS-only':<30} {w_auroc:>12.4f} {w_auroc_full:>12.4f} "
          f"{w_acc:>9.2%} {ci(w_acc):>8.4f}")
    print(f"  {'Spectral-only':<30} {sp_auroc:>12.4f} {sp_auroc_full:>12.4f} "
          f"{'(risk)':>10} {'':>8}")
    print(f"  {'Fusion-3 (W+E+G)':<30} {f3_auroc:>12.4f} {f3_auroc_full:>12.4f} "
          f"{f3_acc:>9.2%} {ci(f3_acc):>8.4f}")
    print(f"  {'Fusion-4 (W+E+G+S)':<30} {f4_auroc:>12.4f} {f4_auroc_full:>12.4f} "
          f"{f4_acc:>9.2%} {ci(f4_acc):>8.4f}")

    # Delta analysis
    d34 = f4_auroc - f3_auroc
    d_w4 = f4_auroc - w_auroc
    print(f"\n  Delta (AUROC):")
    print(f"    Fusion-4 vs WITNESS:   {d_w4:+.4f}")
    print(f"    Fusion-4 vs Fusion-3:  {d34:+.4f}")
    print(f"    Significant (>0.01):   {'YES' if abs(d34) > 0.01 else 'NO'}")

    result = {
        "benchmark": "4-signal fusion on HaluEval-QA full 10K",
        "n": n, "n_test": n_test,
        "witness_auroc": round(w_auroc, 4),
        "spectral_auroc": round(sp_auroc, 4),
        "fusion3_auroc": round(f3_auroc, 4),
        "fusion4_auroc": round(f4_auroc, 4),
        "witness_accuracy": round(w_acc, 4),
        "fusion3_accuracy": round(f3_acc, 4),
        "fusion4_accuracy": round(f4_acc, 4),
        "delta_f4_vs_f3": round(d34, 4),
        "delta_f4_vs_witness": round(d_w4, 4),
    }
    out = RESULTS_DIR / "fusion4_spectral_benchmark.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
