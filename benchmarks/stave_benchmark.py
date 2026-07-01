"""
Benchmark STAVE against HaluEval-QA — measures AUROC improvement over
the current WITNESS + entity-gap fusion on N items.

Usage:
    python benchmarks/stave_benchmark.py          # 500 items (~30s)
    python benchmarks/stave_benchmark.py 2000     # 2K items (~2min)
    python benchmarks/stave_benchmark.py 10000    # full 10K
"""

from __future__ import annotations
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _auroc(scores: list[float], labels: list[int]) -> float:
    paired = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(y for _, y in paired)
    n_neg = len(paired) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks, i = [0.0] * len(paired), 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos_rank_sum = sum(r for r, (_, y) in zip(ranks, paired) if y == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _best_acc(scores, labels):
    thresholds = sorted(set(scores))
    best = 0.0
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        acc = sum(pred == label for pred, label in zip(preds, labels)) / len(labels)
        best = max(best, acc)
    return best


N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 42

print(f"\n{'=' * 70}")
print(f"  STAVE Benchmark — HaluEval-QA  (N={N} items)")
print(f"{'=' * 70}\n")

from datasets import load_dataset  # noqa: E402
from entroly.witness import WitnessAnalyzer  # noqa: E402
from entroly.verifiers.stave import stave_risk  # noqa: E402

ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
items = [
    (
        str(r["knowledge"]),
        str(r["question"]),
        str(r["answer"]),
        1 if r["hallucination"] == "yes" else 0,
    )
    for r in ds
    if r.get("knowledge") and r.get("answer")
]
random.Random(SEED).shuffle(items)
items = items[:N]

analyzer = WitnessAnalyzer(use_nli=False, force_python=True, profile="benchmark_qa")

_ENT_RE = [
    re.compile(r"\b\d+\.?\d*\b"),
    re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b"),
]

w_scores, fus_scores, stave_scores, stave_fus_scores, labels = [], [], [], [], []

t0 = time.perf_counter()
for idx, (k, q, ans, y) in enumerate(items):
    ctx = f"{k}\n\nQuestion: {q}"
    # WITNESS
    res = analyzer.analyze(ctx, ans)
    w_risk = 1.0 - float(res.summary_score)

    # Entity-gap (existing fusion feature)
    ans_ents = set()
    ctx_l = ctx.lower()
    for pat in _ENT_RE:
        for m in pat.finditer(ans):
            ans_ents.add(m.group().lower())
    gap = 0.0
    if ans_ents:
        gap = sum(1 for e in ans_ents if e not in ctx_l) / len(ans_ents)

    # STAVE
    sr = stave_risk(ans, k)

    # Fusions
    fus = 0.50 * w_risk + 0.20 * gap + 0.30 * (sr if sr != 0.5 else w_risk)
    old_fus = 0.50 * w_risk + 0.20 * gap  # baseline (no STAVE)

    w_scores.append(w_risk)
    fus_scores.append(old_fus)
    stave_scores.append(sr)
    stave_fus_scores.append(fus)
    labels.append(y)

elapsed = time.perf_counter() - t0
n_dec = len(labels)

# Compute AUROCs
auroc_w = _auroc(w_scores, labels)
auroc_fus = _auroc(fus_scores, labels)
auroc_stave = _auroc(stave_scores, labels)
auroc_sfus = _auroc(stave_fus_scores, labels)

acc_w = _best_acc(w_scores, labels)
acc_fus = _best_acc(fus_scores, labels)
acc_stave = _best_acc(stave_scores, labels)
acc_sfus = _best_acc(stave_fus_scores, labels)

# STAVE coverage (fraction of decisions where STAVE gave non-neutral signal)
n_signal = sum(1 for s in stave_scores if s != 0.5)
coverage = n_signal / n_dec

print(f"  Decisions  : {n_dec}  ({elapsed * 1000 / n_dec:.2f}ms/decision)")
print(
    f"  STAVE coverage: {n_signal}/{n_dec} decisions have relational signal ({100 * coverage:.1f}%)\n"
)
print(f"  {'System':<30}  {'AUROC':>7}  {'Best-Acc':>9}")
print(f"  {'-' * 52}")
print(f"  {'WITNESS alone':<30}  {auroc_w:>7.4f}  {100 * acc_w:>8.1f}%")
print(
    f"  {'WITNESS + entity-gap (baseline)':<30}  {auroc_fus:>7.4f}  {100 * acc_fus:>8.1f}%"
)
print(f"  {'STAVE alone':<30}  {auroc_stave:>7.4f}  {100 * acc_stave:>8.1f}%")
print(
    f"  {'WITNESS + entity-gap + STAVE':<30}  {auroc_sfus:>7.4f}  {100 * acc_sfus:>8.1f}%"
)
print(f"\n  AUROC delta (STAVE added)  : {auroc_sfus - auroc_fus:+.4f}")
print(f"  Acc   delta (STAVE added)  : {100 * (acc_sfus - acc_fus):+.1f}pp")

# Wrong-slot analysis — split by actual label
from entroly.verifiers.stave import stave_verify  # noqa: E402

n_ws_hall, n_ws_safe, sample = 0, 0, 0
for k, q, ans, y in items[: min(200, N)]:
    detected = stave_verify(ans, k).wrong_slot_detected
    if y == 1 and detected:
        n_ws_hall += 1
    if y == 0 and detected:
        n_ws_safe += 1
    sample += 1
print(f"\n  Wrong-slot gate on {sample} items:")
print(f"    Hallucinations flagged : {n_ws_hall}")
print(f"    Safe answers flagged   : {n_ws_safe}")
prec = n_ws_hall / (n_ws_hall + n_ws_safe + 1e-9)
print(f"    Precision of gate      : {prec:.3f}")
print()

out = {
    "n_items": N,
    "n_decisions": n_dec,
    "ms_per_decision": round(elapsed * 1000 / n_dec, 3),
    "stave_coverage_pct": round(100 * coverage, 2),
    "witness_auroc": round(auroc_w, 4),
    "fusion_auroc": round(auroc_fus, 4),
    "stave_auroc": round(auroc_stave, 4),
    "stave_fusion_auroc": round(auroc_sfus, 4),
    "auroc_delta": round(auroc_sfus - auroc_fus, 4),
    "witness_acc": round(acc_w, 4),
    "fusion_acc": round(acc_fus, 4),
    "stave_acc": round(acc_stave, 4),
    "stave_fusion_acc": round(acc_sfus, 4),
    "acc_delta_pp": round(100 * (acc_sfus - acc_fus), 2),
}
out_path = Path(__file__).parent / "results" / "stave_benchmark.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"  Results -> {out_path.name}")
