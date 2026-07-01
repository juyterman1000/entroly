"""Optimal weight search for 4-signal fusion (cal/test split)."""

from __future__ import annotations
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
SEED, CAL = 42, 2000


def auroc(s, labels):
    p = sorted(zip(s, labels))
    r = [0.0] * len(p)
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[j + 1][0] == p[i][0]:
            j += 1
        a = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[k] = a
        i = j + 1
    n1 = sum(y for _, y in p)
    n0 = len(p) - n1
    if n0 == 0 or n1 == 0:
        return 0.5
    return (sum(r for r, (_, y) in zip(r, p) if y == 1) - n1 * (n1 + 1) / 2) / (n0 * n1)


def main():
    print("=" * 70)
    print("  Weight Optimizer: 4-Signal Fusion (10K HaluEval)")
    print("=" * 70)
    from datasets import load_dataset
    from entroly.witness import WitnessAnalyzer
    from entroly.ravs.ece import compute_fisher_curvature
    from entroly.ravs.spectral import compute_spectral_consistency

    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    items = [
        (
            str(r.get("knowledge", "")),
            str(r.get("question", "")),
            str(r.get("right_answer", "")),
            str(r.get("hallucinated_answer", "")),
        )
        for r in ds
        if all(
            r.get(f)
            for f in ("knowledge", "question", "right_answer", "hallucinated_answer")
        )
    ]
    random.Random(SEED).shuffle(items)
    az = WitnessAnalyzer(use_nli=False, force_python=True, profile="benchmark_qa")
    EP = [
        re.compile(r"\b\d+\.?\d*\b"),
        re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b"),
    ]
    W, E, G, S, L = [], [], [], [], []
    t0 = time.perf_counter()
    for i, (k, q, ra, ha) in enumerate(items):
        ctx = f"{k}\n\nQuestion: {q}"
        for ans, y in ((ra, 0), (ha, 1)):
            res = az.analyze(ctx, ans)
            W.append(1 - float(res.summary_score))
            L.append(y)
            mk, _, _ = compute_fisher_curvature(ans)
            E.append(min(1, mk * 2.5))
            ae = set()
            cl = ctx.lower()
            for p in EP:
                for m in p.finditer(ans):
                    ae.add(m.group().lower())
            G.append(sum(1 for e in ae if e not in cl) / max(len(ae), 1) if ae else 0)
            S.append(1 - compute_spectral_consistency(ctx, ans).score)
        if (i + 1) % 2000 == 0:
            print(
                f"    {i + 1}/{len(items)} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    print(f"  Done: {len(L)} decisions in {time.perf_counter() - t0:.0f}s")
    ce = CAL * 2
    cW, cE, cG, cS, cL = W[:ce], E[:ce], G[:ce], S[:ce], L[:ce]
    tW, tE, tG, tS, tL = W[ce:], E[ce:], G[ce:], S[ce:], L[ce:]
    print(
        f"\n  Signal AUROCs (cal): W={auroc(cW, cL):.4f} E={auroc(cE, cL):.4f} G={auroc(cG, cL):.4f} S={auroc(cS, cL):.4f}"
    )
    best, bw = 0, (0.5, 0.3, 0.2, 0)
    st = 0.05
    for ww in [x * st for x in range(1, int(1 / st) + 1)]:
        for we in [x * st for x in range(0, int((1 - ww) / st) + 1)]:
            for wg in [x * st for x in range(0, int((1 - ww - we) / st) + 1)]:
                ws = round(1 - ww - we - wg, 4)
                if ws < -0.001:
                    continue
                ws = max(0, ws)
                f = [
                    min(1, max(0, ww * cW[i] + we * cE[i] + wg * cG[i] + ws * cS[i]))
                    for i in range(len(cL))
                ]
                a = auroc(f, cL)
                if a > best:
                    best, bw = a, (ww, we, wg, ws)
    print(
        f"  Best cal AUROC: {best:.4f}  weights: W={bw[0]:.2f} E={bw[1]:.2f} G={bw[2]:.2f} S={bw[3]:.2f}"
    )
    tf = [
        min(1, max(0, bw[0] * tW[i] + bw[1] * tE[i] + bw[2] * tG[i] + bw[3] * tS[i]))
        for i in range(len(tL))
    ]
    oa = auroc(tf, tL)
    wa = auroc(tW, tL)
    f3 = [
        min(1, max(0, 0.5 * tW[i] + 0.3 * tE[i] + 0.2 * tG[i])) for i in range(len(tL))
    ]
    f3a = auroc(f3, tL)
    print(f"\n  Test AUROC: WITNESS={wa:.4f}  Fusion3={f3a:.4f}  Fusion4-opt={oa:.4f}")
    print(f"  Lift vs WITNESS: {oa - wa:+.4f}  vs Fusion3: {oa - f3a:+.4f}")
    r = {
        "weights": {"W": bw[0], "E": bw[1], "G": bw[2], "S": bw[3]},
        "test_witness": round(wa, 4),
        "test_f3": round(f3a, 4),
        "test_f4_opt": round(oa, 4),
        "lift_vs_w": round(oa - wa, 4),
        "lift_vs_f3": round(oa - f3a, 4),
    }
    out = Path(__file__).parent / "results" / "fusion4_optimized.json"
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
