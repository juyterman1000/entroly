"""
S1-EXT — Tighter-Budget Sweep for Real-QCCR Verifier Recall
===========================================================

Declared in benchmarks/VERIFIER_ABLATION_PREREGISTRATION.md §10 at commit
38bc576275a4b4d69cfb3bff2c72560f84deea63, before this first run.

Grows the dropped-answer sample from the v2 S1 run (n=11) by shrinking the
QCCR budget over the same 200 eval items. τ is frozen at the v2 dev-selected
operating point — no re-tuning. Reports, per budget: answer survival, flag
rate on dropped items (recall on real compression gaps), and flag rate on
survived items (benign false-alarm rate under real compression).

Criteria (declared):
  S1-EXT-a  flag rate >= 0.75 at every budget with >= 10 dropped items
  S1-EXT-b  pooled unique dropped items >= 50 (correlated-observation caveat)
  S1-EXT-c  false-alarm rate on survived items <= 0.15 at every budget
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
from benchmarks.verifier_ablation_recall import (  # noqa: E402
    N_CAL, N_EVAL, SEED, V2_POOL_START, _wilson, build_item, load_pool,
)

TAU = 0.3608  # v2 dev-selected operating point (verifier_ablation_recall_v2.json)
BUDGET_FRACTIONS = (1.00, 0.75, 0.50, 0.25)
PREREG_COMMIT = "38bc576275a4b4d69cfb3bff2c72560f84deea63"
RESULTS_DIR = _THIS / "results"


def main() -> int:
    from entroly.native_status import QCCR_SYMBOLS, native_status
    status = native_status(QCCR_SYMBOLS)
    if not (status.ok and status.module is not None):
        print("FATAL: native QCCR engine unavailable — S1-EXT requires it")
        return 1
    from entroly import qccr

    print("=" * 74)
    print("  S1-EXT — Tighter-Budget Sweep (real QCCR, frozen tau)")
    print(f"  Preregistration §10 @ {PREREG_COMMIT[:12]}   tau = {TAU}")
    print("=" * 74)

    pool = load_pool()
    rng = random.Random(SEED)
    items, questions = [], []
    cursor = V2_POOL_START
    for cursor in range(V2_POOL_START, len(pool)):
        ctx, ans, q = pool[cursor]
        item, _gate = build_item(ctx, ans, rng)
        if item is not None:
            items.append(item)
            questions.append(q)
        if len(items) == N_EVAL:
            break
    cal_pairs = [(c, a) for c, a, _ in pool[cursor + 1: cursor + 1 + N_CAL]]
    print(f"  Items: {len(items)} (identical to v2 eval set)  cal: {len(cal_pairs)}")

    ana = EICVAnalyzer()
    ana.fit_calibrators(cal_pairs)

    budgets_out = {}
    dropped_union: set[int] = set()
    t0 = time.perf_counter()
    for frac in BUDGET_FRACTIONS:
        surv_flagged = 0
        surv_total = 0
        drop_flagged = 0
        drop_total = 0
        for idx, (item, question) in enumerate(zip(items, questions)):
            frags = [
                {"source": f"s{i}", "content": s}
                for i, s in enumerate(item["sentences"])
            ]
            base_budget = max(16, item["critical_kept_chars"] // 4)
            budget = max(16, int(base_budget * frac))
            selected = qccr.select(frags, budget, query=question)
            evidence = " ".join(str(f.get("content") or "") for f in selected)
            cert = ana.verify(evidence, item["answer"])
            flagged = (1.0 - cert.phi) >= TAU
            if item["answer"].lower() in evidence.lower():
                surv_total += 1
                surv_flagged += int(flagged)
            else:
                drop_total += 1
                drop_flagged += int(flagged)
                dropped_union.add(idx)

        dr, d_lo, d_hi = _wilson(drop_flagged, drop_total)
        sr, s_lo, s_hi = _wilson(surv_flagged, surv_total)
        budgets_out[f"{frac:.2f}"] = {
            "survived": surv_total,
            "dropped": drop_total,
            "dropped_flagged": drop_flagged,
            "drop_flag_rate": round(dr, 4),
            "drop_flag_wilson95": [round(d_lo, 4), round(d_hi, 4)],
            "survived_flagged": surv_flagged,
            "survived_false_alarm_rate": round(sr, 4),
            "survived_fa_wilson95": [round(s_lo, 4), round(s_hi, 4)],
        }
        print(f"  budget x{frac:.2f}: survived {surv_total:3d}  dropped {drop_total:3d}  "
              f"drop-flag {drop_flagged}/{drop_total} = {dr:.3f} [{d_lo:.3f},{d_hi:.3f}]  "
              f"benign-FA {sr:.3f} [{s_lo:.3f},{s_hi:.3f}]")

    elapsed = time.perf_counter() - t0

    eligible = {
        k: v for k, v in budgets_out.items() if v["dropped"] >= 10
    }
    a_pass = bool(eligible) and all(v["drop_flag_rate"] >= 0.75 for v in eligible.values())
    b_pass = len(dropped_union) >= 50
    c_pass = all(v["survived_false_alarm_rate"] <= 0.15 for v in budgets_out.values())

    out = {
        "schema": "verifier-ablation-s1-sweep-v1",
        "preregistration_doc": "benchmarks/VERIFIER_ABLATION_PREREGISTRATION.md#10",
        "preregistration_commit": PREREG_COMMIT,
        "seed": SEED,
        "tau": TAU,
        "n_items": len(items),
        "budget_fractions": list(BUDGET_FRACTIONS),
        "budgets": budgets_out,
        "unique_dropped_items": len(dropped_union),
        "elapsed_s": round(elapsed, 1),
        "criteria": {
            "S1_EXT_a_flag_ge_0.75_per_budget_n_ge_10": a_pass,
            "S1_EXT_b_unique_dropped_ge_50": b_pass,
            "S1_EXT_c_benign_fa_le_0.15": c_pass,
        },
        "verdict_pass": a_pass and b_pass and c_pass,
    }
    out_path = RESULTS_DIR / "verifier_ablation_s1_sweep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  unique dropped items: {len(dropped_union)}")
    print(f"  Saved: {out_path}")
    print(f"  Verdict: {'PASSES' if out['verdict_pass'] else 'FAILS'} "
          f"(a={a_pass}, b={b_pass}, c={c_pass})")
    return 0 if out["verdict_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
