"""
Experiment 1 — Verifier Recall Under Controlled Evidence Ablation (protocol v2)
===============================================================================

Preregistration: benchmarks/VERIFIER_ABLATION_PREREGISTRATION.md
(criteria locked at c45d3073; protocol v2 declared at ac0178c7 — tie-corrected
AUROC, uniform formatting pipeline, fresh eval items. Thresholds unchanged.
The v1 run and its FAIL verdict remain committed in git history and in
benchmarks/results/verifier_ablation_recall.json.)

Measures whether EICV detects *compression-induced evidence gaps*: a correct
claim whose supporting evidence has been removed from the context. This is the
closed-loop failure case (compress -> answer -> verify -> retrieve -> retry),
which the committed right-vs-hallucinated AUROC results do not cover.

Conditions (claim = gold answer span, evidence varies):
  INTACT    full context
  CRITICAL  all answer-bearing sentences removed
  RANDOM    length-matched removal of non-answer sentences (artifact control)
  QCCR-REAL engine-gated secondary: real QCCR selection at matched budget

Criteria (locked):
  P1  AUROC(INTACT vs CRITICAL) >= 0.85            (full eval set, n=200)
  P2  holdout recall >= 0.75 AND holdout FPR <= 0.25 at dev-selected tau
  F1  FAIL if AUROC(INTACT vs RANDOM) > 0.65       (length-artifact detector)
  S1  engine-gated: flag rate >= 0.75 on dropped-answer QCCR items

Exit code 0 iff P1 and P2 and not F1.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from entroly.eicv import EICVAnalyzer  # noqa: E402

SEED = 42
N_EVAL = 200
N_CAL = 200
N_DEV = 100          # eval items [0:100] dev, [100:200] holdout
CHAR_CAP = 0.50      # E3: critical ablation must remove <= 50% of chars
LEN_TOL = 0.30       # E5: random ablation length match +/- 30%
E5_ATTEMPTS = 50
DEV_FPR_CAP = 0.20   # operating-point selection on dev
V2_POOL_START = 492  # v1 consumed pool[0:492] (eval scan + calibration); v2 items are fresh
PREREG_DOC = "benchmarks/VERIFIER_ABLATION_PREREGISTRATION.md"
PREREG_COMMIT = "ac0178c70f7b62f7071b6050f59d8a28b4f9fc5c"  # v2 declaration
RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s for s in (seg.strip() for seg in _SENT_RE.split(text)) if s]


def _auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUROC with midrank tie correction.

    Tied scores receive their midrank regardless of label, so identical score
    distributions yield exactly 0.5. The uncorrected variant (sorted (score,
    label) tuples) breaks ties by label and fabricates high AUROC under heavy
    ties — see benchmarks/results/diagnose_ablation_artifact.json.
    """
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        midrank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    n1 = sum(labels)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return 0.5
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_pool():
    """SQuAD v2 validation pool: (context, answer, question), seed-42 shuffle.

    Mirrors eicv_endtoend.load_squad filtering; questions are retained for the
    engine-gated QCCR condition.
    """
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(SEED)
    pool = []
    for row in ds:
        ctx = str(row.get("context", "") or "")
        ans = row.get("answers", {})
        texts = ans.get("text", []) if isinstance(ans, dict) else []
        q = str(row.get("question", "") or "")
        if ctx and texts and len(texts[0]) > 5:
            pool.append((ctx, texts[0], q))
    rng.shuffle(pool)
    return pool


def build_item(ctx: str, answer: str, rng: random.Random):
    """Apply the E1..E5 eligibility cascade. Returns (item, None) or (None, gate)."""
    low_ans = answer.lower()
    if low_ans not in ctx.lower():
        return None, "E1_answer_not_in_context"
    sents = _sentences(ctx)
    if len(sents) < 4:
        return None, "E2_too_few_sentences"
    crit_idx = [i for i, s in enumerate(sents) if low_ans in s.lower()]
    if not crit_idx:
        return None, "E1b_answer_crosses_sentence_boundary"
    total_chars = sum(len(s) for s in sents)
    removed_chars = sum(len(sents[i]) for i in crit_idx)
    if removed_chars > CHAR_CAP * total_chars:
        return None, "E3_ablation_over_char_cap"
    kept = [s for i, s in enumerate(sents) if i not in crit_idx]
    critical_ctx = " ".join(kept)
    if low_ans in critical_ctx.lower():
        return None, "E4_leakage_after_ablation"

    non_crit = [i for i in range(len(sents)) if i not in crit_idx]
    lo = (1.0 - LEN_TOL) * removed_chars
    hi = (1.0 + LEN_TOL) * removed_chars
    chosen: list[int] | None = None
    for _ in range(E5_ATTEMPTS):
        order = non_crit[:]
        rng.shuffle(order)
        acc: list[int] = []
        total = 0
        for i in order:
            if total >= lo:
                break
            acc.append(i)
            total += len(sents[i])
        if lo <= total <= hi and len(acc) < len(non_crit):
            chosen = acc
            break
    if chosen is None:
        return None, "E5_no_length_matched_random_ablation"
    random_ctx = " ".join(s for i, s in enumerate(sents) if i not in chosen)
    assert low_ans in random_ctx.lower(), "random ablation must not remove the answer"

    return {
        "answer": answer,
        # v2: all conditions share one formatting pipeline (rejoined sentences)
        "intact": " ".join(sents),
        "critical": critical_ctx,
        "random": random_ctx,
        "sentences": sents,
        "critical_kept_chars": total_chars - removed_chars,
    }, None


def _qccr_condition(items, questions, ana, tau):
    """Engine-gated secondary (S1): real QCCR selection at matched budget."""
    from entroly.native_status import QCCR_SYMBOLS, native_status
    status = native_status(QCCR_SYMBOLS)
    if not (status.ok and status.module is not None):
        return {"status": "skipped", "reason": "native QCCR engine unavailable"}

    from entroly import qccr
    dropped_flagged = 0
    dropped_total = 0
    survived = 0
    for item, question in zip(items, questions):
        frags = [
            {"source": f"s{i}", "content": s}
            for i, s in enumerate(item["sentences"])
        ]
        budget = max(16, item["critical_kept_chars"] // 4)  # chars -> approx tokens
        selected = qccr.select(frags, budget, query=question)
        evidence = " ".join(str(f.get("content") or "") for f in selected)
        if item["answer"].lower() in evidence.lower():
            survived += 1
            continue
        dropped_total += 1
        cert = ana.verify(evidence, item["answer"])
        if (1.0 - cert.phi) >= tau:
            dropped_flagged += 1
    rate, lo, hi = _wilson(dropped_flagged, dropped_total)
    return {
        "status": "ran",
        "n_items": len(items),
        "answer_survived": survived,
        "answer_dropped": dropped_total,
        "dropped_flagged_at_tau": dropped_flagged,
        "flag_rate": round(rate, 4),
        "flag_rate_wilson95": [round(lo, 4), round(hi, 4)],
        "s1_pass": dropped_total > 0 and rate >= 0.75,
    }


def main() -> int:
    print("=" * 74)
    print("  Experiment 1 — Verifier Recall Under Controlled Evidence Ablation")
    print(f"  Preregistration: {PREREG_DOC} @ {PREREG_COMMIT[:12]}")
    print("=" * 74)

    print("\n  Loading SQuAD v2 pool...")
    pool = load_pool()
    print(f"  Pool size: {len(pool)}")

    rng = random.Random(SEED)
    exclusions: dict[str, int] = {}
    items = []
    questions = []
    cursor = V2_POOL_START
    for cursor in range(V2_POOL_START, len(pool)):
        ctx, ans, q = pool[cursor]
        item, gate = build_item(ctx, ans, rng)
        if item is None:
            exclusions[gate] = exclusions.get(gate, 0) + 1
            continue
        items.append(item)
        questions.append(q)
        if len(items) == N_EVAL:
            break
    if len(items) < N_EVAL:
        print(f"  FATAL: only {len(items)} eligible items (< {N_EVAL})")
        return 1
    cal_pairs = [(ctx, ans) for ctx, ans, _ in pool[cursor + 1: cursor + 1 + N_CAL]]
    print(f"  Eligible: {len(items)}  (scanned pool[{V2_POOL_START}:{cursor + 1}])")
    print(f"  Exclusions: {exclusions}")
    print(f"  Calibration pairs: {len(cal_pairs)} (disjoint, pool[{cursor + 1}:{cursor + 1 + N_CAL}])")

    ana = EICVAnalyzer()
    print("\n  Fitting calibrators...", flush=True)
    t0 = time.perf_counter()
    ana.fit_calibrators(cal_pairs)
    print(f"  Calibrated in {time.perf_counter() - t0:.1f}s (n={ana.calibration.n_calibration})")

    conds = ("intact", "critical", "random")
    scores: dict[str, list[float]] = {c: [] for c in conds}
    decisions: dict[str, dict[str, int]] = {c: {} for c in conds}
    latencies: list[float] = []
    for item in items:
        for c in conds:
            cert = ana.verify(item[c], item["answer"])
            scores[c].append(1.0 - cert.phi)
            decisions[c][cert.decision] = decisions[c].get(cert.decision, 0) + 1
            latencies.append(cert.elapsed_ms)

    n = len(items)
    auroc_critical = _auroc(scores["intact"] + scores["critical"], [0] * n + [1] * n)
    auroc_random = _auroc(scores["intact"] + scores["random"], [0] * n + [1] * n)

    # Operating point: smallest tau with dev FPR <= cap; evaluate on holdout.
    dev_i, dev_c = scores["intact"][:N_DEV], scores["critical"][:N_DEV]
    hold_i, hold_c = scores["intact"][N_DEV:], scores["critical"][N_DEV:]
    candidates = sorted(set(dev_i + dev_c))
    candidates.append(max(candidates) + 1e-9)
    tau = next(
        t for t in candidates
        if sum(s >= t for s in dev_i) / len(dev_i) <= DEV_FPR_CAP
    )
    dev_recall = sum(s >= tau for s in dev_c) / len(dev_c)
    hold_recall = sum(s >= tau for s in hold_c) / len(hold_c)
    hold_fpr = sum(s >= tau for s in hold_i) / len(hold_i)
    hr, hr_lo, hr_hi = _wilson(sum(s >= tau for s in hold_c), len(hold_c))
    hf, hf_lo, hf_hi = _wilson(sum(s >= tau for s in hold_i), len(hold_i))

    def _flag_rates(c: str) -> dict:
        strict = decisions[c].get("hallucinated", 0)
        liberal = n - decisions[c].get("supported", 0)
        sr, s_lo, s_hi = _wilson(strict, n)
        lr, l_lo, l_hi = _wilson(liberal, n)
        return {
            "decisions": decisions[c],
            "strict_flag_rate": round(sr, 4),
            "strict_wilson95": [round(s_lo, 4), round(s_hi, 4)],
            "liberal_flag_rate": round(lr, 4),
            "liberal_wilson95": [round(l_lo, 4), round(l_hi, 4)],
        }

    p1 = auroc_critical >= 0.85
    p2 = hold_recall >= 0.75 and hold_fpr <= 0.25
    f1 = auroc_random > 0.65
    verdict = p1 and p2 and not f1

    print("\n  Results")
    print(f"    mean(1-phi): intact={sum(scores['intact'])/n:.4f}  "
          f"critical={sum(scores['critical'])/n:.4f}  random={sum(scores['random'])/n:.4f}")
    print(f"    P1  AUROC(intact vs critical) = {auroc_critical:.4f}  "
          f"(>= 0.85: {'PASS' if p1 else 'FAIL'})")
    print(f"    F1  AUROC(intact vs random)   = {auroc_random:.4f}  "
          f"(> 0.65 fails: {'ARTIFACT — EXPERIMENT FAILS' if f1 else 'clean'})")
    print(f"    tau = {tau:.4f} (dev FPR <= {DEV_FPR_CAP}, dev recall {dev_recall:.3f})")
    print(f"    P2  holdout recall = {hold_recall:.3f} [{hr_lo:.3f}, {hr_hi:.3f}]  "
          f"holdout FPR = {hold_fpr:.3f} [{hf_lo:.3f}, {hf_hi:.3f}]  "
          f"({'PASS' if p2 else 'FAIL'})")
    print(f"    mean latency = {sum(latencies)/len(latencies):.2f} ms/verify")

    print("\n  QCCR-REAL secondary (S1)...")
    qccr_res = _qccr_condition(items, questions, ana, tau)
    print(f"    {qccr_res}")

    out = {
        "schema": "verifier-ablation-recall-v2",
        "preregistration_doc": PREREG_DOC,
        "preregistration_commit": PREREG_COMMIT,
        "seed": SEED,
        "pool_start": V2_POOL_START,
        "n_eval": n,
        "n_dev": N_DEV,
        "n_calibration": len(cal_pairs),
        "pool_scanned": cursor + 1,
        "exclusions": exclusions,
        "mean_scores": {c: round(sum(scores[c]) / n, 4) for c in conds},
        "auroc_intact_vs_critical": round(auroc_critical, 4),
        "auroc_intact_vs_random": round(auroc_random, 4),
        "tau": round(tau, 6),
        "dev_recall_at_tau": round(dev_recall, 4),
        "holdout_recall_at_tau": round(hold_recall, 4),
        "holdout_recall_wilson95": [round(hr_lo, 4), round(hr_hi, 4)],
        "holdout_fpr_at_tau": round(hold_fpr, 4),
        "holdout_fpr_wilson95": [round(hf_lo, 4), round(hf_hi, 4)],
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "decision_flag_rates": {c: _flag_rates(c) for c in conds},
        "qccr_real": qccr_res,
        "criteria": {
            "P1_auroc_critical_ge_0.85": p1,
            "P2_holdout_recall_ge_0.75_fpr_le_0.25": p2,
            "F1_random_artifact_auroc_gt_0.65": f1,
            "S1": qccr_res.get("s1_pass", "skipped"),
        },
        "verdict_pass": verdict,
    }
    out_path = RESULTS_DIR / "verifier_ablation_recall_v2.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    print(f"\n  Verdict: {'PASSES' if verdict else 'FAILS'} preregistered criteria "
          f"(P1={p1}, P2={p2}, F1_artifact={f1})")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
