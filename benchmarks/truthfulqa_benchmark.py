"""TruthfulQA Benchmark for WITNESS + Fusion Cascade.

TruthfulQA (Lin et al., ACL 2022) — 817 questions designed so that
humans and LLMs tend to answer incorrectly due to common misconceptions.

Protocol (honest):
  * Load truthful_qa/generation from HuggingFace (Apache 2.0)
  * For each question: we have correct_answers and incorrect_answers
  * Create balanced decision pairs:
      - Faithful:     context = best_answer, response = correct_answer
      - Hallucinated: context = best_answer, response = incorrect_answer
  * WITNESS scores each (context, response) pair
  * Measure: AUROC, accuracy at calibrated threshold, precision, recall
  * Calibration/test split (50/50, fixed seed) — no threshold cheating
  * Compare to published GPT baselines from the TruthfulQA paper

Published baselines (Lin et al., 2022):
  GPT-3 (davinci):     ~58% truthful (generation)
  GPT-3.5-turbo:       ~65% truthful
  GPT-4:               ~78% truthful (estimated from various reports)

What we measure is DIFFERENT: we don't generate answers — we VERIFY
whether a given answer is truthful. This is a detection task, not a
generation task. Our comparison is: can WITNESS detect untruthful
answers better than chance, and how does it compare to LLM-as-judge?
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
RESULTS_DIR = Path(__file__).parent / "results"


def _load_dotenv() -> None:
    p = Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"(?:export\s+)?OPENAI_API_KEY\s*=\s*"
                     r"[\"']?([^\"'\s]+)", line.strip())
        if m:
            os.environ["OPENAI_API_KEY"] = m.group(1)


def _auroc(scores: list[float], labels: list[int]) -> float:
    """Wilcoxon-Mann-Whitney AUROC. Higher score = more likely label=1."""
    from entroly.metrics import tie_corrected_auroc
    return tie_corrected_auroc(scores, labels)


def _ci95(n: int, acc: float) -> float:
    """Wilson 95% CI half-width."""
    import math
    z = 1.96
    denom = 1 + z * z / n
    (acc + z * z / (2 * n)) / denom
    spread = z * math.sqrt((acc * (1 - acc) + z * z / (4 * n)) / n) / denom
    return round(spread, 4)


def main() -> None:
    print("=" * 78)
    print("  TruthfulQA Benchmark: WITNESS Hallucination Detection")
    print("=" * 78)
    _load_dotenv()

    # ── Load dataset ──
    print("  Loading TruthfulQA (generation split)...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    print(f"  Loaded {len(ds)} questions")

    # ── Build decision pairs ──
    # For each question: pair a correct answer (label=0, faithful) with
    # an incorrect answer (label=1, hallucinated), using best_answer as context.
    rng = random.Random(SEED)
    pairs = []  # (context, answer, label)

    for row in ds:
        question = str(row.get("question", ""))
        best = str(row.get("best_answer", ""))
        correct = [str(a) for a in (row.get("correct_answers") or []) if a]
        incorrect = [str(a) for a in (row.get("incorrect_answers") or []) if a]

        if not best or not correct or not incorrect:
            continue

        # Context = question + best answer (this is what WITNESS checks against)
        ctx = f"Question: {question}\nAnswer: {best}"

        # Pick one correct and one incorrect answer for balanced pairs
        c_ans = rng.choice(correct)
        i_ans = rng.choice(incorrect)

        pairs.append((ctx, c_ans, 0))  # faithful
        pairs.append((ctx, i_ans, 1))  # hallucinated

    rng.shuffle(pairs)
    print(f"  Built {len(pairs)} decision pairs "
          f"({len(pairs)//2} questions x 2)")

    # ── Score with WITNESS ──
    print("  Scoring with WITNESS (Python path)...", flush=True)
    from entroly.witness import WitnessAnalyzer
    analyzer = WitnessAnalyzer(use_nli=False, force_python=True,
                               profile="benchmark_qa")

    scores = []  # risk scores (1 - summary_score): higher = more hallucinated
    labels = []
    t0 = time.perf_counter()

    for ctx, ans, y in pairs:
        res = analyzer.analyze(ctx, ans)
        risk = 1.0 - float(res.summary_score)
        scores.append(risk)
        labels.append(y)

    elapsed = time.perf_counter() - t0
    ms_per = elapsed * 1000 / len(pairs)
    print(f"  WITNESS scoring: {elapsed:.1f}s ({ms_per:.2f}ms/decision)")

    # ── Also compute ECE + entity coverage fusion ──
    print("  Computing fusion scores (WITNESS + ECE + entity gap)...",
          flush=True)
    from entroly.ravs.ece import compute_fisher_curvature

    _ENTITY_RE = [
        re.compile(r'\b\d+\.?\d*\b'),
        re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'),
    ]

    fusion_scores = []
    for i, (ctx, ans, y) in enumerate(pairs):
        # ECE Fisher curvature
        mean_k, _, _ = compute_fisher_curvature(ans)
        ece_k = min(1.0, mean_k * 2.5)

        # Entity coverage gap
        ans_ents = set()
        ctx_lower = ctx.lower()
        for pat in _ENTITY_RE:
            for m in pat.finditer(ans):
                ans_ents.add(m.group().lower())
        gap = 0.0
        if ans_ents:
            missing = sum(1 for e in ans_ents if e not in ctx_lower)
            gap = missing / len(ans_ents)

        fusion = 0.50 * scores[i] + 0.30 * ece_k + 0.20 * gap
        fusion_scores.append(min(1.0, max(0.0, fusion)))

    # ── Split (cal / test) ──
    n = len(scores)
    idx = list(range(n))
    rng2 = random.Random(SEED + 7)
    rng2.shuffle(idx)
    half = n // 2

    cal_idx, test_idx = idx[:half], idx[half:]

    def take(ix, arr):
        return [arr[i] for i in ix]

    cal_s, cal_l = take(cal_idx, scores), take(cal_idx, labels)
    test_s, test_l = take(test_idx, scores), take(test_idx, labels)

    f_cal_s, f_cal_l = take(cal_idx, fusion_scores), take(cal_idx, labels)
    f_test_s, f_test_l = take(test_idx, fusion_scores), take(test_idx, labels)

    # ── Calibrate threshold on cal split ──
    def best_threshold(cal_scores, cal_labels):
        best_tau, best_acc = 0.5, 0.0
        for tau in sorted(set(cal_scores)):
            correct = sum(
                int((1 if s > tau else 0) == y)
                for s, y in zip(cal_scores, cal_labels)
            )
            acc = correct / len(cal_labels)
            if acc > best_acc:
                best_acc = acc
                best_tau = tau
        return best_tau, best_acc

    w_tau, w_cal_acc = best_threshold(cal_s, cal_l)
    f_tau, f_cal_acc = best_threshold(f_cal_s, f_cal_l)

    # ── Evaluate on test split ──
    def evaluate(test_scores, test_labels, tau):
        tp = fp = fn = tn = 0
        for s, y in zip(test_scores, test_labels):
            pred = 1 if s > tau else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 1:
                fn += 1
            else:
                tn += 1
        n = len(test_labels)
        acc = (tp + tn) / n
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        return {
            "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4),
            "ci95": _ci95(n, acc),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "threshold": round(tau, 6),
        }

    w_result = evaluate(test_s, test_l, w_tau)
    f_result = evaluate(f_test_s, f_test_l, f_tau)

    w_auroc = _auroc(test_s, test_l)
    f_auroc = _auroc(f_test_s, f_test_l)

    # ── Print results ──
    n_test = len(test_l)
    print(f"\n  === Results (test split, n={n_test}) ===\n")
    print(f"  {'Metric':<20} {'WITNESS':>12} {'Fusion(3-sig)':>14}")
    print(f"  {'-'*20} {'-'*12} {'-'*14}")
    print(f"  {'AUROC':<20} {w_auroc:>12.4f} {f_auroc:>14.4f}")
    print(f"  {'Accuracy':<20} {w_result['accuracy']:>11.2%} "
          f"{f_result['accuracy']:>13.2%}")
    print(f"  {'  +/- 95% CI':<20} {w_result['ci95']:>12.4f} "
          f"{f_result['ci95']:>14.4f}")
    print(f"  {'Precision':<20} {w_result['precision']:>11.2%} "
          f"{f_result['precision']:>13.2%}")
    print(f"  {'Recall':<20} {w_result['recall']:>11.2%} "
          f"{f_result['recall']:>13.2%}")
    print(f"  {'F1':<20} {w_result['f1']:>12.4f} {f_result['f1']:>14.4f}")
    print(f"  {'Threshold':<20} {w_tau:>12.6f} {f_tau:>14.6f}")
    print(f"  {'ms/decision':<20} {ms_per:>12.2f} {'~same':>14}")

    # ── Comparison to published baselines ──
    print("\n  === Comparison to Published Baselines ===\n")
    print("  NOTE: Published TruthfulQA numbers measure GENERATION")
    print("  truthfulness (can the model produce a truthful answer?).")
    print("  Our numbers measure DETECTION (can WITNESS detect")
    print("  whether a given answer is truthful?). These are")
    print("  different tasks. Direct comparison is not valid.")
    print("")
    print("  Published generation truthfulness (Lin et al., 2022):")
    print("    GPT-3 (davinci):  ~58% truthful")
    print("    InstructGPT:      ~65% truthful")
    print("    GPT-4:            ~78% truthful (various reports)")
    print("")
    print("  Our DETECTION accuracy (distinct task):")
    print(f"    WITNESS:          {w_result['accuracy']:.2%} "
          f"+/- {w_result['ci95']:.2%} (AUROC {w_auroc:.4f})")
    print(f"    Fusion:           {f_result['accuracy']:.2%} "
          f"+/- {f_result['ci95']:.2%} (AUROC {f_auroc:.4f})")
    print("    Cost:             $0 (deterministic, no LLM)")

    # ── Save ──
    result_data = {
        "benchmark": "TruthfulQA",
        "dataset": "truthful_qa/generation",
        "license": "Apache-2.0",
        "n_questions": len(ds),
        "n_decisions": n,
        "n_test": n_test,
        "seed": SEED,
        "ms_per_decision": round(ms_per, 2),
        "witness": {
            **w_result,
            "auroc": round(w_auroc, 4),
        },
        "fusion": {
            **f_result,
            "auroc": round(f_auroc, 4),
            "weights": {"witness": 0.50, "ece": 0.30, "entity": 0.20},
        },
        "note": (
            "Published TruthfulQA baselines measure generation truthfulness. "
            "Our numbers measure detection accuracy (different task). "
            "Direct comparison is not valid."
        ),
    }

    out_file = RESULTS_DIR / "truthfulqa_benchmark.json"
    out_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()
