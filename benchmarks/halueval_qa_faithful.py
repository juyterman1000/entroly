"""Faithful HaluEval-QA protocol: WITNESS vs same-data GPT judges.

Why this file exists
--------------------
`run_witness_benchmarks.py` picked an arbitrary `summary_score < 0.35`
cutoff on N=200 and reported accuracy at that single operating point.
That number is not comparable to the HaluEval leaderboard and is
gameable by threshold choice. This runner is the defensible version.

Protocol (HaluEval, Li et al., EMNLP 2023 — `qa` config, 10k items)
-------------------------------------------------------------------
Each item carries (knowledge, question, right_answer,
hallucinated_answer). The standard protocol scores BOTH answers:

    (knowledge, question, right_answer)         -> label 0 (faithful)
    (knowledge, question, hallucinated_answer)  -> label 1 (hallucinated)

giving a balanced 20k-decision set. The paper reports ACCURACY; the
canonical published reference is GPT-3.5-turbo = 62.59% on QA
(no-knowledge setting, ChatGPT judge).

What we measure (no threshold cheating)
---------------------------------------
* WITNESS runs on ALL 10k items (20k decisions). Its detector is
  threshold-parameterised, so the honest, threshold-FREE number is
  **AUROC** (rank statistic, computed exactly here, no sklearn).
* For an accuracy figure we split items: the first `CAL_ITEMS` items
  (shuffled, fixed seed) select the operating threshold by maximising
  balanced accuracy on that calibration split ONLY; accuracy is then
  reported on the disjoint test split. We also print the oracle-best
  test accuracy, explicitly labelled as an optimistic ceiling.
* GPT-3.5-turbo and gpt-4o-mini judge a shared `GPT_ITEMS`-item sample
  drawn from the TEST split (so it never overlaps WITNESS calibration).
  They are given the SAME knowledge WITNESS sees (fair grounded
  comparison); the paper's no-knowledge 62.59% is cited as the
  canonical external reference, not re-derived here.

All comparisons are on identical data. Honest by construction: AUROC is
unspoofable by threshold, and the GPT judge sees exactly what WITNESS
sees.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _load_dotenv_override() -> str | None:
    """Read OPENAI_API_KEY from a project .env and OVERRIDE any stale
    process/user-scope value. The persisted env var on this machine is a
    revoked key (401); the valid key lives only in .env, so .env wins."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
        m = re.match(r"(?:export\s+)?OPENAI_API_KEY\s*=\s*"
                     r"[\"']?([^\"'\s]+)", line.strip())
        if m:
            os.environ["OPENAI_API_KEY"] = m.group(1)
            return m.group(1)
    return None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entroly.witness import WitnessAnalyzer  # noqa: E402

SEED = 42
CAL_ITEMS = 2000          # items reserved for WITNESS threshold selection
GPT_ITEMS = 600           # shared items for the GPT head-to-head
GPT_WORKERS = 8
GPT_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini"]
PUBLISHED_GPT35_QA_ACC = 0.6259   # HaluEval paper, ChatGPT, no-knowledge

# Faithful standard HaluEval-QA judge prompt (grounded variant: the
# judge is given the knowledge, matching what WITNESS sees).
JUDGE_SYSTEM = (
    "I want you to act as an answer judge. Given a knowledge passage, a "
    "question, and a candidate answer, determine whether the answer "
    "contains non-factual or hallucinated information. The answer is "
    "hallucinated if it is not faithful to, not entailed by, or "
    "contradicts the knowledge, or is factually wrong, or fails to "
    "answer the question at an appropriate level of specificity. "
    "Respond with exactly one word: 'Yes' if the answer is "
    "hallucinated, or 'No' if the answer is correct and supported by "
    "the knowledge. Output only Yes or No."
)


def _judge_user(knowledge: str, question: str, answer: str) -> str:
    return (
        f"#Knowledge#: {knowledge}\n"
        f"#Question#: {question}\n"
        f"#Answer#: {answer}\n"
        f"#Your Judgement#:"
    )


def load_qa_items(n_items: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    items = []
    for i, row in enumerate(ds):
        if n_items is not None and i >= n_items:
            break
        k = str(row.get("knowledge", ""))
        q = str(row.get("question", ""))
        ra = str(row.get("right_answer", ""))
        ha = str(row.get("hallucinated_answer", ""))
        if k and q and ra and ha:
            items.append(
                {"knowledge": k, "question": q,
                 "right_answer": ra, "hallucinated_answer": ha}
            )
    rng = random.Random(SEED)
    rng.shuffle(items)
    return items


def auroc(scores: list[float], labels: list[int]) -> float:
    """Exact AUROC via the Mann-Whitney rank statistic with tie-aware
    average ranks. `scores` higher => more likely positive (label 1)."""
    paired = sorted(zip(scores, labels), key=lambda t: t[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(1 for _, y in paired if y == 1)
    n_neg = len(paired) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = sum(r for r, (_, y) in zip(ranks, paired) if y == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_balanced_threshold(scores: list[float], labels: list[int]):
    """Threshold on the score (predict positive if score >= tau) that
    maximises balanced accuracy. Returns (tau, balanced_acc)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5, 0.0
    cands = sorted(set(scores))
    best_tau, best_bacc = cands[0], -1.0
    for tau in cands:
        tpr = sum(1 for s in pos if s >= tau) / len(pos)
        tnr = sum(1 for s in neg if s < tau) / len(neg)
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_tau = bacc, tau
    return best_tau, best_bacc


def acc_at(scores: list[float], labels: list[int], tau: float) -> dict:
    tp = fp = fn = tn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= tau else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    ci = 1.96 * (acc * (1 - acc) / max(n, 1)) ** 0.5
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4), "acc_ci95": round(ci, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4)}


# ── WITNESS over the full set ─────────────────────────────────────────


def run_witness(items: list[dict]) -> dict:
    analyzer = WitnessAnalyzer(use_nli=False, force_python=True,
                               profile="benchmark_qa")
    cal_scores, cal_labels = [], []
    test_scores, test_labels = [], []
    test_index = []  # (item_idx, is_hallucinated) for the test split
    t0 = time.time()
    for idx, it in enumerate(items):
        ctx = f"{it['knowledge']}\n\nQuestion: {it['question']}"
        for ans, label in ((it["right_answer"], 0),
                           (it["hallucinated_answer"], 1)):
            res, _ = analyzer.analyze_and_rewrite(ctx, ans, mode="strict")
            # risk = 1 - groundedness; higher => more likely hallucinated
            risk = 1.0 - float(res.summary_score)
            if idx < CAL_ITEMS:
                cal_scores.append(risk)
                cal_labels.append(label)
            else:
                test_scores.append(risk)
                test_labels.append(label)
                test_index.append((idx, label))
    elapsed = time.time() - t0

    all_scores = cal_scores + test_scores
    all_labels = cal_labels + test_labels
    full_auroc = auroc(all_scores, all_labels)

    tau, cal_bacc = best_balanced_threshold(cal_scores, cal_labels)
    test_cal = acc_at(test_scores, test_labels, tau)
    _, oracle_tau_bacc = best_balanced_threshold(test_scores, test_labels)
    # oracle accuracy (cheating ceiling) at the test-optimal tau
    otau, _ = best_balanced_threshold(test_scores, test_labels)
    test_oracle = acc_at(test_scores, test_labels, otau)

    return {
        "n_items": len(items),
        "n_decisions": len(all_scores),
        "auroc_full": round(full_auroc, 4),
        "calibrated_tau": round(tau, 4),
        "cal_balanced_acc": round(cal_bacc, 4),
        "test_accuracy_calibrated": test_cal,
        "test_accuracy_oracle_ceiling": test_oracle,
        "ms_per_decision": round(1000 * elapsed / max(len(all_scores), 1), 3),
        "_test_scores": test_scores,
        "_test_labels": test_labels,
        "_test_index": test_index,
    }


# ── GPT judges over the shared sample ─────────────────────────────────


def run_gpt(model: str, sample: list[tuple[dict, str, int]]) -> dict:
    from openai import OpenAI

    client = OpenAI()

    def one(payload):
        it, ans, label = payload
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",
             "content": _judge_user(it["knowledge"], it["question"], ans)},
        ]
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=model, messages=msgs,
                    temperature=0.0, max_tokens=4,
                )
                txt = (r.choices[0].message.content or "").strip().lower()
                pred = 1 if txt.startswith("y") or "yes" in txt[:6] else 0
                return label, pred, None
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    return label, None, f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        return label, None, "exhausted"

    tp = fp = fn = tn = 0
    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=GPT_WORKERS) as ex:
        futs = [ex.submit(one, p) for p in sample]
        for f in as_completed(futs):
            label, pred, err = f.result()
            if pred is None:
                errors += 1
                continue
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1
    elapsed = time.time() - t0
    n = tp + fp + fn + tn
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    ci = 1.96 * (acc * (1 - acc) / max(n, 1)) ** 0.5
    return {"model": model, "n": n, "errors": errors,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4), "acc_ci95": round(ci, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "sec": round(elapsed, 1)}


def main() -> None:
    print("=" * 80)
    print("  HaluEval-QA  -  faithful protocol  -  WITNESS vs GPT judges")
    print("=" * 80)
    k = _load_dotenv_override()
    if k:
        print(f"  OpenAI key loaded from .env (prefix {k[:10]}, "
              f"suffix {k[-4:]})")
    print("  Loading pminervini/HaluEval [qa] (10k items, offline cache)...",
          flush=True)
    items = load_qa_items()
    print(f"    {len(items)} items  ->  {2 * len(items)} balanced decisions")
    print(f"    calibration items: {CAL_ITEMS}  |  test items: "
          f"{len(items) - CAL_ITEMS}", flush=True)

    print("\n  Running WITNESS on the full set (shipped Python path)...",
          flush=True)
    w = run_witness(items)
    print(f"    AUROC (threshold-free, full {w['n_decisions']} dec): "
          f"{w['auroc_full']:.4f}")
    tc = w["test_accuracy_calibrated"]
    to = w["test_accuracy_oracle_ceiling"]
    print(f"    Calibrated tau={w['calibrated_tau']:.3f}  ->  test "
          f"accuracy {tc['accuracy']:.4f} +/- {tc['acc_ci95']:.4f} "
          f"(F1 {tc['f1']:.3f}, P {tc['precision']:.3f}, "
          f"R {tc['recall']:.3f})")
    print(f"    Oracle-threshold ceiling (optimistic): "
          f"{to['accuracy']:.4f}")
    print(f"    {w['ms_per_decision']:.3f} ms/decision")

    # Shared GPT sample from the TEST split only.
    rng = random.Random(SEED + 1)
    test_item_idxs = sorted({idx for idx, _ in w["_test_index"]})
    rng.shuffle(test_item_idxs)
    chosen = test_item_idxs[:GPT_ITEMS]
    sample: list[tuple[dict, str, int]] = []
    for idx in chosen:
        it = items[idx]
        sample.append((it, it["right_answer"], 0))
        sample.append((it, it["hallucinated_answer"], 1))
    print(f"\n  Shared GPT sample: {len(chosen)} items -> "
          f"{len(sample)} calls per model", flush=True)

    # WITNESS accuracy on the exact same GPT sample, for a true
    # head-to-head (calibrated tau, no re-fitting).
    samp_scores, samp_labels = [], []
    ti = w["_test_index"]
    ts = w["_test_scores"]
    tl = w["_test_labels"]
    by_item: dict[int, list[int]] = {}
    for pos, (idx, _lab) in enumerate(ti):
        by_item.setdefault(idx, []).append(pos)
    for idx in chosen:
        for pos in by_item.get(idx, []):
            samp_scores.append(ts[pos])
            samp_labels.append(tl[pos])
    w_on_sample = acc_at(samp_scores, samp_labels, w["calibrated_tau"])
    w_auroc_sample = auroc(samp_scores, samp_labels)
    print(f"    WITNESS on this sample: AUROC {w_auroc_sample:.4f}, "
          f"acc {w_on_sample['accuracy']:.4f} +/- "
          f"{w_on_sample['acc_ci95']:.4f}", flush=True)

    gpt_results = []
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [skip] OPENAI_API_KEY not set; GPT baseline skipped.")
    else:
        for model in GPT_MODELS:
            print(f"\n  Judging with {model} "
                  f"({len(sample)} calls, {GPT_WORKERS} workers)...",
                  flush=True)
            gr = run_gpt(model, sample)
            gpt_results.append(gr)
            print(f"    {model}: acc {gr['accuracy']:.4f} +/- "
                  f"{gr['acc_ci95']:.4f}  (F1 {gr['f1']:.3f}, "
                  f"P {gr['precision']:.3f}, R {gr['recall']:.3f}, "
                  f"errs {gr['errors']}, {gr['sec']:.0f}s)")

    print("\n" + "=" * 80)
    print("  HEAD-TO-HEAD (same items, balanced, accuracy)")
    print("=" * 80)
    print(f"  {'system':<34s} {'accuracy':>10s} {'95% CI':>9s} "
          f"{'F1':>7s}")
    print("  " + "-" * 64)
    print(f"  {'WITNESS (calibrated tau, full)':<34s} "
          f"{tc['accuracy']:>10.4f} {tc['acc_ci95']:>9.4f} "
          f"{tc['f1']:>7.3f}")
    print(f"  {'WITNESS (on GPT sample)':<34s} "
          f"{w_on_sample['accuracy']:>10.4f} "
          f"{w_on_sample['acc_ci95']:>9.4f} {w_on_sample['f1']:>7.3f}")
    for gr in gpt_results:
        print(f"  {gr['model'] + ' (grounded judge)':<34s} "
              f"{gr['accuracy']:>10.4f} {gr['acc_ci95']:>9.4f} "
              f"{gr['f1']:>7.3f}")
    print(f"  {'GPT-3.5 paper (no-knowledge ref)':<34s} "
          f"{PUBLISHED_GPT35_QA_ACC:>10.4f} {'--':>9s} {'--':>7s}")
    print("  " + "-" * 64)
    print(f"  WITNESS AUROC (threshold-free, primary): "
          f"{w['auroc_full']:.4f}")

    out = {
        "protocol": "HaluEval-QA faithful, balanced, both answers scored",
        "dataset": "pminervini/HaluEval[qa]",
        "seed": SEED,
        "witness": {k: v for k, v in w.items()
                    if not k.startswith("_")},
        "witness_on_gpt_sample": {
            "auroc": round(w_auroc_sample, 4), **w_on_sample},
        "gpt": gpt_results,
        "published_reference": {
            "gpt35_qa_accuracy_no_knowledge": PUBLISHED_GPT35_QA_ACC,
            "source": "HaluEval, Li et al., EMNLP 2023"},
    }
    out_file = Path(__file__).parent / "results" / "halueval_qa_faithful.json"
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()
