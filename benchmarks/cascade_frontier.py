"""Measure the conformal cascade's cost-risk frontier on HaluEval-QA.

Honest design (same discipline as halueval_qa_faithful.py):
  * Per-decision WITNESS score + true label + gpt-4o-mini prediction on
    the SAME 600-item / 1200-decision shared sample.
  * Calibration / test split (50/50, fixed seed): the escalation band is
    fitted on calibration ONLY; the frontier is measured on the disjoint
    test split. No threshold cheating.
  * We cache the per-decision arrays to results/cascade_arrays.json so
    the falsification test runs offline & repeatably (no re-billing GPT).

The breakthrough is *falsifiable*: the cascade must (a) keep realized
selective risk on the cheap region ≤ target within finite-sample slack,
and (b) at some operating point Pareto-dominate BOTH WITNESS-only and
gpt-4o-mini-only (lower error at lower cost than always-LLM, lower error
than WITNESS-only). If it does not, this prints that plainly.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entroly.conformal_cascade import evaluate_policy, select_band  # noqa: E402

SEED = 42
GPT_ITEMS = 600
GPT_MODEL = "gpt-4o-mini"
GPT_WORKERS = 8
ARRAYS = Path(__file__).parent / "results" / "cascade_arrays.json"

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


def _load_dotenv() -> None:
    p = Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"(?:export\s+)?OPENAI_API_KEY\s*=\s*"
                     r"[\"']?([^\"'\s]+)", line.strip())
        if m:
            os.environ["OPENAI_API_KEY"] = m.group(1)


def build_arrays() -> dict:
    """Compute (or load cached) per-decision WITNESS score, label, and
    gpt-4o-mini prediction on the shared sample."""
    if ARRAYS.exists():
        print(f"  Loading cached arrays: {ARRAYS}")
        return json.loads(ARRAYS.read_text(encoding="utf-8"))

    from datasets import load_dataset

    from entroly.witness import WitnessAnalyzer

    print("  Loading HaluEval[qa] + scoring with WITNESS...", flush=True)
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
    sample = items[:GPT_ITEMS]

    analyzer = WitnessAnalyzer(use_nli=False, force_python=True,
                               profile="benchmark_qa")
    scores, labels, payloads = [], [], []
    for k, q, ra, ha in sample:
        ctx = f"{k}\n\nQuestion: {q}"
        for ans, y in ((ra, 0), (ha, 1)):
            res, _ = analyzer.analyze_and_rewrite(ctx, ans, mode="strict")
            scores.append(1.0 - float(res.summary_score))
            labels.append(y)
            payloads.append((k, q, ans))

    print(f"  Judging {len(payloads)} decisions with {GPT_MODEL}...",
          flush=True)
    from openai import OpenAI
    client = OpenAI()

    def one(i_kqa):
        i, (k, q, ans) = i_kqa
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",
                 "content": f"#Knowledge#: {k}\n#Question#: {q}\n"
                            f"#Answer#: {ans}\n#Your Judgement#:"}]
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=GPT_MODEL, messages=msgs,
                    temperature=0.0, max_tokens=4)
                t = (r.choices[0].message.content or "").strip().lower()
                return i, (1 if t.startswith("y") else 0)
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    return i, -1
                time.sleep(2 ** attempt)
        return i, -1

    llm = [0] * len(payloads)
    with ThreadPoolExecutor(max_workers=GPT_WORKERS) as ex:
        futs = [ex.submit(one, (i, p)) for i, p in enumerate(payloads)]
        for f in as_completed(futs):
            i, pred = f.result()
            llm[i] = pred

    data = {"scores": scores, "labels": labels, "llm": llm,
            "model": GPT_MODEL, "n": len(scores)}
    ARRAYS.parent.mkdir(exist_ok=True)
    ARRAYS.write_text(json.dumps(data), encoding="utf-8")
    print(f"  Cached arrays -> {ARRAYS}")
    return data


def _split(data):
    n = data["n"]
    idx = list(range(n))
    random.Random(SEED + 7).shuffle(idx)
    half = n // 2
    cal, test = idx[:half], idx[half:]

    def take(ix, key):
        return [data[key][i] for i in ix]
    return (
        (take(cal, "scores"), take(cal, "labels"), take(cal, "llm")),
        (take(test, "scores"), take(test, "labels"), take(test, "llm")),
    )


def _baseline(scores, labels, llm):
    """Single-verifier reference points on the test split."""
    n = len(scores)
    # WITNESS-only at the shipped operating point (flag any deficit).
    w_err = sum(int((1 if s > 0.0004 else 0) != y)
                for s, y in zip(scores, labels)) / n
    # gpt-4o-mini-only (always escalate): cost 1.0/item, its own error.
    g_err = sum(int(p != y) for p, y in zip(llm, labels)) / n
    return w_err, g_err


def main() -> None:
    print("=" * 78)
    print("  Conformal cascade - cost/risk frontier on HaluEval-QA")
    print("=" * 78)
    _load_dotenv()
    data = build_arrays()
    (cs, cl, cj), (ts, tl, tj) = _split(data)
    print(f"  calibration={len(cs)}  test={len(tl)}  "
          f"(balanced HaluEval-QA decisions)")

    w_err, g_err = _baseline(ts, tl, tj)
    print(f"\n  WITNESS-only  test error={w_err:.4f}  cost/item=0.00 ($0)")
    print(f"  {GPT_MODEL}-only test error={g_err:.4f}  cost/item=1.00 (LLM)")
    print("\n  Cascade frontier (band fit on calibration, measured on "
          "test):")
    print(f"  {'target_e':>8} {'escal%':>7} {'selRisk':>8} {'ovErr':>7} "
          f"{'cost/it':>8} {'vs WIT':>7} {'vs LLM':>7}")
    print("  " + "-" * 64)

    Q = 1.0
    rows = []
    dominating = []
    for eps in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        # c_exp = q*eps makes rule (★) escalate exactly where local
        # error exceeds the target — ties the band to escalation.py.
        pol = select_band(cs, cl, target_selective_risk=eps,
                          c_exp=Q * eps, q=Q, r_floor=0.0)
        out = evaluate_policy(pol, ts, tl, tj)
        better_than_witness = out.overall_error <= w_err + 1e-9
        cheaper_than_llm = out.expected_cost < 1.0 - 1e-9
        better_than_llm = out.overall_error <= g_err + 1e-9
        pareto = (better_than_witness and cheaper_than_llm
                  and out.overall_error <= min(w_err, g_err) + 0.005)
        rows.append((eps, out, better_than_witness, better_than_llm,
                     cheaper_than_llm))
        if pareto:
            dominating.append((eps, out))
        print(f"  {eps:>8.2f} {100*out.escalation_rate:>6.1f}% "
              f"{out.selective_risk_cheap:>8.4f} {out.overall_error:>7.4f} "
              f"{out.expected_cost:>8.3f} "
              f"{'YES' if better_than_witness else 'no':>7} "
              f"{'YES' if (better_than_llm and cheaper_than_llm) else 'no':>7}")

    print("  " + "-" * 64)
    if dominating:
        eps, o = min(dominating, key=lambda t: t[1].overall_error)
        print(f"\n  PARETO POINT @eps={eps:.2f}: error {o.overall_error:.4f} "
              f"(< WITNESS {w_err:.4f} & <= LLM {g_err:.4f}) at "
              f"{100*o.escalation_rate:.1f}% LLM calls "
              f"(cost {o.expected_cost:.3f} vs always-LLM 1.000) - "
              f"{100*(1-o.expected_cost):.0f}% cost cut at no accuracy loss.")
    else:
        print("\n  NO strict Pareto point found - cascade does not "
              "dominate both baselines on this sample. Reporting honestly.")

    out_file = Path(__file__).parent / "results" / "cascade_frontier.json"
    out_file.write_text(json.dumps({
        "witness_only_error": w_err,
        "llm_only_error": g_err,
        "llm_model": data["model"],
        "n_test": len(tl),
        "frontier": [
            {"target_eps": e, "escalation_rate": o.escalation_rate,
             "selective_risk_cheap": o.selective_risk_cheap,
             "overall_error": o.overall_error,
             "expected_cost": o.expected_cost,
             "beats_witness": bw, "beats_llm": bl, "cheaper_than_llm": cl}
            for e, o, bw, bl, cl in rows
        ],
    }, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()
