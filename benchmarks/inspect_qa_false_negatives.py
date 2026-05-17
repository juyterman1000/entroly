"""Forensic: WHY do hallucinated HaluEval-QA answers escape suppression?

Not a metric runner. For every QA sample it records the pipeline's
suppression decision AND the recomputed feature vector, then contrasts
the feature distribution of false negatives (hallucinated + NOT
suppressed = exposed) against true positives and safe-retained.

The point is to see the *separating* (or non-separating) features so the
next verifier targets the real miss pattern, not a threshold.
"""

from __future__ import annotations

import io
import statistics
import sys
from pathlib import Path

# HaluEval has non-cp1252 chars; force UTF-8 stdout on Windows consoles.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entroly.witness import WitnessAnalyzer  # noqa: E402
from entroly.witness_features import (  # noqa: E402
    ClaimFeatures,
    _longest_contiguous_run,
    _tokens,
    content_words,
    extract_features,
)
import re as _re  # noqa: E402


def _bindings(answer: str, knowledge: str, question: str) -> tuple[float, float]:
    """Mirror feat_qa_alignment's locus pick, return (bind_q, bind_other)."""
    a_tok = _tokens(answer)
    if len(a_tok) < 2 or not question.strip() or not knowledge.strip():
        return (1.0, 0.0)
    K = knowledge.lower()
    q_words = content_words(question)
    sents = [s for s in _re.split(r"(?<=[.!?])\s+", K) if s.strip()] or [K]

    def w(t: str) -> float:
        import math
        return 1.0 / (1.0 + math.log(1 + K.count(t)))

    best, best_s = -1.0, sents[0]
    for s in sents:
        sw = set(_re.findall(r"[A-Za-z][A-Za-z0-9_'-]+", s))
        sc = sum(w(t) for t in q_words if t in sw)
        if sc > best:
            best, best_s = sc, s
    denom = float(len(a_tok))
    bq = _longest_contiguous_run(a_tok, _tokens(best_s)) / denom
    bo = 0.0
    for s in sents:
        if s == best_s:
            continue
        r = _longest_contiguous_run(a_tok, _tokens(s)) / denom
        bo = max(bo, r)
    return (bq, bo)

N = 120


def main() -> None:
    try:
        from datasets import load_dataset
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception as e:  # offline / hub failure
        print(f"[skip] could not load HaluEval-QA: {type(e).__name__}: {e}")
        return

    analyzer = WitnessAnalyzer(use_nli=False, force_python=True, profile="benchmark_qa")

    buckets: dict[str, list[ClaimFeatures]] = {"FN": [], "TP": [], "SAFE_KEEP": [], "SAFE_SUPP": []}
    bind_stats: dict[str, list[float]] = {"FN": [], "TP": [], "SAFE_KEEP": [], "SAFE_SUPP": []}
    fn_examples = []
    fn_qa: list[tuple[str, str]] = []

    n = 0
    for row in ds:
        if n >= N:
            break
        knowledge = str(row.get("knowledge", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not knowledge or not answer:
            continue
        is_hall = str(row.get("hallucination", "")).strip().lower() == "yes"
        n += 1

        context = f"{knowledge}\n\nQuestion: {question}" if question else knowledge
        _result, rewrite = analyzer.analyze_and_rewrite(context, answer, mode="strict")
        suppressed = rewrite.suppressed_count > 0

        feats = extract_features(answer, knowledge, question=question)
        bq, _bo = _bindings(answer, knowledge, question)

        if is_hall and not suppressed:
            buckets["FN"].append(feats)
            bind_stats["FN"].append(bq)
            fn_qa.append((question, answer))
            if len(fn_examples) < 18:
                fn_examples.append((question, answer, knowledge, feats))
        elif is_hall and suppressed:
            buckets["TP"].append(feats)
            bind_stats["TP"].append(bq)
        elif (not is_hall) and (not suppressed):
            buckets["SAFE_KEEP"].append(feats)
            bind_stats["SAFE_KEEP"].append(bq)
        else:
            buckets["SAFE_SUPP"].append(feats)
            bind_stats["SAFE_SUPP"].append(bq)

    print(f"\nN={n}  FN={len(buckets['FN'])} (exposed hallucinations)  "
          f"TP={len(buckets['TP'])}  SAFE_KEEP={len(buckets['SAFE_KEEP'])}  "
          f"SAFE_SUPP={len(buckets['SAFE_SUPP'])}")
    expo = len(buckets["FN"]) / max(len(buckets["FN"]) + len(buckets["TP"]), 1)
    print(f"exposure (sfn/(stp+sfn)) = {expo:.3f}\n")

    names = ClaimFeatures.feature_names()

    def mean_vec(fs: list[ClaimFeatures]) -> list[float]:
        if not fs:
            return [float("nan")] * len(names)
        cols = list(zip(*[f.as_vector() for f in fs]))
        return [statistics.mean(c) for c in cols]

    # Binding separation: the whole thesis is that genuine extractive
    # answers are contiguous spans of their Q-evidence sentence while
    # recombination/wrong-option answers are not. Quantify it.
    print("bind_q distribution (contiguous answer span at Q-locus):")
    for label in ("FN", "TP", "SAFE_KEEP"):
        bqs = bind_stats[label]
        if not bqs:
            continue
        bqs_sorted = sorted(bqs)
        med = bqs_sorted[len(bqs_sorted) // 2]
        frac_lt = sum(1 for v in bqs if v < 0.34) / len(bqs)
        print(f"  {label:<10s} n={len(bqs):>3d}  mean={statistics.mean(bqs):.3f}  "
              f"median={med:.3f}  frac(bind_q<0.34)={frac_lt:.2f}")
    print()

    # Residual-FN composition: is the comparative/selective class the
    # dominant remaining miss, as hypothesized?
    _CMP = _re.compile(
        r"\b(more|less|most|least|fewer|older|younger|newer|earlier|later|"
        r"larger|smaller|longer|shorter|higher|lower|greater|bigger|first|"
        r"last|before|after)\b|\b\w+est\b", _re.I)

    def classify(q: str) -> str:
        ql = q.lower()
        has_or = " or " in ql
        both = "both" in ql and " and " in ql
        cmp_cue = bool(_CMP.search(ql))
        if both:
            return "membership_both"
        if cmp_cue and has_or:
            return "comparative_select"   # "who won more, A or B?"
        if cmp_cue:
            return "comparative"          # superlative/ordinal, single
        if has_or:
            return "selective_or"         # "which is X, A or B?" (no cmp word)
        if _re.match(r"\s*(is|are|was|were|do|does|did|has|have|can|could|will)\b", ql):
            return "yesno"
        return "factoid_other"

    comp: dict[str, list[tuple[str, str]]] = {}
    for q, a in fn_qa:
        comp.setdefault(classify(q), []).append((q, a))
    print("Residual FN composition (exposed after binding gate):")
    for cls in sorted(comp, key=lambda c: -len(comp[c])):
        items = comp[cls]
        print(f"  {cls:<20s} {len(items):>3d}  "
              f"({100*len(items)/max(len(fn_qa),1):.0f}%)")
        for q, a in items[:2]:
            print(f"      Q: {q[:110]}")
            print(f"      A: {a[:90]}")
    print()

    fn_m = mean_vec(buckets["FN"])
    tp_m = mean_vec(buckets["TP"])
    sk_m = mean_vec(buckets["SAFE_KEEP"])

    print(f"{'feature':<20s} {'FN(exposed)':>12s} {'TP(caught)':>12s} "
          f"{'SAFE_keep':>12s} {'FN-TP gap':>10s}")
    print("-" * 70)
    for i, nm in enumerate(names):
        gap = fn_m[i] - tp_m[i]
        print(f"{nm:<20s} {fn_m[i]:>12.3f} {tp_m[i]:>12.3f} "
              f"{sk_m[i]:>12.3f} {gap:>10.3f}")

    print("\n=== Sample exposed hallucinations (FN) ===")
    for q, a, k, f in fn_examples:
        print(f"\nQ: {q[:140]}")
        print(f"A(hallucinated, NOT suppressed): {a[:160]}")
        print(f"K: {k[:240]}...")
        print("  feats: " + ", ".join(
            f"{nm}={v:+.2f}" for nm, v in zip(names, f.as_vector())
        ))


if __name__ == "__main__":
    main()
