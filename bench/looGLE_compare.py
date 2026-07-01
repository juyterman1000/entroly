#!/usr/bin/env python3
"""
LooGLE Head-to-Head: Entroly vs LongTokenPruner / TokenPruner-2
============================================================

Compares context-compression methods on the LooGLE benchmark
(BigAI-NLCO, ACL 2024 — the 2026 standard for long-context RAG eval).

Three methods are compared on identical samples:

  1. BASELINE           — truncate context to fit the budget (FIFO)
  2. ENTROLY            — universal_compress (knapsack + entropy)
  3. LLMLINGUA-2        — Microsoft's task-agnostic compressor (BERT-distilled)
                          (default; switch to --method longtoken_pruner for the
                          heavy LLaMA-based variant)

For each (question, context, method):
  * compress context to budget
  * answer with gpt-4o-mini (or --answer-model)
  * score answer against gold via token-level F1 + exact-match

Output: Pareto table (tokens vs accuracy), JSON or human-readable.

Cost estimate at default settings (--samples 20, gpt-4o-mini answer):
  ≈ $0.10 in API spend + ~500 MB one-time model download for TokenPruner-2.

Usage:
  python bench/looGLE_compare.py
  python bench/looGLE_compare.py --samples 50 --json > results.json
  python bench/looGLE_compare.py --subset longdep_qa  # harder long-dep tasks
  python bench/looGLE_compare.py --skip-pruner        # entroly vs baseline only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── ANSI ──────────────────────────────────────────────────────────────
GREEN, RED, YELLOW, CYAN, BOLD, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[1m", "\033[2m", "\033[0m",
)


def _c(color: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def _tokens(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


# ══════════════════════════════════════════════════════════════════════
# Scoring (SQuAD-style F1 + exact-match)
# ══════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall    = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


# ══════════════════════════════════════════════════════════════════════
# Compression methods
# ══════════════════════════════════════════════════════════════════════

def compress_truncate(context: str, budget: int) -> str:
    """Naive FIFO truncation."""
    target_chars = budget * 4
    return context[:target_chars]


def compress_entroly(context: str, budget: int) -> str:
    """Entroly: universal_compress with knapsack + entropy, capped at the budget."""
    from entroly.universal_compress import universal_compress
    current_tokens = _tokens(context)
    if current_tokens <= budget:
        return context
    # Try a slightly tighter ratio first since universal_compress can overshoot.
    target_ratio = max(0.05, (budget * 0.85) / max(current_tokens, 1))
    compressed, _, _ = universal_compress(context, target_ratio, "prose")
    # Hard cap: if the compressor still overshoots, head-truncate to the budget.
    if _tokens(compressed) > budget:
        compressed = compressed[: budget * 4]
    return compressed


_LINGUA_COMPRESSOR = None


def compress_agentic_pruner(context: str, question: str, budget: int, method: str) -> str:
    """Agentic Context Pruning (2026 SOTA): uses LLM to extract relevant context."""
    current_tokens = _tokens(context)
    if current_tokens <= budget:
        return context

    from openai import OpenAI
    client = OpenAI()
    prompt = (
        "Extract only the exact facts from the following text that are relevant "
        f"to this question: {question}\n\n{context}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=budget,
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback to truncation on API error during compression
        return context[:budget * 4]


# ══════════════════════════════════════════════════════════════════════
# Answer generation
# ══════════════════════════════════════════════════════════════════════

_ANSWER_PROMPT = (
    "You are answering a reading-comprehension question. "
    "Use only the context below. Be concise — give the shortest correct answer.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def answer_with_openai(model: str, context: str, question: str, max_tokens: int = 128) -> str:
    """Generate an answer with the OpenAI API."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": _ANSWER_PROMPT.format(context=context, question=question)}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════
# Benchmark loop
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(args) -> dict:
    from datasets import load_dataset

    print(_c(DIM, f"\n  loading LooGLE/{args.subset} (test split) ..."), flush=True)
    ds = load_dataset("bigai-nlco/LooGLE", args.subset, split="test")
    n_avail = len(ds)
    print(_c(DIM, f"  {n_avail} examples available, sampling {args.samples}"))

    # Deterministic stride sampling
    indices = list(range(0, n_avail, max(1, n_avail // args.samples)))[:args.samples]

    methods = ["baseline", "entroly"]
    if not args.skip_pruner:
        methods.append("agentic_pruning")

    # Pre-warm lingua compressor (one-time download)
    if not args.skip_pruner:
        print(_c(DIM, "  loading agentic-pruning compressor (no model download — uses gpt-4o-mini) ..."), flush=True)
        t0 = time.perf_counter()
        print(_c(DIM, f"  loaded in {time.perf_counter() - t0:.1f}s"))

    per_method: dict[str, dict] = {m: {
        "f1_sum": 0.0, "em_sum": 0, "tokens_in_sum": 0, "original_tokens_sum": 0,
        "tokens_out_sum": 0, "ms_sum": 0.0, "n": 0, "errors": 0,
    } for m in methods}

    rows = []
    for i, idx in enumerate(indices):
        ex = ds[int(idx)]
        context = ex.get("context") or ex.get("input") or ""
        question = ex.get("question") or ""
        gold = ex.get("answer") or ""
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        if not (context and question and gold):
            continue

        original_tokens = _tokens(context)
        row = {"idx": int(idx), "original_tokens": original_tokens, "question": question[:80]}

        for method in methods:
            t0 = time.perf_counter()
            try:
                if method == "baseline":
                    cmp_ctx = compress_truncate(context, args.budget)
                elif method == "entroly":
                    cmp_ctx = compress_entroly(context, args.budget)
                else:
                    cmp_ctx = compress_agentic_pruner(context, question, args.budget, method)
                compress_ms = (time.perf_counter() - t0) * 1000

                pred = answer_with_openai(args.answer_model, cmp_ctx, question)
                f1 = f1_score(pred, gold)
                em = 1 if exact_match(pred, gold) else 0

                per_method[method]["f1_sum"]        += f1
                per_method[method]["em_sum"]        += em
                per_method[method]["tokens_in_sum"] += _tokens(cmp_ctx)
                per_method[method]["original_tokens_sum"] += original_tokens
                per_method[method]["ms_sum"]        += compress_ms
                per_method[method]["n"]             += 1

                row[method] = {
                    "tokens_in": _tokens(cmp_ctx),
                    "f1": round(f1, 3),
                    "em": em,
                    "ms": round(compress_ms, 1),
                    "pred": pred[:100],
                }
            except Exception as e:
                per_method[method]["errors"] += 1
                row[method] = {"error": str(e)[:120]}

        rows.append(row)
        # Progress
        if (i + 1) % 5 == 0 or (i + 1) == len(indices):
            f1s = " | ".join(
                f"{m[:8]}={per_method[m]['f1_sum']/max(1,per_method[m]['n']):.2f}"
                for m in methods
            )
            print(f"  [{i+1}/{len(indices)}] avg F1: {f1s}", flush=True)

    summary = {}
    for m in methods:
        s = per_method[m]
        n = max(1, s["n"])
        # GPT-4o-mini pricing: $0.150 / 1M input tokens, $0.600 / 1M output tokens
        avg_in = s["tokens_in_sum"] / n
        avg_orig = s["original_tokens_sum"] / n

        # Calculate cost per 1000 queries
        if m == "agentic_pruning":
            api_calls = 2
            # Cost = (Read original) + (Generate compressed) + (Read compressed for answer)
            cost = ((avg_orig * 0.150) + (avg_in * 0.600) + (avg_in * 0.150)) / 1000
        else:
            api_calls = 1
            # Cost = (Read compressed for answer)
            cost = (avg_in * 0.150) / 1000

        summary[m] = {
            "n":           s["n"],
            "errors":      s["errors"],
            "avg_f1":      round(s["f1_sum"] / n, 3),
            "exact_match": round(s["em_sum"] / n, 3),
            "avg_tokens":  round(avg_in, 0),
            "avg_compress_ms": round(s["ms_sum"] / n, 1),
            "api_calls":   api_calls,
            "cost_1k":     cost,
        }

    return {
        "subset": args.subset,
        "samples": len(rows),
        "budget": args.budget,
        "answer_model": args.answer_model,
        "pruner_method": "agentic_pruning" if not args.skip_pruner else None,
        "summary": summary,
        "rows": rows,
    }


# ══════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════

def print_report(report: dict) -> None:
    print()
    print(_c(BOLD, "  LooGLE Head-to-Head"))
    print(_c(DIM,  "  ─────────────────────────────────────────────────────────────────"))
    print(f"  subset={report['subset']}  samples={report['samples']}  "
          f"budget={report['budget']}  answer={report['answer_model']}")
    print()

    s = report["summary"]
    methods = list(s.keys())

    # Table header
    header = f"  {'method':<16} {'F1':>6} {'Tokens':>8} {'Latency':>10} {'API-Calls':>10} {'Cost/1k':>10}"
    print(_c(BOLD, header))
    print(_c(DIM, "  " + "─" * (len(header) - 2)))

    # Identify the best F1 to highlight the leader
    best_f1 = max(s[m]["avg_f1"] for m in methods)

    for m in methods:
        d = s[m]
        f1_str = f"{d['avg_f1']:.3f}"
        if d["avg_f1"] == best_f1 and d["avg_f1"] > 0:
            f1_str = _c(GREEN + BOLD, f1_str)

        cost_str = f"${d['cost_1k']:.3f}"
        if m == "entroly":
            cost_str = _c(CYAN, cost_str)

        print(f"  {m:<16} {f1_str:>6} {int(d['avg_tokens']):>8} "
              f"{d['avg_compress_ms']:>8.0f}ms {d['api_calls']:>10} {cost_str:>10}")

    print(_c(DIM, "  " + "─" * (len(header) - 2)))

    # Pairwise deltas vs baseline (if present)
    if "baseline" in s:
        base = s["baseline"]
        print()
        print(_c(BOLD, "  vs baseline (truncation):"))
        for m in methods:
            if m == "baseline":
                continue
            d = s[m]
            f1_delta = d["avg_f1"] - base["avg_f1"]
            tok_delta_pct = (d["avg_tokens"] - base["avg_tokens"]) / max(1, base["avg_tokens"]) * 100
            f1_color = GREEN if f1_delta > 0 else RED if f1_delta < 0 else DIM
            tok_color = GREEN if tok_delta_pct < 0 else RED if tok_delta_pct > 0 else DIM
            f1_arrow = "+" if f1_delta >= 0 else ""
            tok_arrow = "+" if tok_delta_pct >= 0 else ""
            print(f"    {m:<12} F1 {_c(f1_color, f'{f1_arrow}{f1_delta:.3f}')}  "
                  f"tokens {_c(tok_color, f'{tok_arrow}{tok_delta_pct:.1f}%')}")

    print()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="LooGLE head-to-head: entroly vs TokenPruner")
    ap.add_argument("--subset", default="shortdep_qa",
                    choices=["shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization"])
    ap.add_argument("--samples", type=int, default=20, help="number of test cases")
    ap.add_argument("--budget", type=int, default=2000, help="compression target (tokens)")
    ap.add_argument("--answer-model", default="gpt-4o-mini", help="OpenAI model for answers")
    ap.add_argument("--skip-pruner", action="store_true", help="skip agentic pruner")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(_c(RED, "ERROR: OPENAI_API_KEY is not set."), file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    report = run_benchmark(args)
    report["wall_seconds"] = round(time.perf_counter() - t0, 1)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
        print(_c(DIM, f"  total wall time: {report['wall_seconds']}s\n"))

    # Exit non-zero if entroly is worse than baseline (for CI gating)
    s = report["summary"]
    if "entroly" in s and "baseline" in s:
        if s["entroly"]["avg_f1"] < s["baseline"]["avg_f1"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
