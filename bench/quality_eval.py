#!/usr/bin/env python3
"""End-to-end answer-quality eval: baseline (32K grep-dump) vs Entroly.

For each task:
  1. Gather baseline context by grepping the repo for query terms, ranking files
     by match count, and concatenating top files up to a ~32K-token budget.
  2. Gather Entroly context via `entroly optimize --task "<q>" --budget 4096`.
  3. Ask gpt-4o-mini (temp=0) to answer the task with each context.
  4. Have gpt-4o judge each answer against ground truth on correctness (1-5)
     and specificity (1-5).
  5. Report per-task and aggregate deltas.

Usage: OPENAI_API_KEY=sk-... python bench/quality_eval.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from entroly.universal_compress import universal_compress  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ANSWERER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"
BASELINE_TOKEN_BUDGET = 32_000
ENTROLY_TOKEN_BUDGET = 4_096
COMPRESSED_TARGET_TOKENS = 4_096  # universal_compress target for 32K baseline
CHARS_PER_TOKEN = 4  # cheap estimate

TASKS = [
    {
        "id": "knapsack_selector",
        "query": "How does Entroly's knapsack selector decide which fragments to include?",
        "ground_truth": (
            "Entroly uses 0/1 knapsack-style selection. When the problem size "
            "fits (fragments × budget small enough), it runs an exact dynamic-"
            "programming 0/1 knapsack over (score, tokens). When DP is "
            "infeasible, it falls back to a density-greedy approximation: sort "
            "fragments by score/tokens and take them in order until the budget "
            "is exhausted. Knapsack is NP-hard, so both paths are "
            "approximations (DP is pseudo-polynomial; greedy is a heuristic)."
        ),
    },
    {
        "id": "sast_rule_tiers",
        "query": "How does the SAST engine distinguish taint-aware rules from pattern-only rules?",
        "ground_truth": (
            "Each rule has a boolean `taint_flow` field. During scanning, every "
            "finding inherits that flag. `security_report()` counts findings "
            "into `taint_flow_total` vs `pattern_only_total` and returns both "
            "in the JSON output. Of ~151 rules, roughly 46 are taint-aware and "
            "~105 are pattern-only. The distinction matters because taint-aware "
            "rules track data flow from source to sink (higher precision) while "
            "pattern-only rules match syntactic patterns (higher recall, more "
            "false positives)."
        ),
    },
    {
        "id": "jaccard_contradiction",
        "query": "What does the Jaccard contradiction guard compare between two fragments?",
        "ground_truth": (
            "Path tokens. The guard splits each fragment's source path on the "
            "delimiters `/`, `\\`, `.`, `_`, `-`, `:` and computes Jaccard "
            "similarity (|A∩B| / |A∪B|) over the resulting token sets. Two "
            "fragments with a high Jaccard score are considered structurally "
            "related and eligible for the contradiction check; unrelated paths "
            "are skipped to avoid false-positive contradictions across the "
            "repo. The tokenizer lives in entroly-core/src/channel.rs."
        ),
    },
    {
        "id": "context_score_formula",
        "query": "What is the formula for the Context Score shown by `entroly share`?",
        "ground_truth": (
            "The Context Score is the geometric mean of three per-query factors: "
            "stability, coverage, and respect — i.e. (stability · coverage · "
            "respect)^(1/3), scaled to 0–100. It's computed as exp(mean(log x_i)) "
            "over all per-query factor values, which is equivalent to a geometric "
            "mean. The geometric mean penalizes any factor going to zero, unlike "
            "an arithmetic mean."
        ),
    },
    {
        "id": "feedback_command",
        "query": "What does `entroly feedback --outcome success` actually do?",
        "ground_truth": (
            "It reads ~/.entroly/last_optimize.json, which the previous "
            "`entroly optimize` call wrote. That file contains the fragment IDs "
            "selected and the task string. `--outcome success` maps to a score "
            "of 1.0. The command calls EntrolyEngine.record_success(fragment_ids), "
            "which nudges those fragments' relevance scores upward so future "
            "optimizations weight them more heavily. `fail` / `bad` map to 0.0 "
            "(record_failure), `neutral` to 0.5."
        ),
    },
]

STOPWORDS = set("""the a an of to in on for with by how does what why is are and or
fragment fragments context do between two include shown actual actually""".split())


def tokenize_query(q: str) -> list[str]:
    terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", q.lower())
    return [t for t in terms if len(t) > 2 and t not in STOPWORDS]


def gather_baseline_context(query: str, budget_tokens: int) -> str:
    """Rank repo files by query-term match count; concat top files until budget."""
    terms = tokenize_query(query)
    if not terms:
        return ""

    candidates = list(REPO.rglob("*.py")) + list(REPO.rglob("*.rs")) + list(REPO.rglob("*.md"))
    candidates = [p for p in candidates if not any(seg in p.parts for seg in (".git", "node_modules", "target", "__pycache__", ".entroly"))]
    # Exclude the eval harness itself — it contains the ground-truth answers
    # and would trivially dominate any scorer.
    candidates = [p for p in candidates if "quality_eval" not in p.name]

    scores: Counter[Path] = Counter()
    contents: dict[Path, str] = {}
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lo = text.lower()
        hits = sum(lo.count(t) for t in terms)
        if hits > 0:
            scores[p] = hits
            contents[p] = text

    ranked = [p for p, _ in scores.most_common()]

    budget_chars = budget_tokens * CHARS_PER_TOKEN
    out_parts: list[str] = []
    used = 0
    for p in ranked:
        text = contents[p]
        header = f"\n\n===== {p.relative_to(REPO).as_posix()} =====\n"
        remaining = budget_chars - used - len(header)
        if remaining <= 500:
            break
        chunk = text[:remaining]
        out_parts.append(header + chunk)
        used += len(header) + len(chunk)
    return "".join(out_parts)


def gather_entroly_context(query: str) -> str:
    env = os.environ.copy()
    env["ENTROLY_TELEMETRY"] = "0"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        r = subprocess.run(
            [
                "entroly", "optimize",
                "--task", query,
                "--budget", str(ENTROLY_TOKEN_BUDGET),
                "--selector", "qccr",
                "--exclude", "quality_eval",
                "--quiet",
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return ""
    if r.returncode != 0:
        print(f"[entroly error] {r.stderr[-400:]}", file=sys.stderr)
        return ""
    out = r.stdout
    # Strip log lines that the cli emits before the snapshot.
    lines = [ln for ln in out.splitlines() if not ln.startswith(("2026-", "2025-"))]
    return "\n".join(lines)


def est_tokens(s: str) -> int:
    return len(s) // CHARS_PER_TOKEN


def answer(client: OpenAI, query: str, context: str) -> str:
    prompt = (
        "You are answering a technical question about a codebase called 'entroly'. "
        "Answer ONLY from the provided context. If the context doesn't contain the "
        "answer, say so honestly. Be specific: cite function names, field names, and "
        "formulas when possible.\n\n"
        f"=== CONTEXT ===\n{context}\n\n=== QUESTION ===\n{query}\n"
    )
    r = client.chat.completions.create(
        model=ANSWERER_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content or ""


def judge(client: OpenAI, query: str, ground_truth: str, ans: str) -> dict:
    prompt = (
        "You are judging a candidate answer against a ground-truth reference.\n\n"
        f"QUESTION: {query}\n\n"
        f"GROUND TRUTH:\n{ground_truth}\n\n"
        f"CANDIDATE ANSWER:\n{ans}\n\n"
        "Rate the candidate on two dimensions, integer 1-5:\n"
        "  correctness — does it match the ground truth's factual claims?\n"
        "    5 = fully correct, no errors; 3 = partially correct, some wrong/missing; 1 = wrong or hallucinated.\n"
        "  specificity — does it name the right functions/fields/formulas rather than being vague?\n"
        "    5 = names all key identifiers; 3 = names some; 1 = entirely vague or hand-wavy.\n\n"
        "Respond ONLY with JSON: "
        '{"correctness": <1-5>, "specificity": <1-5>, "reasoning": "<one sentence>"}'
    )
    r = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(r.choices[0].message.content or "{}")


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    client = OpenAI()
    rows = []
    for task in TASKS:
        print(f"\n--- {task['id']} ---")
        print(f"Q: {task['query']}")

        baseline_ctx = gather_baseline_context(task["query"], BASELINE_TOKEN_BUDGET)
        entroly_ctx = gather_entroly_context(task["query"])

        # Third condition: baseline 32K → universal_compress → ~4K.
        # Target ratio chosen so the compressed output roughly matches the
        # Entroly-retrieval budget — apples-to-apples on output tokens.
        target_ratio = COMPRESSED_TARGET_TOKENS / max(est_tokens(baseline_ctx), 1)
        target_ratio = max(0.05, min(1.0, target_ratio))
        compressed_ctx, _detected, _savings = universal_compress(
            baseline_ctx, target_ratio=target_ratio, content_type="prose",
        )

        b_tok = est_tokens(baseline_ctx)
        e_tok = est_tokens(entroly_ctx)
        c_tok = est_tokens(compressed_ctx)
        print(f"  baseline    context: ~{b_tok} tokens")
        print(f"  entroly     context: ~{e_tok} tokens")
        print(f"  compressed  context: ~{c_tok} tokens (from {b_tok})")

        if not entroly_ctx:
            print("  [skip] entroly returned nothing")
            continue

        baseline_ans = answer(client, task["query"], baseline_ctx)
        entroly_ans = answer(client, task["query"], entroly_ctx)
        compressed_ans = answer(client, task["query"], compressed_ctx)

        b_judge = judge(client, task["query"], task["ground_truth"], baseline_ans)
        e_judge = judge(client, task["query"], task["ground_truth"], entroly_ans)
        c_judge = judge(client, task["query"], task["ground_truth"], compressed_ans)

        rows.append({
            "id": task["id"],
            "baseline_tokens": b_tok,
            "entroly_tokens": e_tok,
            "compressed_tokens": c_tok,
            "baseline": b_judge,
            "entroly": e_judge,
            "compressed": c_judge,
        })

        print(f"  baseline   : correctness={b_judge.get('correctness')} specificity={b_judge.get('specificity')} — {b_judge.get('reasoning','')[:120]}")
        print(f"  entroly    : correctness={e_judge.get('correctness')} specificity={e_judge.get('specificity')} — {e_judge.get('reasoning','')[:120]}")
        print(f"  compressed : correctness={c_judge.get('correctness')} specificity={c_judge.get('specificity')} — {c_judge.get('reasoning','')[:120]}")

    print("\n\n==== SUMMARY ====")
    print(f"{'task':30} {'b.tok':>7} {'e.tok':>7} {'c.tok':>7} {'b.corr':>6} {'e.corr':>6} {'c.corr':>6} {'b.spec':>6} {'e.spec':>6} {'c.spec':>6}")
    totals = {"b_tok": 0, "e_tok": 0, "c_tok": 0,
              "b_corr": 0, "e_corr": 0, "c_corr": 0,
              "b_spec": 0, "e_spec": 0, "c_spec": 0}
    for r in rows:
        bc = r["baseline"].get("correctness", 0)
        ec = r["entroly"].get("correctness", 0)
        cc = r["compressed"].get("correctness", 0)
        bs = r["baseline"].get("specificity", 0)
        es = r["entroly"].get("specificity", 0)
        cs = r["compressed"].get("specificity", 0)
        print(f"{r['id']:30} {r['baseline_tokens']:>7} {r['entroly_tokens']:>7} {r['compressed_tokens']:>7} {bc:>6} {ec:>6} {cc:>6} {bs:>6} {es:>6} {cs:>6}")
        totals["b_tok"] += r["baseline_tokens"]
        totals["e_tok"] += r["entroly_tokens"]
        totals["c_tok"] += r["compressed_tokens"]
        totals["b_corr"] += bc
        totals["e_corr"] += ec
        totals["c_corr"] += cc
        totals["b_spec"] += bs
        totals["e_spec"] += es
        totals["c_spec"] += cs

    n = len(rows) or 1
    print()
    print(f"avg tokens       baseline={totals['b_tok']//n}  entroly={totals['e_tok']//n}  compressed={totals['c_tok']//n}")
    print(f"avg correctness  baseline={totals['b_corr']/n:.2f}  entroly={totals['e_corr']/n:.2f}  compressed={totals['c_corr']/n:.2f}")
    print(f"avg specificity  baseline={totals['b_spec']/n:.2f}  entroly={totals['e_spec']/n:.2f}  compressed={totals['c_spec']/n:.2f}")
    print(f"delta vs baseline    entroly corr={(totals['e_corr']-totals['b_corr'])/n:+.2f}  compressed corr={(totals['c_corr']-totals['b_corr'])/n:+.2f}")

    out_file = REPO / "bench" / "quality_eval_results.json"
    out_file.write_text(json.dumps({"tasks": rows, "totals": totals, "n": n}, indent=2))
    print(f"\nResults saved: {out_file.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
