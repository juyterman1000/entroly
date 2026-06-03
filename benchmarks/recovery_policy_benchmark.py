"""Measure bounded exact-replay capacity on public extractive QA contexts.

This is an offline evidence-survival diagnostic, not an LLM accuracy claim.
It answers a narrower question:

    When query-conditioned compression drops an answer span, how often can a
    bounded exact replay or marked exact excerpts of the already-selected source
    recover that evidence?

Gold answers are used only for scoring and for the oracle-triggered upper bound.
They are never passed into selection or recovery ranking. Run the real LLM QA
benchmarks separately for end-to-end answer accuracy.

Usage:
    python benchmarks/recovery_policy_benchmark.py --benchmark all
    python benchmarks/recovery_policy_benchmark.py --benchmark squad --samples 200
    python benchmarks/recovery_policy_benchmark.py --benchmark longbench --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from entroly.qccr import select as qccr_select  # noqa: E402
from entroly.ccr import slice_recovery_content  # noqa: E402

CACHE = ROOT / "bench" / ".cache"


def _approx_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def _contains_answer(text: str, answers: list[str]) -> bool:
    lowered = text.lower()
    return any(answer and answer.lower() in lowered for answer in answers)


def chunk_context(text: str, size: int = 400, group_size: int = 8) -> list[dict]:
    """Split a document into source groups that can be replayed exactly."""
    pieces = [text[i:i + size] for i in range(0, len(text), size)]
    return [
        {
            "id": f"f{i}",
            "source": f"chunk_{i // group_size}.txt",
            "content": piece,
            "tokens": _approx_tokens(piece),
        }
        for i, piece in enumerate(pieces)
        if piece
    ]


def compress_with_exact_replay_candidates(
    context: str,
    question: str,
    *,
    token_budget: int,
    recovery_token_budget: int,
) -> tuple[str, str]:
    """Compress by query, then build a bounded exact replay of ranked sources.

    The replay pool expands the query-conditioned sources selected in pass one.
    Oversized originals contribute marked exact lead and query-local excerpts,
    matching the live retry path while preserving strict token accounting.
    This isolates the resolution-loss recovery ceiling. The live proxy also
    registers native query-ranked omitted candidates to repair selection misses.
    """
    fragments = chunk_context(context)
    selected = qccr_select(fragments, token_budget=token_budget, query=question)
    compressed = "\n".join(fragment.get("content", "") for fragment in selected)

    originals_by_source: dict[str, str] = {}
    for fragment in fragments:
        source = fragment["source"]
        originals_by_source[source] = (
            originals_by_source.get(source, "") + fragment["content"]
        )

    replay_parts: list[str] = []
    replay_tokens = 0
    seen: set[str] = set()
    for source in (fragment.get("source", "") for fragment in selected):
        if not source or source in seen:
            continue
        seen.add(source)
        original = originals_by_source.get(source, "")
        remaining_tokens = recovery_token_budget - replay_tokens
        if not original or remaining_tokens <= 0:
            break
        tokens = max(1, (len(original) + 3) // 4)
        if tokens > remaining_tokens:
            original, _ = slice_recovery_content(
                original,
                question,
                remaining_tokens,
            )
            tokens = max(1, (len(original) + 3) // 4) if original else 0
        if not original or tokens > remaining_tokens:
            continue
        replay_parts.append(f"## {source}\n{original}")
        replay_tokens += tokens

    return compressed, "\n\n".join(replay_parts)


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    token_budget: int,
    recovery_token_budget: int,
) -> dict[str, Any]:
    """Return evidence-survival and token-cost metrics for extractive QA cases."""
    first_pass_survived = 0
    recovered = 0
    unresolved = 0
    original_tokens: list[int] = []
    compressed_tokens: list[int] = []
    retry_expansion_tokens: list[int] = []

    for case in cases:
        context = str(case.get("context") or "")
        question = str(case.get("question") or "")
        raw_answers = case.get("answers") or []
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        answers = [str(answer) for answer in raw_answers if answer]
        if not context or not question or not answers:
            continue

        compressed, replay = compress_with_exact_replay_candidates(
            context,
            question,
            token_budget=token_budget,
            recovery_token_budget=recovery_token_budget,
        )
        original_tokens.append(_approx_tokens(context))
        compressed_tokens.append(_approx_tokens(compressed))

        if _contains_answer(compressed, answers):
            first_pass_survived += 1
            retry_expansion_tokens.append(0)
            continue

        expansion_tokens = _approx_tokens(replay)
        retry_expansion_tokens.append(expansion_tokens)
        if _contains_answer(f"{compressed}\n{replay}", answers):
            recovered += 1
        else:
            unresolved += 1

    total = first_pass_survived + recovered + unresolved
    avg_original = sum(original_tokens) / total if total else 0.0
    avg_compressed = sum(compressed_tokens) / total if total else 0.0
    avg_expansion = sum(retry_expansion_tokens) / total if total else 0.0
    effective_tokens = avg_compressed + avg_expansion
    return {
        "metric_scope": "offline_extractable_evidence_survival",
        "trigger": "oracle_answer_span_miss_upper_bound",
        "total": total,
        "token_budget": token_budget,
        "recovery_token_budget": recovery_token_budget,
        "first_pass_survived": first_pass_survived,
        "recovered": recovered,
        "unresolved": unresolved,
        "first_pass_survival": first_pass_survived / total if total else 0.0,
        "bounded_recovery_survival": (
            (first_pass_survived + recovered) / total if total else 0.0
        ),
        "retry_rate": (recovered + unresolved) / total if total else 0.0,
        "avg_original_tokens": round(avg_original, 2),
        "avg_compressed_tokens": round(avg_compressed, 2),
        "avg_retry_expansion_tokens": round(avg_expansion, 2),
        "first_pass_token_savings_pct": round(
            100.0 * (1.0 - avg_compressed / avg_original), 2
        ) if avg_original else 0.0,
        "effective_token_savings_pct": round(
            100.0 * (1.0 - effective_tokens / avg_original), 2
        ) if avg_original else 0.0,
    }


def _load_cases(name: str, samples: int) -> list[dict[str, Any]]:
    path = CACHE / (
        "longbench_hotpotqa.json" if name == "longbench" else "squad.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the public QA benchmark once to populate cache."
        )
    return json.loads(path.read_text(encoding="utf-8"))[:samples]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("squad", "longbench", "all"), default="all")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--recovery-budget", type=int, default=1200)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    names = ("squad", "longbench") if args.benchmark == "all" else (args.benchmark,)
    defaults = {"squad": 100, "longbench": 2000}
    results: dict[str, Any] = {}
    for name in names:
        result = evaluate_cases(
            _load_cases(name, args.samples),
            token_budget=args.budget or defaults[name],
            recovery_token_budget=args.recovery_budget,
        )
        results[name] = result

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print("Offline extractable-evidence recovery upper bound")
    print("Gold answers score survival only; they never influence selection.")
    for name, result in results.items():
        print(
            f"{name:10s} n={result['total']:>4d}  "
            f"first={result['first_pass_survival']:.1%}  "
            f"bounded-replay={result['bounded_recovery_survival']:.1%}  "
            f"retry={result['retry_rate']:.1%}  "
            f"effective-savings={result['effective_token_savings_pct']:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
