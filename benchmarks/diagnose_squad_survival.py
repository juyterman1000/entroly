"""Local diagnosis of Entroly QCCR compression on SQuAD at budget=100.

Asks: when QCCR compresses a SQuAD passage to ~100 tokens, does the answer
span survive in the compressed text? No LLM calls — pure local analysis.

If answer-survival rate is low, the LLM is being asked to answer questions
whose evidence has been compressed away — that's the real bottleneck.

Run:
    python bench/_diagnose_squad.py [N=200]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entroly.qccr import select as qccr_select
from benchmarks.anchor_compress import compress as anchor_compress

CACHE = Path(__file__).resolve().parent / ".cache" / "squad.json"
BUDGET = 100


def chunk(text: str, size: int = 400) -> list[dict]:
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    return [
        {"id": f"f{i}", "source": f"chunk_{i // 8}.txt",
         "content": c, "tokens": len(c) // 4}
        for i, c in enumerate(chunks)
    ]


def qccr_compress(context: str, question: str) -> str:
    selected = qccr_select(chunk(context), token_budget=BUDGET, query=question)
    return "\n".join((s.get("content") or "") for s in selected).strip()


def anchor_compress_at_budget(context: str, question: str) -> str:
    return anchor_compress(context, BUDGET, question)


def evaluate(items: list[dict], compress_fn, label: str) -> dict:
    survived = 0
    lost = []
    avg_compressed_tokens = []
    avg_baseline_tokens = []

    for it in items:
        ctx = it["context"]
        q = it["question"]
        answers = it["answers"] if isinstance(it["answers"], list) else [it["answers"]]
        answers = [str(a) for a in answers if a]
        if not answers:
            continue

        compressed = compress_fn(ctx, q)
        avg_compressed_tokens.append(len(compressed) // 4)
        avg_baseline_tokens.append(len(ctx) // 4)

        if any(a.lower() in compressed.lower() for a in answers):
            survived += 1
        else:
            lost.append({
                "question": q[:80],
                "answers": answers[:2],
                "compressed_tokens": len(compressed) // 4,
            })

    total = survived + len(lost)
    return {
        "label": label,
        "total": total,
        "survived": survived,
        "lost_count": len(lost),
        "survival_rate": survived / total if total else 0,
        "avg_baseline_tokens": sum(avg_baseline_tokens) / len(avg_baseline_tokens) if avg_baseline_tokens else 0,
        "avg_compressed_tokens": sum(avg_compressed_tokens) / len(avg_compressed_tokens) if avg_compressed_tokens else 0,
        "lost_cases": lost,
    }


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    items = json.load(open(CACHE, encoding="utf-8"))[:n]
    print(f"\nDiagnosing SQuAD answer-survival at budget={BUDGET}, n={n}\n")

    for label, fn in [("QCCR (current)", qccr_compress),
                      ("Anchor (proposed)", anchor_compress_at_budget)]:
        r = evaluate(items, fn, label)
        print(f"{label}:")
        print(f"  survived = {r['survived']}/{r['total']} = {r['survival_rate']:.1%}")
        print(f"  avg compressed tokens = {r['avg_compressed_tokens']:.1f}")
        print(f"  lost count = {r['lost_count']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
