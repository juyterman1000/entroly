"""Local answer-survival diagnostic on all benchmarks that have a string answer.

For each benchmark, measure how often the gold answer appears in the
compressed context — QCCR vs anchor. Zero LLM calls. Predicts where
LLM accuracy gains are achievable.

Note: BFCL doesn't have a string answer in context (the gold is a
function call), so it's skipped here.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from entroly.qccr import select as qccr_select  # noqa: E402
from benchmarks.anchor_compress import compress as anchor_compress  # noqa: E402

CACHE = ROOT / "bench" / ".cache"


def chunk(text: str, size: int = 400) -> list[dict]:
    pieces = [text[i:i + size] for i in range(0, len(text), size)]
    return [
        {"id": f"f{i}", "source": f"chunk_{i // 8}.txt",
         "content": p, "tokens": len(p) // 4}
        for i, p in enumerate(pieces)
    ]


def qccr_compress(context: str, question: str, budget: int) -> str:
    sel = qccr_select(chunk(context), token_budget=budget, query=question)
    return "\n".join((s.get("content") or "") for s in sel).strip()


def survival(items: list[tuple[str, str, list[str]]], compress_fn, budget: int) -> dict:
    survived = 0
    total = 0
    avg_tokens = []
    for ctx, q, answers in items:
        if not answers:
            continue
        total += 1
        compressed = compress_fn(ctx, q, budget)
        avg_tokens.append(len(compressed) // 4)
        if any(a and a.lower() in compressed.lower() for a in answers):
            survived += 1
    return {
        "survived": survived,
        "total": total,
        "rate": survived / total if total else 0,
        "avg_tokens": sum(avg_tokens) / len(avg_tokens) if avg_tokens else 0,
    }


def load_squad(n: int) -> list[tuple]:
    d = json.load(open(CACHE / "squad.json", encoding="utf-8"))[:n]
    return [(it["context"], it["question"],
             it["answers"] if isinstance(it["answers"], list) else [it["answers"]])
            for it in d]


def load_longbench(n: int) -> list[tuple]:
    d = json.load(open(CACHE / "longbench_hotpotqa.json", encoding="utf-8"))[:n]
    return [(it["context"], it["question"],
             it["answers"] if isinstance(it["answers"], list) else [it["answers"]])
            for it in d]


def load_needle(n: int) -> list[tuple]:
    """Mimic bench/accuracy.py's needle generator."""
    sys.path.insert(0, str(ROOT))
    from bench.accuracy import _generate_haystack
    needle = "The secret passphrase for Project Aurora is 'crystalline-nebula-7742'."
    items = []
    sizes = [4000, 8000, 16000]  # tokens
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    rng = random.Random(42)
    while len(items) < n:
        size = rng.choice(sizes) * 4  # chars
        depth = rng.choice(depths)
        hay = _generate_haystack(size, needle, depth)
        items.append((hay, "What is the secret passphrase for Project Aurora?",
                      ["'crystalline-nebula-7742'", "crystalline-nebula-7742"]))
    return items[:n]


BENCHMARKS = [
    ("squad",     load_squad,     100, 200),
    ("longbench", load_longbench, 2000, 50),
    ("needle",    load_needle,    2000, 20),
]


def main() -> int:
    print(f"\n{'bench':12s} {'budget':>7s} {'n':>4s}  "
          f"QCCR    survive   anchor    survive   delta")
    print("-" * 76)
    for name, loader, budget, n in BENCHMARKS:
        items = loader(n)
        q_res = survival(items, qccr_compress, budget)
        a_res = survival(items, lambda c, q, b: anchor_compress(c, b, q), budget)
        delta = a_res["rate"] - q_res["rate"]
        marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"{name:12s} {budget:>7d} {q_res['total']:>4d}  "
              f"{q_res['rate']:>5.1%}  ({q_res['survived']:>3d})    "
              f"{a_res['rate']:>5.1%}  ({a_res['survived']:>3d})   "
              f"{delta:+.1%} {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
