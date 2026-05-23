"""End-to-end LLM validation: SQuAD accuracy with the anchor compressor.

Mirrors `bench/accuracy.py` SQuAD evaluation but swaps the compressor for
`benchmarks.anchor_compress.compress`. Tests whether the +2.5pp survival
improvement we measured locally translates to LLM accuracy gains.

  baseline:  full context        → gpt-4o-mini → score
  anchor:    anchor-compressed   → gpt-4o-mini → score

Uses the same scoring helper bench/accuracy.py uses, the same prompt
template, the same seed, and the same model — so the numbers are
directly comparable to `bench.accuracy --benchmark squad`.

Run:
    python bench/_validate_anchor_squad.py [N=50]
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"(?:export\s+)?([A-Z_]+)\s*=\s*['\"]?([^'\"\s]+)",
                     line.strip())
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2)


_load_env()
from benchmarks.anchor_compress import compress as anchor_compress  # noqa: E402
from bench.accuracy import _load_squad  # noqa: E402

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed", file=sys.stderr)
    raise SystemExit(2)


SEED = 42
BUDGET = 100
MODEL = "gpt-4o-mini"
PROMPT_TEMPLATE = (
    "Answer the question using ONLY the given context. Output the "
    "answer as a short span (a few words max), nothing else.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _ask(client: OpenAI, context: str, question: str, max_tokens: int = 40) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == 2:
                return f"__error__: {type(e).__name__}"
            time.sleep(1 + attempt)
    return "__error__: exhausted"


def _normalize(s: str) -> str:
    s = re.sub(r"[\W_]+", " ", s.lower()).strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _correct(prediction: str, acceptable: list[str]) -> bool:
    p = _normalize(prediction)
    for a in acceptable:
        an = _normalize(a)
        if an and (p == an or an in p or p in an):
            return True
    return False


def main() -> int:
    random.seed(SEED)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    items = _load_squad(n)
    print(f"\nValidating ANCHOR compressor on SQuAD: n={n}, model={MODEL}, budget={BUDGET}\n",
          flush=True)

    client = OpenAI()
    baseline_correct = 0
    anchor_correct = 0
    baseline_tokens = []
    anchor_tokens = []
    errors_b = 0
    errors_a = 0

    for i, it in enumerate(items, 1):
        ctx = it["context"]
        q = it["question"]
        answers = it.get("metadata", {}).get("all_answers")
        if not answers:
            answers = it.get("answer")
            answers = [answers] if isinstance(answers, str) else (answers or [])
        if not answers:
            continue

        # 1) baseline
        pred_b = _ask(client, ctx, q)
        if pred_b.startswith("__error__"):
            errors_b += 1
        elif _correct(pred_b, answers):
            baseline_correct += 1
        baseline_tokens.append(len(ctx) // 4)

        # 2) anchor
        ctx_a = anchor_compress(ctx, BUDGET, q)
        pred_a = _ask(client, ctx_a, q)
        if pred_a.startswith("__error__"):
            errors_a += 1
        elif _correct(pred_a, answers):
            anchor_correct += 1
        anchor_tokens.append(len(ctx_a) // 4)

        if i % 10 == 0:
            print(f"  [{i}/{n}] baseline={baseline_correct}  anchor={anchor_correct}",
                  flush=True)

    total = len(items)
    avg_b = sum(baseline_tokens) / len(baseline_tokens) if baseline_tokens else 0
    avg_a = sum(anchor_tokens) / len(anchor_tokens) if anchor_tokens else 0
    sav = (1 - avg_a / avg_b) * 100 if avg_b else 0

    print()
    print(f"  n                = {total}")
    print(f"  baseline acc     = {baseline_correct/total:.1%}  ({baseline_correct}/{total})  errors={errors_b}")
    print(f"  anchor acc       = {anchor_correct/total:.1%}  ({anchor_correct}/{total})  errors={errors_a}")
    print(f"  retention        = {anchor_correct/baseline_correct:.1%}" if baseline_correct else "  retention        = n/a")
    print(f"  avg baseline tok = {avg_b:.1f}")
    print(f"  avg anchor tok   = {avg_a:.1f}")
    print(f"  savings          = {sav:.1f}%")
    print("  README reference : baseline=78.0% / entroly=76.0% / retention=97.4% / savings=39.3%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
