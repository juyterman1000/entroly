"""End-to-end LLM validation: LongBench HotpotQA with the anchor compressor.

Mirrors `bench.accuracy --benchmark longbench` evaluation but swaps the
compressor for `benchmarks.anchor_compress.compress`. Tests whether the +4pp
survival improvement we measured locally translates to LLM accuracy.

Run:
    python bench/_validate_anchor_longbench.py [N=50]
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

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed", file=sys.stderr)
    raise SystemExit(2)


SEED = 42
BUDGET = 2000
MODEL = "gpt-4o-mini"
CACHE = ROOT / "bench" / ".cache" / "longbench_hotpotqa.json"

PROMPT_TEMPLATE = (
    "Answer the question using ONLY the given context. Output the "
    "answer as a short span (a few words max), nothing else.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _ask(client: OpenAI, context: str, question: str, max_tokens: int = 64) -> str:
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
    return "__error__"


def _normalize(s: str) -> str:
    s = re.sub(r"[\W_]+", " ", s.lower()).strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _correct(prediction: str, acceptable: list[str]) -> bool:
    p = _normalize(prediction)
    for a in acceptable:
        an = _normalize(str(a))
        if an and (p == an or an in p or p in an):
            return True
    return False


def main() -> int:
    random.seed(SEED)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    items = json.load(open(CACHE, encoding="utf-8"))[:n]
    print(f"\nValidating ANCHOR on LongBench: n={n}, model={MODEL}, budget={BUDGET}\n",
          flush=True)

    client = OpenAI()
    base_correct = 0
    anch_correct = 0
    base_tokens = []
    anch_tokens = []

    for i, it in enumerate(items, 1):
        ctx = it["context"]
        q = it["question"]
        answers = it["answers"]

        # baseline (full context)
        pred_b = _ask(client, ctx, q)
        if not pred_b.startswith("__error__") and _correct(pred_b, answers):
            base_correct += 1
        base_tokens.append(len(ctx) // 4)

        # anchor
        ctx_a = anchor_compress(ctx, BUDGET, q)
        pred_a = _ask(client, ctx_a, q)
        if not pred_a.startswith("__error__") and _correct(pred_a, answers):
            anch_correct += 1
        anch_tokens.append(len(ctx_a) // 4)

        if i % 10 == 0:
            print(f"  [{i}/{n}] baseline={base_correct}  anchor={anch_correct}",
                  flush=True)

    total = len(items)
    avg_b = sum(base_tokens) / len(base_tokens)
    avg_a = sum(anch_tokens) / len(anch_tokens)
    sav = (1 - avg_a / avg_b) * 100
    print()
    print(f"  n              = {total}")
    print(f"  baseline acc   = {base_correct/total:.1%}  ({base_correct}/{total})")
    print(f"  anchor acc     = {anch_correct/total:.1%}  ({anch_correct}/{total})")
    print(f"  retention      = {anch_correct/base_correct:.1%}" if base_correct else "  retention      = n/a")
    print(f"  avg base tok   = {avg_b:.1f}")
    print(f"  avg anchor tok = {avg_a:.1f}")
    print(f"  savings        = {sav:.1f}%")
    print("  README ref     : baseline=64.0% / entroly=68.0% / retention=106.2% / savings=85.3%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
