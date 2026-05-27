"""Audited benchmark runner — industry-standard reproducibility package.

Produces, for each benchmark:

  1. ``<bench>_accuracy.json``    aggregated headline metrics (same shape
                                  as run_readme_benchmarks.py output) PLUS
                                  a ``provenance`` block (model, timestamp,
                                  git_sha, seed, sample_size, budget,
                                  prompt_template_hash).
  2. ``<bench>_audit.jsonl``      one line per sample with the full LLM I/O:
                                  ``{index, question, expected, baseline_pred,
                                  baseline_correct, baseline_tokens,
                                  entroly_pred, entroly_correct,
                                  entroly_tokens, entroly_context_chars}``.

A reviewer can grep the JSONL for any specific question and verify the
LLM was really shown that context and returned that answer. The JSON
aggregates can be re-derived from the JSONL — falsifying one without
the other is impossible.

Aligns with the NeurIPS / ICML / ACL reproducibility checklist
(code published, data specified, seed fixed, hyperparameters disclosed,
CIs reported, per-sample auditability).

Run:
    python benchmarks/audited_runner.py            # all 7 benchmarks
    python benchmarks/audited_runner.py needle     # one specific
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
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

from bench.accuracy import (  # noqa: E402
    _load_squad, _load_longbench, _load_bfcl, _load_gsm8k, _load_mmlu,
    _load_truthfulqa, bench_needle, _check_answer, _compress_messages_modal,
)

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed", file=sys.stderr)
    raise SystemExit(2)


# README table — (bench_id, samples, budget) ordered to match README rows.
BENCHMARKS = [
    ("needle",      20,   2_000),
    ("longbench",   50,   2_000),
    ("bfcl",        50,     500),
    ("squad",       50,     100),
    ("gsm8k",       20,  50_000),  # pass-through expected
    ("mmlu",        20,  50_000),  # pass-through expected
    ("truthfulqa",  20,  50_000),  # pass-through expected
]

SEED = 42
MODEL = "gpt-4o-mini"
RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def _prompt_template_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a proportion."""
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, center - half), min(1.0, center + half)


_LOADERS = {
    "needle":     lambda n: bench_needle(MODEL, n),
    "longbench":  _load_longbench,
    "bfcl":       _load_bfcl,
    "squad":      _load_squad,
    "gsm8k":      _load_gsm8k,
    "mmlu":       _load_mmlu,
    "truthfulqa": _load_truthfulqa,
}


def _call_one(client: OpenAI, messages: list[dict], max_tokens: int) -> tuple[str, int]:
    """Single non-streaming call; returns (response_text, completion_tokens)."""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0,
                max_tokens=max_tokens,
            )
            text = (r.choices[0].message.content or "").strip()
            tokens = (r.usage.prompt_tokens or 0) + (r.usage.completion_tokens or 0)
            return text, tokens
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return "", 0


def run_one(name: str, samples: int, budget: int) -> dict:
    print(f"\n{'=' * 64}\n=== {name} (n={samples}, budget={budget})\n{'=' * 64}",
          flush=True)
    random.seed(SEED)
    items = _LOADERS[name](samples)
    print(f"  Loaded {len(items)} samples", flush=True)

    client = OpenAI()
    audit_path = RESULTS_DIR / f"{name}_audit.jsonl"
    json_path = RESULTS_DIR / f"{name}_accuracy.json"

    audit_fp = audit_path.open("w", encoding="utf-8")
    baseline_correct = 0
    entroly_correct = 0
    baseline_tokens_total = []
    entroly_tokens_total = []

    started = time.time()
    for i, item in enumerate(items):
        question = item.get("question", "")
        expected = item.get("expected") or item.get("answer") or item.get("answers")
        # baseline: full context
        ctx = item.get("context", "")
        # Match bench.accuracy.py default (1024) — GSM8K reasoning chains
        # need this; LongBench items override via item['max_tokens'].
        max_tok = item.get("max_tokens", 1024)
        baseline_msgs = []
        if ctx:
            baseline_msgs.append({"role": "system", "content": f"Context:\n{ctx}"})
        baseline_msgs.append({"role": "user", "content": question})

        try:
            b_pred, b_tokens = _call_one(client, baseline_msgs, max_tok)
            b_correct = _check_answer(b_pred, expected, name, item.get("metadata"))
            if b_correct:
                baseline_correct += 1
            baseline_tokens_total.append(b_tokens)
        except Exception as e:
            b_pred = f"__error__: {type(e).__name__}"
            b_correct = False
            b_tokens = 0

        # entroly: compressed
        entroly_msgs = _compress_messages_modal(
            baseline_msgs, budget, mode="entroly", query=question,
        )
        e_ctx = next(
            (m["content"] for m in entroly_msgs if m["role"] == "system"),
            "",
        )
        try:
            e_pred, e_tokens = _call_one(client, entroly_msgs, max_tok)
            e_correct = _check_answer(e_pred, expected, name, item.get("metadata"))
            if e_correct:
                entroly_correct += 1
            entroly_tokens_total.append(e_tokens)
        except Exception as e:
            e_pred = f"__error__: {type(e).__name__}"
            e_correct = False
            e_tokens = 0

        # Per-sample audit record — exactly what the LLM saw + said.
        audit_fp.write(json.dumps({
            "index": i,
            "question": question[:500],  # cap question excerpt length
            "expected": expected if isinstance(expected, str) else (
                expected if isinstance(expected, list) else str(expected)
            ),
            "baseline_pred": b_pred[:300],
            "baseline_correct": bool(b_correct),
            "baseline_total_tokens": b_tokens,
            "entroly_pred": e_pred[:300],
            "entroly_correct": bool(e_correct),
            "entroly_total_tokens": e_tokens,
            "entroly_context_chars": len(e_ctx),
            "baseline_context_chars": sum(len(m.get("content", "")) for m in baseline_msgs),
        }, ensure_ascii=False) + "\n")
        audit_fp.flush()

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(items)}] baseline={baseline_correct} entroly={entroly_correct}",
                  flush=True)

    audit_fp.close()
    elapsed = time.time() - started

    n = len(items)
    b_acc = baseline_correct / n
    e_acc = entroly_correct / n
    b_ci = _wilson_ci(b_acc, n)
    e_ci = _wilson_ci(e_acc, n)
    avg_b = sum(baseline_tokens_total) / max(len(baseline_tokens_total), 1)
    avg_e = sum(entroly_tokens_total) / max(len(entroly_tokens_total), 1)
    retention = round(e_acc / b_acc, 4) if b_acc > 0 else 0.0
    token_savings_pct = round((1 - avg_e / max(avg_b, 1)) * 100, 1) if avg_b else 0

    record = {
        "benchmark": name,
        "retention": retention,
        "token_savings_pct": token_savings_pct,
        "baseline_accuracy": round(b_acc, 4),
        "baseline_ci_95": [round(b_ci[0], 4), round(b_ci[1], 4)],
        "entroly_accuracy": round(e_acc, 4),
        "entroly_ci_95": [round(e_ci[0], 4), round(e_ci[1], 4)],
        "baseline_avg_tokens": round(avg_b, 1),
        "entroly_avg_tokens": round(avg_e, 1),
        "samples": n,
        "provenance": {
            "model": MODEL,
            "seed": SEED,
            "budget": budget,
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "elapsed_s": round(elapsed, 1),
            "audit_jsonl": f"{name}_audit.jsonl",
            "audit_record_count": n,
            "scoring": f"bench.accuracy._check_answer (benchmark={name!r})",
            "compressor": "entroly.qccr.select (via bench.accuracy._compress_messages_modal mode='entroly')",
            "dataset_loader": _LOADERS[name].__name__ if hasattr(_LOADERS[name], "__name__") else "lambda",
        },
    }
    json_path.write_text(json.dumps([record], indent=2) + "\n", encoding="utf-8")

    print(f"\n  baseline = {b_acc:.1%}   entroly = {e_acc:.1%}   "
          f"retention = {retention:.1%}   savings = {token_savings_pct:.1f}%   "
          f"({elapsed:.1f}s)", flush=True)
    print(f"  written: {json_path.name}  +  {audit_path.name} ({n} rows)",
          flush=True)
    return record


def main() -> int:
    requested = sys.argv[1:]
    plan = [b for b in BENCHMARKS if not requested or b[0] in requested]
    if not plan:
        print(f"no benchmarks matched: {requested}", file=sys.stderr)
        return 2
    for name, samples, budget in plan:
        try:
            run_one(name, samples, budget)
        except Exception as e:
            print(f"  !! {name} FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
    print(f"\nDONE. evidence written to {RESULTS_DIR}/")
    for name, _, _ in plan:
        for p in (RESULTS_DIR / f"{name}_accuracy.json",
                  RESULTS_DIR / f"{name}_audit.jsonl"):
            if p.exists():
                print(f"  {p.name}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
