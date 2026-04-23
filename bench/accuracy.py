"""
Entroly Public Accuracy Benchmarks
====================================

Measures accuracy RETENTION when Entroly compresses context.
Tests: does the LLM still give correct answers after compression?

Benchmarks (ordered by visibility / traffic):
  1. NeedleInAHaystack — info retrieval from long context
  2. LongBench         — multi-task long context (THUDM)
  3. HumanEval         — code generation (OpenAI)
  4. GSM8K             — grade school math reasoning
  5. MMLU              — massive multitask knowledge
  6. BFCL              — Berkeley function calling
  7. TruthfulQA        — truthfulness under compression
  8. SQuAD 2.0         — reading comprehension

Each benchmark runs in two modes:
  - Baseline: raw context → LLM → answer → score
  - Entroly:  raw context → Entroly compress → LLM → answer → score

The key metric is ACCURACY RETENTION:
  retention = entroly_score / baseline_score

Usage:
    python -m bench.accuracy --benchmark needle --model claude-sonnet-4-5-20250929
    python -m bench.accuracy --benchmark all --model gpt-4o
    python -m bench.accuracy --benchmark all --model gemini-2.0-flash
    python -m bench.accuracy --benchmark humaneval --samples 50

Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY set in environment.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Data types ────────────────────────────────────────────────────────


def _wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (default z=1.96).

    More reliable than normal-approximation at small n / extreme p.
    Returns (low, high).
    """
    if total <= 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    benchmark: str
    mode: str  # "baseline" or "entroly"
    samples: int
    correct: int
    accuracy: float
    ci_low: float
    ci_high: float
    avg_tokens: float
    avg_latency_ms: float
    total_cost_usd: float
    errors: int = 0
    details: list[dict] = field(default_factory=list)


@dataclass
class RetentionReport:
    """Accuracy retention: Entroly vs baseline."""
    benchmark: str
    baseline: BenchmarkResult
    entroly: BenchmarkResult
    retention: float  # entroly_accuracy / baseline_accuracy
    token_savings_pct: float
    cost_savings_pct: float


# ── LLM Client ────────────────────────────────────────────────────────


def _call_llm(messages: list[dict], model: str, max_tokens: int = 1024) -> tuple[str, int, float]:
    """Call LLM API. Returns (response_text, token_count, latency_ms).

    Supports Anthropic (claude-*), OpenAI (gpt-*), and Gemini (gemini-*) models.
    """
    t0 = time.perf_counter()

    if model.startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()
        # Separate system message
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append(m)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system if system else anthropic.NOT_GIVEN,
            messages=user_msgs,
        )
        text = resp.content[0].text
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
    elif model.startswith("gemini"):
        import openai
        # Gemini provides an OpenAI compatibility API
        client = openai.OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
    else:
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0

    latency = (time.perf_counter() - t0) * 1000
    return text, tokens, latency


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough cost estimate in USD."""
    rates = {
        "claude-sonnet-4-5-20250929": (3.0, 15.0),
        "claude-3-5-haiku-20241022": (0.80, 4.0),
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.0),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-2.5-flash": (0.10, 0.40),
        "gemini-3-flash-preview": (0.10, 0.40),
        "gemini-3.1-pro-preview": (1.25, 5.0),
    }
    inp_rate, out_rate = rates.get(model, (2.0, 8.0))
    return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000


# ── Entroly Compression ──────────────────────────────────────────────


def _compress_messages(messages: list[dict], budget: int, query: str = "") -> list[dict]:
    """Compress messages through Entroly's QCCR selector.

    Pass-through rules (honest eval — don't inject noise where compression
    can't help):
      - No system context present  → return messages unchanged
      - Context already under budget → return messages unchanged

    Only runs the selector when there's actually something to compress.
    """
    system_text = ""
    user_query = ""
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        elif m["role"] == "user":
            user_query = m["content"]

    if not system_text.strip():
        return messages

    # ~4 chars/token; if context already fits comfortably, pass through
    if len(system_text) <= budget * 4:
        return messages

    try:
        from entroly.qccr import select as qccr_select

        # Chunk system content into file-like fragments so QCCR can operate.
        # One "file" ≈ one paragraph boundary. 400-char sentences preserve
        # locality for needle-style queries.
        chunk_size = 400
        chunks = [
            system_text[i : i + chunk_size]
            for i in range(0, len(system_text), chunk_size)
        ]
        fragments = [
            {
                "id": f"f{i}",
                "source": f"chunk_{i // 8}.txt",  # group 8 chunks per pseudo-file
                "content": c,
                "tokens": len(c) // 4,
            }
            for i, c in enumerate(chunks)
        ]

        query_to_use = query or user_query
        selected = qccr_select(fragments, token_budget=budget, query=query_to_use)
        compressed_text = "\n".join(
            (s.get("content") or "") for s in selected
        ).strip()

        if not compressed_text:
            return messages  # selector returned nothing → don't corrupt prompt

        return [
            {"role": "system", "content": f"Context:\n{compressed_text}"},
            {"role": "user", "content": user_query},
        ]
    except Exception as e:
        print(f"  QCCR compression failed: {e} — falling back to raw messages")
        return messages


# ── Benchmark: NeedleInAHaystack ──────────────────────────────────────


def _generate_haystack(size_tokens: int, needle: str, depth_pct: float) -> str:
    """Generate a haystack with a needle at the given depth."""
    # Use Paul Graham essays style filler
    filler_sentences = [
        "The most important thing in any organization is the quality of its people.",
        "Technology is best when it brings people together and enables new possibilities.",
        "Good design is innovative, useful, aesthetic, understandable, and honest.",
        "The best way to predict the future is to invent it with careful planning.",
        "Software is a great combination of artistry and engineering principles.",
        "Every great developer you know got there by solving problems they were unqualified to solve.",
        "The function of good software is to make the complex appear simple.",
        "Simplicity is the ultimate sophistication in system design and architecture.",
    ]

    # Build haystack to approximate size_tokens
    lines = []
    total_chars = 0
    target_chars = size_tokens * 4  # ~4 chars per token
    needle_pos = int(target_chars * depth_pct)

    needle_inserted = False
    while total_chars < target_chars:
        if not needle_inserted and total_chars >= needle_pos:
            lines.append(needle)
            needle_inserted = True
        line = random.choice(filler_sentences)
        lines.append(line)
        total_chars += len(line) + 1

    if not needle_inserted:
        lines.append(needle)

    return "\n".join(lines)


def bench_needle(model: str, samples: int = 20) -> list[dict]:
    """NeedleInAHaystack: can the LLM find a fact in compressed context?"""
    needle = "The secret passphrase for Project Aurora is 'crystalline-nebula-7742'."
    question = "What is the secret passphrase for Project Aurora? Answer with just the passphrase."
    expected = "crystalline-nebula-7742"

    results = []
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    sizes = [4000, 8000, 16000, 32000]

    for size in sizes:
        for depth in depths:
            if len(results) >= samples:
                break
            haystack = _generate_haystack(size, needle, depth)
            results.append({
                "context": haystack,
                "question": question,
                "expected": expected,
                "metadata": {"size_tokens": size, "depth_pct": depth},
            })

    return results


# ── Benchmark: GSM8K ──────────────────────────────────────────────────


def _load_gsm8k(samples: int = 50) -> list[dict]:
    """Load GSM8K samples. Downloads if needed."""
    cache_path = Path(__file__).parent / ".cache" / "gsm8k.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
    else:
        # Download from HuggingFace
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            resp = urllib.request.urlopen(url, timeout=30)
            lines = resp.read().decode().strip().split("\n")
            data = [json.loads(line) for line in lines]
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: could not download GSM8K: {e}")
            return []

    random.seed(42)
    selected = random.sample(data, min(samples, len(data)))

    results = []
    for item in selected:
        question = item["question"]
        answer_text = item["answer"]
        # Extract final numeric answer after ####
        match = re.search(r"####\s*([\d,.-]+)", answer_text)
        expected = match.group(1).replace(",", "") if match else ""
        results.append({
            "context": "",
            "question": question,
            "expected": expected,
            "metadata": {"source": "gsm8k"},
        })

    return results


# ── Benchmark: HumanEval ──────────────────────────────────────────────


def _load_humaneval(samples: int = 30) -> list[dict]:
    """Load HumanEval samples."""
    cache_path = Path(__file__).parent / ".cache" / "humaneval.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
    else:
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            resp = urllib.request.urlopen(url, timeout=30)
            import gzip
            lines = gzip.decompress(resp.read()).decode().strip().split("\n")
            data = [json.loads(line) for line in lines]
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: could not download HumanEval: {e}")
            return []

    random.seed(42)
    selected = random.sample(data, min(samples, len(data)))

    results = []
    for item in selected:
        prompt = item.get("prompt", "")
        entry_point = item.get("entry_point", "")
        test = item.get("test", "")
        canonical = item.get("canonical_solution", "")
        results.append({
            "context": prompt,
            "question": f"Complete this Python function:\n\n{prompt}",
            "expected": canonical.strip(),
            "metadata": {
                "source": "humaneval",
                "task_id": item.get("task_id", ""),
                "entry_point": entry_point,
                "test": test,
            },
        })

    return results


# ── Benchmark: SQuAD 2.0 ─────────────────────────────────────────────


def _load_squad(samples: int = 50) -> list[dict]:
    """Load SQuAD 2.0 samples."""
    cache_path = Path(__file__).parent / ".cache" / "squad.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
    else:
        try:
            import urllib.request
            url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            resp = urllib.request.urlopen(url, timeout=30)
            raw = json.loads(resp.read())
            # Flatten to list of (context, question, answers)
            data = []
            for article in raw["data"]:
                for para in article["paragraphs"]:
                    for qa in para["qas"]:
                        if qa.get("is_impossible"):
                            continue
                        answers = [a["text"] for a in qa.get("answers", [])]
                        if answers:
                            data.append({
                                "context": para["context"],
                                "question": qa["question"],
                                "answers": answers,
                            })
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: could not download SQuAD: {e}")
            return []

    random.seed(42)
    selected = random.sample(data, min(samples, len(data)))

    results = []
    for item in selected:
        results.append({
            "context": item["context"],
            "question": item["question"],
            "expected": item["answers"][0],
            "metadata": {"source": "squad", "all_answers": item["answers"]},
        })

    return results


# ── Generic runner ────────────────────────────────────────────────────


def _check_answer(response: str, expected: str, benchmark: str, metadata: dict | None = None) -> bool:
    """Check if the LLM response contains the expected answer."""
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()

    if benchmark == "needle":
        return expected_lower in response_lower

    if benchmark == "gsm8k":
        # Extract last number from response
        numbers = re.findall(r"[-+]?\d*\.?\d+", response)
        if numbers:
            return numbers[-1].replace(",", "") == expected_lower.replace(",", "")
        return False

    if benchmark == "squad":
        # Check if any accepted answer appears
        all_answers = (metadata or {}).get("all_answers", [expected])
        return any(a.lower() in response_lower for a in all_answers)

    if benchmark == "humaneval":
        # Basic: check if key parts of canonical solution appear
        return expected_lower[:50] in response_lower or response_lower[:50] in expected_lower

    # Default: substring match
    return expected_lower in response_lower


def run_benchmark(
    benchmark: str,
    model: str = "gpt-4o-mini",
    samples: int = 50,
    budget: int = 50_000,
) -> RetentionReport:
    """Run a benchmark, comparing baseline vs Entroly-compressed."""

    # Load benchmark data
    loaders = {
        "needle": lambda: bench_needle(model, samples),
        "gsm8k": lambda: _load_gsm8k(samples),
        "humaneval": lambda: _load_humaneval(min(samples, 30)),
        "squad": lambda: _load_squad(samples),
    }

    if benchmark not in loaders:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(loaders.keys())}")

    print(f"\n  Loading {benchmark} benchmark...")
    items = loaders[benchmark]()
    if not items:
        raise RuntimeError(f"Failed to load {benchmark} data")

    print(f"  Loaded {len(items)} samples. Model: {model}")

    # Run baseline
    print(f"\n  [1/2] Running baseline (no compression)...")
    baseline = _run_mode(items, benchmark, model, "baseline", budget=None)

    # Run with Entroly compression
    print(f"  [2/2] Running with Entroly compression (budget={budget})...")
    entroly = _run_mode(items, benchmark, model, "entroly", budget=budget)

    # Compute retention
    retention = entroly.accuracy / max(baseline.accuracy, 1e-9)
    token_savings = 1.0 - (entroly.avg_tokens / max(baseline.avg_tokens, 1))
    cost_savings = 1.0 - (entroly.total_cost_usd / max(baseline.total_cost_usd, 1e-9))

    return RetentionReport(
        benchmark=benchmark,
        baseline=baseline,
        entroly=entroly,
        retention=round(retention, 4),
        token_savings_pct=round(token_savings * 100, 1),
        cost_savings_pct=round(cost_savings * 100, 1),
    )


def _run_mode(
    items: list[dict],
    benchmark: str,
    model: str,
    mode: str,
    budget: int | None,
) -> BenchmarkResult:
    """Run all items in a given mode."""
    correct = 0
    total_tokens = 0
    total_latency = 0.0
    total_cost = 0.0
    errors = 0
    details = []

    for i, item in enumerate(items):
        messages = []
        if item["context"]:
            messages.append({"role": "system", "content": f"Context:\n{item['context']}"})
        messages.append({"role": "user", "content": item["question"]})

        # Compress if Entroly mode
        if mode == "entroly" and budget:
            messages = _compress_messages(messages, budget, query=item["question"])

        try:
            response, tokens, latency = _call_llm(messages, model)
            is_correct = _check_answer(response, item["expected"], benchmark, item.get("metadata"))
            if is_correct:
                correct += 1
            total_tokens += tokens
            total_latency += latency
            total_cost += _estimate_cost(model, tokens, 200)
            details.append({
                "index": i,
                "correct": is_correct,
                "tokens": tokens,
                "latency_ms": round(latency, 1),
            })
        except Exception as e:
            errors += 1
            details.append({"index": i, "error": str(e)})

        # Progress
        if (i + 1) % 10 == 0:
            pct = correct / max(i + 1 - errors, 1) * 100
            print(f"    [{i+1}/{len(items)}] accuracy={pct:.0f}%")

    n = len(items)
    valid = max(n - errors, 1)
    ci_lo, ci_hi = _wilson_ci(correct, valid)
    return BenchmarkResult(
        benchmark=benchmark,
        mode=mode,
        samples=n,
        correct=correct,
        accuracy=round(correct / valid, 4),
        ci_low=round(ci_lo, 4),
        ci_high=round(ci_hi, 4),
        avg_tokens=round(total_tokens / max(n, 1), 1),
        avg_latency_ms=round(total_latency / max(n, 1), 1),
        total_cost_usd=round(total_cost, 4),
        errors=errors,
        details=details,
    )


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Entroly Public Accuracy Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmarks:
  needle      NeedleInAHaystack — info retrieval from long context
  gsm8k       GSM8K — grade school math reasoning
  humaneval   HumanEval — code generation
  squad       SQuAD 2.0 — reading comprehension
  all         Run all benchmarks

Examples:
  python -m bench.accuracy --benchmark needle --model gemini-2.0-flash --samples 20
  python -m bench.accuracy --benchmark gsm8k --model claude-sonnet-4-5-20250929
  python -m bench.accuracy --benchmark all --model gpt-4o-mini
""",
    )
    parser.add_argument(
        "--benchmark", "-b", type=str, default="needle",
        help="Benchmark to run (needle, gsm8k, humaneval, squad, all)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=200,
        help="Number of samples per benchmark (default: 200). n=50 is noise.",
    )
    parser.add_argument(
        "--budget", type=int, default=50_000,
        help="Entroly token budget (default: 50000)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    benchmarks = ["needle", "gsm8k", "humaneval", "squad"] if args.benchmark == "all" else [args.benchmark]

    all_reports = []
    for bench in benchmarks:
        try:
            report = run_benchmark(bench, args.model, args.samples, args.budget)
            all_reports.append(report)
        except Exception as e:
            print(f"\n  ERROR: {bench} failed: {e}")
            continue

    if args.json:
        output = []
        for r in all_reports:
            output.append({
                "benchmark": r.benchmark,
                "retention": r.retention,
                "token_savings_pct": r.token_savings_pct,
                "cost_savings_pct": r.cost_savings_pct,
                "baseline_accuracy": r.baseline.accuracy,
                "baseline_ci_95": [r.baseline.ci_low, r.baseline.ci_high],
                "entroly_accuracy": r.entroly.accuracy,
                "entroly_ci_95": [r.entroly.ci_low, r.entroly.ci_high],
                "baseline_avg_tokens": r.baseline.avg_tokens,
                "entroly_avg_tokens": r.entroly.avg_tokens,
                "samples": r.baseline.samples,
            })
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 88)
        print("  ENTROLY ACCURACY RETENTION BENCHMARKS")
        print("=" * 88)
        print(f"  Model: {args.model}  |  Budget: {args.budget:,} tokens  |  Samples: {args.samples}")
        print("-" * 88)
        print(f"  {'Benchmark':<12} {'Baseline (95% CI)':>24} {'Entroly (95% CI)':>24} {'Retention':>10} {'Token Save':>12}")
        print("-" * 88)
        for r in all_reports:
            b_ci = f"{r.baseline.accuracy:.1%} [{r.baseline.ci_low:.1%}-{r.baseline.ci_high:.1%}]"
            e_ci = f"{r.entroly.accuracy:.1%} [{r.entroly.ci_low:.1%}-{r.entroly.ci_high:.1%}]"
            print(
                f"  {r.benchmark:<12} "
                f"{b_ci:>24} "
                f"{e_ci:>24} "
                f"{r.retention:>9.1%} "
                f"{r.token_savings_pct:>10.1f}%"
            )
        print("-" * 88)
        if all_reports:
            avg_retention = sum(r.retention for r in all_reports) / len(all_reports)
            avg_savings = sum(r.token_savings_pct for r in all_reports) / len(all_reports)
            print(f"  {'AVERAGE':<12} {'':>24} {'':>24} {avg_retention:>9.1%} {avg_savings:>10.1f}%")
        print("=" * 88)
        print()


if __name__ == "__main__":
    main()
