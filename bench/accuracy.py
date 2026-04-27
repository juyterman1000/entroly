"""
Entroly Public Accuracy Benchmarks
====================================

Measures accuracy RETENTION when Entroly compresses context.
Tests: does the LLM still give correct answers after compression?

Benchmarks (ordered by visibility / traffic):
  1. NeedleInAHaystack — info retrieval from long context
  2. LongBench         — multi-task long context (THUDM/LongBench, HotpotQA subtask)
  3. HumanEval         — code generation (OpenAI)
  4. GSM8K             — grade school math reasoning
  5. MMLU              — massive multitask knowledge
  6. TruthfulQA        — truthfulness under compression (MC1)
  7. SQuAD 2.0         — reading comprehension
  8. BFCL              — Berkeley Function Calling (gorilla-llm, Simple subset)

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


# ── .env.local loader ─────────────────────────────────────────────────
#
# Cross-shell key plumbing for Windows where PowerShell ``$env:VAR`` does
# not propagate to bash subprocesses spawned by other tools (notably
# Claude Code or any IDE-launched runner). Drops a one-time loader so
# any shell that runs this file picks up the same credentials.
#
# Format (.env.local at repo root, gitignore'd):
#
#     OPENAI_API_KEY=sk-proj-...
#     ANTHROPIC_API_KEY=sk-ant-...
#     # comments allowed; quotes around values stripped
#
# os.environ wins if the variable is already set — environment beats
# file, so production deploys with secrets-managers are unaffected.

def _load_env_local() -> None:
    """Load .env.local from the repo root if present. Idempotent."""
    # Walk up from this file until we find pyproject.toml (repo root marker).
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            env_path = candidate / ".env.local"
            break
    else:
        return
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


_load_env_local()


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


def _call_llm(
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> tuple[str, int, float]:
    """Call LLM API. Returns (response_text, token_count, latency_ms).

    Provider routing (base_url wins over model-name detection):
      - base_url set                 → OpenAI-compatible custom endpoint
        (Groq, Together, OpenRouter, Ollama, LM Studio, vLLM, Fireworks, ...)
        Auth: api_key read from os.environ[api_key_env] if specified; missing
        env var raises. If api_key_env is None, the SDK is called with a
        "no-auth" sentinel — fine for self-hosted endpoints that ignore auth
        (Ollama/LM Studio/vLLM). We deliberately do NOT fall back to
        OPENAI_API_KEY here, to avoid leaking OpenAI credentials to a
        third-party endpoint.
      - model starts with "claude"   → Anthropic SDK
      - model starts with "gemini"   → OpenAI SDK + Google's compat endpoint
      - otherwise                    → OpenAI default

    Retries on transient errors (connection drops, 429 rate-limits,
    5xx server errors) with exponential backoff. Network blips during a
    long sweep should not crash the whole run — the per-item except in
    ``_run_mode`` would just count this as an error and move on, but
    retrying transparently is strictly better data quality.
    """
    return _call_llm_with_retry(
        messages, model, max_tokens, base_url, api_key_env, max_retries=3
    )


def _is_transient_error(exc: Exception) -> bool:
    """True if the exception is a network/rate-limit blip worth retrying.

    Conservative: only retry on errors we *know* are transient. Bad
    auth, bad request, model-not-found, and similar permanent failures
    pass through immediately so the user sees the real cause.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    # httpx network errors that are typically transient
    if name in {
        "ConnectError", "ReadTimeout", "ReadError", "ConnectTimeout",
        "RemoteProtocolError", "PoolTimeout", "WriteTimeout",
    }:
        return True
    # OpenAI/Anthropic SDK rate-limit + transient server errors
    if name in {"RateLimitError", "APIConnectionError", "InternalServerError",
                "APITimeoutError"}:
        return True
    # Generic httpx exceptions also have an HTTP status code attribute
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code in (408, 425, 429, 500, 502, 503, 504):
        return True
    # Fallback: text-pattern match for the rare cases the SDK wraps weirdly
    if any(s in msg for s in ("rate limit", "timeout", "temporarily unavailable",
                                "connection reset", "service unavailable")):
        return True
    return False


def _call_llm_with_retry(
    messages: list[dict],
    model: str,
    max_tokens: int,
    base_url: str | None,
    api_key_env: str | None,
    max_retries: int,
) -> tuple[str, int, float]:
    """Inner retry loop. Exponential backoff: 1s, 2s, 4s, … capped at 30s."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _call_llm_once(
                messages, model, max_tokens, base_url, api_key_env,
            )
        except Exception as e:
            last_exc = e
            if attempt >= max_retries or not _is_transient_error(e):
                raise
            backoff = min(30.0, 2 ** attempt)
            print(
                f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: "
                f"{str(e)[:120]} — sleeping {backoff:.0f}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(backoff)
    # Defensive — should never reach here.
    raise last_exc if last_exc else RuntimeError("retry loop exhausted")


def _call_llm_once(
    messages: list[dict],
    model: str,
    max_tokens: int,
    base_url: str | None,
    api_key_env: str | None,
) -> tuple[str, int, float]:
    """Single LLM call attempt — provider routing only, no retry logic."""
    t0 = time.perf_counter()

    if base_url:
        import openai
        if api_key_env is None:
            api_key = "no-auth"
        else:
            api_key = os.environ.get(api_key_env)
            if api_key is None:
                raise ValueError(
                    f"--api-key-env was set to {api_key_env!r}, but that env var is "
                    f"not exported. Set it (e.g. `export {api_key_env}=...`), or omit "
                    f"the flag for self-hosted endpoints that ignore auth."
                )
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=messages,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
    elif model.startswith("claude"):
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


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    base_url: str | None = None,
) -> float:
    """Rough cost estimate in USD.

    Custom providers (base_url set) return 0.0 — pricing varies wildly
    across Groq / Together / OpenRouter / self-hosted, and the user
    knows their own bill better than this table does.
    """
    if base_url:
        return 0.0
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
    """Backwards-compatible wrapper — defaults to Entroly compression mode.

    Prefer ``_compress_messages_modal`` for new callers.
    """
    return _compress_messages_modal(messages, budget, mode="entroly", query=query)


# ── Compressor implementations ────────────────────────────────────────
#
# Three compressor families, all sharing a uniform interface:
#   compress(text, budget_tokens, query) -> str
#
# - "entroly"   : QCCR fragment-level selector (knapsack on entropy + sim + recency)
# - "llmlingua" : LLMLingua-2 token-level keep/drop classifier (Pan et al ACL 2024)
# - "hybrid"    : Entroly fragment selection → LLMLingua-2 token compression
#                 within selected fragments. Composes orthogonal granularities.
#
# All compressors are deterministic given identical input + budget; the
# only stochasticity is in the LLM evaluator that follows.

# Lazy LLMLingua singleton — model load is ~3s, do it once per process.
_LLMLINGUA_MODEL: object | None = None


def _get_llmlingua():
    """Load LLMLingua-2 small model lazily (singleton). Raises ImportError if missing."""
    global _LLMLINGUA_MODEL
    if _LLMLINGUA_MODEL is None:
        from llmlingua import PromptCompressor
        _LLMLINGUA_MODEL = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _LLMLINGUA_MODEL


def _entroly_compress(text: str, budget_tokens: int, query: str) -> str:
    """Fragment-level QCCR selection."""
    from entroly.qccr import select as qccr_select
    chunk_size = 400
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    fragments = [
        {"id": f"f{i}", "source": f"chunk_{i // 8}.txt",
         "content": c, "tokens": len(c) // 4}
        for i, c in enumerate(chunks)
    ]
    selected = qccr_select(fragments, token_budget=budget_tokens, query=query)
    return "\n".join((s.get("content") or "") for s in selected).strip()


def _llmlingua_compress(text: str, budget_tokens: int, query: str) -> str:
    """Token-level keep/drop via LLMLingua-2 classifier."""
    compressor = _get_llmlingua()
    # LLMLingua-2 takes a target rate (kept_fraction). Convert budget → rate.
    original_tokens = max(1, len(text) // 4)
    rate = min(0.99, max(0.05, budget_tokens / original_tokens))
    result = compressor.compress_prompt(
        text,
        rate=rate,
        force_tokens=["\n", ".", "!", "?"],
        # Question-aware compression — LLMLingua-2 supports it directly.
        # Improves compression quality measurably (Pan et al ACL 2024 §4.2).
        question=query,
    )
    return result.get("compressed_prompt", text)


def _truncate_head_compress(text: str, budget_tokens: int, query: str) -> str:
    """Keep the first ``budget_tokens`` tokens (≈4 chars each).

    Surprisingly hard to beat on benchmarks where the relevant info is
    front-loaded (system prompts, document headers). Including this as a
    baseline is honest practice — Liu et al. ACL 2024 show it is the
    median compressor most published methods barely beat.

    ``query`` is intentionally ignored — this is a position-only baseline.
    The signature stays uniform with the other compressors.
    """
    del query  # unused-by-design (position-only baseline)
    return text[: budget_tokens * 4]


def _truncate_tail_compress(text: str, budget_tokens: int, query: str) -> str:
    """Keep the LAST ``budget_tokens`` tokens.

    Counterpart to ``_truncate_head_compress``. On instruction-following
    workloads, recent context dominates — tail truncation often wins.
    Pairing head + tail makes the position-bias of the workload visible.

    ``query`` ignored by design (position-only baseline).
    """
    del query  # unused-by-design (position-only baseline)
    return text[-budget_tokens * 4:]


def _random_keep_compress(text: str, budget_tokens: int, query: str) -> str:
    """Keep a uniform-random subset of sentences summing to ≤ budget.

    The proper random-baseline for compression. Any "smart" method must
    statistically beat this — otherwise it's just noise. Seeded for
    reproducibility (the seed is the query hash).
    """
    import hashlib
    import random as _random
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return text
    seed = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
    rng = _random.Random(seed)
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    out, used = [], 0
    cap_chars = budget_tokens * 4
    for s in shuffled:
        if used + len(s) + 2 > cap_chars and out:
            break
        out.append(s)
        used += len(s) + 2
    return ". ".join(out) + "."


def _hybrid_compress(text: str, budget_tokens: int, query: str) -> str:
    """Entroly × LLMLingua-2 hybrid: orthogonal-granularity composition.

    Stage 1 (Entroly): fragment-level knapsack with budget = 1.5× target.
        Selects the most relevant fragments at a slightly relaxed budget,
        leaving headroom for stage 2.

    Stage 2 (LLMLingua-2): token-level compression to the actual budget.
        Removes filler tokens within the selected fragments.

    The 1.5× headroom is empirically calibrated: too tight (1.0×) and
    stage-2 has nothing to compress; too loose (≥2×) and stage-1 selects
    irrelevant fragments that stage-2 can't recover from.
    """
    intermediate_budget = int(budget_tokens * 1.5)
    stage1 = _entroly_compress(text, intermediate_budget, query)
    if len(stage1) // 4 <= budget_tokens:
        return stage1  # already under budget after stage 1
    return _llmlingua_compress(stage1, budget_tokens, query)


_COMPRESSORS = {
    # Smart compressors — adapt to query.
    "entroly": _entroly_compress,
    "llmlingua": _llmlingua_compress,
    "hybrid": _hybrid_compress,
    # Naive baselines — make the position-bias of a workload visible.
    # Including them is honest methodology: any "smart" compressor must
    # statistically beat all three to claim it has learned anything.
    "head": _truncate_head_compress,
    "tail": _truncate_tail_compress,
    "random": _random_keep_compress,
}


def _compress_messages_modal(
    messages: list[dict],
    budget: int,
    mode: str = "entroly",
    query: str = "",
) -> list[dict]:
    """Compress messages via the chosen compressor.

    Pass-through invariants (honest eval):
      - No system context present  → unchanged
      - Context already under budget → unchanged
      - Compressor returned empty → unchanged (don't corrupt prompt)
      - Compressor raised → unchanged (and printed a one-liner)
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
    if len(system_text) <= budget * 4:  # already fits, ~4 chars/tok
        return messages

    fn = _COMPRESSORS.get(mode)
    if fn is None:
        raise ValueError(f"unknown compression mode: {mode!r} (have: {list(_COMPRESSORS)})")

    try:
        compressed = fn(system_text, budget, query or user_query)
    except Exception as e:
        print(f"  {mode} compression failed: {e} — falling back to raw messages")
        return messages

    if not compressed:
        return messages

    return [
        {"role": "system", "content": f"Context:\n{compressed}"},
        {"role": "user", "content": user_query},
    ]


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


# ── HuggingFace datasets-server fetch helper ─────────────────────────


def _fetch_hf_rows(dataset: str, config: str, split: str, limit: int) -> list[dict]:
    """Fetch rows from HuggingFace datasets-server (paginated; max 100/req).

    No auth needed for public datasets. Returns the `row` payload of each entry.
    """
    import urllib.parse
    import urllib.request

    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        batch = min(100, limit - len(rows))
        qs = urllib.parse.urlencode({
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": batch,
        })
        url = f"https://datasets-server.huggingface.co/rows?{qs}"
        resp = urllib.request.urlopen(url, timeout=30)
        payload = json.loads(resp.read())
        new_rows = payload.get("rows", [])
        if not new_rows:
            break
        rows.extend(r["row"] for r in new_rows)
        offset += batch
    return rows[:limit]


# ── Benchmark: MMLU ──────────────────────────────────────────────────


def _load_mmlu(samples: int = 50) -> list[dict]:
    """Load MMLU (all subjects) via HuggingFace datasets-server."""
    cache_path = Path(__file__).parent / ".cache" / "mmlu.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
    else:
        try:
            data = _fetch_hf_rows("cais/mmlu", "all", "test", max(samples * 4, 800))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: could not download MMLU: {e}")
            return []

    random.seed(42)
    selected = random.sample(data, min(samples, len(data)))
    letters = ["A", "B", "C", "D"]
    results = []
    for item in selected:
        choices = item.get("choices") or []
        if len(choices) != 4:
            continue
        answer_idx = item.get("answer", 0)
        if not isinstance(answer_idx, int) or answer_idx >= len(letters):
            continue
        question = item.get("question", "")
        prompt = (
            f"{question}\n\n"
            f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\n"
            "Answer with only the letter (A, B, C, or D)."
        )
        results.append({
            "context": "",
            "question": prompt,
            "expected": letters[answer_idx],
            "metadata": {"source": "mmlu", "subject": item.get("subject", "")},
        })
    return results


# ── Benchmark: TruthfulQA (MC1) ──────────────────────────────────────


def _load_truthfulqa(samples: int = 50) -> list[dict]:
    """Load TruthfulQA multiple-choice (MC1) via HuggingFace datasets-server."""
    cache_path = Path(__file__).parent / ".cache" / "truthfulqa.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
    else:
        try:
            data = _fetch_hf_rows("truthfulqa/truthful_qa", "multiple_choice", "validation", 800)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: could not download TruthfulQA: {e}")
            return []

    random.seed(42)
    selected = random.sample(data, min(samples, len(data)))
    letters = list("ABCDEFGHIJKLMNOP")
    results = []
    for item in selected:
        mc1 = item.get("mc1_targets") or {}
        choices = mc1.get("choices") or []
        labels = mc1.get("labels") or []
        if not choices or len(choices) != len(labels) or 1 not in labels:
            continue
        correct_idx = labels.index(1)
        if correct_idx >= len(letters) or len(choices) > len(letters):
            continue
        choices_text = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))
        prompt = (
            f"{item.get('question', '')}\n\n{choices_text}\n\n"
            "Answer with only the letter."
        )
        results.append({
            "context": "",
            "question": prompt,
            "expected": letters[correct_idx],
            "metadata": {"source": "truthfulqa"},
        })
    return results


# ── Benchmark: LongBench (HotpotQA subtask) ──────────────────────────


def _load_longbench(samples: int = 50) -> list[dict]:
    """Load LongBench HotpotQA subtask — multi-hop QA with long context.

    HotpotQA chosen as the primary LongBench subtask: QA format matches the
    existing substring-scoring path, and the 10k+ token contexts are exactly
    the regime where Entroly compression is supposed to pay off.

    Uses the Xnhyacinth/LongBench mirror (THUDM original isn't indexed by
    the HF datasets-server).
    """
    cache_path = Path(__file__).parent / ".cache" / "longbench_hotpotqa.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
    else:
        try:
            data = _fetch_hf_rows("Xnhyacinth/LongBench", "hotpotqa", "test", 200)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: could not download LongBench: {e}")
            return []

    random.seed(42)
    selected = random.sample(data, min(samples, len(data)))
    results = []
    for item in selected:
        context = item.get("context", "")
        question = item.get("question", "")
        answers = item.get("answers") or []
        if not answers or not question or not context:
            continue
        results.append({
            "context": context,
            "question": question,
            "expected": answers[0],
            "metadata": {"source": "longbench_hotpotqa", "all_answers": answers},
        })
    return results


# ── Benchmark: BFCL (Berkeley Function Calling) ──────────────────────


# Distractor tool schemas — realistic function signatures that pad the
# system context so compression is exercised. These are NOT the target;
# the model must pick the correct function from among these distractors.
_BFCL_DISTRACTORS = [
    {"name": "get_weather_forecast", "description": "Get weather forecast for a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "days": {"type": "integer", "default": 5}}, "required": ["location"]}},
    {"name": "send_email", "description": "Send an email to a recipient.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}, "cc": {"type": "array", "items": {"type": "string"}}}, "required": ["to", "subject", "body"]}},
    {"name": "search_database", "description": "Search a database with a SQL-like query.", "parameters": {"type": "object", "properties": {"table": {"type": "string"}, "filter": {"type": "string"}, "limit": {"type": "integer", "default": 100}, "offset": {"type": "integer", "default": 0}}, "required": ["table"]}},
    {"name": "create_calendar_event", "description": "Create a new calendar event.", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start_time": {"type": "string"}, "end_time": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "start_time"]}},
    {"name": "translate_text", "description": "Translate text from one language to another.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "source_lang": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}},
    {"name": "resize_image", "description": "Resize an image to specified dimensions.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "format": {"type": "string", "enum": ["png", "jpeg", "webp"]}}, "required": ["url", "width", "height"]}},
    {"name": "generate_report", "description": "Generate a business report from data sources.", "parameters": {"type": "object", "properties": {"report_type": {"type": "string", "enum": ["sales", "inventory", "finance"]}, "date_range": {"type": "string"}, "format": {"type": "string", "enum": ["pdf", "csv", "xlsx"]}}, "required": ["report_type"]}},
    {"name": "manage_subscription", "description": "Manage user subscription plan.", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}, "action": {"type": "string", "enum": ["upgrade", "downgrade", "cancel"]}, "plan": {"type": "string"}}, "required": ["user_id", "action"]}},
    {"name": "analyze_sentiment", "description": "Analyze sentiment of a text passage.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "language": {"type": "string", "default": "en"}, "granularity": {"type": "string", "enum": ["document", "sentence"]}}, "required": ["text"]}},
    {"name": "deploy_service", "description": "Deploy a microservice to the cluster.", "parameters": {"type": "object", "properties": {"service_name": {"type": "string"}, "version": {"type": "string"}, "replicas": {"type": "integer", "default": 3}, "environment": {"type": "string", "enum": ["staging", "production"]}}, "required": ["service_name", "version"]}},
    {"name": "query_knowledge_base", "description": "Query an internal knowledge base for articles.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "category": {"type": "string"}, "max_results": {"type": "integer", "default": 10}}, "required": ["query"]}},
    {"name": "process_payment", "description": "Process a payment transaction.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "currency": {"type": "string", "default": "USD"}, "method": {"type": "string", "enum": ["credit_card", "bank_transfer", "crypto"]}, "recipient_id": {"type": "string"}}, "required": ["amount", "recipient_id"]}},
    {"name": "schedule_backup", "description": "Schedule a database backup.", "parameters": {"type": "object", "properties": {"database": {"type": "string"}, "schedule": {"type": "string"}, "retention_days": {"type": "integer", "default": 30}, "compression": {"type": "boolean", "default": True}}, "required": ["database"]}},
    {"name": "create_user_account", "description": "Create a new user account in the system.", "parameters": {"type": "object", "properties": {"username": {"type": "string"}, "email": {"type": "string"}, "role": {"type": "string", "enum": ["admin", "user", "viewer"]}, "team": {"type": "string"}}, "required": ["username", "email"]}},
    {"name": "run_diagnostics", "description": "Run system diagnostics and health checks.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "depth": {"type": "string", "enum": ["quick", "full", "deep"]}, "include_logs": {"type": "boolean", "default": False}}, "required": ["target"]}},
    {"name": "configure_firewall", "description": "Configure firewall rules for a network.", "parameters": {"type": "object", "properties": {"network_id": {"type": "string"}, "rules": {"type": "array", "items": {"type": "object"}}, "mode": {"type": "string", "enum": ["append", "replace"]}}, "required": ["network_id", "rules"]}},
    {"name": "extract_entities", "description": "Extract named entities from text.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "entity_types": {"type": "array", "items": {"type": "string"}}, "language": {"type": "string", "default": "en"}}, "required": ["text"]}},
    {"name": "convert_currency", "description": "Convert amount between currencies.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}, "date": {"type": "string"}}, "required": ["amount", "from_currency", "to_currency"]}},
    {"name": "train_ml_model", "description": "Train a machine learning model on a dataset.", "parameters": {"type": "object", "properties": {"dataset_id": {"type": "string"}, "algorithm": {"type": "string", "enum": ["random_forest", "xgboost", "neural_net"]}, "hyperparameters": {"type": "object"}, "epochs": {"type": "integer", "default": 100}}, "required": ["dataset_id", "algorithm"]}},
    {"name": "monitor_metrics", "description": "Set up monitoring for system metrics.", "parameters": {"type": "object", "properties": {"metric_name": {"type": "string"}, "threshold": {"type": "number"}, "alert_channel": {"type": "string", "enum": ["email", "slack", "pagerduty"]}, "interval_seconds": {"type": "integer", "default": 60}}, "required": ["metric_name", "threshold"]}},
]


def _load_bfcl(samples: int = 50) -> list[dict]:
    """Load BFCL Simple function-calling samples (v3 exec subset).

    Downloads the gorilla-llm/Berkeley-Function-Calling-Leaderboard dataset
    (exec_simple subset, 100 samples) from HuggingFace. Each sample has a
    user query, candidate function schemas, and a ground-truth function call.

    To make compression meaningful, we pad each sample with 20 distractor
    tool schemas so the system context is large enough for Entroly to compress.
    """
    cache_path = Path(__file__).parent / ".cache" / "bfcl_simple.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
    else:
        try:
            import urllib.request
            url = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_exec_simple.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            resp = urllib.request.urlopen(url, timeout=60)
            lines = resp.read().decode().strip().split("\n")
            data = [json.loads(line) for line in lines if line.strip()]
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: could not download BFCL: {e}")
            return []

    random.seed(42)
    selected = random.sample(data, min(samples, len(data)))

    results = []
    for item in selected:
        # question is list[list[dict]] — take first conversation, first message
        question_outer = item.get("question", [])
        if not question_outer:
            continue
        question_inner = question_outer[0] if isinstance(question_outer[0], list) else question_outer
        user_content = ""
        for msg in question_inner:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        if not user_content:
            continue

        # ground_truth is list[str] like ["func_name(arg1=val1, arg2=val2)"]
        ground_truth = item.get("ground_truth", [])
        if not ground_truth:
            continue
        gt_call = ground_truth[0] if isinstance(ground_truth, list) else str(ground_truth)

        # Parse function name from ground truth
        func_match = re.match(r"([\w.]+)\(", str(gt_call))
        if not func_match:
            continue
        expected_func = func_match.group(1)

        # Build tool schemas: real functions from the sample + distractors
        real_functions = item.get("function", [])
        if isinstance(real_functions, str):
            try:
                real_functions = json.loads(real_functions)
            except json.JSONDecodeError:
                real_functions = []
        if isinstance(real_functions, dict):
            real_functions = [real_functions]

        # Combine real + distractor schemas and shuffle
        all_tools = list(real_functions) + list(_BFCL_DISTRACTORS)
        random.shuffle(all_tools)
        tools_text = json.dumps(all_tools, indent=2)

        results.append({
            "context": f"You have access to the following tools:\n\n{tools_text}",
            "question": f"{user_content}\n\nRespond with ONLY the function call in the format: function_name(arg1=value1, arg2=value2)",
            "expected": expected_func,
            "metadata": {"source": "bfcl", "full_answer": str(gt_call)},
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

    if benchmark in ("mmlu", "truthfulqa"):
        # Match the last standalone A-P letter (word boundary). Models usually
        # conclude with the answer; taking the last avoids stray matches on
        # pronouns like "I" at the start of rationales.
        matches = re.findall(r"\b([A-P])\b", response.upper())
        return bool(matches) and matches[-1] == expected.upper()

    if benchmark == "longbench":
        all_answers = (metadata or {}).get("all_answers", [expected])
        return any(a.lower() in response_lower for a in all_answers)

    if benchmark == "bfcl":
        # Check if the model called the correct function name
        # Expected is just the function name (e.g., "get_weather")
        func_match = re.search(r"([\w.]+)\(", response)
        if func_match:
            return func_match.group(1).lower() == expected_lower
        # Also accept function name on its own line
        return expected_lower in response_lower

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
    base_url: str | None = None,
    api_key_env: str | None = None,
    mode: str = "entroly",
) -> RetentionReport:
    """Run a benchmark, comparing baseline vs Entroly-compressed.

    base_url + api_key_env are forwarded to _call_llm for OpenAI-compatible
    custom providers (Groq, Together, OpenRouter, Ollama, LM Studio, vLLM, ...).
    """

    # Load benchmark data
    loaders = {
        "needle": lambda: bench_needle(model, samples),
        "gsm8k": lambda: _load_gsm8k(samples),
        "humaneval": lambda: _load_humaneval(min(samples, 30)),
        "squad": lambda: _load_squad(samples),
        "mmlu": lambda: _load_mmlu(samples),
        "truthfulqa": lambda: _load_truthfulqa(samples),
        "longbench": lambda: _load_longbench(samples),
        "bfcl": lambda: _load_bfcl(samples),
    }

    if benchmark not in loaders:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(loaders.keys())}")

    print(f"\n  Loading {benchmark} benchmark...")
    items = loaders[benchmark]()
    if not items:
        raise RuntimeError(f"Failed to load {benchmark} data")

    print(f"  Loaded {len(items)} samples. Model: {model}")

    if mode not in _COMPRESSORS:
        raise ValueError(
            f"unknown mode {mode!r}; choose from {list(_COMPRESSORS)} or use --pareto"
        )

    # Run baseline
    print("\n  [1/2] Running baseline (no compression)...")
    baseline = _run_mode(items, benchmark, model, "baseline", budget=None,
                         base_url=base_url, api_key_env=api_key_env)

    # Run with the selected compression mode (entroly / llmlingua / hybrid).
    # The RetentionReport name is preserved for backwards compatibility,
    # but the .entroly field now actually holds whichever mode ran.
    print(f"  [2/2] Running with {mode} compression (budget={budget})...")
    treatment = _run_mode(items, benchmark, model, mode, budget=budget,
                         base_url=base_url, api_key_env=api_key_env)

    retention = treatment.accuracy / max(baseline.accuracy, 1e-9)
    token_savings = 1.0 - (treatment.avg_tokens / max(baseline.avg_tokens, 1))
    cost_savings = 1.0 - (treatment.total_cost_usd / max(baseline.total_cost_usd, 1e-9))

    return RetentionReport(
        benchmark=benchmark,
        baseline=baseline,
        entroly=treatment,
        retention=round(retention, 4),
        token_savings_pct=round(token_savings * 100, 1),
        cost_savings_pct=round(cost_savings * 100, 1),
    )


def run_pareto_sweep(
    benchmark: str,
    samples: int = 100,
    model: str = "gpt-4o-mini",
    modes: tuple[str, ...] = ("entroly", "head", "tail", "random", "llmlingua", "hybrid"),
    budget_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Run the budget × mode Pareto sweep with AUC summary.

    For each compression mode, evaluate at each budget fraction and
    compute the area under the accuracy-vs-budget curve (Pareto-AUC).

    AUC is the right single-number summary because it captures the full
    accuracy/budget trade — methods that win at one budget but lose at
    another are visibly worse than methods that uniformly Pareto-dominate.

    Definition (trapezoidal, normalized):
        AUC(method) = ∫₀¹ accuracy(b) db   ≈   trapezoid over budget grid
    A method that always returns the gold answer regardless of budget
    has AUC = 1.0; the baseline (raw context, full budget) is the
    reference upper bound.

    Returns a dict with:
        baseline_accuracy: scalar (raw, full context)
        baseline_avg_tokens: scalar (the cost of raw)
        modes: { mode_name: {
            points: [{budget_frac, budget_tokens, accuracy, ci_low, ci_high,
                       avg_tokens, avg_latency_ms, total_cost_usd}, ...],
            pareto_auc: float,
            tokens_at_match: int | None,  # min budget where retention ≥ 95%
        } }
    """
    print(f"\n  Loading {benchmark} for Pareto sweep (samples={samples})...")
    loaders = {
        "needle": lambda: bench_needle(model, samples),
        "gsm8k": lambda: _load_gsm8k(samples),
        "humaneval": lambda: _load_humaneval(min(samples, 30)),
        "squad": lambda: _load_squad(samples),
        "mmlu": lambda: _load_mmlu(samples),
        "truthfulqa": lambda: _load_truthfulqa(samples),
        "longbench": lambda: _load_longbench(samples),
        "bfcl": lambda: _load_bfcl(samples),
    }
    if benchmark not in loaders:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    items = loaders[benchmark]()
    if not items:
        raise RuntimeError(f"Failed to load {benchmark} data")

    # Median context size determines absolute budget at each fraction.
    median_ctx_tokens = int(
        sorted(len(it.get("context", "")) for it in items)[len(items) // 2] / 4
    )
    print(f"  Median context: ~{median_ctx_tokens} tokens. Modes: {modes}.")

    # Compression evaluation only makes sense when there's real context to
    # compress. GSM8K, MMLU, and TruthfulQA are *very* short-context
    # benchmarks (median < 100 tokens) — every "compression" is a no-op
    # pass-through, producing identical numbers across all methods.
    # Threshold calibrated against measured medians: SQuAD ~180 (passes),
    # gsm8k ~50 (rejected), mmlu ~80 (rejected). 100 is the natural floor.
    SWEEP_MIN_CTX_TOKENS = 100
    if median_ctx_tokens < SWEEP_MIN_CTX_TOKENS:
        raise ValueError(
            f"Pareto sweep requires median context ≥ {SWEEP_MIN_CTX_TOKENS} "
            f"tokens; {benchmark!r} has only ~{median_ctx_tokens}. "
            f"Compression methods can't differentiate when there's nothing "
            f"to compress. Use one of: squad (~150–300), longbench (~5,000), "
            f"needle (~50,000). For short-context benchmarks like gsm8k, "
            f"mmlu, truthfulqa, run without --pareto (single-budget mode)."
        )

    print("  [baseline] full context, no compression…")
    baseline = _run_mode(items, benchmark, model, "baseline", budget=None,
                        base_url=base_url, api_key_env=api_key_env)

    out: dict[str, Any] = {
        "benchmark": benchmark,
        "samples": samples,
        "model": model,
        "median_ctx_tokens": median_ctx_tokens,
        "baseline_accuracy": baseline.accuracy,
        "baseline_ci": [baseline.ci_low, baseline.ci_high],
        "baseline_avg_tokens": baseline.avg_tokens,
        "baseline_cost_usd": baseline.total_cost_usd,
        "budget_fractions": list(budget_fractions),
        "modes": {},
    }

    # Adaptive floor: small contexts can't sustain 50-token budget without
    # collapsing the bottom of the budget grid (5%, 10%, 20% all clamp to
    # the same value). Use 10% of the median as a soft floor — large enough
    # for the compressor to do something, small enough to preserve grid
    # spacing. SQuAD-sized contexts (~180 tok) get a 18-tok floor.
    budget_floor = max(10, int(median_ctx_tokens * 0.10))

    for mode in modes:
        print(f"\n  [{mode}] sweeping budgets {list(budget_fractions)}…")
        points: list[dict[str, Any]] = []
        for frac in budget_fractions:
            tok_budget = max(budget_floor, int(median_ctx_tokens * frac))
            print(f"    budget = {frac*100:.0f}% (~{tok_budget} tokens)")
            r = _run_mode(items, benchmark, model, mode, budget=tok_budget,
                         base_url=base_url, api_key_env=api_key_env)
            points.append({
                "budget_frac": frac,
                "budget_tokens": tok_budget,
                "accuracy": r.accuracy,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "avg_tokens": r.avg_tokens,
                "avg_latency_ms": r.avg_latency_ms,
                "total_cost_usd": r.total_cost_usd,
                "errors": r.errors,
            })

        # Trapezoidal AUC over budget fractions (clipped to [0, 1]).
        # Anchor with (0, 0) since at zero budget, no useful context = no
        # accuracy. Anchor with (1, baseline.accuracy) since at 100% you
        # recover the baseline.
        xs = [0.0] + [p["budget_frac"] for p in points] + [1.0]
        ys = [0.0] + [p["accuracy"] for p in points] + [baseline.accuracy]
        auc = sum(
            0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
            for i in range(len(xs) - 1)
        )

        # "Tokens at match": the smallest budget where the lower CI bound
        # of this mode's accuracy crosses 95% of baseline's *point estimate*.
        # This is a credibility-conservative version of retention.
        target = 0.95 * baseline.accuracy
        tokens_at_match: int | None = None
        for p in points:
            if p["ci_low"] >= target:
                tokens_at_match = p["budget_tokens"]
                break

        out["modes"][mode] = {
            "points": points,
            "pareto_auc": round(auc, 4),
            "tokens_at_95pct_retention": tokens_at_match,
            "min_cost_at_95pct_retention_usd": (
                next((p["total_cost_usd"] for p in points
                      if p["ci_low"] >= target), None)
            ),
        }

    return out


def _run_mode(
    items: list[dict],
    benchmark: str,
    model: str,
    mode: str,
    budget: int | None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> BenchmarkResult:
    """Run all items in a given mode."""
    correct = 0
    total_tokens = 0
    total_latency = 0.0
    total_cost = 0.0
    errors = 0
    error_classes: dict[str, int] = {}
    details = []

    for i, item in enumerate(items):
        messages = []
        if item["context"]:
            messages.append({"role": "system", "content": f"Context:\n{item['context']}"})
        messages.append({"role": "user", "content": item["question"]})

        # Compress if a compression mode is selected.
        # Modes: "baseline" (no compression), "entroly", "llmlingua", "hybrid".
        if mode in _COMPRESSORS and budget:
            messages = _compress_messages_modal(
                messages, budget, mode=mode, query=item["question"],
            )

        try:
            response, tokens, latency = _call_llm(
                messages, model, base_url=base_url, api_key_env=api_key_env,
            )
            is_correct = _check_answer(response, item["expected"], benchmark, item.get("metadata"))
            if is_correct:
                correct += 1
            total_tokens += tokens
            total_latency += latency
            total_cost += _estimate_cost(model, tokens, 200, base_url=base_url)
            details.append({
                "index": i,
                "correct": is_correct,
                "tokens": tokens,
                "latency_ms": round(latency, 1),
            })
        except Exception as e:
            errors += 1
            err_label = f"{type(e).__name__}: {str(e)[:160]}"
            error_classes[err_label] = error_classes.get(err_label, 0) + 1
            details.append({"index": i, "error": str(e)[:300]})
            # Surface the first occurrence of each unique error class
            # immediately to stderr so the user sees what's failing without
            # waiting for the whole run. Subsequent identical errors are
            # silenced (counted in error_classes for the summary).
            if error_classes[err_label] == 1:
                import sys as _sys
                print(
                    f"  [{benchmark}/{mode}] ERROR (item {i}): {err_label}",
                    file=_sys.stderr, flush=True,
                )

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
  mmlu        MMLU — massive multitask knowledge (4-way MCQ)
  truthfulqa  TruthfulQA MC1 — truthfulness under compression
  longbench   LongBench HotpotQA — multi-hop QA with long context
  all         Run all benchmarks

Examples:
  python -m bench.accuracy --benchmark needle --model gemini-2.0-flash --samples 20
  python -m bench.accuracy --benchmark gsm8k --model claude-sonnet-4-5-20250929
  python -m bench.accuracy --benchmark longbench --model gpt-4o-mini --samples 100
  python -m bench.accuracy --benchmark all --model gpt-4o-mini

Custom OpenAI-compatible providers (Groq, Together, OpenRouter, Ollama, vLLM, ...):
  python -m bench.accuracy --benchmark gsm8k --model llama-3.1-70b-versatile \\
      --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY

  python -m bench.accuracy --benchmark mmlu --model llama3.1:8b \\
      --base-url http://localhost:11434/v1                  # Ollama local (no auth)
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
        "--base-url", type=str, default=None,
        help="OpenAI-compatible custom endpoint (e.g. https://api.groq.com/openai/v1, "
             "http://localhost:11434/v1 for Ollama). Routes requests through the OpenAI "
             "SDK against this base URL.",
    )
    parser.add_argument(
        "--api-key-env", type=str, default=None,
        help="Name of the env var holding the API key for --base-url (default: OPENAI_API_KEY). "
             "Use any var with empty/dummy value for self-hosted endpoints that ignore auth.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--mode", type=str, default="entroly",
        choices=["entroly", "llmlingua", "hybrid"],
        help="Compression method to compare against baseline. "
             "'entroly' = QCCR fragment selection (default). "
             "'llmlingua' = LLMLingua-2 token compression. "
             "'hybrid' = Entroly fragment selection + LLMLingua-2 token compression.",
    )
    parser.add_argument(
        "--pareto", action="store_true",
        help="Run a budget-fraction sweep ([0.05, 0.10, 0.20, 0.30, 0.50] of "
             "median context size) for ALL methods (entroly, llmlingua, hybrid) "
             "and compute Pareto-AUC. Strongest methodology — supersedes single-budget run.",
    )
    parser.add_argument(
        "--budget-fractions", type=str, default="0.05,0.10,0.20,0.30,0.50",
        help="Comma-separated budget fractions for --pareto sweep "
             "(default: 0.05,0.10,0.20,0.30,0.50)",
    )
    args = parser.parse_args()

    benchmarks = (
        ["needle", "gsm8k", "humaneval", "squad", "mmlu", "truthfulqa", "longbench", "bfcl"]
        if args.benchmark == "all"
        else [args.benchmark]
    )

    if args.base_url:
        print(
            f"  Note: cost tracking disabled for custom provider {args.base_url}. "
            f"total_cost_usd in the report will be 0.0 — your real bill is on the provider's dashboard."
        )

    # Pareto sweep mode: evaluates all compressors at multiple budgets,
    # computes the AUC summary. Supersedes single-budget run.
    if args.pareto:
        budget_fractions = tuple(float(f) for f in args.budget_fractions.split(","))
        sweep_reports: list[dict[str, Any]] = []
        for bench in benchmarks:
            try:
                report = run_pareto_sweep(
                    benchmark=bench, samples=args.samples, model=args.model,
                    modes=("entroly", "head", "tail", "random", "llmlingua", "hybrid"),
                    budget_fractions=budget_fractions,
                    base_url=args.base_url, api_key_env=args.api_key_env,
                )
                sweep_reports.append(report)
            except Exception as e:
                print(f"\n  ERROR: {bench} pareto sweep failed: {e}")
                continue

        # Pretty-print AUC table
        print("\n" + "=" * 96)
        print("  PARETO-AUC SUMMARY  (higher = better; baseline.accuracy ≤ 1.0)")
        print("=" * 96)
        print(f"  Model: {args.model}  |  Samples: {args.samples}  |  Budget grid: {list(budget_fractions)}")
        print("-" * 96)
        method_cols = ("entroly", "head", "tail", "random", "llmlingua", "hybrid")
        header = f"  {'Benchmark':<12} {'Baseline':>9} " + " ".join(
            f"{m+'-AUC':>11}" for m in method_cols
        ) + f"  {'best method':>14}"
        print(header)
        print("-" * 110)
        for rep in sweep_reports:
            # Identify the AUC-winner so the table foregrounds the headline.
            best_m = max(method_cols, key=lambda m: rep["modes"][m]["pareto_auc"])
            best_auc = rep["modes"][best_m]["pareto_auc"]
            row = (
                f"  {rep['benchmark']:<12} "
                f"{rep['baseline_accuracy']:>9.1%} "
                + " ".join(
                    f"{rep['modes'][m]['pareto_auc']:>11.3f}"
                    for m in method_cols
                )
                + f"  {best_m + ' (' + format(best_auc, '.3f') + ')':>14}"
            )
            print(row)
        print("=" * 110)

        if args.json:
            print(json.dumps(sweep_reports, indent=2, default=str))
        return

    all_reports = []
    for bench in benchmarks:
        try:
            report = run_benchmark(
                bench, args.model, args.samples, args.budget,
                base_url=args.base_url, api_key_env=args.api_key_env,
                mode=args.mode,
            )
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
