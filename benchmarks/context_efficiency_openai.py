"""Run paired LongBench trials through OpenAI and Entroly Context Commits.

This is a paid, networked benchmark. It writes each trial before continuing so
an interrupted run can resume without re-billing completed conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmarks.context_efficiency_frontier import (
    SCHEMA_VERSION,
    Trial,
    analyze_frontier,
    load_trials,
)
from benchmarks.context_efficiency_report import render_markdown
from entroly.context_commit import create_context_commit, replay_context

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_BUDGET = 2_000
DEFAULT_SEED = 42
PRICING_REFERENCE = (
    "openai:gpt-4o-mini:2026-07-11:usd-per-1m="
    "input-0.15,cached-input-0.075,output-0.60;"
    "source=https://developers.openai.com/api/docs/models/gpt-4o-mini"
)
INPUT_USD_PER_MILLION = 0.15
CACHED_INPUT_USD_PER_MILLION = 0.075
OUTPUT_USD_PER_MILLION = 0.60
SYSTEM_PROMPT = (
    "Answer the question using only the supplied context. Return only the "
    "shortest correct answer with no explanation."
)


@dataclass(frozen=True)
class WorkloadItem:
    task_id: str
    context: str
    question: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class ProviderObservation:
    response_text: str
    request_id: str
    prompt_tokens: int
    cached_prompt_tokens: int
    reasoning_tokens: int
    completion_tokens: int
    latency_ms: float


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_longbench_items(rows: Iterable[dict[str, Any]]) -> list[WorkloadItem]:
    items: list[WorkloadItem] = []
    for index, row in enumerate(rows):
        context = str(row.get("context", ""))
        question = str(row.get("question", ""))
        metadata = row.get("metadata")
        raw_answers = metadata.get("all_answers", []) if isinstance(metadata, dict) else []
        if not raw_answers and row.get("expected"):
            raw_answers = [row["expected"]]
        answers = tuple(str(answer).strip() for answer in raw_answers if str(answer).strip())
        if not context or not question or not answers:
            continue
        identity = _stable_digest(
            {"context": context, "question": question, "answers": answers}
        )[:12]
        items.append(
            WorkloadItem(
                task_id=f"hotpotqa-{index:04d}-{identity}",
                context=context,
                question=question,
                answers=answers,
            )
        )
    if not items:
        raise ValueError("LongBench loader produced no valid items")
    return items


def workload_version(items: Iterable[WorkloadItem]) -> str:
    manifest = [
        {"task_id": item.task_id, "question": item.question, "answers": item.answers}
        for item in items
    ]
    return "longbench-hotpotqa-test-seed42-" + _stable_digest(manifest)[:16]


def _selected_context(commit: dict[str, Any]) -> str:
    return "\n\n".join(
        str(chunk.get("text", ""))
        for chunk in replay_context(commit)
        if str(chunk.get("text", "")).strip()
    )


def _answer_present(text: str, answers: Iterable[str]) -> bool:
    folded = text.casefold()
    return any(answer.casefold() in folded for answer in answers)


def _usage_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"provider response has invalid {name}")
    return value


def call_openai(client: Any, *, model: str, context: str, question: str) -> ProviderObservation:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=64,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
    )
    latency_ms = (time.perf_counter() - started) * 1_000
    usage = getattr(response, "usage", None)
    if usage is None:
        raise ValueError("provider response is missing usage")
    prompt_tokens = _usage_int(getattr(usage, "prompt_tokens", None), "prompt_tokens")
    completion_tokens = _usage_int(
        getattr(usage, "completion_tokens", None), "completion_tokens"
    )
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
    reasoning = (
        getattr(completion_details, "reasoning_tokens", 0)
        if completion_details
        else 0
    )
    cached = 0 if cached is None else _usage_int(cached, "cached_tokens")
    reasoning = 0 if reasoning is None else _usage_int(reasoning, "reasoning_tokens")
    if cached > prompt_tokens:
        raise ValueError("cached_tokens exceeds prompt_tokens")
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("provider response has no choices")
    text = str(getattr(choices[0].message, "content", "") or "").strip()
    request_id = str(getattr(response, "id", "") or "").strip()
    if not request_id:
        raise ValueError("provider response is missing request id")
    return ProviderObservation(
        response_text=text,
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached,
        reasoning_tokens=reasoning,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


def _cost_usd(observation: ProviderObservation) -> float:
    uncached = observation.prompt_tokens - observation.cached_prompt_tokens
    return (
        uncached * INPUT_USD_PER_MILLION
        + observation.cached_prompt_tokens * CACHED_INPUT_USD_PER_MILLION
        + observation.completion_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def _success_trial(
    *,
    item: WorkloadItem,
    condition: str,
    model: str,
    version: str,
    selected_context: str,
    commit_id: str | None,
    observation: ProviderObservation,
) -> Trial:
    correct = _answer_present(observation.response_text, item.answers)
    return Trial.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "workload": "LongBench HotpotQA",
            "workload_version": version,
            "task_id": item.task_id,
            "provider": "openai",
            "model": model,
            "provider_request_id": observation.request_id,
            "usage_source": "provider_response",
            "cost_source": "pricing_snapshot",
            "cost_source_reference": PRICING_REFERENCE,
            "outcome": "success",
            "error_type": None,
            "replicate": 0,
            "condition": condition,
            "scorer": "longbench-hotpotqa-answer-only-v1",
            "task_score": float(correct),
            "evidence_recall": float(_answer_present(selected_context, item.answers)),
            "unsupported_claim_rate": float(not correct),
            "context_tokens": observation.prompt_tokens,
            "reasoning_tokens": observation.reasoning_tokens,
            "output_tokens": observation.completion_tokens,
            "billed_cost_usd": _cost_usd(observation),
            "latency_ms": observation.latency_ms,
            "context_commit_id": commit_id,
        }
    )


def _error_trial(
    *,
    item: WorkloadItem,
    condition: str,
    model: str,
    version: str,
    selected_context: str,
    commit_id: str | None,
    error: Exception,
) -> Trial:
    error_type = type(error).__name__
    error_id = _stable_digest(
        {"task": item.task_id, "condition": condition, "error": error_type}
    )[:20]
    usage_source = "provider_error" if error_type != "ContextPreparationError" else "runner_error"
    return Trial.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "workload": "LongBench HotpotQA",
            "workload_version": version,
            "task_id": item.task_id,
            "provider": "openai",
            "model": model,
            "provider_request_id": f"error_{error_id}",
            "usage_source": usage_source,
            "cost_source": "pricing_snapshot",
            "cost_source_reference": PRICING_REFERENCE,
            "outcome": "error",
            "error_type": error_type,
            "replicate": 0,
            "condition": condition,
            "scorer": "longbench-hotpotqa-answer-only-v1",
            "task_score": 0.0,
            "evidence_recall": float(_answer_present(selected_context, item.answers)),
            "unsupported_claim_rate": 1.0,
            "context_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "billed_cost_usd": 0.0,
            "latency_ms": 0.0,
            "context_commit_id": commit_id,
        }
    )


class ContextPreparationError(RuntimeError):
    """Raised when Entroly cannot produce replayable selected context."""


def _append_trial(path: Path, trial: Trial) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(trial.as_record(), sort_keys=True) + "\n")


def run_trials(
    *,
    items: list[WorkloadItem],
    client: Any,
    output: Path,
    model: str = DEFAULT_MODEL,
    token_budget: int = DEFAULT_BUDGET,
    seed: int = DEFAULT_SEED,
    resume: bool = False,
) -> list[Trial]:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; use --resume or choose another path")
    existing = load_trials(output) if output.exists() else []
    completed = {(trial.task_id, trial.condition) for trial in existing}
    version = workload_version(items)
    if any(trial.workload_version != version or trial.model != model for trial in existing):
        raise ValueError("existing trials do not match this workload/model configuration")

    commits_dir = output.parent / f"{output.stem}_context_commits"
    commits_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index, item in enumerate(items, 1):
        conditions = ["raw", "entroly"]
        rng.shuffle(conditions)
        for condition in conditions:
            if (item.task_id, condition) in completed:
                continue
            selected_context = item.context
            commit: dict[str, Any] | None = None
            commit_id: str | None = None
            try:
                if condition == "entroly":
                    chunk_tokens = min(360, token_budget)
                    commit = create_context_commit(
                        [(f"{item.task_id}.txt", item.context)],
                        query=item.question,
                        token_budget=token_budget,
                        chunk_tokens=chunk_tokens,
                        overlap_tokens=min(32, max(1, chunk_tokens // 8)),
                    )
                    commit_id = str(commit["commit_id"])
                    selected_context = _selected_context(commit)
                    if not selected_context:
                        raise ContextPreparationError("Context Commit selected no context")
                    commit_path = commits_dir / f"{item.task_id}.json"
                    commit_path.write_text(
                        json.dumps(commit, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                observation = call_openai(
                    client,
                    model=model,
                    context=selected_context,
                    question=item.question,
                )
                trial = _success_trial(
                    item=item,
                    condition=condition,
                    model=model,
                    version=version,
                    selected_context=selected_context,
                    commit_id=commit_id,
                    observation=observation,
                )
            except Exception as error:  # Every failed request remains in the matrix.
                trial = _error_trial(
                    item=item,
                    condition=condition,
                    model=model,
                    version=version,
                    selected_context=selected_context,
                    commit_id=commit_id,
                    error=error,
                )
            _append_trial(output, trial)
            existing.append(trial)
            completed.add((item.task_id, condition))
            print(
                f"[{index}/{len(items)}] {condition}: {trial.outcome} "
                f"score={trial.task_score:.0f} context={trial.context_tokens}"
            )
    return existing


def _load_longbench(samples: int) -> list[WorkloadItem]:
    from bench.accuracy import _load_longbench as load_rows

    return prepare_longbench_items(load_rows(samples))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    from openai import OpenAI

    items = _load_longbench(args.samples)
    trials = run_trials(
        items=items,
        client=OpenAI(max_retries=2),
        output=args.output,
        model=args.model,
        token_budget=args.budget,
        seed=args.seed,
        resume=args.resume,
    )
    report = analyze_frontier(trials)
    report_path = args.output.with_suffix(".report.json")
    markdown_path = args.output.with_suffix(".report.md")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output}, {report_path}, and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
