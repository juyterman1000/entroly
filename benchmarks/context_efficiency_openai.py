"""Run paired LongBench trials through OpenAI and Entroly Context Commits.

This is a paid, networked benchmark. It writes each trial before continuing so
an interrupted run can resume without re-billing completed conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import string
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmarks.context_efficiency_frontier import (
    COST_SOURCES,
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
SELECTIONS = ("random", "shortest-context")
LONG_BENCH_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONG_BENCH_METRICS_SHA256 = (
    "587b87e8ea520f6093ebd061a7a99b0fb53ade84356ba16e95c518b28fe23d85"
)
SCORER = (
    "longbench-v1-hotpotqa-official-qa-f1@"
    f"{LONG_BENCH_REVISION}:metrics-sha256:{LONG_BENCH_METRICS_SHA256}"
)
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


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    cost_source: str
    cost_source_reference: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float


@dataclass(frozen=True)
class FrozenBaseline:
    condition: str
    name: str
    version: str
    source: str
    manifest_sha256: str
    contexts: dict[str, str]

    def experiment_identity(self) -> dict[str, str]:
        return {
            "condition": self.condition,
            "manifest_sha256": self.manifest_sha256,
            "name": self.name,
            "source": self.source,
            "version": self.version,
        }


OPENAI_PROVIDER = ProviderConfig(
    name="openai",
    cost_source="pricing_snapshot",
    cost_source_reference=PRICING_REFERENCE,
    input_usd_per_million=INPUT_USD_PER_MILLION,
    cached_input_usd_per_million=CACHED_INPUT_USD_PER_MILLION,
    output_usd_per_million=OUTPUT_USD_PER_MILLION,
)


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_digest(path),
            }
        )
    return _stable_digest(entries)


def _git_value(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def entroly_runtime_identity() -> dict[str, Any]:
    """Bind every trial to the source and native implementation actually loaded."""
    import entroly

    package_root = Path(entroly.__file__).resolve().parent
    repository_root_text = _git_value(package_root, "rev-parse", "--show-toplevel")
    repository_root = Path(repository_root_text) if repository_root_text else None
    git_status = (
        _git_value(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
        if repository_root
        else None
    )
    native_version: str | None
    native_artifacts_sha256: str | None = None
    try:
        native_version = importlib.metadata.version("entroly-core")
        import entroly_core

        native_module = Path(entroly_core.__file__).resolve()
        native_root = native_module.parent
        native_artifacts = sorted(
            path
            for path in native_root.rglob("*")
            if path.is_file() and path.suffix in {".dylib", ".pyd", ".so"}
        )
        if native_module.suffix in {".dylib", ".pyd", ".so"}:
            native_artifacts = sorted(set(native_artifacts) | {native_module})
        if native_artifacts:
            native_artifacts_sha256 = _stable_digest(
                [
                    {
                        "path": path.relative_to(native_root).as_posix(),
                        "sha256": _file_digest(path),
                    }
                    for path in native_artifacts
                ]
            )
    except (ImportError, importlib.metadata.PackageNotFoundError):
        native_version = None

    return {
        "benchmark_frontier_sha256": _file_digest(
            Path(__file__).with_name("context_efficiency_frontier.py")
        ),
        "benchmark_runner_sha256": _file_digest(Path(__file__)),
        "entroly_git_dirty": bool(git_status),
        "entroly_git_revision": (
            _git_value(repository_root, "rev-parse", "HEAD")
            if repository_root
            else None
        ),
        "entroly_git_status_sha256": (
            hashlib.sha256(git_status.encode("utf-8")).hexdigest()
            if git_status
            else None
        ),
        "entroly_package_version": getattr(entroly, "__version__", None),
        "entroly_python_source_sha256": _source_tree_digest(package_root),
        "native_artifacts_sha256": native_artifacts_sha256,
        "native_distribution_version": native_version,
    }


def _manifest_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline manifest {name} must be a non-empty string")
    return value.strip()


def load_frozen_baseline(
    path: Path, items: Iterable[WorkloadItem]
) -> FrozenBaseline:
    """Validate exact precomputed contexts from an algorithmic or external baseline."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load baseline manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("baseline manifest must be a JSON object")
    if payload.get("schema_version") != "entroly.context-efficiency-baseline.v1":
        raise ValueError("unsupported baseline manifest schema_version")
    condition = _manifest_text(payload, "condition")
    if condition not in {"algorithmic_baseline", "external_baseline"}:
        raise ValueError(
            "baseline manifest condition must be algorithmic_baseline or "
            "external_baseline"
        )
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("baseline manifest baseline must be an object")
    name = _manifest_text(baseline, "name")
    version = _manifest_text(baseline, "version")
    source = _manifest_text(baseline, "source")
    config = baseline.get("config")
    if not isinstance(config, dict) or not config:
        raise ValueError("baseline manifest config must be a non-empty object")
    rows = payload.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("baseline manifest tasks must be a non-empty array")

    materialized = list(items)
    expected = {item.task_id: item for item in materialized}
    contexts: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each baseline task must be an object")
        task_id = _manifest_text(row, "task_id")
        if task_id in contexts:
            raise ValueError(f"duplicate baseline task {task_id!r}")
        item = expected.get(task_id)
        if item is None:
            raise ValueError(f"unexpected baseline task {task_id!r}")
        source_digest = _manifest_text(row, "source_context_sha256")
        if source_digest != hashlib.sha256(item.context.encode("utf-8")).hexdigest():
            raise ValueError(f"baseline source digest mismatch for {task_id!r}")
        selected = row.get("selected_context")
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError(f"baseline selected context is empty for {task_id!r}")
        selected_digest = _manifest_text(row, "selected_context_sha256")
        if selected_digest != hashlib.sha256(selected.encode("utf-8")).hexdigest():
            raise ValueError(f"baseline selected digest mismatch for {task_id!r}")
        contexts[task_id] = selected

    missing = sorted(set(expected).difference(contexts))
    if missing:
        raise ValueError("baseline manifest is missing tasks: " + ", ".join(missing))
    return FrozenBaseline(
        condition=condition,
        name=name,
        version=version,
        source=source,
        manifest_sha256=_stable_digest(payload),
        contexts=contexts,
    )


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


def workload_version(
    items: Iterable[WorkloadItem], *, selection_policy: str
) -> str:
    manifest = [
        {
            "task_id": item.task_id,
            "context_sha256": hashlib.sha256(item.context.encode("utf-8")).hexdigest(),
            "question": item.question,
            "answers": item.answers,
        }
        for item in items
    ]
    return (
        f"longbench-v1-hotpotqa:{LONG_BENCH_REVISION}:{selection_policy}:"
        + _stable_digest(manifest)
    )


def _selected_context(commit: dict[str, Any]) -> str:
    return "\n\n".join(
        str(chunk.get("text", ""))
        for chunk in replay_context(commit)
        if str(chunk.get("text", "")).strip()
    )


def normalize_answer(text: str) -> str:
    """Match the official LongBench English-QA normalization."""
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation)
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(ground_truth).split()
    if not predicted or not expected:
        return float(predicted == expected)
    shared = Counter(predicted) & Counter(expected)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _answer_present(text: str, answers: Iterable[str]) -> bool:
    normalized_text = normalize_answer(text)
    return any(
        normalized_answer and normalized_answer in normalized_text
        for answer in answers
        if (normalized_answer := normalize_answer(answer))
    )


def _exact_answer(prediction: str, answers: Iterable[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return any(
        normalized_prediction == normalize_answer(answer) for answer in answers
    )


def _answer_score(prediction: str, answers: Iterable[str]) -> float:
    return max(qa_f1_score(prediction, answer) for answer in answers)


def _response_supported(response: str, context: str) -> bool:
    normalized_response = normalize_answer(response)
    return bool(normalized_response) and normalized_response in normalize_answer(context)


def _usage_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"provider response has invalid {name}")
    return value


def call_openai(
    client: Any,
    *,
    model: str,
    context: str,
    question: str,
    max_output_tokens: int = 64,
    reasoning_effort: str | None = None,
) -> ProviderObservation:
    started = time.perf_counter()
    request: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**request)
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


def _cost_usd(observation: ProviderObservation, provider: ProviderConfig) -> float:
    uncached = observation.prompt_tokens - observation.cached_prompt_tokens
    return (
        uncached * provider.input_usd_per_million
        + observation.cached_prompt_tokens * provider.cached_input_usd_per_million
        + observation.completion_tokens * provider.output_usd_per_million
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
    provider: ProviderConfig,
    experiment_config: str,
) -> Trial:
    score = _answer_score(observation.response_text, item.answers)
    task_success = _exact_answer(observation.response_text, item.answers)
    has_claim = bool(normalize_answer(observation.response_text))
    return Trial.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "workload": "LongBench HotpotQA",
            "workload_version": version,
            "task_id": item.task_id,
            "provider": provider.name,
            "model": model,
            "provider_request_id": observation.request_id,
            "usage_source": "provider_response",
            "cost_source": provider.cost_source,
            "cost_source_reference": provider.cost_source_reference,
            "outcome": "success",
            "error_type": None,
            "context_sha256": hashlib.sha256(selected_context.encode("utf-8")).hexdigest(),
            "response_text": observation.response_text,
            "response_sha256": hashlib.sha256(
                observation.response_text.encode("utf-8")
            ).hexdigest(),
            "replicate": 0,
            "condition": condition,
            "scorer": SCORER,
            "task_score": score,
            "task_success": task_success,
            "evidence_recall": float(_answer_present(selected_context, item.answers)),
            "unsupported_claim_rate": float(
                has_claim and not _response_supported(observation.response_text, selected_context)
            ),
            "experiment_config": experiment_config,
            "usage_observed": True,
            "cost_observed": True,
            "error_fingerprint": None,
            "context_tokens": observation.prompt_tokens,
            "reasoning_tokens": observation.reasoning_tokens,
            "output_tokens": observation.completion_tokens,
            "billed_cost_usd": _cost_usd(observation, provider),
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
    provider: ProviderConfig,
    experiment_config: str,
    latency_ms: float,
) -> Trial:
    error_type = type(error).__name__
    error_fingerprint = _stable_digest(
        {
            "error_type": error_type,
            "message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "status_code": getattr(error, "status_code", None),
        }
    )
    error_id = _stable_digest(
        {
            "task": item.task_id,
            "condition": condition,
            "error_fingerprint": error_fingerprint,
        }
    )[:20]
    usage_source = "provider_error" if error_type != "ContextPreparationError" else "runner_error"
    return Trial.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "workload": "LongBench HotpotQA",
            "workload_version": version,
            "task_id": item.task_id,
            "provider": provider.name,
            "model": model,
            "provider_request_id": f"error_{error_id}",
            "usage_source": usage_source,
            "cost_source": provider.cost_source,
            "cost_source_reference": provider.cost_source_reference,
            "outcome": "error",
            "error_type": error_type,
            "context_sha256": hashlib.sha256(selected_context.encode("utf-8")).hexdigest(),
            "response_text": None,
            "response_sha256": None,
            "replicate": 0,
            "condition": condition,
            "scorer": SCORER,
            "task_score": 0.0,
            "task_success": False,
            "evidence_recall": float(_answer_present(selected_context, item.answers)),
            "unsupported_claim_rate": 0.0,
            "experiment_config": experiment_config,
            "usage_observed": False,
            "cost_observed": False,
            "error_fingerprint": error_fingerprint,
            "context_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "billed_cost_usd": 0.0,
            "latency_ms": latency_ms,
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
    provider: ProviderConfig = OPENAI_PROVIDER,
    max_output_tokens: int = 64,
    reasoning_effort: str | None = None,
    selection_policy: str = "preselected",
    frozen_baseline: FrozenBaseline | None = None,
) -> list[Trial]:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    task_ids = [item.task_id for item in items]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values must be unique")
    missing_evidence = [
        item.task_id for item in items if not _answer_present(item.context, item.answers)
    ]
    if missing_evidence:
        raise ValueError(
            "raw context does not contain normalized answer evidence for: "
            + ", ".join(missing_evidence)
        )
    experiment_config = json.dumps(
        {
            "cached_input_usd_per_million": provider.cached_input_usd_per_million,
            "cost_source": provider.cost_source,
            "cost_source_reference": provider.cost_source_reference,
            "input_usd_per_million": provider.input_usd_per_million,
            "entroly_runtime": entroly_runtime_identity(),
            "longbench_metrics_sha256": LONG_BENCH_METRICS_SHA256,
            "longbench_revision": LONG_BENCH_REVISION,
            "max_output_tokens": max_output_tokens,
            "output_usd_per_million": provider.output_usd_per_million,
            "reasoning_effort": reasoning_effort,
            "runner_contract": "context-efficiency-openai-v2",
            "seed": seed,
            "selection_policy": selection_policy,
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "temperature": 0,
            "token_budget": token_budget,
            "frozen_baseline": (
                frozen_baseline.experiment_identity() if frozen_baseline else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; use --resume or choose another path")
    existing = load_trials(output) if output.exists() else []
    completed = {(trial.task_id, trial.condition) for trial in existing}
    version = workload_version(items, selection_policy=selection_policy)
    if any(
        trial.workload_version != version
        or trial.model != model
        or trial.provider != provider.name
        or trial.experiment_config != experiment_config
        for trial in existing
    ):
        raise ValueError("existing trials do not match this workload/model configuration")

    commits_dir = output.parent / f"{output.stem}_context_commits"
    commits_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index, item in enumerate(items, 1):
        conditions = ["raw", "entroly"]
        if frozen_baseline is not None:
            conditions.append(frozen_baseline.condition)
        rng.shuffle(conditions)
        for condition in conditions:
            if (item.task_id, condition) in completed:
                continue
            selected_context = item.context
            commit: dict[str, Any] | None = None
            commit_id: str | None = None
            provider_started: float | None = None
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
                elif frozen_baseline is not None and condition == frozen_baseline.condition:
                    selected_context = frozen_baseline.contexts[item.task_id]
                provider_started = time.perf_counter()
                observation = call_openai(
                    client,
                    model=model,
                    context=selected_context,
                    question=item.question,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )
                trial = _success_trial(
                    item=item,
                    condition=condition,
                    model=model,
                    version=version,
                    selected_context=selected_context,
                    commit_id=commit_id,
                    observation=observation,
                    provider=provider,
                    experiment_config=experiment_config,
                )
            except Exception as error:  # Every failed request remains in the matrix.
                failed_latency_ms = (
                    (time.perf_counter() - provider_started) * 1_000
                    if provider_started is not None
                    else 0.0
                )
                trial = _error_trial(
                    item=item,
                    condition=condition,
                    model=model,
                    version=version,
                    selected_context=selected_context,
                    commit_id=commit_id,
                    error=error,
                    provider=provider,
                    experiment_config=experiment_config,
                    latency_ms=failed_latency_ms,
                )
            _append_trial(output, trial)
            existing.append(trial)
            completed.add((item.task_id, condition))
            print(
                f"[{index}/{len(items)}] {condition}: {trial.outcome} "
                f"score={trial.task_score:.0f} context={trial.context_tokens}"
            )
    return existing


def select_longbench_rows(
    rows: Iterable[dict[str, Any]], samples: int, selection: str
) -> list[dict[str, Any]]:
    candidates = list(rows)
    if selection == "shortest-context":
        candidates.sort(
            key=lambda row: (
                len(str(row.get("context", ""))),
                _stable_digest(row),
            )
        )
    elif selection != "random":
        raise ValueError(f"unsupported LongBench selection: {selection}")
    return candidates[:samples]


def _load_longbench(samples: int, selection: str = "random") -> list[WorkloadItem]:
    from bench.accuracy import _load_longbench as load_rows

    source_rows = load_rows(200 if selection == "shortest-context" else samples)
    return prepare_longbench_items(select_longbench_rows(source_rows, samples, selection))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--selection",
        choices=SELECTIONS,
        default="random",
        help="Dataset selection; shortest-context is a biased, quick calibration subset.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--provider")
    parser.add_argument("--api-key-env")
    parser.add_argument("--cost-source", choices=COST_SOURCES)
    parser.add_argument("--cost-source-reference")
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--cached-input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument(
        "--frozen-baseline-manifest",
        type=Path,
        help="Validated JSON manifest of exact algorithmic or external baseline contexts.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high"),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60).",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")

    from openai import OpenAI

    if args.base_url:
        if not args.provider:
            parser.error("--provider is required with --base-url")
        if not args.cost_source_reference:
            parser.error("--cost-source-reference is required with --base-url")
        if args.api_key_env and not os.environ.get(args.api_key_env):
            parser.error(f"environment variable {args.api_key_env!r} is not set")
        api_key = os.environ[args.api_key_env] if args.api_key_env else "no-auth"
        client = OpenAI(
            base_url=args.base_url,
            api_key=api_key,
            max_retries=0,
            timeout=args.request_timeout,
        )
        provider = ProviderConfig(
            name=args.provider,
            cost_source=args.cost_source or "self_hosted_no_api_fee",
            cost_source_reference=args.cost_source_reference,
            input_usd_per_million=args.input_usd_per_million or 0.0,
            cached_input_usd_per_million=args.cached_input_usd_per_million or 0.0,
            output_usd_per_million=args.output_usd_per_million or 0.0,
        )
    else:
        client = OpenAI(max_retries=2, timeout=args.request_timeout)
        provider = OPENAI_PROVIDER

    items = _load_longbench(args.samples, args.selection)
    frozen_baseline = (
        load_frozen_baseline(args.frozen_baseline_manifest, items)
        if args.frozen_baseline_manifest
        else None
    )
    trials = run_trials(
        items=items,
        client=client,
        output=args.output,
        model=args.model,
        token_budget=args.budget,
        seed=args.seed,
        resume=args.resume,
        provider=provider,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        selection_policy=args.selection,
        frozen_baseline=frozen_baseline,
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
