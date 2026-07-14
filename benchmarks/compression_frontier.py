"""Auditable matched-target context-compression frontier.

The benchmark builds long contexts from a frozen, answerable SQuAD v2 subset,
runs byte-identical inputs through isolated Entroly and Headroom environments,
and measures the achieved keep ratio and exact retained-answer recall.  An
optional local Ollama pass evaluates short-answer quality without sending the
dataset or compressed outputs to a hosted service.

The publication gate is deliberately fail closed.  A participant is never
called superior because it merely ignored a requested compression target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA_VERSION = "entroly.compression-frontier.v1"
DATASET_ID = "rajpurkar/squad_v2"
DATASET_SPLIT = "validation"
TOKENIZER = "o200k_base"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TARGET_RATIOS = (0.5, 0.25, 0.125)
DEFAULT_TRIALS = 60
DEFAULT_DISTRACTORS = 15
DEFAULT_SEED = 20260715
DEFAULT_RUNS = 2
DEFAULT_WARMUPS = 1
DEFAULT_ANSWER_RATIO = 0.25
DEFAULT_ANSWER_TRIALS = 8
_SUBPROCESS_ENV_ALLOWLIST = (
    "APPDATA",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_ANSWER_PROMPT_TEMPLATE = (
    "Use only the context below. Answer the question with only the shortest "
    "supported answer phrase. If the context does not contain the answer, "
    "reply UNKNOWN.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)


@dataclass(frozen=True, slots=True)
class SourceRow:
    example_id: str
    question: str
    context: str
    answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Trial:
    trial_id: str
    example_id: str
    question: str
    answers: tuple[str, ...]
    content: str
    gold_document: int
    document_ids: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value["answers"] = list(self.answers)
        value["document_ids"] = list(self.document_ids)
        value["content_sha256"] = _sha256(self.content)
        return value


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _distribution_record_sha256(package: str) -> str | None:
    try:
        record = importlib.metadata.distribution(package).read_text("RECORD")
    except importlib.metadata.PackageNotFoundError:
        return None
    return _sha256(record) if record else None


def _versions(packages: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _implementation_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        # Git may materialize text files with CRLF on Windows and LF in CI.
        # Fingerprint source semantics rather than checkout-specific newlines.
        source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


def _token_counter() -> tuple[Callable[[str], int], str]:
    import tiktoken

    encoding = tiktoken.get_encoding(TOKENIZER)

    def count(value: str) -> int:
        return len(encoding.encode(value, disallowed_special=()))

    return count, importlib.metadata.version("tiktoken")


def _load_source_rows() -> tuple[list[SourceRow], str]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - optional environment
        raise RuntimeError("install the neural-benchmark extra") from error
    try:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    except Exception as error:  # pragma: no cover - cache state dependent
        raise RuntimeError(
            "SQuAD v2 is not cached; this benchmark refuses an implicit download"
        ) from error

    rows: list[SourceRow] = []
    seen_contexts: set[str] = set()
    for record in dataset:
        raw_answers = record.get("answers", {}).get("text", [])
        answers = tuple(dict.fromkeys(str(value).strip() for value in raw_answers))
        answers = tuple(value for value in answers if len(value) >= 2)
        context = str(record.get("context", "")).strip()
        if not answers or not context or context in seen_contexts:
            continue
        if not any(answer.casefold() in context.casefold() for answer in answers):
            continue
        seen_contexts.add(context)
        rows.append(
            SourceRow(
                example_id=str(record["id"]),
                question=str(record["question"]).strip(),
                context=context,
                answers=answers,
            )
        )
    return rows, str(getattr(dataset, "_fingerprint", "unknown"))


def build_trials(
    rows: Sequence[SourceRow],
    *,
    trials: int,
    distractors: int,
    seed: int,
) -> list[Trial]:
    if trials < 2:
        raise ValueError("trials must be at least 2")
    if distractors < 3:
        raise ValueError("distractors must be at least 3")
    if len(rows) < trials + distractors + 1:
        raise ValueError("dataset does not contain enough unique contexts")

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    gold_rows = shuffled[:trials]
    pool = shuffled[trials : min(len(shuffled), trials + max(2_000, trials * 20))]
    built: list[Trial] = []
    for source in gold_rows:
        answer_keys = tuple(answer.casefold() for answer in source.answers)
        eligible = [
            candidate
            for candidate in pool
            if candidate.example_id != source.example_id
            and not any(key in candidate.context.casefold() for key in answer_keys)
        ]
        if len(eligible) < distractors:
            raise ValueError(f"insufficient distractors for {source.example_id}")
        selected = rng.sample(eligible, distractors)
        gold_document = rng.randrange(distractors + 1)
        documents = [candidate.context for candidate in selected]
        document_ids = [candidate.example_id for candidate in selected]
        documents.insert(gold_document, source.context)
        document_ids.insert(gold_document, source.example_id)
        content = json.dumps(
            {
                "query_result": {
                    "documents": [
                        {
                            "document_id": document_id,
                            "rank": index + 1,
                            "retrieval_score": round(1.0 - index / 100.0, 4),
                            "text": text,
                        }
                        for index, (document_id, text) in enumerate(
                            zip(document_ids, documents)
                        )
                    ],
                    "returned": len(documents),
                }
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        trial_id = f"squad-{_sha256(source.example_id + content)[:16]}"
        built.append(
            Trial(
                trial_id=trial_id,
                example_id=source.example_id,
                question=source.question,
                answers=source.answers,
                content=content,
                gold_document=gold_document,
                document_ids=tuple(document_ids),
            )
        )
    return built


def _messages(trial: Trial) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": trial.question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{trial.trial_id}",
                    "type": "function",
                    "function": {"name": "search_corpus", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{trial.trial_id}",
            "content": trial.content,
        },
    ]


def _trial_from_record(record: dict[str, Any]) -> Trial:
    return Trial(
        trial_id=str(record["trial_id"]),
        example_id=str(record["example_id"]),
        question=str(record["question"]),
        answers=tuple(str(value) for value in record["answers"]),
        content=str(record["content"]),
        gold_document=int(record["gold_document"]),
        document_ids=tuple(str(value) for value in record["document_ids"]),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _run_repeated(
    compress: Callable[[], tuple[str, dict[str, Any]]],
    *,
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        compress()
    outputs: list[str] = []
    latencies: list[float] = []
    metadata: dict[str, Any] = {}
    for _ in range(runs):
        started = time.perf_counter()
        output, metadata = compress()
        latencies.append((time.perf_counter() - started) * 1_000)
        outputs.append(output)
    return {
        "output_text": outputs[-1],
        "output_sha256": _sha256(outputs[-1]),
        "deterministic": len({_sha256(value) for value in outputs}) == 1,
        "latency_ms": {
            "samples": [round(value, 3) for value in latencies],
            "p50": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "native_metrics": metadata,
    }


def _entroly_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    from entroly import __version__
    from entroly.compression_proxy import compress_proxy_payload

    trials = [_trial_from_record(value) for value in payload["trials"]]
    ratios = [float(value) for value in payload["target_ratios"]]
    count_tokens, _ = _token_counter()
    results: list[dict[str, Any]] = []
    for trial in trials:
        input_tokens = count_tokens(trial.content)
        for ratio in ratios:
            target_tokens = max(64, int(math.floor(input_tokens * ratio)))

            def compress() -> tuple[str, dict[str, Any]]:
                control_budget = target_tokens
                attempts: list[dict[str, int]] = []
                for _ in range(3):
                    result = compress_proxy_payload(
                        {"model": payload["model"], "messages": _messages(trial)},
                        query=trial.question,
                        budget_tokens=control_budget,
                        include_receipt_header=False,
                    )
                    output = str(result.body["messages"][-1]["content"])
                    observed_tokens = count_tokens(output)
                    attempts.append(
                        {
                            "control_budget": control_budget,
                            "observed_tokens": observed_tokens,
                        }
                    )
                    if observed_tokens <= target_tokens:
                        break
                    control_budget = max(
                        64,
                        int(control_budget * target_tokens / observed_tokens * 0.97),
                    )
                return output, {
                    "changed": bool(result.changed),
                    "compressed_blocks": int(result.receipt.compressed_blocks),
                    "receipt_count": len(result.receipt.receipts),
                    "requested_target_tokens": target_tokens,
                    "control_attempts": attempts,
                }

            result = _run_repeated(
                compress,
                warmups=int(payload["warmups"]),
                runs=int(payload["runs"]),
            )
            results.append(
                {"trial_id": trial.trial_id, "target_ratio": ratio, **result}
            )

    root = Path(__file__).resolve().parents[1]
    return {
        "system": "entroly",
        "package": "entroly",
        "version": __version__,
        "release_status": "source candidate; not yet published",
        "algorithm": "candidate query-conditioned evidence-lock compression",
        "config": {
            "budget": "shared-tokenizer cap with up to 3 feedback attempts",
            "receipt": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "import_origin": "repository source checkout",
            "native_acceleration_used": False,
            "dependencies": _versions(("tiktoken",)),
            "implementation_sha256": _implementation_sha256(
                (
                    Path(__file__).resolve(),
                    root / "entroly" / "compression_proxy.py",
                    root / "entroly" / "evidence_locked_compression.py",
                )
            ),
        },
        "results": results,
    }


def _headroom_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    from headroom import compress as headroom_compress

    trials = [_trial_from_record(value) for value in payload["trials"]]
    ratios = [float(value) for value in payload["target_ratios"]]
    count_tokens, _ = _token_counter()
    results: list[dict[str, Any]] = []
    for trial in trials:
        input_tokens = count_tokens(trial.content)
        for ratio in ratios:
            target_tokens = max(64, int(math.floor(input_tokens * ratio)))

            def compress() -> tuple[str, dict[str, Any]]:
                control_ratio = ratio
                attempts: list[dict[str, float | int]] = []
                for _ in range(3):
                    result = headroom_compress(
                        json.loads(json.dumps(_messages(trial))),
                        model=str(payload["model"]),
                        protect_recent=0,
                        savings_profile="agent-90",
                        target_ratio=control_ratio,
                        min_tokens_to_compress=0,
                    )
                    output = str(result.messages[-1]["content"])
                    observed_tokens = count_tokens(output)
                    attempts.append(
                        {
                            "control_ratio": round(control_ratio, 8),
                            "observed_tokens": observed_tokens,
                        }
                    )
                    if observed_tokens <= target_tokens:
                        break
                    control_ratio = max(
                        0.01,
                        control_ratio * target_tokens / observed_tokens * 0.97,
                    )
                return output, {
                    "tokens_before": int(result.tokens_before),
                    "tokens_after": int(result.tokens_after),
                    "transforms": list(result.transforms_applied),
                    "requested_target_tokens": target_tokens,
                    "control_attempts": attempts,
                }

            result = _run_repeated(
                compress,
                warmups=int(payload["warmups"]),
                runs=int(payload["runs"]),
            )
            results.append(
                {"trial_id": trial.trial_id, "target_ratio": ratio, **result}
            )

    return {
        "system": "headroom",
        "package": "headroom-ai",
        "version": importlib.metadata.version("headroom-ai"),
        "release_status": "published PyPI release",
        "algorithm": "released public compress() pipeline with agent-90 profile",
        "config": {
            "protect_recent": 0,
            "savings_profile": "agent-90",
            "target_ratio": "matrix value",
            "min_tokens_to_compress": 0,
            "cap_controller": "up to 3 shared-tokenizer feedback attempts",
        },
        "runtime": {
            "python": platform.python_version(),
            "dependencies": _versions(
                ("headroom-ai", "litellm", "onnxruntime", "tiktoken", "transformers")
            ),
            "distribution_record_sha256": _distribution_record_sha256("headroom-ai"),
        },
        "results": results,
    }


def run_adapter(system: str, payload: dict[str, Any]) -> dict[str, Any]:
    if system == "entroly":
        return _entroly_adapter(payload)
    if system == "headroom":
        return _headroom_adapter(payload)
    raise ValueError(f"unsupported adapter: {system}")


def _adapter_command(python: str, system: str) -> list[str]:
    return [python, "-m", "benchmarks.compression_frontier", "adapter", "--system", system]


def _subprocess_env() -> dict[str, str]:
    environment = {
        key: os.environ[key] for key in _SUBPROCESS_ENV_ALLOWLIST if key in os.environ
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _invoke_adapter(
    command: list[str], payload: dict[str, Any], *, root: Path, timeout: float
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=_subprocess_env(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4_000:]
        raise RuntimeError(f"adapter failed ({completed.returncode}): {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"adapter returned invalid JSON: {completed.stdout[-2_000:]}"
        ) from error
    if not isinstance(report, dict):
        raise RuntimeError("adapter report must be a JSON object")
    report["stderr_sha256"] = _sha256(completed.stderr)
    report["stderr_tail"] = completed.stderr[-1_000:]
    return report


def _answer_retained(output: str, answers: Sequence[str]) -> bool:
    normalized_output = " ".join(output.casefold().split())
    return any(" ".join(answer.casefold().split()) in normalized_output for answer in answers)


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def _aggregate_rows(
    rows: Sequence[dict[str, Any]], participants: Sequence[str], ratios: Sequence[float]
) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for system in participants:
        by_ratio: dict[str, Any] = {}
        for ratio in ratios:
            selected = [
                row
                for row in rows
                if row["system"] == system and float(row["target_ratio"]) == ratio
            ]
            input_total = sum(int(row["input_tokens"]) for row in selected)
            output_total = sum(int(row["output_tokens"]) for row in selected)
            by_ratio[f"{ratio:g}"] = {
                "trials": len(selected),
                "valid_trials": sum(bool(row["valid"]) for row in selected),
                "answer_retained": sum(bool(row["answer_retained"]) for row in selected),
                "answer_retention": round(
                    statistics.mean(bool(row["answer_retained"]) for row in selected), 6
                ),
                "achieved_keep_ratio": round(output_total / input_total, 6),
                "weighted_savings_ratio": round(1.0 - output_total / input_total, 6),
                "target_met_trials": sum(bool(row["target_met"]) for row in selected),
                "target_attainment": round(
                    statistics.mean(bool(row["target_met"]) for row in selected), 6
                ),
                "p50_latency_ms": round(
                    statistics.median(float(row["latency_ms"]["p50"]) for row in selected),
                    3,
                ),
                "passed": all(bool(row["valid"]) for row in selected),
            }
        aggregates[system] = by_ratio
    return aggregates


def _paired_statistics(
    rows: Sequence[dict[str, Any]], ratios: Sequence[float]
) -> dict[str, Any]:
    statistics_by_ratio: dict[str, Any] = {}
    for ratio in ratios:
        selected = [row for row in rows if float(row["target_ratio"]) == ratio]
        keyed = {
            (str(row["system"]), str(row["trial_id"])): bool(row["answer_retained"])
            for row in selected
        }
        trial_ids = sorted({str(row["trial_id"]) for row in selected})
        entroly_only = sum(
            keyed[("entroly", trial_id)] and not keyed[("headroom", trial_id)]
            for trial_id in trial_ids
        )
        headroom_only = sum(
            keyed[("headroom", trial_id)] and not keyed[("entroly", trial_id)]
            for trial_id in trial_ids
        )
        statistics_by_ratio[f"{ratio:g}"] = {
            "entroly_only": entroly_only,
            "headroom_only": headroom_only,
            "discordant": entroly_only + headroom_only,
            "mcnemar_exact_two_sided_p": round(
                _mcnemar_exact(entroly_only, headroom_only), 18
            ),
        }
    return statistics_by_ratio


def _normalize_answer(value: str) -> str:
    lowered = value.casefold()
    without_punctuation = _PUNCT_RE.sub(" ", lowered)
    without_articles = _ARTICLES_RE.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def _answer_scores(prediction: str, answers: Sequence[str]) -> tuple[bool, float]:
    prediction_normalized = _normalize_answer(prediction)
    exact = any(prediction_normalized == _normalize_answer(answer) for answer in answers)
    prediction_tokens = prediction_normalized.split()
    best_f1 = 0.0
    for answer in answers:
        answer_tokens = _normalize_answer(answer).split()
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if not prediction_tokens or not answer_tokens:
            score = float(prediction_tokens == answer_tokens)
        elif not overlap:
            score = 0.0
        else:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(answer_tokens)
            score = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, score)
    return exact, best_f1


def _answer_prompt(context: str, question: str) -> str:
    return _ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)


def _aggregate_downstream(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for system in ("raw", "entroly", "headroom"):
        selected = [row for row in rows if row["system"] == system]
        if not selected:
            raise ValueError(f"downstream matrix has no {system} rows")
        aggregates[system] = {
            "trials": len(selected),
            "exact_matches": sum(bool(row["exact_match"]) for row in selected),
            "exact_match": round(
                statistics.mean(bool(row["exact_match"]) for row in selected), 6
            ),
            "mean_token_f1": round(
                statistics.mean(float(row["token_f1"]) for row in selected), 6
            ),
            "errors": sum(row["error"] is not None for row in selected),
        }
    return aggregates


def _ollama_post(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ollama request failed for {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Ollama returned a non-object for {path}")
    return value


def _ollama_identity(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    tags_request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
    with urllib.request.urlopen(tags_request, timeout=timeout) as response:
        tags = json.loads(response.read().decode("utf-8"))
    matches = [value for value in tags.get("models", []) if value.get("name") == model]
    if not matches:
        raise RuntimeError(f"Ollama model {model!r} is not installed")
    match = matches[0]
    show = _ollama_post(base_url, "/api/show", {"model": model}, timeout)
    return {
        "name": model,
        "digest": match.get("digest"),
        "size": match.get("size"),
        "details": match.get("details", {}),
        "show_sha256": _canonical_sha256(show),
    }


def _run_downstream(
    *,
    trials: Sequence[Trial],
    rows: Sequence[dict[str, Any]],
    model: str,
    base_url: str,
    ratio: float,
    trial_limit: int,
    timeout: float,
) -> dict[str, Any]:
    selected_trials = list(trials[:trial_limit])
    row_lookup = {
        (str(row["system"]), str(row["trial_id"]), float(row["target_ratio"])): row
        for row in rows
    }
    identity = _ollama_identity(base_url, model, timeout)
    prompts: list[tuple[str, Trial, str]] = []
    for trial in selected_trials:
        prompts.append(("raw", trial, trial.content))
        for system in ("entroly", "headroom"):
            prompts.append(
                (system, trial, str(row_lookup[(system, trial.trial_id, ratio)]["output_text"]))
            )
    random.Random(DEFAULT_SEED + 1).shuffle(prompts)

    answer_rows: list[dict[str, Any]] = []
    for system, trial, context in prompts:
        prompt = _answer_prompt(context, trial.question)
        started = time.perf_counter()
        error_message: str | None = None
        try:
            response = _ollama_post(
                base_url,
                "/api/generate",
                {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "seed": 0, "num_predict": 40},
                },
                timeout,
            )
        except RuntimeError as error:
            response = {}
            error_message = str(error)
        if error_message is None and not bool(response.get("done")):
            error_message = "Ollama generation did not complete"
        latency_ms = (time.perf_counter() - started) * 1_000
        prediction = str(response.get("response", "")).strip()
        exact, f1 = (
            _answer_scores(prediction, trial.answers)
            if error_message is None
            else (False, 0.0)
        )
        answer_rows.append(
            {
                "system": system,
                "trial_id": trial.trial_id,
                "target_ratio": None if system == "raw" else ratio,
                "context_sha256": _sha256(context),
                "prompt_sha256": _sha256(prompt),
                "prediction": prediction,
                "prediction_sha256": _sha256(prediction),
                "exact_match": exact,
                "token_f1": round(f1, 6),
                "latency_ms": round(latency_ms, 3),
                "eval_count": response.get("eval_count"),
                "done": bool(response.get("done")),
                "error": error_message,
            }
        )

    aggregates = _aggregate_downstream(answer_rows)
    return {
        "enabled": True,
        "scope": "local short-answer guard; not a hosted frontier-model benchmark",
        "model": identity,
        "base_url": base_url,
        "target_ratio": ratio,
        "prompt_template_sha256": _sha256(_ANSWER_PROMPT_TEMPLATE),
        "generation": {"temperature": 0, "seed": 0, "num_predict": 40},
        "rows": answer_rows,
        "aggregates": aggregates,
    }


def _superiority_gate(
    *,
    aggregates: dict[str, dict[str, Any]],
    paired_statistics: dict[str, Any],
    ratios: Sequence[float],
    downstream: dict[str, Any] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    for ratio in ratios:
        key = f"{ratio:g}"
        entroly = aggregates["entroly"][key]
        headroom = aggregates["headroom"][key]
        if not entroly["passed"] or not headroom["passed"]:
            reasons.append(f"incomplete or invalid matrix at target ratio {key}")
        if entroly["target_attainment"] != 1.0:
            reasons.append(f"Entroly misses the token cap at target ratio {key}")
        if entroly["answer_retention"] <= headroom["answer_retention"]:
            reasons.append(f"Entroly lacks a strict answer-quality win at target ratio {key}")
        if paired_statistics[key]["mcnemar_exact_two_sided_p"] > 0.05:
            reasons.append(
                f"paired answer-quality win is not significant at target ratio {key}"
            )

    if downstream is None:
        reasons.append("downstream answer guard was not run")
    else:
        values = downstream["aggregates"]
        if values["raw"]["exact_match"] < 0.5:
            reasons.append("raw-context downstream exact match is below 50%")
        if values["entroly"]["exact_match"] < values["headroom"]["exact_match"]:
            reasons.append("Entroly downstream exact match is below Headroom")
        if values["entroly"]["mean_token_f1"] < values["headroom"]["mean_token_f1"]:
            reasons.append("Entroly downstream token F1 is below Headroom")
        if any(value["errors"] for value in values.values()):
            reasons.append("downstream evaluation contains errors")

    passed = not reasons
    return {
        "passed": passed,
        "label": (
            "Entroly wins every matched token-cap quality gate against Headroom"
            if passed
            else "Superiority not established"
        ),
        "claim_scope": (
            "SQuAD v2 long-context answer retention at 2x/4x/8x requested "
            "compression, with a local short-answer guard"
        ),
        "reasons": reasons,
        "not_claimed": [
            "general LLM quality",
            "all context compressors",
            "hosted frontier-model accuracy",
            "production cost savings",
        ],
    }


def analyze(
    *,
    trials: Sequence[Trial],
    adapters: Sequence[dict[str, Any]],
    ratios: Sequence[float],
    protocol: dict[str, Any],
    downstream_options: dict[str, Any] | None,
) -> dict[str, Any]:
    count_tokens, tokenizer_version = _token_counter()
    trial_by_id = {trial.trial_id: trial for trial in trials}
    participant_meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    expected = {(trial.trial_id, ratio) for trial in trials for ratio in ratios}
    for adapter in adapters:
        system = str(adapter["system"])
        participant_meta[system] = {
            key: adapter[key]
            for key in (
                "package",
                "version",
                "release_status",
                "algorithm",
                "config",
                "runtime",
                "stderr_sha256",
                "stderr_tail",
            )
            if key in adapter
        }
        seen: set[tuple[str, float]] = set()
        for result in adapter.get("results", []):
            trial_id = str(result["trial_id"])
            ratio = float(result["target_ratio"])
            pair = (trial_id, ratio)
            if trial_id not in trial_by_id or pair in seen or pair not in expected:
                raise ValueError(f"{system} returned unknown or duplicate row {pair!r}")
            seen.add(pair)
            trial = trial_by_id[trial_id]
            output = str(result["output_text"])
            input_tokens = count_tokens(trial.content)
            output_tokens = count_tokens(output)
            retained = _answer_retained(output, trial.answers)
            valid = (
                bool(result.get("deterministic"))
                and 0 <= output_tokens <= input_tokens
                and str(result.get("output_sha256")) == _sha256(output)
            )
            rows.append(
                {
                    "system": system,
                    "trial_id": trial_id,
                    "target_ratio": ratio,
                    "target_tokens": max(64, int(math.floor(input_tokens * ratio))),
                    "input_sha256": _sha256(trial.content),
                    "output_sha256": _sha256(output),
                    "output_text": output,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "achieved_keep_ratio": round(output_tokens / input_tokens, 6),
                    "target_met": output_tokens
                    <= max(64, int(math.floor(input_tokens * ratio))),
                    "answer_retained": retained,
                    "deterministic": bool(result.get("deterministic")),
                    "valid": valid,
                    "latency_ms": result["latency_ms"],
                    "native_metrics": result.get("native_metrics", {}),
                }
            )
        if seen != expected:
            missing = sorted(expected.difference(seen))
            raise ValueError(f"{system} returned incomplete matrix: {missing[:5]}")

    participants = sorted(participant_meta)
    if participants != ["entroly", "headroom"]:
        raise ValueError("frontier requires both entroly and headroom adapters")
    aggregates = _aggregate_rows(rows, participants, ratios)
    paired = _paired_statistics(rows, ratios)
    downstream = None
    if downstream_options is not None:
        downstream = _run_downstream(
            trials=trials,
            rows=rows,
            **downstream_options,
        )
    gate = _superiority_gate(
        aggregates=aggregates,
        paired_statistics=paired,
        ratios=ratios,
        downstream=downstream,
    )
    trial_records = [trial.record() for trial in trials]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            **protocol,
            "dataset": DATASET_ID,
            "dataset_split": DATASET_SPLIT,
            "trial_manifest_sha256": _canonical_sha256(trial_records),
            "tokenizer": TOKENIZER,
            "tokenizer_package": f"tiktoken=={tokenizer_version}",
            "target_ratios": list(ratios),
            "quality_metric": "any accepted answer string retained, case-insensitive",
            "winner_gate": (
                "complete deterministic non-inflating matrix + Entroly meets every "
                "maximum token cap + strict retained-answer win with paired exact "
                "p<=0.05 at every target + downstream raw-capability and "
                "non-inferiority guards"
            ),
        },
        "participants": participant_meta,
        "trials": trial_records,
        "rows": rows,
        "aggregates": aggregates,
        "paired_statistics": paired,
        "downstream": downstream,
        "superiority_gate": gate,
        "limitations": [
            "SQuAD v2 contexts test extractive answer retention, not full agent behavior.",
            "Requested target ratios are controls, not assumed outcomes; achieved ratios are primary.",
            "The local Ollama answer guard is not evidence about hosted frontier models.",
            "No retrieval or post-compression recovery is invoked in this active-context benchmark.",
            "Headroom is measured through its released public compress() API and declared config.",
            "Entroly is a 1.0.59 source candidate; publication must follow before claiming released-package parity.",
        ],
    }
    report["payload_sha256"] = _canonical_sha256(report)
    return report


def _verify_downstream(
    downstream: dict[str, Any] | None,
    *,
    trials: Sequence[Trial],
    compression_rows: Sequence[dict[str, Any]],
    ratios: Sequence[float],
) -> None:
    if downstream is None:
        return
    if downstream.get("prompt_template_sha256") != _sha256(_ANSWER_PROMPT_TEMPLATE):
        raise ValueError("downstream prompt template hash mismatch")
    if downstream.get("generation") != {
        "temperature": 0,
        "seed": 0,
        "num_predict": 40,
    }:
        raise ValueError("downstream generation configuration mismatch")
    base_url = str(downstream.get("base_url", ""))
    if not base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("downstream base_url must be loopback-local")
    model = downstream.get("model", {})
    if not model.get("name") or not model.get("digest"):
        raise ValueError("downstream model identity is incomplete")

    ratio = float(downstream["target_ratio"])
    if ratio not in ratios:
        raise ValueError("downstream target ratio is outside the compression matrix")
    aggregates = downstream["aggregates"]
    sample_sizes = {
        int(aggregates[system]["trials"])
        for system in ("raw", "entroly", "headroom")
    }
    if len(sample_sizes) != 1:
        raise ValueError("downstream systems have different sample sizes")
    sample_size = sample_sizes.pop()
    if sample_size < 1 or sample_size > len(trials):
        raise ValueError("downstream sample size is invalid")

    trial_by_id = {trial.trial_id: trial for trial in trials[:sample_size]}
    compression_lookup = {
        (str(row["system"]), str(row["trial_id"]), float(row["target_ratio"])): row
        for row in compression_rows
    }
    expected = {
        (system, trial_id)
        for system in ("raw", "entroly", "headroom")
        for trial_id in trial_by_id
    }
    seen: set[tuple[str, str]] = set()
    for row in downstream["rows"]:
        system = str(row["system"])
        trial_id = str(row["trial_id"])
        key = (system, trial_id)
        if key not in expected or key in seen:
            raise ValueError(f"unknown or duplicate downstream row {key!r}")
        seen.add(key)
        trial = trial_by_id[trial_id]
        if system == "raw":
            context = trial.content
            expected_ratio = None
        else:
            context = str(compression_lookup[(system, trial_id, ratio)]["output_text"])
            expected_ratio = ratio
        prediction = str(row["prediction"])
        prompt = _answer_prompt(context, trial.question)
        error = row.get("error")
        exact, f1 = (
            (False, 0.0)
            if error is not None
            else _answer_scores(prediction, trial.answers)
        )
        checks = {
            "target_ratio": expected_ratio,
            "context_sha256": _sha256(context),
            "prompt_sha256": _sha256(prompt),
            "prediction_sha256": _sha256(prediction),
            "exact_match": exact,
            "token_f1": round(f1, 6),
        }
        for field, expected_value in checks.items():
            if row.get(field) != expected_value:
                raise ValueError(f"downstream {field} mismatch for {key!r}")
        if error is None and not bool(row.get("done")):
            raise ValueError(f"downstream completion flag mismatch for {key!r}")
    if seen != expected:
        raise ValueError("downstream result matrix is incomplete")
    if aggregates != _aggregate_downstream(downstream["rows"]):
        raise ValueError("downstream aggregate mismatch")


def verify_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    stored_hash = report.get("payload_sha256")
    unhashed = {key: value for key, value in report.items() if key != "payload_sha256"}
    if stored_hash != _canonical_sha256(unhashed):
        raise ValueError("payload_sha256 mismatch")

    trials = [_trial_from_record(value) for value in report["trials"]]
    trial_by_id = {trial.trial_id: trial for trial in trials}
    if len(trial_by_id) != len(trials):
        raise ValueError("duplicate trial ids")
    expected_manifest = _canonical_sha256([trial.record() for trial in trials])
    if report["protocol"].get("trial_manifest_sha256") != expected_manifest:
        raise ValueError("trial manifest hash mismatch")
    count_tokens, _ = _token_counter()
    ratios = [float(value) for value in report["protocol"]["target_ratios"]]
    expected = {
        (system, trial.trial_id, ratio)
        for system in ("entroly", "headroom")
        for trial in trials
        for ratio in ratios
    }
    seen: set[tuple[str, str, float]] = set()
    for row in report["rows"]:
        key = (str(row["system"]), str(row["trial_id"]), float(row["target_ratio"]))
        if key not in expected or key in seen:
            raise ValueError(f"unknown or duplicate result row {key!r}")
        seen.add(key)
        trial = trial_by_id[key[1]]
        output = str(row["output_text"])
        input_tokens = count_tokens(trial.content)
        output_tokens = count_tokens(output)
        checks = {
            "input_sha256": _sha256(trial.content),
            "output_sha256": _sha256(output),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "target_tokens": max(64, int(math.floor(input_tokens * key[2]))),
            "achieved_keep_ratio": round(output_tokens / input_tokens, 6),
            "target_met": output_tokens
            <= max(64, int(math.floor(input_tokens * key[2]))),
            "answer_retained": _answer_retained(output, trial.answers),
        }
        for field, expected_value in checks.items():
            if row.get(field) != expected_value:
                raise ValueError(f"{field} mismatch for {key!r}")
        expected_valid = (
            bool(row["deterministic"])
            and output_tokens <= input_tokens
            and output_tokens >= 0
        )
        if bool(row["valid"]) != expected_valid:
            raise ValueError(f"valid mismatch for {key!r}")
    if seen != expected:
        raise ValueError("result matrix is incomplete")

    participants = sorted(report["participants"])
    expected_aggregates = _aggregate_rows(report["rows"], participants, ratios)
    if report["aggregates"] != expected_aggregates:
        raise ValueError("aggregate mismatch")
    if report["paired_statistics"] != _paired_statistics(report["rows"], ratios):
        raise ValueError("paired statistics mismatch")
    _verify_downstream(
        report.get("downstream"),
        trials=trials,
        compression_rows=report["rows"],
        ratios=ratios,
    )
    expected_gate = _superiority_gate(
        aggregates=expected_aggregates,
        paired_statistics=report["paired_statistics"],
        ratios=ratios,
        downstream=report.get("downstream"),
    )
    if report["superiority_gate"] != expected_gate:
        raise ValueError("superiority gate mismatch")


def render_markdown(report: dict[str, Any]) -> str:
    verify_report(report)
    protocol = report["protocol"]
    gate = report["superiority_gate"]
    participants = report["participants"]
    lines = [
        "# Matched token-cap quality frontier",
        "",
        f"**{gate['label']}.**",
        "",
        (
            f"{len(report['trials'])} frozen SQuAD v2 long-context trials; "
            f"Entroly {participants['entroly']['version']} source candidate vs "
            f"released Headroom {participants['headroom']['version']}; achieved ratios use "
            f"`{protocol['tokenizer']}`."
        ),
        (
            "Active-context scope: Headroom's CCR pointers remain in its output, "
            "but retrieval recovery is not invoked; this is not an end-to-end "
            "Headroom CCR comparison."
        ),
        "",
        "| Requested compression | System | Answer retained | Actual tokens kept | p50 latency |",
        "|---:|---|---:|---:|---:|",
    ]
    for ratio in protocol["target_ratios"]:
        key = f"{float(ratio):g}"
        for system in ("entroly", "headroom"):
            aggregate = report["aggregates"][system][key]
            lines.append(
                f"| {1 / float(ratio):.0f}x | {system.title()} | "
                f"{aggregate['answer_retention']:.1%} | "
                f"{aggregate['achieved_keep_ratio']:.1%} | "
                f"{aggregate['p50_latency_ms']:.1f} ms |"
            )
    lines.extend(["", "## Paired retained-answer statistics", ""])
    lines.extend(
        [
            "| Target | Entroly only | Headroom only | Exact McNemar p |",
            "|---:|---:|---:|---:|",
        ]
    )
    for ratio in protocol["target_ratios"]:
        key = f"{float(ratio):g}"
        value = report["paired_statistics"][key]
        lines.append(
            f"| {1 / float(ratio):.0f}x | {value['entroly_only']} | "
            f"{value['headroom_only']} | {value['mcnemar_exact_two_sided_p']:.4g} |"
        )
    downstream = report.get("downstream")
    if downstream:
        lines.extend(
            [
                "",
                "## Local short-answer guard",
                "",
                (
                    f"Model: `{downstream['model']['name']}` at the "
                    f"{1 / float(downstream['target_ratio']):.0f}x target. "
                    "This is a local-model guard, not a hosted frontier-model claim."
                ),
                "",
                "| Context | Exact match | Token F1 | Trials |",
                "|---|---:|---:|---:|",
            ]
        )
        for system in ("raw", "entroly", "headroom"):
            value = downstream["aggregates"][system]
            lines.append(
                f"| {system.title()} | {value['exact_match']:.1%} | "
                f"{value['mean_token_f1']:.1%} | {value['trials']} |"
            )
    if not gate["passed"]:
        lines.extend(["", "## Why the gate did not pass", ""])
        lines.extend(f"- {reason}" for reason in gate["reasons"])
    lines.extend(
        [
            "",
            "## Reproduce and verify",
            "",
            "```bash",
            (
                "python -m benchmarks.compression_frontier verify "
                "benchmarks/results/compression_frontier.json"
            ),
            "```",
            "",
            "## Scope limits",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in report["limitations"])
    lines.extend(
        [
            "",
            f"Payload SHA-256: `{report['payload_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_svg(report: dict[str, Any]) -> str:
    verify_report(report)
    gate = report["superiority_gate"]
    protocols = report["protocol"]
    ratios = [float(value) for value in protocols["target_ratios"]]
    participants = report["participants"]
    title_color = "#6ee7b7" if gate["passed"] else "#fbbf24"
    status = "EVIDENCE GATE PASSED" if gate["passed"] else "SUPERIORITY NOT ESTABLISHED"
    rows: list[str] = []
    y = 244
    for ratio in ratios:
        key = f"{ratio:g}"
        left = report["aggregates"]["entroly"][key]
        right = report["aggregates"]["headroom"][key]
        rows.append(
            f'<text x="70" y="{y}" class="target">{1 / ratio:.0f}× target</text>'
            f'<text x="410" y="{y}" class="value">{left["answer_retention"]:.0%} recall · '
            f'{left["achieved_keep_ratio"]:.1%} kept</text>'
            f'<text x="890" y="{y}" class="value muted">{right["answer_retention"]:.0%} recall · '
            f'{right["achieved_keep_ratio"]:.1%} kept</text>'
        )
        y += 68
    downstream = report.get("downstream")
    downstream_text = "Downstream guard not run"
    if downstream:
        left = downstream["aggregates"]["entroly"]
        right = downstream["aggregates"]["headroom"]
        downstream_text = (
            f'Local answer EM @ {1 / float(downstream["target_ratio"]):.0f}×: '
            f'Entroly {left["exact_match"]:.0%} · '
            f'Headroom {right["exact_match"]:.0%} · n={left["trials"]}'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="610" viewBox="0 0 1400 610" role="img" aria-labelledby="title desc">
  <title id="title">Entroly and Headroom matched token-cap quality frontier</title>
  <desc id="desc">{escape(gate["label"])}. Results show retained-answer recall and actual tokens kept.</desc>
  <rect width="1400" height="610" rx="28" fill="#07111f"/>
  <style>
    text {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; fill: #e5eef8; }}
    .eyebrow {{ font-size: 20px; font-weight: 750; letter-spacing: 2px; fill: {title_color}; }}
    .title {{ font-size: 42px; font-weight: 780; }}
    .subtitle {{ font-size: 19px; fill: #9fb1c7; }}
    .header {{ font-size: 20px; font-weight: 750; }}
    .target {{ font-size: 22px; font-weight: 700; fill: #b8c8db; }}
    .value {{ font-size: 24px; font-weight: 760; fill: #6ee7b7; }}
    .muted {{ fill: #c3cedb; }}
    .foot {{ font-size: 17px; fill: #8da2ba; }}
  </style>
  <text x="70" y="64" class="eyebrow">{status}</text>
  <text x="70" y="119" class="title">Matched token-cap quality frontier</text>
  <text x="70" y="157" class="subtitle">{len(report["trials"])} frozen SQuAD v2 long-context trials · achieved o200k token ratios · exact retained answers</text>
  <line x1="70" y1="187" x2="1330" y2="187" stroke="#24364b"/>
  <text x="410" y="211" class="header">Entroly {escape(str(participants["entroly"]["version"]))} candidate</text>
  <text x="890" y="211" class="header">Headroom {escape(str(participants["headroom"]["version"]))} release</text>
  {''.join(rows)}
  <rect x="70" y="458" width="1260" height="62" rx="14" fill="#0d1b2c" stroke="#20344b"/>
  <text x="98" y="497" class="header">{escape(downstream_text)}</text>
  <text x="70" y="558" class="foot">Scope: extractive answer retention + local-model guard. Not a claim about all tasks, models, or production cost.</text>
  <text x="70" y="585" class="foot">Artifact: benchmarks/results/compression_frontier.json · fail-closed verifier included</text>
</svg>'''


def _write_outputs(report: dict[str, Any], args: argparse.Namespace) -> None:
    output = args.output or Path("benchmarks/results/compression_frontier.json")
    markdown = args.markdown or Path("benchmarks/results/compression_frontier.md")
    svg = args.svg or Path("docs/assets/compression_frontier.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    svg.write_text(render_svg(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "markdown": str(markdown),
                "svg": str(svg),
                "superiority_gate": report["superiority_gate"],
            },
            indent=2,
        )
    )


def _run(args: argparse.Namespace) -> int:
    rows, fingerprint = _load_source_rows()
    trials = build_trials(
        rows, trials=args.trials, distractors=args.distractors, seed=args.seed
    )
    ratios = tuple(float(value) for value in args.target_ratios)
    if sorted(ratios, reverse=True) != list(ratios):
        raise ValueError("target ratios must be supplied in descending order")
    if any(value <= 0 or value >= 1 for value in ratios):
        raise ValueError("target ratios must be between zero and one")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "warmups": args.warmups,
        "runs": args.runs,
        "target_ratios": ratios,
        "trials": [trial.record() for trial in trials],
    }
    root = Path(__file__).resolve().parents[1]
    adapters = [
        _invoke_adapter(
            _adapter_command(sys.executable, "entroly"),
            payload,
            root=root,
            timeout=args.timeout,
        )
    ]
    if args.headroom_python:
        adapters.append(
            _invoke_adapter(
                _adapter_command(str(Path(args.headroom_python).resolve()), "headroom"),
                payload,
                root=root,
                timeout=args.timeout,
            )
        )
    if len(adapters) != 2:
        raise RuntimeError("--headroom-python is required for the public frontier")
    downstream_options = None
    if args.ollama_model:
        if args.answer_ratio not in ratios:
            raise ValueError("--answer-ratio must be present in --target-ratios")
        downstream_options = {
            "model": args.ollama_model,
            "base_url": args.ollama_url,
            "ratio": args.answer_ratio,
            "trial_limit": min(args.answer_trials, len(trials)),
            "timeout": args.ollama_timeout,
        }
    report = analyze(
        trials=trials,
        adapters=adapters,
        ratios=ratios,
        protocol={
            "dataset_fingerprint": fingerprint,
            "seed": args.seed,
            "development_seed": 20260714,
            "holdout_seed_frozen_after_development": args.seed != 20260714,
            "distractors_per_trial": args.distractors,
            "model_routing_label": args.model,
            "token_cap_controller": (
                "up to 3 feedback attempts per public API using measured "
                f"{TOKENIZER} output tokens"
            ),
            "runs": args.runs,
            "warmups": args.warmups,
            "preregistered_before_result": True,
        },
        downstream_options=downstream_options,
    )
    verify_report(report)
    _write_outputs(report, args)
    return 0


def _adapter(args: argparse.Namespace) -> int:
    payload = json.load(sys.stdin)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    print(json.dumps(run_adapter(args.system, payload), sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    report = json.loads(args.input.read_text(encoding="utf-8"))
    verify_report(report)
    print(
        f"verified {len(report['trials'])} trials; "
        f"gate_passed={report['superiority_gate']['passed']}; "
        f"payload_sha256={report['payload_sha256']}"
    )
    return 0


def _render(args: argparse.Namespace) -> int:
    report = json.loads(args.input.read_text(encoding="utf-8"))
    verify_report(report)
    markdown = render_markdown(report)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.svg:
        args.svg.write_text(render_svg(report), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the isolated matched-target frontier")
    run.add_argument("--headroom-python", required=True)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    run.add_argument("--distractors", type=int, default=DEFAULT_DISTRACTORS)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument(
        "--target-ratios", type=float, nargs="+", default=DEFAULT_TARGET_RATIOS
    )
    run.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    run.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    run.add_argument("--timeout", type=float, default=900.0)
    run.add_argument("--ollama-model")
    run.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    run.add_argument("--ollama-timeout", type=float, default=180.0)
    run.add_argument("--answer-ratio", type=float, default=DEFAULT_ANSWER_RATIO)
    run.add_argument("--answer-trials", type=int, default=DEFAULT_ANSWER_TRIALS)
    run.add_argument("--output", type=Path)
    run.add_argument("--markdown", type=Path)
    run.add_argument("--svg", type=Path)
    run.set_defaults(func=_run)

    adapter = subparsers.add_parser("adapter", help=argparse.SUPPRESS)
    adapter.add_argument("--system", choices=("entroly", "headroom"), required=True)
    adapter.set_defaults(func=_adapter)

    verify = subparsers.add_parser("verify", help="Verify the full artifact")
    verify.add_argument("input", type=Path)
    verify.set_defaults(func=_verify)

    render = subparsers.add_parser("render", help="Render Markdown and SVG")
    render.add_argument("input", type=Path)
    render.add_argument("--output", type=Path)
    render.add_argument("--svg", type=Path)
    render.set_defaults(func=_render)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
