"""Opt-in paired context trials using OpenAI Decisions choice answers.

The collector never runs during normal Entroly selection. Its output is
descriptive evidence, not an assurance certificate or production risk bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping

from .context_receipts.models import stable_hash
from .decision_evaluation import (
    DecisionScope,
    JsonFieldProjection,
    context_regret,
    observe_decisions,
)

MANIFEST_SCHEMA = "entroly.decisions-trial-manifest.v1"
REPLAY_SCHEMA = "entroly.decisions-trial-replay.v1"
REPORT_SCHEMA = "entroly.decisions-trial-report.v1"
MODEL = "gpt-6-luna"
PROJECTION = JsonFieldProjection("openai-decisions-choice.v1", ("choice",))
CONDITIONS = ("full", "selected", "full_repeat")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a probability")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must be a probability")
    return result


def validate_manifest(value: Any) -> Mapping[str, Any]:
    manifest = _object(value, "manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported manifest schema")
    for field in ("task_family", "context_policy", "tokenizer"):
        _text(manifest.get(field), field)
    budget = _count(manifest.get("budget"), "budget")
    if budget == 0:
        raise ValueError("budget must be positive")
    trials = manifest.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("trials must be a nonempty array")
    trial_ids: set[str] = set()
    for index, raw in enumerate(trials):
        trial = _object(raw, f"trials[{index}]")
        trial_id = _text(trial.get("trial_id"), "trial_id")
        _text(trial.get("task_id"), "task_id")
        _text(trial.get("question"), "question")
        _text(trial.get("full_context"), "full_context")
        _text(trial.get("selected_context"), "selected_context")
        if trial_id in trial_ids:
            raise ValueError("duplicate trial_id")
        trial_ids.add(trial_id)
        full_tokens = _count(trial.get("full_context_tokens"), "full_context_tokens")
        selected_tokens = _count(trial.get("selected_tokens"), "selected_tokens")
        if selected_tokens > full_tokens:
            raise ValueError("selected_tokens cannot exceed full_context_tokens")
        if selected_tokens > budget:
            raise ValueError("selected_tokens exceeds budget")
        decision = _object(trial.get("decision"), "decision")
        _text(decision.get("name"), "decision.name")
        _text(decision.get("instructions"), "decision.instructions")
        choices = decision.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError("choices must contain at least two options")
        values: set[str] = set()
        for raw_choice in choices:
            choice = _object(raw_choice, "choice")
            choice_value = _text(choice.get("value"), "choice.value")
            _text(choice.get("description"), "choice.description")
            if choice_value in values:
                raise ValueError("duplicate choice value")
            values.add(choice_value)
        if "truth_choice" in trial and trial["truth_choice"] not in values:
            raise ValueError("truth_choice is not a declared option")
    return manifest


def _parse_answer(
    response: Any, trial: Mapping[str, Any]
) -> tuple[str, list[float], int | None, str]:
    value = _object(response, "provider response")
    model = _text(value.get("model"), "response.model")
    if model != MODEL:
        raise ValueError("provider model differs from the frozen trial model")
    answers = value.get("answers")
    if not isinstance(answers, list) or len(answers) != 1:
        raise ValueError("response must contain exactly one answer")
    answer = _object(answers[0], "answer")
    if answer.get("type") == "refusal":
        raise ValueError("provider refused a trial condition")
    decision = trial["decision"]
    if answer.get("type") != "choice" or answer.get("name") != decision["name"]:
        raise ValueError("answer type or name differs from the trial")
    options = [item["value"] for item in decision["choices"]]
    choice = answer.get("choice")
    if choice not in options:
        raise ValueError("answer choice is not a declared option")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != len(options):
        raise ValueError("answer must contain every option probability")
    by_value: dict[str, float] = {}
    for item in probabilities:
        entry = _object(item, "option probability")
        option = entry.get("value")
        if not isinstance(option, str) or option not in options or option in by_value:
            raise ValueError("unknown or duplicate probability option")
        by_value[option] = _probability(entry.get("probability"), "option probability")
    if not math.isclose(sum(by_value.values()), 1.0, abs_tol=1e-3, rel_tol=0):
        raise ValueError("option probabilities must sum to one")
    _probability(answer.get("confidence"), "confidence")
    usage = value.get("usage")
    input_tokens = None
    if usage is not None:
        input_tokens = _count(
            _object(usage, "usage").get("input_tokens"), "usage.input_tokens"
        )
    return choice, [by_value[option] for option in options], input_tokens, model


def _js_divergence(left: list[float], right: list[float]) -> float:
    """Jensen-Shannon divergence in bits; descriptive, not calibrated risk."""
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]

    def relative_entropy(values: list[float]) -> float:
        return sum(
            probability * math.log2(probability / middle)
            for probability, middle in zip(values, midpoint)
            if probability > 0
        )

    return (relative_entropy(left) + relative_entropy(right)) / 2


def _request_payload(trial: Mapping[str, Any], condition: str) -> dict[str, Any]:
    context = (
        trial["selected_context"] if condition == "selected" else trial["full_context"]
    )
    decision = trial["decision"]
    return {
        "model": MODEL,
        "input": f"Context:\n{context}\n\nQuestion:\n{trial['question']}",
        "questions": [
            {
                "type": "choice",
                "name": decision["name"],
                "instructions": decision["instructions"],
                "choices": [
                    {"value": item["value"], "description": item["description"]}
                    for item in decision["choices"]
                ],
            }
        ],
    }


def _call_openai(
    payload: Mapping[str, Any], api_key: str
) -> tuple[Mapping[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/decisions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as result:
        response = json.load(result)
    return _object(response, "provider response"), (time.perf_counter() - start) * 1000


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _seal_report(report: dict[str, Any]) -> dict[str, Any]:
    report.pop("report_id", None)
    report["report_id"] = stable_hash(report)
    return report


def evaluate(
    manifest: Mapping[str, Any],
    *,
    workload_sha256: str,
    entroly_sha: str,
    replay: Mapping[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate complete triplets. A missing/refused condition aborts the report."""
    validate_manifest(manifest)
    if (replay is None) == (api_key is None):
        raise ValueError("choose exactly one of offline replay or live collection")
    if api_key is not None:
        _text(api_key, "api_key")
    if replay is not None:
        if _object(replay, "replay").get("schema") != REPLAY_SCHEMA:
            raise ValueError("unsupported replay schema")
        replay_trials = replay.get("trials")
        if not isinstance(replay_trials, list) or len(replay_trials) != len(
            manifest["trials"]
        ):
            raise ValueError("replay must have exactly one entry per trial")
        replay_by_id = {}
        for raw in replay_trials:
            row = _object(raw, "replay trial")
            identifier = _text(row.get("trial_id"), "replay trial_id")
            if identifier in replay_by_id or any(key not in row for key in CONDITIONS):
                raise ValueError("duplicate or incomplete replay trial")
            replay_by_id[identifier] = row
        if set(replay_by_id) != {row["trial_id"] for row in manifest["trials"]}:
            raise ValueError("replay trial ids differ from manifest")
        collection_id = hashlib.sha256(
            json.dumps(
                replay, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    else:
        replay_by_id = {}
        collection_id = uuid.uuid4().hex

    scope = DecisionScope(
        model=MODEL,
        model_version=MODEL,
        settings_json=json.dumps(
            {
                "collection_id": collection_id,
                "endpoint": "/v1/decisions",
                "question_type": "choice",
            }
        ),
        task_family=manifest["task_family"],
        evaluator="openai-decisions-choice.v1",
        projection=PROJECTION.protocol_id,
        tokenizer=manifest["tokenizer"],
        context_policy=manifest["context_policy"],
        workload_sha256=workload_sha256,
        entroly_sha=entroly_sha,
    )
    observations: list[dict[str, Any]] = []
    probability_shifts: list[dict[str, Any]] = []
    for trial in manifest["trials"]:
        parsed = {}
        latencies = {}
        for condition in CONDITIONS:
            if replay is None:
                response, latency = _call_openai(
                    _request_payload(trial, condition), api_key or ""
                )
            else:
                response, latency = replay_by_id[trial["trial_id"]][condition], None
            parsed[condition] = _parse_answer(response, trial)
            latencies[condition] = latency
        usage = {condition: parsed[condition][2] for condition in CONDITIONS}
        truth = trial.get("truth_choice")
        observation = observe_decisions(
            scope=scope,
            trial_id=trial["trial_id"],
            task_id=trial["task_id"],
            query=trial["question"],
            full_output={"choice": parsed["full"][0]},
            selected_output={"choice": parsed["selected"][0]},
            full_repeat_output={"choice": parsed["full_repeat"][0]},
            projection=PROJECTION,
            full_context_tokens=trial["full_context_tokens"],
            selected_tokens=trial["selected_tokens"],
            budget=manifest["budget"],
            assurance_verdict="unverified_external_decision",
            task_loss=(lambda decision, expected: float(decision["choice"] != expected))
            if truth is not None
            else None,
            truth=truth,
            latency_ms=sum(value or 0 for value in latencies.values()),
            provider_observed_usage={"input_tokens": usage},
        )
        observations.append(observation)
        probability_shifts.append(
            {
                "trial_id": trial["trial_id"],
                "request_commitments": {
                    condition: stable_hash(_request_payload(trial, condition))
                    for condition in CONDITIONS
                },
                "full_selected_js_bits": _js_divergence(
                    parsed["full"][1], parsed["selected"][1]
                ),
                "full_repeat_js_bits": _js_divergence(
                    parsed["full"][1], parsed["full_repeat"][1]
                ),
            }
        )
    return _seal_report(
        {
            "schema": REPORT_SCHEMA,
            "source": "offline_replay"
            if replay is not None
            else "openai_decisions_api",
            "collection_id": collection_id,
            "model_version_status": "provider alias; underlying revision not pinned",
            "token_count_source": "caller_supplied_unverified",
            "truth_source": "caller_supplied_unverified"
            if any("truth_choice" in trial for trial in manifest["trials"])
            else "unavailable",
            "latency_source": "unavailable"
            if replay is not None
            else "client_wall_clock",
            "production_authority": False,
            "workload_sha256": workload_sha256,
            "scope_id": scope.scope_id,
            "context_regret": context_regret(observations),
            "probability_shift_summary": {
                "mean_full_selected_js_bits": sum(
                    item["full_selected_js_bits"] for item in probability_shifts
                )
                / len(probability_shifts),
                "mean_full_repeat_js_bits": sum(
                    item["full_repeat_js_bits"] for item in probability_shifts
                )
                / len(probability_shifts),
            },
            "probability_shifts": probability_shifts,
            "observations": observations,
        }
    )


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Replace the output only after a complete serializable report exists."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--entroly-sha", help="Exact Git SHA, required outside a source checkout"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--responses", type=Path, help="Offline captured/fixture responses; no network"
    )
    source.add_argument(
        "--send-content",
        action="store_true",
        help="Send three text contexts per trial to OpenAI",
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists; use a new path for each run")
        manifest_bytes = args.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        replay = (
            json.loads(args.responses.read_text(encoding="utf-8"))
            if args.responses
            else None
        )
        key = os.environ.get("OPENAI_API_KEY") if args.send_content else None
        if args.send_content and not key:
            raise ValueError("OPENAI_API_KEY is required for live collection")
        report = evaluate(
            manifest,
            workload_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            entroly_sha=args.entroly_sha or _git_sha() or "",
            replay=replay,
            api_key=key,
        )
        report["entroly_sha_source"] = (
            "caller_supplied_unverified" if args.entroly_sha else "git_head"
        )
        report["source_tree_dirty"] = _git_dirty()
        if args.responses:
            report["responses_sha256"] = hashlib.sha256(
                args.responses.read_bytes()
            ).hexdigest()
        _seal_report(report)
        _write_report(args.output, report)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"Decision trial failed: {error}", file=sys.stderr)
        return 1
    except OSError:
        print(
            "Decision trial failed: file or provider operation failed", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
