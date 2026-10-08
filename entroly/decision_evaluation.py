"""Paired context decision observations, independent of optimizer certificates.

Full/full-repeat controls measure model self-divergence. Observed full/selected
divergence is not causally noise-corrected by subtracting that control. This
module records supplied outputs only: it never invokes a provider or executes
an output, and it does not enable production risk enforcement.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from .context_assurance import ASSURANCE_SCHEMA
from .context_receipts.models import byte_digest, stable_hash

DECISION_OBSERVATION_SCHEMA = "entroly.context-decision-observation.v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class DecisionScope:
    model: str
    model_version: str
    settings_json: str
    task_family: str
    evaluator: str
    projection: str
    tokenizer: str
    context_policy: str
    workload_sha256: str
    entroly_sha: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required for a context-risk observation")
        if not isinstance(json.loads(self.settings_json), dict):
            raise ValueError("settings_json must be an object")
        object.__setattr__(
            self, "settings_json", _canonical(json.loads(self.settings_json))
        )
        if len(self.workload_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.workload_sha256
        ):
            raise ValueError("workload_sha256 must be a SHA-256 digest")
        if len(self.entroly_sha) != 40 or any(
            c not in "0123456789abcdef" for c in self.entroly_sha
        ):
            raise ValueError("entroly_sha must be an exact Git SHA")

    @property
    def scope_id(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class JsonFieldProjection:
    """Structured decision projection; ignores non-decision prose by declaration."""

    protocol_id: str
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.protocol_id
            or not self.fields
            or len(set(self.fields)) != len(self.fields)
        ):
            raise ValueError("a projection requires a protocol id and unique fields")
        if any(not isinstance(field, str) or not field for field in self.fields):
            raise ValueError("projection fields must be nonempty strings")
        object.__setattr__(self, "fields", tuple(sorted(self.fields)))

    def __call__(self, output: Any) -> dict[str, Any]:
        value = json.loads(output) if isinstance(output, str) else output
        if not isinstance(value, Mapping):
            raise ValueError("JSON decision projection requires a structured object")
        if any(field not in value for field in self.fields):
            raise ValueError("missing decision projection field")
        result = {field: value[field] for field in self.fields}
        _canonical(result)  # Reject non-finite/non-JSON outcomes before recording.
        return result


def _loss(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("loss must be a finite number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("loss must be a finite number in [0, 1]")
    return number


def _count(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def observe_decisions(
    *,
    scope: DecisionScope,
    trial_id: str,
    task_id: str,
    query: str,
    full_output: Any,
    selected_output: Any,
    full_repeat_output: Any,
    projection: JsonFieldProjection,
    full_context_tokens: int,
    selected_tokens: int,
    budget: int,
    assurance_verdict: str,
    task_loss: Callable[[Any, Any], float] | None = None,
    truth: Any = None,
    recovered_tokens: int = 0,
    latency_ms: float = 0.0,
    seeds: Sequence[int | None] = (None, None, None),
    provider_observed_usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record C/S/C-repeat decisions with hashes instead of raw outcomes.

    Trial samples must share the supplied scope and settings. A custom prose
    evaluator must first supply independently declared structured judgments;
    prose byte equality is not a supported decision projection here.
    """
    if scope.projection != projection.protocol_id:
        raise ValueError("projection does not match the declared scope")
    if not trial_id or not task_id:
        raise ValueError("trial_id and task_id are required")
    if len(seeds) != 3 or any(
        seed is not None and type(seed) is not int for seed in seeds
    ):
        raise ValueError("seeds must identify the three paired runs")
    if isinstance(latency_ms, bool) or not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be finite and nonnegative")
    full, reduced, repeat = map(
        projection, (full_output, selected_output, full_repeat_output)
    )
    keys = [_canonical(value) for value in (full, reduced, repeat)]
    task_losses = (
        None
        if task_loss is None
        else {
            "full": _loss(task_loss(full, truth)),
            "selected": _loss(task_loss(reduced, truth)),
            "full_repeat": _loss(task_loss(repeat, truth)),
        }
    )
    payload = {
        "schema": DECISION_OBSERVATION_SCHEMA,
        "scope": asdict(scope),
        "scope_id": scope.scope_id,
        "trial_id": trial_id,
        "task_id": task_id,
        "query_sha256": byte_digest(query),
        "projection_commitment": stable_hash(asdict(projection)),
        "decision_commitments": {
            name: byte_digest(value)
            for name, value in zip(("full", "selected", "full_repeat"), keys)
        },
        "decision_divergence": float(keys[0] != keys[1]),
        "model_self_divergence": float(keys[0] != keys[2]),
        "task_loss": task_losses,
        "task_context_regret": None
        if task_losses is None
        else task_losses["selected"] - task_losses["full"],
        "full_context_tokens": _count(full_context_tokens, "full_context_tokens"),
        "selected_tokens": _count(selected_tokens, "selected_tokens"),
        "recovered_tokens": _count(recovered_tokens, "recovered_tokens"),
        "budget": _count(budget, "budget"),
        "seeds": list(seeds),
        "assurance_verdict": assurance_verdict,
        "certificate_version": ASSURANCE_SCHEMA,
        "latency_ms": latency_ms,
        "provider_observed_usage": dict(provider_observed_usage)
        if provider_observed_usage is not None
        else None,
        "risk_bound": None,
    }
    return {**payload, "observation_id": stable_hash(payload)}


def context_regret(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum one declared scope only; missing truth never becomes zero task regret."""
    if not observations:
        raise ValueError("observations cannot be empty")
    if len({item["scope_id"] for item in observations}) != 1:
        raise ValueError("cross-scope context regret is unsupported")
    if len({item["projection_commitment"] for item in observations}) != 1:
        raise ValueError("cross-projection context regret is unsupported")
    for item in observations:
        if DecisionScope(**item["scope"]).scope_id != item["scope_id"]:
            raise ValueError("observation scope commitment mismatch")
        payload = {k: v for k, v in item.items() if k != "observation_id"}
        if item.get("observation_id") != stable_hash(payload):
            raise ValueError("observation integrity mismatch")
    if len({item["trial_id"] for item in observations}) != len(observations):
        raise ValueError("duplicate trial ids cannot increase evidence")
    measured_task = [item["task_context_regret"] for item in observations]
    return {
        "scope_id": observations[0]["scope_id"],
        "trials": len(observations),
        "internal_selection_regret": None,
        "decision_divergence_regret": sum(
            item["decision_divergence"] for item in observations
        ),
        "task_context_regret": None
        if any(x is None for x in measured_task)
        else sum(measured_task),
        "model_self_divergence": sum(
            item["model_self_divergence"] for item in observations
        ),
        "risk_bound": None,
        "causal_attribution": "unestablished; interpret divergence relative to full/full control",
    }


def continuity_debt(
    *,
    previously_omitted: Sequence[str],
    newly_required: Sequence[str],
    visible_before_decision: Sequence[str],
    verified_recovered: Sequence[str],
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Measured evidence availability for one turn; relevance is caller-declared.

    Only prior omissions newly required this turn incur debt. A recovery must be
    verified AND visible before the decision to discharge it. This does not infer
    task success, improve a memory model, or persist a second continuity store.
    """
    relevant = set(previously_omitted) & set(newly_required)
    discharged = relevant & set(visible_before_decision) & set(verified_recovered)
    missed = relevant - discharged
    actual_weights = {}
    for cid in relevant:
        value = 1.0 if weights is None else weights[cid]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("evidence weights must be finite and nonnegative")
        actual_weights[cid] = float(value)
    return {
        "newly_relevant_count": len(relevant),
        "recovered_before_decision_count": len(discharged),
        "missed_context_debt": sum(actual_weights[cid] for cid in missed),
        "relevance_authority": "caller_declared",
        "missed_commitment": stable_hash(sorted(missed)),
    }
