"""Evidence-first benchmark records and paired statistics.

Provider billing totals are deliberately separate from compressor output. The
schema is JSONL-friendly and preserves enough per-sample detail to audit every
regression rather than hiding it in aggregate averages.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.total_tokens) < 0:
            raise ValueError("provider token counts must be non-negative")
        if self.total_tokens < self.input_tokens + self.output_tokens:
            raise ValueError("provider total_tokens cannot be below input + output")


@dataclass(frozen=True)
class SampleEvidence:
    sample_id: str
    dataset: str
    split: str
    model: str
    seed: int
    raw_context_tokens: int
    selected_tokens_pre_trim: int
    emitted_context_tokens: int
    prompt_input_tokens: int
    output_tokens: int
    provider_total_tokens: int
    baseline_correct: bool
    treatment_correct: bool
    answer_present_raw: bool | None = None
    answer_present_pre_trim: bool | None = None
    answer_present_post_trim: bool | None = None
    compression_decision: str = ""
    certificate_scope: str = ""
    certificate_verdict: str = ""
    source_spans: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    latency_ms: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        counts = (
            self.raw_context_tokens,
            self.selected_tokens_pre_trim,
            self.emitted_context_tokens,
            self.prompt_input_tokens,
            self.output_tokens,
            self.provider_total_tokens,
        )
        if min(counts) < 0:
            raise ValueError("token fields must be non-negative")
        if self.provider_total_tokens < self.prompt_input_tokens + self.output_tokens:
            raise ValueError("provider total is inconsistent with prompt/output usage")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

    @property
    def emitted_savings(self) -> float:
        if self.raw_context_tokens == 0:
            return 0.0
        return 1.0 - self.emitted_context_tokens / self.raw_context_tokens

    @property
    def paired_outcome(self) -> str:
        if self.baseline_correct and self.treatment_correct:
            return "both_correct"
        if self.baseline_correct and not self.treatment_correct:
            return "regression"
        if not self.baseline_correct and self.treatment_correct:
            return "gain"
        return "both_wrong"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_spans"] = list(self.source_spans)
        payload["emitted_savings"] = self.emitted_savings
        payload["paired_outcome"] = self.paired_outcome
        return payload


def canonical_fingerprint(records: Iterable[SampleEvidence]) -> str:
    encoded = "\n".join(
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda item: item.sample_id)
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= total:
        raise ValueError("successes must be inside [0, total]")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def exact_mcnemar(records: Sequence[SampleEvidence], two_sided: bool = True) -> float:
    regressions = sum(r.baseline_correct and not r.treatment_correct for r in records)
    gains = sum(not r.baseline_correct and r.treatment_correct for r in records)
    n = regressions + gains
    if n == 0:
        return 1.0
    k = min(regressions, gains)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail) if two_sided else tail


def paired_bootstrap_delta(
    records: Sequence[SampleEvidence],
    *,
    iterations: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    if not records:
        return (0.0, 0.0, 0.0)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sample = [records[rng.randrange(len(records))] for _ in records]
        deltas.append(
            sum(int(r.treatment_correct) - int(r.baseline_correct) for r in sample)
            / len(sample)
        )
    deltas.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = deltas[min(len(deltas) - 1, int(alpha * len(deltas)))]
    upper = deltas[min(len(deltas) - 1, int((1.0 - alpha) * len(deltas)))]
    observed = sum(int(r.treatment_correct) - int(r.baseline_correct) for r in records) / len(records)
    return (observed, lower, upper)


def summarize(records: Sequence[SampleEvidence]) -> dict[str, Any]:
    if not records:
        return {"samples": 0}
    baseline = sum(record.baseline_correct for record in records)
    treatment = sum(record.treatment_correct for record in records)
    regressions = sum(record.paired_outcome == "regression" for record in records)
    gains = sum(record.paired_outcome == "gain" for record in records)
    mean_raw = sum(record.raw_context_tokens for record in records) / len(records)
    mean_emitted = sum(record.emitted_context_tokens for record in records) / len(records)
    delta, low, high = paired_bootstrap_delta(records)
    return {
        "samples": len(records),
        "baseline_accuracy": baseline / len(records),
        "treatment_accuracy": treatment / len(records),
        "accuracy_delta": delta,
        "accuracy_delta_bootstrap_95": [low, high],
        "regressions": regressions,
        "gains": gains,
        "mcnemar_exact_two_sided_p": exact_mcnemar(records),
        "mean_raw_context_tokens": mean_raw,
        "mean_emitted_context_tokens": mean_emitted,
        "mean_emitted_savings": 1.0 - mean_emitted / mean_raw if mean_raw else 0.0,
        "fingerprint": canonical_fingerprint(records),
    }
