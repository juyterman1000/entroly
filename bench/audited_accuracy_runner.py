"""Paired benchmark runner for audited context selection.

The model/evaluator is injected, making the harness provider-neutral. Expected
answers are used only after selection to attribute failures; they are never
passed to the compressor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from bench.benchmark_evidence import ProviderUsage, SampleEvidence
from bench.squad_failure_audit import answer_present
from entroly.audited_qccr import select_with_audit


@dataclass(frozen=True)
class QASample:
    sample_id: str
    dataset: str
    split: str
    query: str
    context: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    correct: bool
    response: str
    usage: ProviderUsage


Evaluator = Callable[[str, str, str, int], EvaluationResult]


def context_tokens(text: str) -> int:
    """Benchmark-local emitted-context estimate, never provider usage."""
    return max(0, (len(text) + 3) // 4)


def chunk_context(text: str, sample_id: str, size: int = 1600) -> list[dict]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [
        {
            "fragment_id": f"{sample_id}::{start}",
            "source": f"sample:{sample_id}",
            "content": text[start : start + size],
            "start_byte": len(text[:start].encode("utf-8")),
            "end_byte": len(text[: start + size].encode("utf-8")),
            "token_count": context_tokens(text[start : start + size]),
        }
        for start in range(0, len(text), size)
    ]


def _selected_pre_trim(envelope: dict, fragments: Sequence[dict]) -> str:
    by_id = {str(fragment["fragment_id"]): fragment for fragment in fragments}
    recovered: list[tuple[str, int, str]] = []
    for candidate in envelope.get("candidates", []):
        if not isinstance(candidate, dict) or not candidate.get("selected"):
            continue
        fragment = by_id.get(str(candidate.get("fragment_id") or ""))
        if fragment is None:
            continue
        base = int(fragment.get("start_byte") or 0)
        start = int(candidate.get("start_byte") or 0) - base
        end = int(candidate.get("end_byte") or 0) - base
        content = str(fragment.get("content") or "")
        encoded = content.encode("utf-8")
        if 0 <= start < end <= len(encoded):
            recovered.append(
                (
                    str(candidate.get("source_id") or ""),
                    int(candidate.get("start_byte") or 0),
                    encoded[start:end].decode("utf-8"),
                )
            )
    recovered.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(item[2] for item in recovered)


def run_sample(
    sample: QASample,
    *,
    budget: int,
    model: str,
    seed: int,
    evaluator: Evaluator,
) -> tuple[SampleEvidence, dict]:
    baseline = evaluator(sample.query, sample.context, model, seed)
    fragments = chunk_context(sample.context, sample.sample_id)
    envelope = select_with_audit(fragments, budget, sample.query)
    selected = envelope.get("selected", [])
    emitted = "\n".join(
        str(fragment.get("content") or "")
        for fragment in selected
        if isinstance(fragment, dict)
    )
    treatment = evaluator(sample.query, emitted, model, seed)
    pre_trim = _selected_pre_trim(envelope, fragments) or emitted
    metrics = envelope.get("metrics", {})
    evidence = SampleEvidence(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        split=sample.split,
        model=model,
        seed=seed,
        raw_context_tokens=context_tokens(sample.context),
        selected_tokens_pre_trim=context_tokens(pre_trim),
        emitted_context_tokens=int(
            envelope.get("emitted_tokens", context_tokens(emitted))
        ),
        prompt_input_tokens=treatment.usage.input_tokens,
        output_tokens=treatment.usage.output_tokens,
        provider_total_tokens=treatment.usage.total_tokens,
        baseline_correct=baseline.correct,
        treatment_correct=treatment.correct,
        answer_present_raw=answer_present(sample.context, sample.answers),
        answer_present_pre_trim=answer_present(pre_trim, sample.answers),
        answer_present_post_trim=answer_present(emitted, sample.answers),
        compression_decision=str(envelope.get("selection_mode") or ""),
        certificate_scope=str(metrics.get("scope") or ""),
        certificate_verdict=str(metrics.get("verdict") or ""),
        source_spans=tuple(
            span
            for fragment in selected
            if isinstance(fragment, dict)
            for span in fragment.get("source_spans", [])
            if isinstance(span, dict)
        ),
    )
    return evidence, envelope


def write_jsonl(path: str | Path, records: Sequence[SampleEvidence]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )
