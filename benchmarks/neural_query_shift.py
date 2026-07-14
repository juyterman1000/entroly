"""Evaluate future-query retention and exact rehydration after compression.

Each trial uses two answerable SQuAD v2 questions from the same paragraph whose
answers occur in different sentences. The paragraph is compressed for q1; q2
is revealed only afterward. The benchmark records direct q2 retention and the
cost/success of retrieving exact omitted source spans from the receipt store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from entroly.compression_retrieval_store import CompressionRetrievalStore
from entroly.neural_evidence_selector import (
    EvidenceSpan,
    LocalTransformerEncoder,
    NeuralSelectionPolicy,
    persist_neural_selection,
    score_lexical_evidence,
    select_neural_evidence,
)

SCHEMA_VERSION = "entroly.neural-query-shift.v1"
_SENTENCE_RE = re.compile(r"\S(?:.*?\S)?(?:[.!?](?=\s|$)|$)", re.DOTALL)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|nor|except|unless|without)\b", re.I
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains(text: str, answer: str) -> bool:
    return answer.casefold() in text.casefold()


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def _sentence_ranges(context: str) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for match in _SENTENCE_RE.finditer(context):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        text = context[start:end]
        if text:
            ranges.append((start, end, text))
    return ranges


def _answer_sentence(
    ranges: Sequence[tuple[int, int, str]], answer_start: int
) -> int | None:
    for index, (start, end, _) in enumerate(ranges):
        if start <= answer_start < end:
            return index
    return None


def _evidence_features(text: str) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    if _NUMBER_RE.search(text):
        reasons.append("numeric_fact")
    if _NEGATION_RE.search(text):
        reasons.append("logical_exception")
    if '"' in text or "'" in text:
        reasons.append("quoted_fact")
    value = min(1.0, 0.35 * len(reasons))
    return value, tuple(reasons)


@dataclass(frozen=True, slots=True)
class Pair:
    pair_id: str
    context: str
    q1: str
    a1: str
    q2: str
    a2: str
    q1_sentence: int
    q2_sentence: int


@dataclass
class PrecomputedEncoder:
    expected: Sequence[str]
    vectors: Sequence[Sequence[float]]
    model_id: str
    fingerprint: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if list(texts) != list(self.expected):
            raise ValueError("precomputed encoder input mismatch")
        return self.vectors


def _load_pairs(*, trials: int, seed: int) -> tuple[list[Pair], str]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("install the neural-benchmark extra") from error
    try:
        dataset = load_dataset("rajpurkar/squad_v2", split="validation")
    except Exception as error:  # pragma: no cover - cache state dependent
        raise RuntimeError(
            "SQuAD v2 must already be cached; downloads are disabled"
        ) from error

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dataset:
        answers = record.get("answers", {})
        if not answers.get("text") or not answers.get("answer_start"):
            continue
        grouped[str(record["context"])].append(record)

    pairs: list[Pair] = []
    for context, records in grouped.items():
        ranges = _sentence_ranges(context)
        if len(ranges) < 3:
            continue
        candidates: list[tuple[dict[str, Any], int]] = []
        for record in records:
            sentence = _answer_sentence(
                ranges, int(record["answers"]["answer_start"][0])
            )
            if sentence is not None:
                candidates.append((record, sentence))
        chosen: tuple[dict[str, Any], int, dict[str, Any], int] | None = None
        for left, left_sentence in candidates:
            for right, right_sentence in candidates:
                if left["id"] != right["id"] and left_sentence != right_sentence:
                    chosen = (left, left_sentence, right, right_sentence)
                    break
            if chosen:
                break
        if not chosen:
            continue
        left, left_sentence, right, right_sentence = chosen
        pairs.append(
            Pair(
                pair_id=f"{left['id']}->{right['id']}",
                context=context,
                q1=str(left["question"]),
                a1=str(left["answers"]["text"][0]),
                q2=str(right["question"]),
                a2=str(right["answers"]["text"][0]),
                q1_sentence=left_sentence,
                q2_sentence=right_sentence,
            )
        )
    random.Random(seed).shuffle(pairs)
    if len(pairs) < trials:
        raise ValueError(
            f"only {len(pairs)} distinct-sentence query pairs are available"
        )
    return pairs[:trials], str(getattr(dataset, "_fingerprint", "unknown"))


def _spans(context: str) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for index, (start, end, text) in enumerate(_sentence_ranges(context)):
        evidence_value, reasons = _evidence_features(text)
        spans.append(
            EvidenceSpan(
                span_id=f"sentence-{index}",
                source="squad-v2-context",
                text=text,
                token_count=max(1, len(text) // 4),
                ordinal=index,
                start_char=start,
                end_char=end,
                evidence_value=evidence_value,
                lock_reasons=reasons,
            )
        )
    return spans


def _budget_select(
    spans: Sequence[EvidenceSpan], scores: Sequence[float], budget: int
) -> list[int]:
    order = sorted(
        range(len(spans)),
        key=lambda index: (-float(scores[index]), spans[index].token_count, index),
    )
    selected: list[int] = []
    used = 0
    for index in order:
        if used + spans[index].token_count <= budget:
            selected.append(index)
            used += spans[index].token_count
    return selected


def _selected_text(spans: Sequence[EvidenceSpan], indices: Sequence[int]) -> str:
    return "\n".join(spans[index].text for index in sorted(set(indices)))


def _recovery_indices(
    q2: str,
    spans: Sequence[EvidenceSpan],
    omitted: Sequence[int],
    q2_embedding: Sequence[float],
    span_embeddings: Sequence[Sequence[float]],
) -> list[int]:
    if not omitted:
        return []
    omitted_spans = [spans[index] for index in omitted]
    lexical = score_lexical_evidence(q2, omitted_spans)
    neural = [
        sum(float(a) * float(b) for a, b in zip(q2_embedding, span_embeddings[index]))
        for index in omitted
    ]
    lexical_top = max(range(len(omitted)), key=lambda index: (lexical[index], -index))
    neural_top = max(range(len(omitted)), key=lambda index: (neural[index], -index))
    selected = {omitted[lexical_top], omitted[neural_top]}
    return sorted(selected)


def run(
    *,
    model_path: Path,
    trials: int,
    seed: int,
    active_ratio: float,
) -> dict[str, Any]:
    if not 0.05 <= active_ratio <= 0.9:
        raise ValueError("active_ratio must be between 0.05 and 0.9")
    pairs, dataset_fingerprint = _load_pairs(trials=trials, seed=seed)
    model = LocalTransformerEncoder(model_path)
    trial_spans = [_spans(pair.context) for pair in pairs]
    texts: list[str] = []
    for pair, spans in zip(pairs, trial_spans):
        texts.extend([pair.q1, pair.q2, *[span.text for span in spans]])
    embeddings = list(model.encode(texts))

    cursor = 0
    rows: list[dict[str, Any]] = []
    modes: Counter[str] = Counter()
    for pair, spans in zip(pairs, trial_spans):
        q1_embedding = embeddings[cursor]
        q2_embedding = embeddings[cursor + 1]
        span_embeddings = embeddings[cursor + 2 : cursor + 2 + len(spans)]
        cursor += 2 + len(spans)
        total_tokens = sum(span.token_count for span in spans)
        budget = max(1, round(total_tokens * active_ratio))

        lexical_scores = score_lexical_evidence(pair.q1, spans)
        lexical_indices = _budget_select(spans, lexical_scores, budget)
        lexical_text = _selected_text(spans, lexical_indices)

        precomputed = PrecomputedEncoder(
            expected=[pair.q1, *[span.text for span in spans]],
            vectors=[q1_embedding, *span_embeddings],
            model_id=model.model_id,
            fingerprint=model.fingerprint,
        )
        selection = select_neural_evidence(
            pair.q1,
            spans,
            budget_tokens=budget,
            encoder=precomputed,
            policy=NeuralSelectionPolicy(),
        )
        selected_ids = set(selection.receipt["selected_span_ids"])
        prism_indices = [
            index for index, span in enumerate(spans) if span.span_id in selected_ids
        ]
        prism_text = _selected_text(spans, prism_indices)
        modes[str(selection.receipt["reason"])] += 1

        omitted = [index for index in range(len(spans)) if index not in prism_indices]
        recovery = _recovery_indices(
            pair.q2, spans, omitted, q2_embedding, span_embeddings
        )
        store = CompressionRetrievalStore()
        stored = persist_neural_selection(
            store,
            original_text=pair.context,
            spans=spans,
            result=selection,
            metadata={"benchmark": "neural_query_shift", "pair_id": pair.pair_id},
        )
        recovered_contents: list[str] = []
        for index in recovery:
            recovered = store.retrieve_span(
                stored.receipt_id,
                spans[index].span_id,
                retrieval_id=f"{pair.pair_id}:q2:{spans[index].span_id}",
            )
            if recovered is None:
                raise RuntimeError("stored omitted span could not be rehydrated")
            recovered_contents.append(recovered.content)
        recovered_text = "\n".join([prism_text, *recovered_contents])
        savings = store.realized_savings(stored.receipt_id)
        assert savings is not None
        recovery_tokens = int(savings["retrieved_tokens"])

        rows.append(
            {
                "pair_id": pair.pair_id,
                "context_sha256": _sha256(pair.context),
                "q1_sha256": _sha256(pair.q1),
                "q2_sha256": _sha256(pair.q2),
                "a1_sha256": _sha256(pair.a1.casefold()),
                "a2_sha256": _sha256(pair.a2.casefold()),
                "q1_sentence": pair.q1_sentence,
                "q2_sentence": pair.q2_sentence,
                "sentence_count": len(spans),
                "input_tokens_approx": total_tokens,
                "active_budget_tokens_approx": budget,
                "lexical_selected": [spans[index].span_id for index in lexical_indices],
                "prism_r_selected": [spans[index].span_id for index in prism_indices],
                "recovered": [spans[index].span_id for index in recovery],
                "stored_omitted_spans": len(stored.spans),
                "lexical_q1_retained": _contains(lexical_text, pair.a1),
                "lexical_q2_retained": _contains(lexical_text, pair.a2),
                "prism_r_q1_retained": _contains(prism_text, pair.a1),
                "prism_r_q2_retained": _contains(prism_text, pair.a2),
                "q2_retained_after_rehydration": _contains(recovered_text, pair.a2),
                "active_tokens_approx": selection.selected_tokens,
                "recovery_tokens_approx": recovery_tokens,
                "selection_mode": selection.receipt["mode"],
                "selection_reason": selection.receipt["reason"],
                "receipt_sha256": _sha256(
                    json.dumps(selection.receipt, sort_keys=True, separators=(",", ":"))
                ),
            }
        )

    def rate(field: str) -> float:
        return round(sum(bool(row[field]) for row in rows) / len(rows), 6)

    input_tokens = sum(int(row["input_tokens_approx"]) for row in rows)
    active_tokens = sum(int(row["active_tokens_approx"]) for row in rows)
    recovery_tokens = sum(int(row["recovery_tokens_approx"]) for row in rows)
    prism_only_q1 = sum(
        row["prism_r_q1_retained"] and not row["lexical_q1_retained"] for row in rows
    )
    lexical_only_q1 = sum(
        row["lexical_q1_retained"] and not row["prism_r_q1_retained"] for row in rows
    )
    recovered_only_q2 = sum(
        row["q2_retained_after_rehydration"] and not row["prism_r_q2_retained"]
        for row in rows
    )
    active_only_q2 = sum(
        row["prism_r_q2_retained"] and not row["q2_retained_after_rehydration"]
        for row in rows
    )
    q1_p = _mcnemar_exact(prism_only_q1, lexical_only_q1)
    recovery_p = _mcnemar_exact(recovered_only_q2, active_only_q2)
    active_plus_recovery = (active_tokens + recovery_tokens) / input_tokens
    pilot_gate_passed = (
        rate("prism_r_q1_retained") > rate("lexical_q1_retained")
        and q1_p < 0.05
        and rate("q2_retained_after_rehydration") > rate("prism_r_q2_retained")
        and recovery_p < 0.05
        and active_plus_recovery < 1.0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "rajpurkar/squad_v2",
            "split": "validation",
            "dataset_fingerprint": dataset_fingerprint,
            "trials": trials,
            "seed": seed,
            "active_ratio": active_ratio,
            "query_shift": "q2 is a different question with an answer in a different sentence",
            "offline_only": True,
        },
        "model": {
            "id": model.model_id,
            "fingerprint_sha256": model.fingerprint,
        },
        "metrics": {
            "lexical_q1_retention": rate("lexical_q1_retained"),
            "lexical_future_q2_retention": rate("lexical_q2_retained"),
            "prism_r_q1_retention": rate("prism_r_q1_retained"),
            "prism_r_future_q2_retention": rate("prism_r_q2_retained"),
            "prism_r_q2_after_exact_rehydration": rate("q2_retained_after_rehydration"),
            "active_token_ratio_approx": round(active_tokens / input_tokens, 6),
            "rehydration_token_ratio_approx": round(recovery_tokens / input_tokens, 6),
            "active_plus_rehydration_ratio_approx": round(active_plus_recovery, 6),
            "paired": {
                "prism_only_q1": prism_only_q1,
                "lexical_only_q1": lexical_only_q1,
                "q1_mcnemar_exact_p": q1_p,
                "rehydration_only_q2": recovered_only_q2,
                "active_only_q2": active_only_q2,
                "rehydration_mcnemar_exact_p": recovery_p,
            },
            "selection_reasons": dict(sorted(modes.items())),
        },
        "pilot_gate_passed": pilot_gate_passed,
        "headline_eligible": False,
        "claim_scope": (
            "Exact answer-string retention before and after source-span rehydration "
            "under same-document future-query shift."
        ),
        "caveats": [
            "This measures exact evidence retention, not generated answer correctness.",
            "Approximate token counts use four characters per token.",
            "SQuAD paragraphs are short; long-agent repeated-compaction evaluation remains required.",
            "Future q2 is hidden from initial selection but comes from the same source paragraph.",
        ],
        "trials": rows,
    }


def verify_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    rows = report.get("trials")
    if not isinstance(rows, list) or not rows:
        raise ValueError("report contains no trials")
    if report.get("headline_eligible") is not False:
        raise ValueError("query-shift pilot must remain headline_eligible=false")
    if report.get("protocol", {}).get("trials") != len(rows):
        raise ValueError("protocol trial count does not match trial rows")
    if len({str(row.get("pair_id")) for row in rows}) != len(rows):
        raise ValueError("pair_id values must be unique")
    hash_fields = (
        "a1_sha256",
        "a2_sha256",
        "context_sha256",
        "q1_sha256",
        "q2_sha256",
        "receipt_sha256",
    )
    for index, row in enumerate(rows):
        for field in hash_fields:
            value = row.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"trial {index} has invalid {field}")
        sentence_count = int(row["sentence_count"])
        if sentence_count < 2:
            raise ValueError(f"trial {index} requires at least two sentences")
        if row["q1_sentence"] == row["q2_sentence"]:
            raise ValueError(f"trial {index} does not contain a real query shift")
        valid_span_ids = {f"sentence-{value}" for value in range(sentence_count)}
        for field in ("lexical_selected", "prism_r_selected", "recovered"):
            span_ids = row.get(field)
            if (
                not isinstance(span_ids, list)
                or len(set(span_ids)) != len(span_ids)
                or not set(span_ids).issubset(valid_span_ids)
            ):
                raise ValueError(f"trial {index} has invalid {field}")
        selected = set(row["prism_r_selected"])
        recovered = set(row["recovered"])
        if selected & recovered:
            raise ValueError(f"trial {index} recovers an already-active span")
        if row["stored_omitted_spans"] != sentence_count - len(selected):
            raise ValueError(f"trial {index} stored omitted count is inconsistent")
        input_tokens = int(row["input_tokens_approx"])
        active_tokens = int(row["active_tokens_approx"])
        recovery_tokens = int(row["recovery_tokens_approx"])
        if input_tokens <= 0 or not 0 <= active_tokens <= input_tokens:
            raise ValueError(f"trial {index} has invalid active token accounting")
        if not 0 <= recovery_tokens <= input_tokens:
            raise ValueError(f"trial {index} has invalid recovery token accounting")

    def rate(field: str) -> float:
        return round(sum(bool(row[field]) for row in rows) / len(rows), 6)

    expected = {
        "lexical_q1_retention": rate("lexical_q1_retained"),
        "lexical_future_q2_retention": rate("lexical_q2_retained"),
        "prism_r_q1_retention": rate("prism_r_q1_retained"),
        "prism_r_future_q2_retention": rate("prism_r_q2_retained"),
        "prism_r_q2_after_exact_rehydration": rate("q2_retained_after_rehydration"),
    }
    for key, value in expected.items():
        if report["metrics"].get(key) != value:
            raise ValueError(f"{key} does not match trial rows")
    input_tokens = sum(int(row["input_tokens_approx"]) for row in rows)
    active_tokens = sum(int(row["active_tokens_approx"]) for row in rows)
    recovery_tokens = sum(int(row["recovery_tokens_approx"]) for row in rows)
    ratios = {
        "active_token_ratio_approx": round(active_tokens / input_tokens, 6),
        "rehydration_token_ratio_approx": round(recovery_tokens / input_tokens, 6),
        "active_plus_rehydration_ratio_approx": round(
            (active_tokens + recovery_tokens) / input_tokens, 6
        ),
    }
    for key, value in ratios.items():
        if report["metrics"].get(key) != value:
            raise ValueError(f"{key} does not match trial rows")
    prism_only_q1 = sum(
        row["prism_r_q1_retained"] and not row["lexical_q1_retained"] for row in rows
    )
    lexical_only_q1 = sum(
        row["lexical_q1_retained"] and not row["prism_r_q1_retained"] for row in rows
    )
    recovered_only_q2 = sum(
        row["q2_retained_after_rehydration"] and not row["prism_r_q2_retained"]
        for row in rows
    )
    active_only_q2 = sum(
        row["prism_r_q2_retained"] and not row["q2_retained_after_rehydration"]
        for row in rows
    )
    paired = {
        "prism_only_q1": prism_only_q1,
        "lexical_only_q1": lexical_only_q1,
        "q1_mcnemar_exact_p": _mcnemar_exact(prism_only_q1, lexical_only_q1),
        "rehydration_only_q2": recovered_only_q2,
        "active_only_q2": active_only_q2,
        "rehydration_mcnemar_exact_p": _mcnemar_exact(
            recovered_only_q2, active_only_q2
        ),
    }
    if report["metrics"].get("paired") != paired:
        raise ValueError("paired significance metrics do not match trial rows")
    selection_reasons = dict(
        sorted(Counter(str(row["selection_reason"]) for row in rows).items())
    )
    if report["metrics"].get("selection_reasons") != selection_reasons:
        raise ValueError("selection reason metrics do not match trial rows")
    pilot_gate = (
        expected["prism_r_q1_retention"] > expected["lexical_q1_retention"]
        and paired["q1_mcnemar_exact_p"] < 0.05
        and expected["prism_r_q2_after_exact_rehydration"]
        > expected["prism_r_future_q2_retention"]
        and paired["rehydration_mcnemar_exact_p"] < 0.05
        and ratios["active_plus_rehydration_ratio_approx"] < 1.0
    )
    if report.get("pilot_gate_passed") is not pilot_gate:
        raise ValueError("pilot_gate_passed does not match trial rows")


def render_markdown(report: dict[str, Any]) -> str:
    verify_report(report)
    metrics = report["metrics"]
    lines = [
        "# PRISM-R Query-Shift and Rehydration Pilot",
        "",
        (
            "**PILOT GATE PASSED — NOT YET A BREAKTHROUGH CLAIM.**"
            if report["pilot_gate_passed"]
            else "**RESEARCH PILOT — NO BREAKTHROUGH CLAIM.**"
        ),
        "",
        report["claim_scope"],
        "",
        "| Method | Current q1 evidence | Unseen q2 evidence | q2 after recovery |",
        "|---|---:|---:|---:|",
        (
            f"| Lexical compression | {metrics['lexical_q1_retention']:.1%} | "
            f"{metrics['lexical_future_q2_retention']:.1%} | n/a |"
        ),
        (
            f"| PRISM-R active context | {metrics['prism_r_q1_retention']:.1%} | "
            f"{metrics['prism_r_future_q2_retention']:.1%} | n/a |"
        ),
        (
            f"| PRISM-R + exact rehydration | {metrics['prism_r_q1_retention']:.1%} | "
            f"{metrics['prism_r_future_q2_retention']:.1%} | "
            f"{metrics['prism_r_q2_after_exact_rehydration']:.1%} |"
        ),
        "",
        f"- Active context ratio: {metrics['active_token_ratio_approx']:.1%}",
        f"- Additional recovery ratio: {metrics['rehydration_token_ratio_approx']:.1%}",
        (
            "- Active + recovery ratio: "
            f"{metrics['active_plus_rehydration_ratio_approx']:.1%}"
        ),
        f"- Current-q1 exact McNemar p: {metrics['paired']['q1_mcnemar_exact_p']:.3e}",
        (
            "- Rehydration exact McNemar p: "
            f"{metrics['paired']['rehydration_mcnemar_exact_p']:.3e}"
        ),
        "",
        "## Caveats",
        "",
        *[f"- {value}" for value in report["caveats"]],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--trials", type=int, default=200)
    run_parser.add_argument("--seed", type=int, default=91_337)
    run_parser.add_argument("--active-ratio", type=float, default=0.25)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--markdown", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("input", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        report = run(
            model_path=args.model_path,
            trials=args.trials,
            seed=args.seed,
            active_ratio=args.active_ratio,
        )
        verify_report(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        markdown = render_markdown(report)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(markdown, encoding="utf-8")
        print(markdown, end="")
        return 0
    report = json.loads(args.input.read_text(encoding="utf-8"))
    verify_report(report)
    print(f"VERIFIED {args.input}: {len(report['trials'])} query-shift trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
