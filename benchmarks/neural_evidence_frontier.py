"""Offline semantic-evidence retrieval frontier on SQuAD v2.

This benchmark measures whether a local transformer ranks the paragraph that
contains a known answer above distractor paragraphs. It does not call an LLM
and therefore does not measure downstream answer quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from entroly.neural_evidence_selector import (
    EvidenceSpan,
    LocalTransformerEncoder,
    score_lexical_evidence,
)

SCHEMA_VERSION = "entroly.neural-evidence-frontier.v1"
DATASET_ID = "rajpurkar/squad_v2"
DATASET_SPLIT = "validation"


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _rank(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def _wilson_upper(errors: int, total: int, *, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 1.0
    proportion = errors / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return min(1.0, (center + radius) / denominator)


def _mcnemar_exact(neural_only: int, lexical_only: int) -> float:
    discordant = neural_only + lexical_only
    if discordant == 0:
        return 1.0
    tail = min(neural_only, lexical_only)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


@dataclass(frozen=True, slots=True)
class SourceRow:
    example_id: str
    question: str
    context: str
    answer: str


def _load_rows() -> tuple[list[SourceRow], str]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - optional environment
        raise RuntimeError(
            "install the neural-benchmark extra to run this benchmark"
        ) from error
    try:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    except Exception as error:  # pragma: no cover - cache state dependent
        raise RuntimeError(
            "SQuAD v2 is not cached locally; this benchmark refuses an implicit download"
        ) from error

    rows: list[SourceRow] = []
    seen_contexts: set[str] = set()
    for record in dataset:
        answers = record.get("answers", {}).get("text", [])
        if not answers:
            continue
        context = str(record["context"]).strip()
        answer = str(answers[0]).strip()
        if len(answer) < 3 or context in seen_contexts:
            continue
        seen_contexts.add(context)
        rows.append(
            SourceRow(
                example_id=str(record["id"]),
                question=str(record["question"]),
                context=context,
                answer=answer,
            )
        )
    fingerprint = str(getattr(dataset, "_fingerprint", "unknown"))
    return rows, fingerprint


def _build_trials(
    rows: Sequence[SourceRow], *, trials: int, distractors: int, seed: int
) -> list[dict[str, Any]]:
    if trials < 2:
        raise ValueError("trials must be at least 2")
    if distractors < 1:
        raise ValueError("distractors must be positive")
    if len(rows) < trials + distractors + 1:
        raise ValueError("dataset does not contain enough unique answerable contexts")

    rng = random.Random(seed)
    trial_rows = list(rows[:trials])
    pool = list(rows[: min(len(rows), max(trials * 5, trials + distractors + 1))])
    built: list[dict[str, Any]] = []
    for trial_index, row in enumerate(trial_rows):
        eligible = [
            candidate
            for candidate in pool
            if candidate.example_id != row.example_id
            and row.answer.casefold() not in candidate.context.casefold()
        ]
        if len(eligible) < distractors:
            raise ValueError(f"insufficient distractors for {row.example_id}")
        selected_distractors = rng.sample(eligible, distractors)
        gold_index = rng.randrange(distractors + 1)
        candidates = [candidate.context for candidate in selected_distractors]
        candidate_ids = [candidate.example_id for candidate in selected_distractors]
        candidates.insert(gold_index, row.context)
        candidate_ids.insert(gold_index, row.example_id)
        built.append(
            {
                "trial_index": trial_index,
                "example_id": row.example_id,
                "question": row.question,
                "answer": row.answer,
                "candidates": candidates,
                "candidate_ids": candidate_ids,
                "gold_index": gold_index,
            }
        )
    return built


def _calibrate_override(
    rows: Sequence[dict[str, Any]],
    *,
    max_error_upper: float,
    minimum_overrides: int,
) -> dict[str, Any]:
    disagreements = [row for row in rows if row["neural_top"] != row["lexical_top"]]
    thresholds = sorted({float(row["neural_margin"]) for row in disagreements})
    candidates: list[dict[str, Any]] = []
    lexical_accuracy = sum(row["lexical_correct"] for row in rows) / len(rows)
    for threshold in thresholds:
        overrides = [
            row for row in disagreements if float(row["neural_margin"]) >= threshold
        ]
        if len(overrides) < minimum_overrides:
            continue
        errors = sum(not row["neural_correct"] for row in overrides)
        upper = _wilson_upper(errors, len(overrides))
        gated_correct = sum(
            (
                row["neural_correct"]
                if row["neural_top"] != row["lexical_top"]
                and float(row["neural_margin"]) >= threshold
                else row["lexical_correct"]
            )
            for row in rows
        )
        accuracy = gated_correct / len(rows)
        if upper <= max_error_upper and accuracy >= lexical_accuracy:
            candidates.append(
                {
                    "threshold": threshold,
                    "overrides": len(overrides),
                    "override_errors": errors,
                    "override_error_upper_95": upper,
                    "accuracy": accuracy,
                }
            )
    if not candidates:
        return {
            "eligible": False,
            "threshold": None,
            "reason": "no threshold passed the finite-sample risk and non-inferiority gates",
            "max_override_error_upper_95": max_error_upper,
            "minimum_overrides": minimum_overrides,
        }
    winner = max(
        candidates,
        key=lambda candidate: (
            candidate["accuracy"],
            candidate["overrides"],
            -candidate["override_error_upper_95"],
            candidate["threshold"],
        ),
    )
    return {
        "eligible": True,
        **winner,
        "max_override_error_upper_95": max_error_upper,
        "minimum_overrides": minimum_overrides,
        "method": "Wilson upper bound on neural-only override error; not a conformal guarantee",
    }


def _apply_gate(row: dict[str, Any], calibration: dict[str, Any]) -> int:
    threshold = calibration.get("threshold")
    if (
        calibration.get("eligible")
        and row["neural_top"] != row["lexical_top"]
        and float(row["neural_margin"]) >= float(threshold)
    ):
        return int(row["neural_top"])
    return int(row["lexical_top"])


def _metrics(rows: Sequence[dict[str, Any]], system: str) -> dict[str, Any]:
    top_field = f"{system}_top"
    order_field = f"{system}_order"
    correct = sum(int(row[top_field]) == int(row["gold_index"]) for row in rows)
    reciprocal_rank = 0.0
    top2 = 0
    for row in rows:
        order = [int(value) for value in row[order_field]]
        rank = order.index(int(row["gold_index"])) + 1
        reciprocal_rank += 1.0 / rank
        top2 += rank <= 2
    return {
        "trials": len(rows),
        "top1_correct": correct,
        "top1_recall": round(correct / len(rows), 6),
        "top2_recall": round(top2 / len(rows), 6),
        "mrr": round(reciprocal_rank / len(rows), 6),
    }


def _gated_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(int(row["gated_top"]) == int(row["gold_index"]) for row in rows)
    return {
        "trials": len(rows),
        "top1_correct": correct,
        "top1_recall": round(correct / len(rows), 6),
    }


def _guarded_metrics(
    rows: Sequence[dict[str, Any]], *, candidates_per_trial: int
) -> dict[str, Any]:
    correct = sum(
        bool(row["lexical_correct"]) or bool(row["neural_correct"]) for row in rows
    )
    disagreements = sum(row["lexical_top"] != row["neural_top"] for row in rows)
    selected = len(rows) + disagreements
    average_selected = selected / len(rows)
    return {
        "trials": len(rows),
        "answer_passage_correct": correct,
        "answer_passage_recall": round(correct / len(rows), 6),
        "disagreements_guarded": disagreements,
        "average_selected_passages": round(average_selected, 6),
        "passage_compression_ratio": round(candidates_per_trial / average_selected, 6),
    }


def run(
    *,
    model_path: Path,
    trials: int,
    distractors: int,
    seed: int,
    calibration_fraction: float,
    max_override_error_upper: float,
    minimum_overrides: int,
) -> dict[str, Any]:
    if not 0.1 <= calibration_fraction <= 0.9:
        raise ValueError("calibration_fraction must be between 0.1 and 0.9")
    source_rows, dataset_fingerprint = _load_rows()
    trial_rows = _build_trials(
        source_rows, trials=trials, distractors=distractors, seed=seed
    )
    encoder = LocalTransformerEncoder(model_path)

    encoded_inputs: list[str] = []
    for trial in trial_rows:
        encoded_inputs.append(str(trial["question"]))
        encoded_inputs.extend(str(value) for value in trial["candidates"])
    embeddings = list(encoder.encode(encoded_inputs))

    cursor = 0
    result_rows: list[dict[str, Any]] = []
    for trial in trial_rows:
        question_embedding = embeddings[cursor]
        candidate_embeddings = embeddings[
            cursor + 1 : cursor + 2 + distractors
        ]
        cursor += distractors + 2
        neural_scores = [
            max(0.0, min(1.0, _dot(question_embedding, vector)))
            for vector in candidate_embeddings
        ]
        spans = [
            EvidenceSpan(
                span_id=str(candidate_id),
                source=str(candidate_id),
                text=str(context),
                token_count=max(1, len(str(context)) // 4),
                ordinal=index,
            )
            for index, (candidate_id, context) in enumerate(
                zip(trial["candidate_ids"], trial["candidates"])
            )
        ]
        lexical_scores = list(score_lexical_evidence(str(trial["question"]), spans))
        lexical_order = _rank(lexical_scores)
        neural_order = _rank(neural_scores)
        result_rows.append(
            {
                "trial_index": trial["trial_index"],
                "example_id": trial["example_id"],
                "question_sha256": _sha256(str(trial["question"])),
                "answer_sha256": _sha256(str(trial["answer"]).casefold()),
                "candidate_ids": list(trial["candidate_ids"]),
                "candidate_sha256": [
                    _sha256(str(context)) for context in trial["candidates"]
                ],
                "gold_index": trial["gold_index"],
                "lexical_order": lexical_order,
                "neural_order": neural_order,
                "lexical_top": lexical_order[0],
                "neural_top": neural_order[0],
                "lexical_correct": lexical_order[0] == trial["gold_index"],
                "neural_correct": neural_order[0] == trial["gold_index"],
                "neural_margin": round(
                    neural_scores[neural_order[0]] - neural_scores[neural_order[1]], 8
                ),
            }
        )

    split_index = max(1, min(len(result_rows) - 1, round(len(result_rows) * calibration_fraction)))
    calibration_rows = result_rows[:split_index]
    test_rows = result_rows[split_index:]
    calibration = _calibrate_override(
        calibration_rows,
        max_error_upper=max_override_error_upper,
        minimum_overrides=minimum_overrides,
    )
    for index, row in enumerate(result_rows):
        row["partition"] = "calibration" if index < split_index else "test"
        row["gated_top"] = _apply_gate(row, calibration)
        row["gated_correct"] = row["gated_top"] == row["gold_index"]

    lexical_metrics = _metrics(test_rows, "lexical")
    neural_metrics = _metrics(test_rows, "neural")
    gated_metrics = _gated_metrics(test_rows)
    guarded_metrics = _guarded_metrics(
        test_rows, candidates_per_trial=distractors + 1
    )
    neural_only = sum(
        row["neural_correct"] and not row["lexical_correct"] for row in test_rows
    )
    lexical_only = sum(
        row["lexical_correct"] and not row["neural_correct"] for row in test_rows
    )
    p_value = _mcnemar_exact(neural_only, lexical_only)
    headline_eligible = (
        neural_metrics["top1_recall"] > lexical_metrics["top1_recall"]
        and p_value < 0.05
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": DATASET_ID,
            "split": DATASET_SPLIT,
            "dataset_fingerprint": dataset_fingerprint,
            "trials": trials,
            "distractors_per_trial": distractors,
            "candidates_per_trial": distractors + 1,
            "seed": seed,
            "calibration_fraction": calibration_fraction,
            "calibration_trials": len(calibration_rows),
            "test_trials": len(test_rows),
            "offline_only": True,
        },
        "model": {
            "id": encoder.model_id,
            "fingerprint_sha256": encoder.fingerprint,
            "path_disclosed": False,
        },
        "calibration": calibration,
        "test_metrics": {
            "lexical_bm25": lexical_metrics,
            "local_transformer": neural_metrics,
            "risk_gated": gated_metrics,
            "dual_channel_guard": guarded_metrics,
            "paired": {
                "transformer_only_correct": neural_only,
                "lexical_only_correct": lexical_only,
                "mcnemar_exact_p": round(p_value, 8),
            },
        },
        "headline_eligible": headline_eligible,
        "claim_scope": (
            "Answer-bearing paragraph retrieval under fixed one-of-N selection on a "
            "frozen SQuAD v2 validation subset."
        ),
        "caveats": [
            "This benchmark measures answer-bearing paragraph retrieval, not downstream LLM answers.",
            "The local transformer is a pretrained semantic encoder, not an Entroly-trained compressor.",
            "The calibration policy uses a Wilson error bound and is not a conformal guarantee.",
            "A headline requires a positive transformer delta and paired McNemar p < 0.05 on the held-out partition.",
        ],
        "trials": result_rows,
    }


def _recompute_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    test_rows = [row for row in rows if row.get("partition") == "test"]
    return {
        "lexical_bm25": _metrics(test_rows, "lexical"),
        "local_transformer": _metrics(test_rows, "neural"),
        "risk_gated": _gated_metrics(test_rows),
    }


def verify_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    rows = report.get("trials")
    if not isinstance(rows, list) or not rows:
        raise ValueError("report contains no trials")
    candidate_count = int(report["protocol"]["candidates_per_trial"])
    seen: set[int] = set()
    for row in rows:
        trial_index = int(row["trial_index"])
        if trial_index in seen:
            raise ValueError("duplicate trial_index")
        seen.add(trial_index)
        expected_order = list(range(candidate_count))
        if sorted(row["lexical_order"]) != expected_order:
            raise ValueError(f"invalid lexical order for trial {trial_index}")
        if sorted(row["neural_order"]) != expected_order:
            raise ValueError(f"invalid neural order for trial {trial_index}")
        if row["lexical_top"] != row["lexical_order"][0]:
            raise ValueError(f"lexical top mismatch for trial {trial_index}")
        if row["neural_top"] != row["neural_order"][0]:
            raise ValueError(f"neural top mismatch for trial {trial_index}")
        if row["lexical_correct"] is not (row["lexical_top"] == row["gold_index"]):
            raise ValueError(f"lexical correctness mismatch for trial {trial_index}")
        if row["neural_correct"] is not (row["neural_top"] == row["gold_index"]):
            raise ValueError(f"neural correctness mismatch for trial {trial_index}")
        if row["gated_correct"] is not (row["gated_top"] == row["gold_index"]):
            raise ValueError(f"gated correctness mismatch for trial {trial_index}")

    recomputed = _recompute_metrics(rows)
    for system, metrics in recomputed.items():
        if report["test_metrics"][system] != metrics:
            raise ValueError(f"{system} metrics do not match trial rows")
    guarded = _guarded_metrics(
        [row for row in rows if row["partition"] == "test"],
        candidates_per_trial=candidate_count,
    )
    if report["test_metrics"].get("dual_channel_guard") != guarded:
        raise ValueError("dual_channel_guard metrics do not match trial rows")
    test_rows = [row for row in rows if row["partition"] == "test"]
    neural_only = sum(
        row["neural_correct"] and not row["lexical_correct"] for row in test_rows
    )
    lexical_only = sum(
        row["lexical_correct"] and not row["neural_correct"] for row in test_rows
    )
    paired = report["test_metrics"]["paired"]
    if paired["transformer_only_correct"] != neural_only:
        raise ValueError("paired transformer-only count does not match trials")
    if paired["lexical_only_correct"] != lexical_only:
        raise ValueError("paired lexical-only count does not match trials")
    p_value = round(_mcnemar_exact(neural_only, lexical_only), 8)
    if paired["mcnemar_exact_p"] != p_value:
        raise ValueError("paired McNemar p-value does not match trials")
    eligible = (
        recomputed["local_transformer"]["top1_recall"]
        > recomputed["lexical_bm25"]["top1_recall"]
        and p_value < 0.05
    )
    if report["headline_eligible"] is not eligible:
        raise ValueError("headline_eligible does not match the statistical gate")


def render_markdown(report: dict[str, Any]) -> str:
    verify_report(report)
    metrics = report["test_metrics"]
    paired = metrics["paired"]
    status = (
        "**HELD-OUT SEMANTIC RETRIEVAL WIN.**"
        if report["headline_eligible"]
        else "**NO BREAKTHROUGH CLAIM: statistical gate not passed.**"
    )
    lines = [
        "# Neural Evidence Retrieval Frontier",
        "",
        status,
        "",
        report["claim_scope"],
        "",
        "| Selector | Top-1 answer-passage recall | Top-2 recall | MRR |",
        "|---|---:|---:|---:|",
        (
            f"| Deterministic BM25 | {metrics['lexical_bm25']['top1_recall']:.1%} | "
            f"{metrics['lexical_bm25']['top2_recall']:.1%} | {metrics['lexical_bm25']['mrr']:.4f} |"
        ),
        (
            f"| Local transformer | {metrics['local_transformer']['top1_recall']:.1%} | "
            f"{metrics['local_transformer']['top2_recall']:.1%} | {metrics['local_transformer']['mrr']:.4f} |"
        ),
        (
            f"| Risk-gated selector | {metrics['risk_gated']['top1_recall']:.1%} | n/a | n/a |"
        ),
        (
            f"| Dual-channel disagreement guard | "
            f"{metrics['dual_channel_guard']['answer_passage_recall']:.1%} | n/a | n/a |"
        ),
        "",
        "## Paired test",
        "",
        f"- Transformer-only correct: {paired['transformer_only_correct']}",
        f"- Lexical-only correct: {paired['lexical_only_correct']}",
        f"- Exact McNemar p-value: {paired['mcnemar_exact_p']:.8f}",
        f"- Held-out trials: {report['protocol']['test_trials']}",
        f"- Distractors per trial: {report['protocol']['distractors_per_trial']}",
        (
            "- Dual-channel guard average selected passages: "
            f"{metrics['dual_channel_guard']['average_selected_passages']:.3f} / "
            f"{report['protocol']['candidates_per_trial']}"
        ),
        (
            "- Dual-channel guard passage compression: "
            f"{metrics['dual_channel_guard']['passage_compression_ratio']:.2f}x"
        ),
        "",
        "## Calibration",
        "",
        "```json",
        json.dumps(report["calibration"], indent=2, sort_keys=True),
        "```",
        "",
        "## Caveats",
        "",
        *[f"- {caveat}" for caveat in report["caveats"]],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--trials", type=int, default=1_000)
    run_parser.add_argument("--distractors", type=int, default=15)
    run_parser.add_argument("--seed", type=int, default=73_021)
    run_parser.add_argument("--calibration-fraction", type=float, default=0.5)
    run_parser.add_argument("--max-override-error-upper", type=float, default=0.10)
    run_parser.add_argument("--minimum-overrides", type=int, default=40)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--markdown", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("input", type=Path)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("input", type=Path)
    render_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "run":
        report = run(
            model_path=args.model_path,
            trials=args.trials,
            distractors=args.distractors,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
            max_override_error_upper=args.max_override_error_upper,
            minimum_overrides=args.minimum_overrides,
        )
        verify_report(report)
        rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered_json, encoding="utf-8")
        rendered_markdown = render_markdown(report)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(rendered_markdown, encoding="utf-8")
        print(rendered_markdown, end="")
        return 0
    report = json.loads(args.input.read_text(encoding="utf-8"))
    verify_report(report)
    if args.command == "verify":
        print(
            f"VERIFIED {args.input}: {report['protocol']['trials']} trials, "
            f"headline_eligible={report['headline_eligible']}"
        )
        return 0
    rendered = render_markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
