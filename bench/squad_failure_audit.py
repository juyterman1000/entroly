"""Failure attribution for answerable extractive-QA compression samples."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable, Sequence


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def answer_present(text: str, answers: Iterable[str]) -> bool:
    normalized_text = normalize_answer(text)
    return any(
        normalized and normalized in normalized_text
        for normalized in (normalize_answer(answer) for answer in answers)
    )


@dataclass(frozen=True)
class FailureAttribution:
    answer_present_raw: bool
    answer_present_pre_trim: bool
    answer_present_post_trim: bool
    baseline_correct: bool
    treatment_correct: bool
    category: str


def classify_failure(
    *,
    raw_context: str,
    selected_pre_trim: str,
    emitted_post_trim: str,
    answers: Sequence[str],
    baseline_correct: bool,
    treatment_correct: bool,
) -> FailureAttribution:
    raw = answer_present(raw_context, answers)
    pre = answer_present(selected_pre_trim, answers)
    post = answer_present(emitted_post_trim, answers)
    if not raw:
        category = "dataset_or_answer_normalization_mismatch"
    elif not pre:
        category = "retrieval_or_ranking_loss"
    elif not post:
        category = "trim_or_boundary_loss"
    elif baseline_correct and not treatment_correct:
        category = "utilization_order_or_generation_failure"
    elif not baseline_correct and treatment_correct:
        category = "compression_gain_or_baseline_variance"
    elif baseline_correct and treatment_correct:
        category = "preserved"
    else:
        category = "both_wrong_or_evaluator_variance"
    return FailureAttribution(raw, pre, post, baseline_correct, treatment_correct, category)
