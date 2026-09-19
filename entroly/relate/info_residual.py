"""Information residual: concrete content unique to an omitted fragment.

An omission is causally unsafe when it removes information that cannot be
reconstructed from the retained set.  This module extracts structured
information tokens (constraints, values, states, paths) and computes which
are unique to the omitted fragment.

This is NOT semantic similarity.  It is structural type matching: does the
retained set contain the same KINDS of concrete information?
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = re.compile(
    r"(?<![A-Za-z0-9])\$?\d+(?:[.,]\d+)?"
    r"(?:\s*(?:%|rpm|rps|ms|seconds?|minutes?|hours?|days?|weeks?|months?|years?|GB|MB|KB|TB))?"
    r"(?![A-Za-z0-9])",
    re.I,
)
_TEMPORAL = re.compile(
    r"\b\d+\s+(?:business\s+)?(?:days?|weeks?|months?|hours?|minutes?|seconds?)\b",
    re.I,
)
_PATH = re.compile(r"(?:/[\w.-]+){2,}")
_CONSTRAINT = re.compile(
    r"\b(?:must(?:\s+never)?|require[sd]?|shall(?:\s+not)?|prohibit(?:ed|s)?|only\s+if|unless|cannot)\b",
    re.I,
)
_BOOLEAN_STATE = re.compile(
    r"\b(?:enabled|disabled|allowed|denied|approved|rejected|active|inactive)\b",
    re.I,
)
_STATE_ANTONYM = {
    "enabled": "disabled", "disabled": "enabled",
    "allowed": "denied", "denied": "allowed",
    "approved": "rejected", "rejected": "approved",
    "active": "inactive", "inactive": "active",
}
_STOPWORDS = frozenset(
    "that this with from have been were being their about would could should "
    "which there these those than some more very just also only other into "
    "does when what where".split()
)
_ACTION_VERBS = frozenset(
    "restart deploy configure update grant revoke process execute install "
    "remove delete create start stop migrate".split()
)
_VALUE_QUESTIONS = (
    "how many", "how much", "what is the", "what are the",
    "what port", "what rate", "what limit", "what amount",
    "what cost", "what price", "what version",
)


def _extract(pattern: re.Pattern[str], text: str) -> frozenset[str]:
    return frozenset(m.group(0).strip().lower() for m in pattern.finditer(text))


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", text)
    ) - _STOPWORDS


@dataclass(frozen=True)
class InfoResidual:
    unique_constraints: tuple[str, ...]
    unique_numbers: tuple[str, ...]
    unique_temporals: tuple[str, ...]
    unique_paths: tuple[str, ...]

    @property
    def has_constraint_residual(self) -> bool:
        return bool(self.unique_constraints)

    @property
    def has_value_residual(self) -> bool:
        return bool(self.unique_numbers or self.unique_temporals)

    @property
    def has_path_residual(self) -> bool:
        return bool(self.unique_paths)


def compute_residual(omitted_text: str, retained_text: str) -> InfoResidual:
    return InfoResidual(
        unique_constraints=tuple(sorted(_extract(_CONSTRAINT, omitted_text) - _extract(_CONSTRAINT, retained_text))),
        unique_numbers=tuple(sorted(_extract(_NUM, omitted_text) - _extract(_NUM, retained_text))),
        unique_temporals=tuple(sorted(_extract(_TEMPORAL, omitted_text) - _extract(_TEMPORAL, retained_text))),
        unique_paths=tuple(sorted(_extract(_PATH, omitted_text) - _extract(_PATH, retained_text))),
    )


def detect_state_conflict(omitted_text: str, retained_texts: list[str]) -> list[str]:
    """Detect when omission hides a boolean state contradiction."""
    reasons: list[str] = []
    omit_states = _extract(_BOOLEAN_STATE, omitted_text)
    omit_words = _content_words(omitted_text)

    for ret_text in retained_texts:
        ret_states = _extract(_BOOLEAN_STATE, ret_text)
        ret_words = _content_words(ret_text)
        shared = omit_words & ret_words
        if len(shared) < 2:
            continue
        for s in omit_states:
            antonym = _STATE_ANTONYM.get(s)
            if antonym and antonym in ret_states:
                reasons.append(f"state_conflict:{s}_vs_{antonym}")
        omit_nums = _extract(_NUM, omitted_text)
        ret_nums = _extract(_NUM, ret_text)
        if omit_nums and ret_nums and omit_nums != ret_nums:
            reasons.append(f"numeric_conflict:{','.join(sorted(omit_nums))}_vs_{','.join(sorted(ret_nums))}")
    return reasons


def task_asks_for_value(task: str) -> bool:
    task_lower = task.lower()
    return any(vq in task_lower for vq in _VALUE_QUESTIONS)


def task_is_action(task: str) -> bool:
    return bool(_ACTION_VERBS & set(task.lower().split()))
