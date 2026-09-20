from __future__ import annotations

import re
from collections.abc import Iterable
from .types import AuthorityEnvelope, ConstraintKind, ConstraintSpan, QueryContract

_EXCLUSION_RE = re.compile(r"\b(?:except|excluding|exclude|without|but not|other than)\b\s+(?P<target>[^.;!?\"']+)", re.I)
_CONDITION_RE = re.compile(r"\b(?:if|when|unless|only if|provided that)\b\s+(?P<target>[^.;]+)", re.I)
_SENTENCE_RE = re.compile(r"[^.!?;]+")


def _quoted_ranges(text: str) -> list[range]:
    ranges: list[range] = []
    for quote in ('"', "'"):
        start = None
        for idx, ch in enumerate(text):
            if ch == quote:
                if start is None:
                    start = idx
                else:
                    ranges.append(range(start, idx + 1))
                    start = None
    return ranges


def _inside_quoted(idx: int, ranges: list[range]) -> bool:
    return any(idx in r for r in ranges)


def _trim_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    while end > start and text[end - 1] in ".,;!?":
        end -= 1
    return text[start:end], start, end


def _spans(pattern: re.Pattern[str], text: str, kind: ConstraintKind) -> tuple[ConstraintSpan, ...]:
    quoted = _quoted_ranges(text)
    out: list[ConstraintSpan] = []
    for m in pattern.finditer(text):
        if _inside_quoted(m.start(), quoted):
            continue
        s, e = m.span("target")
        target, s, e = _trim_span(text, s, e)
        if target:
            out.append(ConstraintSpan(kind, target, s, e))
    return tuple(out)


def _obligations(text: str, exclusions: tuple[ConstraintSpan, ...], conditions: tuple[ConstraintSpan, ...]) -> tuple[ConstraintSpan, ...]:
    blocked = [range(x.start, x.end) for x in (*exclusions, *conditions)]
    quoted = _quoted_ranges(text)
    out: list[ConstraintSpan] = []
    for m in _SENTENCE_RE.finditer(text):
        if _inside_quoted(m.start(), quoted):
            continue
        raw, s, e = _trim_span(text, m.start(), m.end())
        if not raw:
            continue
        if any(s in r or max(s, e - 1) in r for r in blocked):
            continue
        out.append(ConstraintSpan(ConstraintKind.OBLIGATION, raw, s, e, confidence=0.7))
    return tuple(out)


def compile_query_contract(
    task: str,
    *,
    allowed_capabilities: Iterable[str] = (),
    denied_capabilities: Iterable[str] = (),
) -> QueryContract:
    """Compile natural-language task constraints without granting authority from text."""
    exclusions = _spans(_EXCLUSION_RE, task, ConstraintKind.EXCLUSION)
    conditions = _spans(_CONDITION_RE, task, ConstraintKind.CONDITION)
    obligations = _obligations(task, exclusions, conditions)
    authority = AuthorityEnvelope(tuple(allowed_capabilities), tuple(denied_capabilities))
    return QueryContract(task, obligations, exclusions, conditions, authority)
