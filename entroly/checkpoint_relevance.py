"""Query-conditioned checkpoint selection and safe continuity rendering."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Iterable, Mapping, Sequence

from .hardening import sanitize_injected_context

_WORD_RE = re.compile(r"[A-Za-z0-9_:-]+")
_STOPWORDS = frozenset(
    {
        "about", "after", "again", "also", "and", "are", "but", "can",
        "code", "did", "does", "for", "from", "have", "into", "issue",
        "make", "need", "project", "that", "the", "this", "was", "what",
        "when", "where", "which", "with", "would",
    }
)


@dataclass(frozen=True, slots=True)
class CheckpointRelevancePolicy:
    minimum_score: float = 0.28
    recency_half_life_seconds: float = 7 * 24 * 3600
    max_age_seconds: float = 90 * 24 * 3600
    max_decisions: int = 20

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be between zero and one")
        if self.recency_half_life_seconds <= 0 or self.max_age_seconds <= 0:
            raise ValueError("checkpoint time windows must be positive")
        if self.max_decisions < 1:
            raise ValueError("max_decisions must be positive")


@dataclass(frozen=True, slots=True)
class CheckpointMatch:
    checkpoint: Any
    score: float
    lexical_score: float
    source_score: float
    project_score: float
    recency_score: float
    matched_terms: tuple[str, ...]


DECISION_CHAR_LIMIT = 500

_TRUNCATION_NOTE = " [truncated: {kept} of {total} chars]"
_OVERFLOW_NOTE = "[{count} further decision(s) not carried: raise max_decisions]"
# Parsed, not merely skipped, so a count lost at an earlier checkpoint is still
# reported after a later merge drops the note itself.
_OVERFLOW_RE = re.compile(r"^\[(\d+) further decision\(s\) not carried")


def bounded_text(value: str, limit: int) -> str:
    """Bound ``value`` and say so when it does not fit.

    Bounds keep a checkpoint from growing without limit, which is correct. Cutting
    silently is not: a resuming agent receives a decision that stops mid-sentence
    with no indication that anything is missing, and cannot tell a complete
    instruction from a severed one. Marking the cut costs a few characters and
    turns invisible loss into a visible, actionable one.
    """
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + _TRUNCATION_NOTE.format(kept=limit, total=len(value))


def normalize_decisions(
    value: Any, *, limit: int = 20, char_limit: int = DECISION_CHAR_LIMIT
) -> list[str]:
    """Normalize explicit decisions without inferring intent from arbitrary code.

    Both bounds here are reported rather than applied silently, because this is
    the handoff path: whatever is dropped is dropped from the only record the
    next agent will read.

    Deduplication keys on the full text, not the bounded text. Keying on the
    bounded form made two decisions that differ only past the limit collide, so
    the second was discarded as a duplicate — a correctness bug, not just a
    visibility one, and the longer the decisions the likelier it fired.
    """
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = value
    else:
        return []

    unique: list[str] = []
    seen: set[str] = set()
    carried_over = 0
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        full = " ".join(candidate.strip().split())
        if not full:
            continue
        overflow = _OVERFLOW_RE.match(full)
        if overflow:
            carried_over += int(overflow.group(1))
            continue
        key = full.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(full)

    output = [bounded_text(text, char_limit) for text in unique[:limit]]
    dropped = len(unique) - len(output) + carried_over
    if dropped > 0:
        # Exceeds ``limit`` by one on purpose: the note is an accounting record,
        # not a decision, and losing it would re-hide the loss it reports.
        output.append(_OVERFLOW_NOTE.format(count=dropped))
    return output


def merge_checkpoint_metadata(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    max_decisions: int = 20,
) -> dict[str, Any]:
    """Carry explicit decisions forward while allowing current metadata to win."""
    # Checkpoint state such as engine snapshots and current steps must never be
    # inherited implicitly. Only explicit decisions have cross-checkpoint
    # continuity; current metadata remains authoritative for everything else.
    merged = dict(current or {})
    previous_scope = str(
        (previous or {}).get("_decision_scope")
        or (previous or {}).get("task", "")
    ).strip()
    current_task = str((current or {}).get("task", "")).strip()
    previous_decisions = normalize_decisions(
        (previous or {}).get("decisions"), limit=max_decisions
    )
    if not _same_task_scope(previous or {}, current or {}):
        previous_decisions = []
    decisions = normalize_decisions(
        [
            *previous_decisions,
            *normalize_decisions((current or {}).get("decisions"), limit=max_decisions),
        ],
        limit=max_decisions,
    )
    if decisions:
        merged["decisions"] = decisions
    if current_task:
        merged["_decision_scope"] = current_task
    elif previous_scope and previous_decisions:
        merged["_decision_scope"] = previous_scope
    merged.pop("reset_decisions", None)
    return merged


def _same_task_scope(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    if bool(current.get("reset_decisions")):
        return False
    previous_task = str(
        previous.get("_decision_scope") or previous.get("task", "")
    ).strip()
    current_task = str(current.get("task", "")).strip()
    if not previous_task or not current_task:
        return True
    previous_terms = _terms(previous_task)
    current_terms = _terms(current_task)
    overlap = len(previous_terms & current_terms)
    return overlap / max(1, min(len(previous_terms), len(current_terms))) >= 0.5


def select_relevant_checkpoint(
    checkpoints: Iterable[Any],
    query: str,
    *,
    project: str = "",
    now: float | None = None,
    policy: CheckpointRelevancePolicy | None = None,
) -> CheckpointMatch | None:
    """Select the best checkpoint using bounded, explainable features."""
    active_policy = policy or CheckpointRelevancePolicy()
    timestamp = time.time() if now is None else float(now)
    query_terms = _terms(query)
    if not query_terms:
        return None

    candidates: list[CheckpointMatch] = []
    for checkpoint in checkpoints:
        age = max(0.0, timestamp - float(getattr(checkpoint, "timestamp", 0.0)))
        if age > active_policy.max_age_seconds:
            continue
        metadata = getattr(checkpoint, "metadata", {}) or {}
        recorded_project = str(metadata.get("project", "")).strip()
        if (
            project
            and recorded_project
            and PurePath(project).name.casefold()
            != PurePath(recorded_project).name.casefold()
        ):
            continue
        fragments = getattr(checkpoint, "fragments", []) or []
        metadata_text = _metadata_text(metadata)
        metadata_terms = _terms(metadata_text)
        source_terms = _terms(" ".join(str(fragment.get("source", "")) for fragment in fragments))
        lexical_matches = query_terms & metadata_terms
        source_matches = query_terms & source_terms
        lexical = len(lexical_matches) / max(1, len(query_terms))
        source = len(source_matches) / max(1, len(query_terms))
        project_score = _project_match(project, metadata, fragments)
        recency = math.exp(-math.log(2.0) * age / active_policy.recency_half_life_seconds)
        score = 0.55 * lexical + 0.25 * source + 0.10 * project_score + 0.10 * recency
        candidates.append(
            CheckpointMatch(
                checkpoint=checkpoint,
                score=round(score, 6),
                lexical_score=round(lexical, 6),
                source_score=round(source, 6),
                project_score=round(project_score, 6),
                recency_score=round(recency, 6),
                matched_terms=tuple(sorted(lexical_matches | source_matches)),
            )
        )
    if not candidates:
        return None
    best = max(candidates, key=lambda match: (match.score, match.checkpoint.timestamp))
    return best if best.score >= active_policy.minimum_score else None


def render_recovery_context(match: CheckpointMatch) -> str:
    """Render only continuity metadata, fenced as untrusted recovered data."""
    metadata = getattr(match.checkpoint, "metadata", {}) or {}
    lines = [
        f"checkpoint_id: {getattr(match.checkpoint, 'checkpoint_id', '')}",
        f"relevance_score: {match.score:.3f}",
    ]
    for key in ("task", "step", "remaining_work"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {bounded_text(' '.join(value.split()), 1000)}")
    decisions = normalize_decisions(metadata.get("decisions"))
    if decisions:
        lines.append("decisions:")
        lines.extend(f"- {decision}" for decision in decisions)
    files = metadata.get("modified_files")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes, bytearray)):
        # A silently cut path is worse than an absent one: it looks openable and
        # is not. Mark it, and say how many files did not fit at all.
        safe_files = [bounded_text(str(path), 300) for path in files[:50]]
        if safe_files:
            lines.append("modified_files:")
            lines.extend(f"- {path}" for path in safe_files)
            hidden = len(files) - len(safe_files)
            if hidden > 0:
                lines.append(f"- [{hidden} further modified file(s) not listed]")
    rendered, _report = sanitize_injected_context("\n".join(lines), fence=True)
    return rendered


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(text or "")
        if len(token) >= 3 and token.casefold() not in _STOPWORDS
    }


def _metadata_text(metadata: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "task", "_decision_scope", "step", "remaining_work", "query", "project"
    ):
        value = metadata.get(key)
        if isinstance(value, str):
            values.append(value)
    values.extend(normalize_decisions(metadata.get("decisions")))
    files = metadata.get("modified_files")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes, bytearray)):
        values.extend(str(path) for path in files[:100])
    return " ".join(values)


def _project_match(project: str, metadata: Mapping[str, Any], fragments: Sequence[Any]) -> float:
    if not project:
        return 0.0
    wanted = PurePath(project).name.casefold()
    recorded = str(metadata.get("project", "")).casefold()
    if wanted and (wanted == PurePath(recorded).name.casefold() or wanted in recorded):
        return 1.0
    for fragment in fragments:
        if wanted and wanted in str(fragment.get("source", "")).casefold():
            return 1.0
    return 0.0


__all__ = [
    "CheckpointMatch",
    "CheckpointRelevancePolicy",
    "merge_checkpoint_metadata",
    "normalize_decisions",
    "render_recovery_context",
    "select_relevant_checkpoint",
]
