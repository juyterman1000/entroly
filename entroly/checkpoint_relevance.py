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


def normalize_decisions(value: Any, *, limit: int = 20) -> list[str]:
    """Normalize explicit decisions without inferring intent from arbitrary code."""
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = value
    else:
        return []
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        decision = " ".join(candidate.strip().split())[:500]
        key = decision.casefold()
        if not decision or key in seen:
            continue
        seen.add(key)
        output.append(decision)
        if len(output) >= limit:
            break
    return output


def merge_checkpoint_metadata(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    max_decisions: int = 20,
) -> dict[str, Any]:
    """Carry explicit decisions forward while allowing current metadata to win."""
    merged = dict(previous or {})
    merged.update(dict(current or {}))
    decisions = normalize_decisions(
        [
            *normalize_decisions((previous or {}).get("decisions"), limit=max_decisions),
            *normalize_decisions((current or {}).get("decisions"), limit=max_decisions),
        ],
        limit=max_decisions,
    )
    if decisions:
        merged["decisions"] = decisions
    return merged


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
            lines.append(f"{key}: {' '.join(value.split())[:1000]}")
    decisions = normalize_decisions(metadata.get("decisions"))
    if decisions:
        lines.append("decisions:")
        lines.extend(f"- {decision}" for decision in decisions)
    files = metadata.get("modified_files")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes, bytearray)):
        safe_files = [str(path)[:300] for path in files[:50]]
        if safe_files:
            lines.append("modified_files:")
            lines.extend(f"- {path}" for path in safe_files)
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
    for key in ("task", "step", "remaining_work", "query", "project"):
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
