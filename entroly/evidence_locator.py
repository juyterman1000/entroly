"""Exact-source evidence location for Entroly.

The locator separates two responsibilities:

* a ranker may estimate which source passages are useful for a query;
* Entroly owns passage boundaries, exact source offsets, freshness checks,
  recovery, and the claim boundary around that estimate.

No ranker may return generated text.  It can only score a supplied passage and
choose one of that passage's supplied sentences.  Invalid or incomplete
ranker output causes abstention instead of a lexical or fabricated fallback.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

from .codec import RecoveryReference, RecoveryStore, estimate_tokens
from .neural_evidence_selector import SemanticEncoder


_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]*", re.UNICODE)
_SENTENCE_END = re.compile(r"(?:[.!?]+|[。！？]+)(?=\s|$)")
_STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be because been before being
    between both but by can could did do does doing during each few for from
    further had has have having how if in into is it its itself just more most
    no nor not of off on once only or other our out over own same should so some
    such than that the their them then there these they this those through to too
    under until up very was we were what when where which while who why will with
    would you your
    """.split()
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _terms(value: str) -> list[str]:
    return [
        match.group(0).lower()
        for match in _WORD.finditer(value)
        if match.group(0).lower() not in _STOPWORDS
    ]


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


@dataclass(frozen=True, slots=True)
class SourceSentence:
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True, slots=True)
class SourcePassage:
    passage_id: str
    source_id: str
    start_char: int
    end_char: int
    text: str
    sentences: tuple[SourceSentence, ...]
    ordinal: int


@dataclass(frozen=True, slots=True)
class PassageJudgment:
    """Ranker output constrained to caller-supplied passage structure."""

    passage_id: str
    relevance: float
    focus_sentence: int


class EvidenceRanker(Protocol):
    @property
    def ranker_id(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    def rank(
        self, query: str, passages: Sequence[SourcePassage]
    ) -> Sequence[PassageJudgment]: ...


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    passage_id: str
    source_id: str
    relevance: float
    passage_start_char: int
    passage_end_char: int
    focus_start_char: int
    focus_end_char: int
    passage_text: str
    focus_text: str
    passage_sha256: str
    focus_sha256: str

    def verify(self, source_text: str) -> bool:
        passage = source_text[self.passage_start_char : self.passage_end_char]
        focus = source_text[self.focus_start_char : self.focus_end_char]
        return (
            passage == self.passage_text
            and focus == self.focus_text
            and _sha256(passage) == self.passage_sha256
            and _sha256(focus) == self.focus_sha256
        )


LocationStatus = Literal["selected", "no_match", "abstained"]


@dataclass(frozen=True, slots=True)
class EvidenceLocationResult:
    status: LocationStatus
    reason: str
    source_id: str
    query_sha256: str
    source_sha256: str
    ranker_id: str
    ranker_fingerprint: str
    threshold: float | None
    calibration_id: str | None
    source_coverage_complete: bool
    candidate_count: int
    matches: tuple[EvidenceMatch, ...]
    recovery: RecoveryReference | None = None

    @property
    def calibrated(self) -> bool:
        return bool(self.calibration_id and self.threshold is not None)

    @property
    def exact_recovery(self) -> bool:
        return self.recovery is not None

    def render_context(self, *, relevance_order: bool = False) -> str:
        rows = self.matches
        if not relevance_order:
            rows = tuple(sorted(rows, key=lambda row: row.passage_start_char))
        return "\n\n".join(row.passage_text for row in rows)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "entroly.evidence-location.v1",
            "status": self.status,
            "reason": self.reason,
            "source_id": self.source_id,
            "query_sha256": self.query_sha256,
            "source_sha256": self.source_sha256,
            "source_coverage_complete": self.source_coverage_complete,
            "candidate_count": self.candidate_count,
            "selected_count": len(self.matches),
            "ranker": {
                "id": self.ranker_id,
                "fingerprint_sha256": self.ranker_fingerprint,
            },
            "selection": {
                "threshold": self.threshold,
                "calibration_id": self.calibration_id,
                "calibrated": self.calibrated,
            },
            "matches": [
                {
                    "passage_id": row.passage_id,
                    "source_id": row.source_id,
                    "relevance": row.relevance,
                    "passage_start_char": row.passage_start_char,
                    "passage_end_char": row.passage_end_char,
                    "focus_start_char": row.focus_start_char,
                    "focus_end_char": row.focus_end_char,
                    "passage_sha256": row.passage_sha256,
                    "focus_sha256": row.focus_sha256,
                }
                for row in self.matches
            ],
            "exact_recovery": self.exact_recovery,
            "recovery_digest": self.recovery.digest if self.recovery else None,
            "claim_boundary": (
                "The ranker selected existing source spans. Scores are ranking signals, "
                "not proof of relevance, completeness, factual correctness, or task success."
            ),
        }

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        result = self.receipt()
        if include_text:
            for payload, match in zip(result["matches"], self.matches):
                payload["focus_text"] = match.focus_text
                payload["passage_text"] = match.passage_text
        return result


def sentence_spans(text: str, *, offset: int = 0) -> tuple[SourceSentence, ...]:
    """Return exact sentence spans without normalizing the source text."""
    spans: list[SourceSentence] = []
    cursor = 0
    for boundary in _SENTENCE_END.finditer(text):
        trimmed = _trimmed_span(text, cursor, boundary.end())
        if trimmed:
            start, end = trimmed
            spans.append(SourceSentence(offset + start, offset + end, text[start:end]))
        cursor = boundary.end()
    trimmed = _trimmed_span(text, cursor, len(text))
    if trimmed:
        start, end = trimmed
        spans.append(SourceSentence(offset + start, offset + end, text[start:end]))
    if not spans and text:
        trimmed = _trimmed_span(text, 0, len(text))
        if trimmed:
            start, end = trimmed
            spans.append(SourceSentence(offset + start, offset + end, text[start:end]))
    return tuple(spans)


def _base_spans(text: str, mode: str) -> list[tuple[int, int]]:
    if mode not in {"auto", "paragraph", "line"}:
        raise ValueError("passage_mode must be auto, paragraph, or line")
    if mode == "line":
        return [match.span() for match in re.finditer(r"[^\r\n]+", text)]

    paragraphs: list[tuple[int, int]] = []
    cursor = 0
    for separator in re.finditer(r"(?:\r?\n){2,}", text):
        trimmed = _trimmed_span(text, cursor, separator.start())
        if trimmed:
            paragraphs.append(trimmed)
        cursor = separator.end()
    tail = _trimmed_span(text, cursor, len(text))
    if tail:
        paragraphs.append(tail)
    if mode == "paragraph":
        return paragraphs
    # Accessibility snapshots and logs usually contain many meaningful lines
    # with no blank separators.  Preserve those source boundaries instead of
    # treating the entire snapshot as one oversized paragraph.
    if len(paragraphs) <= 1 and text.count("\n") >= 3:
        return [match.span() for match in re.finditer(r"[^\r\n]+", text)]
    return paragraphs


def _split_oversized(
    source_text: str, start: int, end: int, max_passage_chars: int
) -> list[tuple[int, int]]:
    trimmed = _trimmed_span(source_text, start, end)
    if not trimmed:
        return []
    start, end = trimmed
    if end - start <= max_passage_chars:
        return [(start, end)]

    sentences = sentence_spans(source_text[start:end], offset=start)
    chunks: list[tuple[int, int]] = []
    chunk_start: int | None = None
    chunk_end: int | None = None
    for sentence in sentences:
        if sentence.end_char - sentence.start_char > max_passage_chars:
            if chunk_start is not None and chunk_end is not None:
                chunks.append((chunk_start, chunk_end))
                chunk_start = chunk_end = None
            cursor = sentence.start_char
            while cursor < sentence.end_char:
                proposed = min(sentence.end_char, cursor + max_passage_chars)
                if proposed < sentence.end_char:
                    whitespace = source_text.rfind(" ", cursor, proposed)
                    if whitespace > cursor:
                        proposed = whitespace
                piece = _trimmed_span(source_text, cursor, proposed)
                if piece:
                    chunks.append(piece)
                cursor = max(proposed, cursor + 1)
            continue
        if chunk_start is None:
            chunk_start, chunk_end = sentence.start_char, sentence.end_char
        elif sentence.end_char - chunk_start <= max_passage_chars:
            chunk_end = sentence.end_char
        else:
            chunks.append((chunk_start, chunk_end or chunk_start))
            chunk_start, chunk_end = sentence.start_char, sentence.end_char
    if chunk_start is not None and chunk_end is not None:
        chunks.append((chunk_start, chunk_end))
    return chunks


def segment_source(
    source_text: str,
    *,
    source_id: str = "source",
    passage_mode: str = "auto",
    max_passage_chars: int = 2_200,
    max_passages: int = 160,
    max_source_chars: int = 60_000,
) -> tuple[tuple[SourcePassage, ...], bool]:
    """Segment source text while retaining exact original character offsets."""
    if not isinstance(source_text, str):
        raise TypeError("source_text must be a string")
    if not source_id.strip():
        raise ValueError("source_id must be nonempty")
    if max_passage_chars < 64 or max_passages < 1 or max_source_chars < 1:
        raise ValueError("passage and source limits must be positive")

    coverage_end = min(len(source_text), max_source_chars)
    base = _base_spans(source_text[:coverage_end], passage_mode)
    bounded: list[tuple[int, int]] = []
    for start, end in base:
        bounded.extend(_split_oversized(source_text, start, end, max_passage_chars))
        if len(bounded) >= max_passages:
            bounded = bounded[:max_passages]
            break
    passages = tuple(
        SourcePassage(
            passage_id=f"p{ordinal}",
            source_id=source_id,
            start_char=start,
            end_char=end,
            text=source_text[start:end],
            sentences=sentence_spans(source_text[start:end], offset=start),
            ordinal=ordinal,
        )
        for ordinal, (start, end) in enumerate(bounded)
    )
    covered_end = max((passage.end_char for passage in passages), default=0)
    complete = coverage_end == len(source_text) and covered_end >= len(source_text.rstrip())
    return passages, complete


def _bm25_scores(query: str, texts: Sequence[str]) -> list[float]:
    query_terms = frozenset(_terms(query))
    if not query_terms:
        return [0.0 for _ in texts]
    frequencies = [Counter(_terms(text)) for text in texts]
    document_frequency = Counter(term for row in frequencies for term in row)
    average_length = max(
        1.0, sum(sum(row.values()) for row in frequencies) / max(1, len(frequencies))
    )
    raw: list[float] = []
    for row in frequencies:
        document_length = sum(row.values())
        score = 0.0
        for term in query_terms:
            frequency = row.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1.0 + (len(texts) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (
                1.0 - 0.75 + 0.75 * document_length / average_length
            )
            score += inverse_frequency * (frequency * 2.2) / denominator
        raw.append(score)
    maximum = max(raw, default=0.0)
    return [score / maximum if maximum > 0 else 0.0 for score in raw]


class LexicalEvidenceRanker:
    """Deterministic ranker available in every Entroly installation."""

    ranker_id = "entroly.lexical-evidence.v1"
    fingerprint = _sha256("bm25:k1=1.2:b=0.75:exact-sentence-focus:v1")

    def rank(
        self, query: str, passages: Sequence[SourcePassage]
    ) -> tuple[PassageJudgment, ...]:
        passage_scores = _bm25_scores(query, [passage.text for passage in passages])
        judgments: list[PassageJudgment] = []
        for passage, relevance in zip(passages, passage_scores):
            sentence_scores = _bm25_scores(
                query, [sentence.text for sentence in passage.sentences]
            )
            focus = max(
                range(len(sentence_scores)),
                key=lambda index: (sentence_scores[index], -index),
                default=0,
            )
            judgments.append(PassageJudgment(passage.passage_id, relevance, focus))
        return tuple(judgments)


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("semantic encoder returned a zero or non-finite vector")
    result = tuple(float(value) / norm for value in vector)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("semantic encoder returned non-finite values")
    return result


class EncoderEvidenceRanker:
    """Adapter for Entroly's local, no-download SemanticEncoder contract."""

    def __init__(self, encoder: SemanticEncoder) -> None:
        self.encoder = encoder

    @property
    def ranker_id(self) -> str:
        return f"entroly.local-semantic:{self.encoder.model_id}"

    @property
    def fingerprint(self) -> str:
        return self.encoder.fingerprint

    def rank(
        self, query: str, passages: Sequence[SourcePassage]
    ) -> tuple[PassageJudgment, ...]:
        sentences = [sentence for passage in passages for sentence in passage.sentences]
        encoded = list(
            self.encoder.encode(
                [query, *[passage.text for passage in passages], *[row.text for row in sentences]]
            )
        )
        expected = 1 + len(passages) + len(sentences)
        if len(encoded) != expected:
            raise ValueError("semantic encoder returned the wrong number of vectors")
        vectors = [_normalize(row) for row in encoded]
        query_vector = vectors[0]
        passage_vectors = vectors[1 : 1 + len(passages)]
        sentence_vectors = vectors[1 + len(passages) :]
        if any(len(row) != len(query_vector) for row in vectors):
            raise ValueError("semantic encoder returned inconsistent dimensions")

        judgments: list[PassageJudgment] = []
        sentence_cursor = 0
        for passage, passage_vector in zip(passages, passage_vectors):
            sentence_count = len(passage.sentences)
            local_vectors = sentence_vectors[
                sentence_cursor : sentence_cursor + sentence_count
            ]
            sentence_cursor += sentence_count
            sentence_scores = [
                max(0.0, min(1.0, sum(a * b for a, b in zip(query_vector, row))))
                for row in local_vectors
            ]
            focus = max(
                range(len(sentence_scores)),
                key=lambda index: (sentence_scores[index], -index),
                default=0,
            )
            relevance = max(
                0.0,
                min(1.0, sum(a * b for a, b in zip(query_vector, passage_vector))),
            )
            judgments.append(PassageJudgment(passage.passage_id, relevance, focus))
        return tuple(judgments)


def _validate_judgments(
    passages: Sequence[SourcePassage], judgments: Sequence[PassageJudgment]
) -> dict[str, PassageJudgment]:
    expected = {passage.passage_id: passage for passage in passages}
    received: dict[str, PassageJudgment] = {}
    for judgment in judgments:
        if judgment.passage_id not in expected or judgment.passage_id in received:
            raise ValueError("ranker returned an unknown or duplicate passage id")
        if not math.isfinite(judgment.relevance) or not 0.0 <= judgment.relevance <= 1.0:
            raise ValueError("ranker relevance must be finite and between zero and one")
        passage = expected[judgment.passage_id]
        if not 0 <= judgment.focus_sentence < len(passage.sentences):
            raise ValueError("ranker focus must select a supplied sentence")
        received[judgment.passage_id] = judgment
    if received.keys() != expected.keys():
        raise ValueError("ranker must return exactly one judgment per passage")
    return received


def _empty_result(
    *,
    status: LocationStatus,
    reason: str,
    source_id: str,
    query: str,
    source_text: str,
    ranker: EvidenceRanker,
    threshold: float | None,
    calibration_id: str | None,
    complete: bool,
    candidate_count: int,
) -> EvidenceLocationResult:
    ranker_id, ranker_fingerprint = _ranker_metadata(ranker)
    return EvidenceLocationResult(
        status=status,
        reason=reason,
        source_id=source_id,
        query_sha256=_sha256(query),
        source_sha256=_sha256(source_text),
        ranker_id=ranker_id,
        ranker_fingerprint=ranker_fingerprint,
        threshold=threshold,
        calibration_id=calibration_id,
        source_coverage_complete=complete,
        candidate_count=candidate_count,
        matches=(),
    )


def _ranker_metadata(ranker: EvidenceRanker) -> tuple[str, str]:
    """Read untrusted adapter metadata without weakening the abstention path."""
    try:
        ranker_id = str(ranker.ranker_id)
    except Exception:
        ranker_id = "unavailable"
    try:
        ranker_fingerprint = str(ranker.fingerprint)
    except Exception:
        ranker_fingerprint = _sha256("unavailable-ranker-fingerprint")
    return ranker_id, ranker_fingerprint


def locate_evidence(
    source_text: str,
    query: str,
    *,
    source_id: str = "source",
    ranker: EvidenceRanker | None = None,
    min_score: float | None = None,
    calibration_id: str | None = None,
    max_matches: int = 5,
    budget_tokens: int = 2_000,
    passage_mode: str = "auto",
    max_passage_chars: int = 2_200,
    max_passages: int = 160,
    max_source_chars: int = 60_000,
    recovery_store: RecoveryStore | None = None,
) -> EvidenceLocationResult:
    """Locate exact source passages while keeping ranker claims bounded."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty")
    if min_score is not None and (
        not math.isfinite(min_score) or not 0.0 <= min_score <= 1.0
    ):
        raise ValueError("min_score must be between zero and one")
    if max_matches < 1 or budget_tokens < 1:
        raise ValueError("max_matches and budget_tokens must be positive")
    ranker = ranker or LexicalEvidenceRanker()
    passages, complete = segment_source(
        source_text,
        source_id=source_id,
        passage_mode=passage_mode,
        max_passage_chars=max_passage_chars,
        max_passages=max_passages,
        max_source_chars=max_source_chars,
    )
    if not passages:
        return _empty_result(
            status="no_match",
            reason="no_candidate_passages",
            source_id=source_id,
            query=query,
            source_text=source_text,
            ranker=ranker,
            threshold=min_score,
            calibration_id=calibration_id,
            complete=complete,
            candidate_count=0,
        )
    try:
        judgments = _validate_judgments(passages, ranker.rank(query, passages))
    except Exception as error:
        return _empty_result(
            status="abstained",
            reason=f"ranker_failed:{type(error).__name__}",
            source_id=source_id,
            query=query,
            source_text=source_text,
            ranker=ranker,
            threshold=min_score,
            calibration_id=calibration_id,
            complete=complete,
            candidate_count=len(passages),
        )
    try:
        ranker_id = str(ranker.ranker_id)
        ranker_fingerprint = str(ranker.fingerprint)
        if not ranker_id or not ranker_fingerprint:
            raise ValueError("ranker metadata must be nonempty")
    except Exception as error:
        return _empty_result(
            status="abstained",
            reason=f"ranker_metadata_failed:{type(error).__name__}",
            source_id=source_id,
            query=query,
            source_text=source_text,
            ranker=ranker,
            threshold=min_score,
            calibration_id=calibration_id,
            complete=complete,
            candidate_count=len(passages),
        )

    floor = min_score if min_score is not None else 0.0
    ranked = sorted(
        passages,
        key=lambda passage: (-judgments[passage.passage_id].relevance, passage.ordinal),
    )
    selected: list[EvidenceMatch] = []
    used_tokens = 0
    eligible = 0
    for passage in ranked:
        judgment = judgments[passage.passage_id]
        if judgment.relevance <= 0.0 or judgment.relevance < floor:
            continue
        eligible += 1
        cost = estimate_tokens(passage.text)
        if used_tokens + cost > budget_tokens:
            continue
        focus = passage.sentences[judgment.focus_sentence]
        selected.append(
            EvidenceMatch(
                passage_id=passage.passage_id,
                source_id=passage.source_id,
                relevance=round(judgment.relevance, 8),
                passage_start_char=passage.start_char,
                passage_end_char=passage.end_char,
                focus_start_char=focus.start_char,
                focus_end_char=focus.end_char,
                passage_text=passage.text,
                focus_text=focus.text,
                passage_sha256=_sha256(passage.text),
                focus_sha256=_sha256(focus.text),
            )
        )
        used_tokens += cost
        if len(selected) >= max_matches:
            break

    if not selected:
        reason = "budget_too_small" if eligible else "no_score_cleared_threshold"
        return _empty_result(
            status="abstained" if eligible else "no_match",
            reason=reason,
            source_id=source_id,
            query=query,
            source_text=source_text,
            ranker=ranker,
            threshold=min_score,
            calibration_id=calibration_id,
            complete=complete,
            candidate_count=len(passages),
        )

    recovery: RecoveryReference | None = None
    if recovery_store is not None and (
        not complete or len(selected) < len(passages)
    ):
        try:
            recovery = recovery_store.put(
                source_text,
                item_count=len(passages) - len(selected),
                item_label="source passage(s) restored",
                note=f"complete source for evidence location: {source_id}",
            )
            if recovery_store.recover(recovery) != source_text:
                raise ValueError("recovered source differs")
        except Exception as error:
            return _empty_result(
                status="abstained",
                reason=f"recovery_failed:{type(error).__name__}",
                source_id=source_id,
                query=query,
                source_text=source_text,
                ranker=ranker,
                threshold=min_score,
                calibration_id=calibration_id,
                complete=complete,
                candidate_count=len(passages),
            )

    return EvidenceLocationResult(
        status="selected",
        reason="calibrated_threshold" if calibration_id and min_score is not None else "ranked_candidates",
        source_id=source_id,
        query_sha256=_sha256(query),
        source_sha256=_sha256(source_text),
        ranker_id=ranker_id,
        ranker_fingerprint=ranker_fingerprint,
        threshold=min_score,
        calibration_id=calibration_id,
        source_coverage_complete=complete,
        candidate_count=len(passages),
        matches=tuple(selected),
        recovery=recovery,
    )


def verify_evidence_location(
    result: EvidenceLocationResult, current_source: str
) -> dict[str, Any]:
    """Verify freshness and exact spans against the current source bytes."""
    source_fresh = _sha256(current_source) == result.source_sha256
    match_results = [match.verify(current_source) for match in result.matches]
    return {
        "schema_version": "entroly.evidence-location-verification.v1",
        "source_fresh": source_fresh,
        "matches_exact": all(match_results),
        "match_results": match_results,
        "valid": source_fresh and all(match_results),
    }


__all__ = [
    "EncoderEvidenceRanker",
    "EvidenceLocationResult",
    "EvidenceMatch",
    "EvidenceRanker",
    "LexicalEvidenceRanker",
    "PassageJudgment",
    "SourcePassage",
    "SourceSentence",
    "locate_evidence",
    "segment_source",
    "sentence_spans",
    "verify_evidence_location",
]
