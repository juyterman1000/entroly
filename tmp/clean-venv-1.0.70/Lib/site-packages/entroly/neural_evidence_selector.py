"""Evidence-constrained local transformer selection.

This module is intentionally opt-in. It combines a local semantic encoder with
hard evidence locks and a monotone D-optimal objective, while retaining a
deterministic lexical fallback. Model loading is local-files-only; a Hugging
Face model id is rejected rather than downloaded.

For candidate spans ``i`` with normalized embeddings ``e_i``, current-query
utility ``u_i``, evidence value ``a_i``, future utility ``h_i``, and semantic
reliability ``s_i``, the selector approximately maximizes

    F(S) = sum(u_i + gamma*a_i + rho*h_i for i in S)
           + lambda * log det(I + sum(w_i * e_i e_i^T for i in S))

where ``w_i = s_i * max(0, u_i - tau)^p``. Separating the modular utility
from the determinant weight avoids double-counting relevance, while ``tau``
prevents irrelevant but orthogonal spans from winning on novelty alone.

under a token knapsack and mandatory-evidence constraints. The modular term
rewards relevance. The log-determinant term is monotone submodular and rewards
semantic directions not already represented by the selected set.

This is a selection primitive, not a quality claim. A neural override on a
lexical disagreement is disabled unless a caller supplies a threshold tied to
an external calibration artifact. Without one, the selector keeps both
champions when the budget permits or abstains to the lexical baseline.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence

if TYPE_CHECKING:
    from .compression_retrieval_store import CompressionRetrievalStore, StoredCompression

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]{2,}|\d+(?:\.\d+)*")
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


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(value) if token.lower() not in _STOPWORDS]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("semantic encoder returned a zero or non-finite vector")
    normalized = tuple(float(value) / norm for value in vector)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("semantic encoder returned non-finite values")
    return normalized


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """A source-addressable unit eligible for selection."""

    span_id: str
    source: str
    text: str
    token_count: int
    ordinal: int
    start_char: int = 0
    end_char: int | None = None
    mandatory: bool = False
    evidence_value: float = 0.0
    future_utility: float = 0.0
    semantic_confidence: float = 1.0
    lock_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.span_id:
            raise ValueError("span_id must not be empty")
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char is not None and self.end_char < self.start_char:
            raise ValueError("end_char must not precede start_char")
        for name, value in (
            ("evidence_value", self.evidence_value),
            ("future_utility", self.future_utility),
            ("semantic_confidence", self.semantic_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class NeuralSelectionPolicy:
    """Safety and optimization policy for neural evidence selection."""

    diversity_weight: float = 0.20
    evidence_weight: float = 0.50
    future_utility_weight: float = 0.25
    relevance_floor: float = 0.15
    diversity_power: float = 2.0
    min_neural_margin: float = 0.03
    calibrated_override_margin: float | None = None
    calibration_id: str | None = None
    guard_disagreements: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("diversity_weight", self.diversity_weight),
            ("evidence_weight", self.evidence_weight),
            ("future_utility_weight", self.future_utility_weight),
            ("relevance_floor", self.relevance_floor),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.relevance_floor > 1:
            raise ValueError("relevance_floor must not exceed 1")
        if self.diversity_power < 1:
            raise ValueError("diversity_power must be at least 1")
        if self.min_neural_margin < 0:
            raise ValueError("min_neural_margin must be non-negative")
        if self.calibrated_override_margin is not None:
            if self.calibrated_override_margin < self.min_neural_margin:
                raise ValueError(
                    "calibrated_override_margin must be at least min_neural_margin"
                )
            if not self.calibration_id:
                raise ValueError("a neural override threshold requires calibration_id")


class SemanticEncoder(Protocol):
    """Minimal local semantic-encoder contract."""

    @property
    def model_id(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class LocalTransformerEncoder:
    """Lazy SentenceTransformer adapter that refuses remote model identifiers."""

    def __init__(self, model_path: str | Path, *, device: str = "cpu") -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(
                "model_path must be an existing local directory; remote model ids are disabled"
            )
        self._path = path
        self._device = device
        self._model: Any | None = None
        self._fingerprint: str | None = None

    @property
    def model_id(self) -> str:
        return self._path.name

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            digest = hashlib.sha256()
            files = sorted(path for path in self._path.rglob("*") if path.is_file())
            if not files:
                raise ValueError("local model directory contains no files")
            for path in files:
                relative = path.relative_to(self._path).as_posix()
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                digest.update(b"\0")
            self._fingerprint = digest.hexdigest()
        return self._fingerprint

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "local transformer selection requires the optional neural dependencies"
                ) from error
            self._model = SentenceTransformer(
                str(self._path),
                device=self._device,
                local_files_only=True,
            )
        encoded = self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in encoded]


@dataclass(frozen=True, slots=True)
class NeuralSelectionResult:
    selected: tuple[EvidenceSpan, ...]
    receipt: dict[str, Any]

    @property
    def text(self) -> str:
        return "\n\n".join(span.text for span in self.selected)

    @property
    def selected_tokens(self) -> int:
        return sum(span.token_count for span in self.selected)


def _lexical_scores(query: str, spans: Sequence[EvidenceSpan]) -> list[float]:
    query_terms = frozenset(_tokens(query))
    if not query_terms:
        return [0.0 for _ in spans]
    term_frequencies = [Counter(_tokens(f"{span.source} {span.text}")) for span in spans]
    document_frequency = Counter(
        term for frequencies in term_frequencies for term in frequencies
    )
    average_length = max(
        1.0,
        sum(sum(frequencies.values()) for frequencies in term_frequencies)
        / max(1, len(term_frequencies)),
    )
    raw_scores: list[float] = []
    for frequencies in term_frequencies:
        document_length = sum(frequencies.values())
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (len(spans) - df + 0.5) / (df + 0.5)
            )
            norm = 0.25 + 0.75 * document_length / average_length
            score += inverse_frequency * (frequency * 2.5) / (frequency + 1.5 * norm)
        raw_scores.append(score)
    maximum = max(raw_scores, default=0.0)
    return [score / maximum if maximum > 0 else 0.0 for score in raw_scores]


def score_lexical_evidence(
    query: str, spans: Sequence[EvidenceSpan]
) -> tuple[float, ...]:
    """Return deterministic normalized BM25 scores for audit and calibration."""
    return tuple(_lexical_scores(query, spans))


def _rank(scores: Sequence[float], spans: Sequence[EvidenceSpan]) -> list[int]:
    return sorted(
        range(len(spans)),
        key=lambda index: (
            -float(scores[index]),
            spans[index].token_count,
            spans[index].ordinal,
            spans[index].span_id,
        ),
    )


def _mandatory_indices(
    spans: Sequence[EvidenceSpan], required_evidence: Sequence[str]
) -> tuple[set[int], list[str]]:
    mandatory = {index for index, span in enumerate(spans) if span.mandatory}
    missing: list[str] = []
    for needle in required_evidence:
        matches = [index for index, span in enumerate(spans) if needle in span.text]
        if not matches:
            missing.append(needle)
            continue
        mandatory.update(matches)
    return mandatory, missing


def _ordered(indices: Sequence[int], spans: Sequence[EvidenceSpan]) -> tuple[EvidenceSpan, ...]:
    unique = set(indices)
    return tuple(
        spans[index]
        for index in sorted(
            unique,
            key=lambda value: (
                spans[value].ordinal,
                spans[value].source,
                spans[value].start_char,
                spans[value].span_id,
            ),
        )
    )


def _baseline_indices(
    spans: Sequence[EvidenceSpan],
    scores: Sequence[float],
    budget_tokens: int,
    mandatory: set[int],
) -> list[int]:
    selected = sorted(mandatory)
    used = sum(spans[index].token_count for index in selected)
    for index in _rank(scores, spans):
        if index in mandatory:
            continue
        cost = spans[index].token_count
        if used + cost <= budget_tokens:
            selected.append(index)
            used += cost
    return selected


def _logdet_identity_gram(
    indices: Sequence[int],
    embeddings: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> float:
    """Compute log det(I + E E^T) through the small selected-set Gram matrix."""
    if not indices:
        return 0.0
    size = len(indices)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for row, left_index in enumerate(indices):
        left_weight = math.sqrt(max(0.0, weights[left_index]))
        for column, right_index in enumerate(indices):
            right_weight = math.sqrt(max(0.0, weights[right_index]))
            value = left_weight * right_weight * _dot(
                embeddings[left_index], embeddings[right_index]
            )
            matrix[row][column] = value + (1.0 if row == column else 0.0)

    # Cholesky is stable here because I + E E^T is positive definite.
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            residual = matrix[row][column] - sum(
                lower[row][inner] * lower[column][inner] for inner in range(column)
            )
            if row == column:
                lower[row][column] = math.sqrt(max(residual, 1e-12))
            else:
                lower[row][column] = residual / lower[column][column]
    return 2.0 * sum(math.log(lower[index][index]) for index in range(size))


def _objective(
    indices: Sequence[int],
    utilities: Sequence[float],
    embeddings: Sequence[Sequence[float]],
    diversity_weights: Sequence[float],
    diversity_weight: float,
) -> float:
    modular = sum(max(0.0, utilities[index]) for index in indices)
    diversity = _logdet_identity_gram(indices, embeddings, diversity_weights)
    return modular + diversity_weight * diversity


def _greedy_schedule(
    spans: Sequence[EvidenceSpan],
    utilities: Sequence[float],
    embeddings: Sequence[Sequence[float]],
    diversity_weights: Sequence[float],
    budget_tokens: int,
    forced: set[int],
    diversity_weight: float,
    *,
    density: bool,
) -> list[int]:
    selected = sorted(forced)
    used = sum(spans[index].token_count for index in selected)
    candidates = [index for index in range(len(spans)) if index not in forced]
    current = _objective(
        selected, utilities, embeddings, diversity_weights, diversity_weight
    )
    while candidates:
        best: tuple[float, float, int] | None = None
        for index in candidates:
            cost = spans[index].token_count
            if used + cost > budget_tokens:
                continue
            objective = _objective(
                [*selected, index],
                utilities,
                embeddings,
                diversity_weights,
                diversity_weight,
            )
            gain = max(0.0, objective - current)
            priority = gain / cost if density else gain
            candidate = (priority, gain, -index)
            if best is None or candidate > best:
                best = candidate
        if best is None or best[1] <= 1e-12:
            break
        index = -best[2]
        selected.append(index)
        candidates.remove(index)
        used += spans[index].token_count
        current += best[1]
    return selected


def _neural_indices(
    spans: Sequence[EvidenceSpan],
    utilities: Sequence[float],
    embeddings: Sequence[Sequence[float]],
    diversity_weights: Sequence[float],
    budget_tokens: int,
    forced: set[int],
    diversity_weight: float,
) -> list[int]:
    schedules = [
        _greedy_schedule(
            spans,
            utilities,
            embeddings,
            diversity_weights,
            budget_tokens,
            forced,
            diversity_weight,
            density=True,
        ),
        _greedy_schedule(
            spans,
            utilities,
            embeddings,
            diversity_weights,
            budget_tokens,
            forced,
            diversity_weight,
            density=False,
        ),
    ]
    schedules.sort(
        key=lambda indices: (
            _objective(
                indices, utilities, embeddings, diversity_weights, diversity_weight
            ),
            -sum(spans[index].token_count for index in indices),
            tuple(-index for index in indices),
        ),
        reverse=True,
    )
    return schedules[0]


def _receipt(
    *,
    query: str,
    spans: Sequence[EvidenceSpan],
    selected_indices: Sequence[int],
    budget_tokens: int,
    mode: str,
    reason: str | None,
    lexical_scores: Sequence[float],
    neural_scores: Sequence[float] | None,
    selection_utilities: Sequence[float] | None,
    diversity_weights: Sequence[float] | None,
    encoder: SemanticEncoder | None,
    policy: NeuralSelectionPolicy,
    mandatory: set[int],
    missing_evidence: Sequence[str],
    neural_margin: float | None,
    lexical_champion: int | None,
    neural_champion: int | None,
) -> dict[str, Any]:
    selected = set(selected_indices)
    span_records = []
    for index, span in enumerate(spans):
        record = asdict(span)
        record.pop("text")
        record.update(
            {
                "content_sha256": _sha256(span.text),
                "selected": index in selected,
                "lexical_score": round(float(lexical_scores[index]), 8),
                "neural_score": (
                    round(float(neural_scores[index]), 8)
                    if neural_scores is not None
                    else None
                ),
                "selection_utility": (
                    round(float(selection_utilities[index]), 8)
                    if selection_utilities is not None
                    else None
                ),
                "diversity_weight": (
                    round(float(diversity_weights[index]), 8)
                    if diversity_weights is not None
                    else None
                ),
            }
        )
        span_records.append(record)
    return {
        "schema_version": "entroly.neural-evidence-selection.v1",
        "query_sha256": _sha256(query),
        "mode": mode,
        "abstained": mode != "neural",
        "reason": reason,
        "budget_tokens": budget_tokens,
        "input_tokens": sum(span.token_count for span in spans),
        "selected_tokens": sum(spans[index].token_count for index in selected),
        "budget_exceeded": sum(spans[index].token_count for index in selected) > budget_tokens,
        "selected_span_ids": [span.span_id for span in _ordered(selected_indices, spans)],
        "mandatory_span_ids": [spans[index].span_id for index in sorted(mandatory)],
        "missing_required_evidence": list(missing_evidence),
        "model": (
            {"id": encoder.model_id, "fingerprint_sha256": encoder.fingerprint}
            if encoder is not None and neural_scores is not None
            else None
        ),
        "policy": asdict(policy),
        "gate": {
            "neural_margin": round(neural_margin, 8) if neural_margin is not None else None,
            "lexical_champion": (
                spans[lexical_champion].span_id if lexical_champion is not None else None
            ),
            "neural_champion": (
                spans[neural_champion].span_id if neural_champion is not None else None
            ),
        },
        "spans": span_records,
    }


def select_neural_evidence(
    query: str,
    spans: Sequence[EvidenceSpan],
    *,
    budget_tokens: int,
    encoder: SemanticEncoder | None,
    policy: NeuralSelectionPolicy | None = None,
    required_evidence: Sequence[str] = (),
) -> NeuralSelectionResult:
    """Select evidence spans with explicit abstention and deterministic fallback."""
    if budget_tokens < 1:
        raise ValueError("budget_tokens must be positive")
    if not spans:
        return NeuralSelectionResult(
            selected=(),
            receipt={
                "schema_version": "entroly.neural-evidence-selection.v1",
                "mode": "empty",
                "abstained": True,
                "reason": "no_candidate_spans",
                "budget_tokens": budget_tokens,
                "input_tokens": 0,
                "selected_tokens": 0,
                "budget_exceeded": False,
                "selected_span_ids": [],
            },
        )

    policy = policy or NeuralSelectionPolicy()
    lexical_scores = _lexical_scores(query, spans)
    mandatory, missing_evidence = _mandatory_indices(spans, required_evidence)
    mandatory_tokens = sum(spans[index].token_count for index in mandatory)

    def fallback(reason: str, *, selected: Sequence[int] | None = None) -> NeuralSelectionResult:
        indices = list(selected) if selected is not None else _baseline_indices(
            spans, lexical_scores, budget_tokens, mandatory
        )
        receipt = _receipt(
            query=query,
            spans=spans,
            selected_indices=indices,
            budget_tokens=budget_tokens,
            mode="lexical_fallback",
            reason=reason,
            lexical_scores=lexical_scores,
            neural_scores=None,
            selection_utilities=None,
            diversity_weights=None,
            encoder=None,
            policy=policy,
            mandatory=mandatory,
            missing_evidence=missing_evidence,
            neural_margin=None,
            lexical_champion=_rank(lexical_scores, spans)[0],
            neural_champion=None,
        )
        return NeuralSelectionResult(_ordered(indices, spans), receipt)

    if missing_evidence:
        return fallback("required_evidence_not_found", selected=range(len(spans)))
    if mandatory_tokens > budget_tokens:
        return fallback("mandatory_evidence_exceeds_budget", selected=sorted(mandatory))
    if encoder is None:
        return fallback("neural_encoder_not_configured")
    if not _tokens(query):
        return fallback("query_has_no_semantic_terms")

    try:
        encoded = list(encoder.encode([query, *[span.text for span in spans]]))
        if len(encoded) != len(spans) + 1:
            raise ValueError("semantic encoder returned the wrong number of vectors")
        query_embedding = _normalize(encoded[0])
        embeddings = [_normalize(vector) for vector in encoded[1:]]
        if any(len(vector) != len(query_embedding) for vector in embeddings):
            raise ValueError("semantic encoder returned inconsistent dimensions")
    except Exception as error:
        return fallback(f"neural_encoder_failed:{type(error).__name__}")

    neural_scores = [
        max(0.0, min(1.0, _dot(query_embedding, vector))) for vector in embeddings
    ]
    selection_utilities = [
        neural_score
        + policy.evidence_weight * span.evidence_value
        + policy.future_utility_weight * span.future_utility
        for neural_score, span in zip(neural_scores, spans)
    ]
    diversity_weights = [
        span.semantic_confidence
        * max(0.0, neural_score - policy.relevance_floor) ** policy.diversity_power
        for neural_score, span in zip(neural_scores, spans)
    ]
    neural_order = _rank(neural_scores, spans)
    lexical_order = _rank(lexical_scores, spans)
    neural_champion = neural_order[0]
    lexical_champion = lexical_order[0]
    second_score = neural_scores[neural_order[1]] if len(neural_order) > 1 else 0.0
    neural_margin = neural_scores[neural_champion] - second_score
    forced = set(mandatory)

    if neural_champion == lexical_champion:
        forced.add(neural_champion)
        gate_reason = "lexical_neural_agreement"
    else:
        override_threshold = policy.calibrated_override_margin
        if override_threshold is not None and neural_margin >= override_threshold:
            forced.add(neural_champion)
            gate_reason = "calibrated_neural_override"
        elif policy.guard_disagreements:
            guarded = {neural_champion, lexical_champion}
            guarded_tokens = sum(
                spans[index].token_count for index in mandatory.union(guarded)
            )
            if neural_margin >= policy.min_neural_margin and guarded_tokens <= budget_tokens:
                forced.update(guarded)
                gate_reason = "disagreement_dual_channel_guard"
            else:
                return fallback("uncertain_neural_disagreement")
        else:
            return fallback("uncalibrated_neural_disagreement")

    selected_indices = _neural_indices(
        spans,
        selection_utilities,
        embeddings,
        diversity_weights,
        budget_tokens,
        forced,
        policy.diversity_weight,
    )
    selected_text = "\n\n".join(spans[index].text for index in selected_indices)
    missing_after_selection = [needle for needle in required_evidence if needle not in selected_text]
    if missing_after_selection:
        return fallback("neural_selection_failed_evidence_gate")

    receipt = _receipt(
        query=query,
        spans=spans,
        selected_indices=selected_indices,
        budget_tokens=budget_tokens,
        mode="neural",
        reason=gate_reason,
        lexical_scores=lexical_scores,
        neural_scores=neural_scores,
        selection_utilities=selection_utilities,
        diversity_weights=diversity_weights,
        encoder=encoder,
        policy=policy,
        mandatory=mandatory,
        missing_evidence=(),
        neural_margin=neural_margin,
        lexical_champion=lexical_champion,
        neural_champion=neural_champion,
    )
    return NeuralSelectionResult(_ordered(selected_indices, spans), receipt)


def persist_neural_selection(
    store: CompressionRetrievalStore,
    *,
    original_text: str,
    spans: Sequence[EvidenceSpan],
    result: NeuralSelectionResult,
    metadata: dict[str, object] | None = None,
) -> StoredCompression:
    """Persist every omitted source span with exact offset/hash validation."""
    selected_ids = {span.span_id for span in result.selected}
    omitted: list[dict[str, Any]] = []
    for span in spans:
        if span.span_id in selected_ids:
            continue
        if span.end_char is None:
            raise ValueError(f"span {span.span_id} requires end_char for exact recovery")
        content = original_text[span.start_char : span.end_char]
        if content != span.text:
            raise ValueError(f"source offsets do not match span {span.span_id}")
        omitted.append(
            {
                "span_id": span.span_id,
                "source": span.source,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "content_sha256": _sha256(span.text),
                "reason": "neural_budget",
            }
        )
    return store.put_exact_spans(
        original_text=original_text,
        compressed_text=result.text,
        receipt=result.receipt,
        spans=omitted,
        metadata=metadata,
    )


__all__ = [
    "EvidenceSpan",
    "LocalTransformerEncoder",
    "NeuralSelectionPolicy",
    "NeuralSelectionResult",
    "SemanticEncoder",
    "score_lexical_evidence",
    "select_neural_evidence",
    "persist_neural_selection",
]
