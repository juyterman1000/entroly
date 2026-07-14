from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import pytest

from entroly.compression_retrieval_store import CompressionRetrievalStore
from entroly.neural_evidence_selector import (
    EvidenceSpan,
    LocalTransformerEncoder,
    NeuralSelectionPolicy,
    persist_neural_selection,
    select_neural_evidence,
)


@dataclass
class FakeEncoder:
    vectors: dict[str, Sequence[float]]
    model_id: str = "fixture-transformer"
    fingerprint: str = "a" * 64

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return [self.vectors[text] for text in texts]


class BrokenEncoder:
    model_id = "broken"
    fingerprint = "b" * 64

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        raise RuntimeError("model unavailable")


def _span(
    span_id: str,
    text: str,
    *,
    tokens: int = 3,
    ordinal: int = 0,
    mandatory: bool = False,
    evidence_value: float = 0.0,
    future_utility: float = 0.0,
) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=span_id,
        source=f"{span_id}.txt",
        text=text,
        token_count=tokens,
        ordinal=ordinal,
        end_char=len(text),
        mandatory=mandatory,
        evidence_value=evidence_value,
        future_utility=future_utility,
    )


def test_local_transformer_rejects_remote_model_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local directory"):
        LocalTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")

    empty = tmp_path / "empty-model"
    empty.mkdir()
    encoder = LocalTransformerEncoder(empty)
    with pytest.raises(ValueError, match="no files"):
        _ = encoder.fingerprint


def test_lexical_neural_agreement_uses_neural_selector() -> None:
    query = "auth refresh timeout"
    spans = [
        _span("answer", "auth refresh timeout at retry boundary"),
        _span("noise", "billing invoice completed", ordinal=1),
    ]
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [1.0, 0.0],
            spans[1].text: [0.0, 1.0],
        }
    )

    result = select_neural_evidence(query, spans, budget_tokens=3, encoder=encoder)

    assert [span.span_id for span in result.selected] == ["answer"]
    assert result.receipt["mode"] == "neural"
    assert result.receipt["reason"] == "lexical_neural_agreement"
    assert result.receipt["model"]["fingerprint_sha256"] == "a" * 64


def test_uncertain_disagreement_abstains_to_lexical_baseline() -> None:
    query = "auth timeout"
    spans = [
        _span("lexical", "auth timeout exact terms"),
        _span("semantic", "credential renewal stalled", ordinal=1),
    ]
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [0.79, 0.61],
            spans[1].text: [0.80, 0.60],
        }
    )

    result = select_neural_evidence(query, spans, budget_tokens=3, encoder=encoder)

    assert [span.span_id for span in result.selected] == ["lexical"]
    assert result.receipt["mode"] == "lexical_fallback"
    assert result.receipt["reason"] == "uncertain_neural_disagreement"
    assert result.receipt["abstained"] is True


def test_calibrated_margin_can_override_lexical_champion() -> None:
    query = "auth timeout"
    spans = [
        _span("lexical", "auth timeout exact terms"),
        _span("semantic", "credential renewal stalled", ordinal=1),
    ]
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [0.4, 0.9165],
            spans[1].text: [1.0, 0.0],
        }
    )
    policy = NeuralSelectionPolicy(
        calibrated_override_margin=0.10,
        calibration_id="squad-v2-calibration-fixture",
    )

    result = select_neural_evidence(
        query, spans, budget_tokens=3, encoder=encoder, policy=policy
    )

    assert [span.span_id for span in result.selected] == ["semantic"]
    assert result.receipt["reason"] == "calibrated_neural_override"
    assert result.receipt["policy"]["calibration_id"] == "squad-v2-calibration-fixture"


def test_uncalibrated_disagreement_keeps_both_channels_when_budget_allows() -> None:
    query = "auth timeout"
    spans = [
        _span("lexical", "auth timeout exact terms"),
        _span("semantic", "credential renewal stalled", ordinal=1),
        _span("noise", "invoice completed", ordinal=2),
    ]
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [0.4, 0.9165],
            spans[1].text: [1.0, 0.0],
            spans[2].text: [0.0, 1.0],
        }
    )

    result = select_neural_evidence(query, spans, budget_tokens=6, encoder=encoder)

    assert [span.span_id for span in result.selected] == ["lexical", "semantic"]
    assert result.receipt["reason"] == "disagreement_dual_channel_guard"


def test_mandatory_evidence_over_budget_is_visible_and_never_dropped() -> None:
    query = "incident"
    spans = [
        _span("locked", "incident INC-77", tokens=5, mandatory=True),
        _span("noise", "normal request", tokens=2, ordinal=1),
    ]

    result = select_neural_evidence(query, spans, budget_tokens=3, encoder=None)

    assert [span.span_id for span in result.selected] == ["locked"]
    assert result.receipt["reason"] == "mandatory_evidence_exceeds_budget"
    assert result.receipt["budget_exceeded"] is True


def test_missing_declared_evidence_abstains_to_full_passthrough() -> None:
    spans = [
        _span("one", "first source"),
        _span("two", "second source", ordinal=1),
    ]

    result = select_neural_evidence(
        "find INC-404",
        spans,
        budget_tokens=3,
        encoder=None,
        required_evidence=["INC-404"],
    )

    assert [span.span_id for span in result.selected] == ["one", "two"]
    assert result.receipt["reason"] == "required_evidence_not_found"
    assert result.receipt["missing_required_evidence"] == ["INC-404"]
    assert result.receipt["budget_exceeded"] is True


def test_encoder_failure_is_explicit_and_falls_back() -> None:
    query = "auth timeout"
    spans = [
        _span("answer", "auth timeout exact terms"),
        _span("noise", "billing complete", ordinal=1),
    ]

    result = select_neural_evidence(
        query, spans, budget_tokens=3, encoder=BrokenEncoder()
    )

    assert [span.span_id for span in result.selected] == ["answer"]
    assert result.receipt["reason"] == "neural_encoder_failed:RuntimeError"


def test_doptimal_term_prefers_a_nonredundant_semantic_direction() -> None:
    query = "topic alpha"
    spans = [
        _span("champion", "topic alpha primary", tokens=2),
        _span("duplicate", "topic alpha duplicate", tokens=2, ordinal=1),
        _span("complement", "topic alpha complementary", tokens=2, ordinal=2),
    ]
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [1.0, 0.0],
            spans[1].text: [0.995, 0.1],
            spans[2].text: [0.72, 0.694],
        }
    )
    policy = NeuralSelectionPolicy(
        diversity_weight=20.0, relevance_floor=0.0, diversity_power=1.0
    )

    result = select_neural_evidence(
        query, spans, budget_tokens=4, encoder=encoder, policy=policy
    )

    assert [span.span_id for span in result.selected] == ["champion", "complement"]


def test_future_utility_is_separate_from_current_query_relevance() -> None:
    query = "current incident"
    spans = [
        _span("current", "current incident evidence", tokens=2),
        _span("duplicate", "current incident repeated", tokens=2, ordinal=1),
        _span(
            "future",
            "durable decision needed by the next task",
            tokens=2,
            ordinal=2,
            future_utility=1.0,
        ),
    ]
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [1.0, 0.0],
            spans[1].text: [0.9, 0.4359],
            spans[2].text: [0.1, 0.995],
        }
    )
    policy = NeuralSelectionPolicy(
        diversity_weight=0.0,
        future_utility_weight=1.0,
    )

    result = select_neural_evidence(
        query, spans, budget_tokens=4, encoder=encoder, policy=policy
    )

    assert [span.span_id for span in result.selected] == ["current", "future"]
    future_record = next(
        record for record in result.receipt["spans"] if record["span_id"] == "future"
    )
    assert future_record["future_utility"] == 1.0
    assert future_record["selection_utility"] == pytest.approx(1.1, abs=1e-5)


def test_omitted_spans_persist_and_rehydrate_by_exact_character_range(
    tmp_path: Path,
) -> None:
    original = "auth timeout answer\nbilling history omitted"
    spans = [
        EvidenceSpan(
            span_id="answer",
            source="incident.txt",
            text="auth timeout answer",
            token_count=3,
            ordinal=0,
            start_char=0,
            end_char=19,
        ),
        EvidenceSpan(
            span_id="history",
            source="incident.txt",
            text="billing history omitted",
            token_count=3,
            ordinal=1,
            start_char=20,
            end_char=len(original),
        ),
    ]
    query = "auth timeout"
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [1.0, 0.0],
            spans[1].text: [0.0, 1.0],
        }
    )
    result = select_neural_evidence(query, spans, budget_tokens=3, encoder=encoder)
    path = tmp_path / "neural-recovery.json"
    store = CompressionRetrievalStore(path)

    stored = persist_neural_selection(
        store,
        original_text=original,
        spans=spans,
        result=result,
    )

    assert [span.span_id for span in stored.spans] == ["history"]
    assert stored.spans[0].content == "billing history omitted"
    assert stored.spans[0].start_char == 20
    assert stored.spans[0].content_sha256

    restored = CompressionRetrievalStore(path)
    recovered = restored.retrieve_span(
        stored.receipt_id, "history", retrieval_id="query-shift-q2"
    )
    assert recovered is not None
    assert recovered.content == "billing history omitted"
    assert restored.realized_savings(stored.receipt_id)["retrieved_tokens"] > 0


def test_exact_span_persistence_is_atomic_on_offset_mismatch(tmp_path: Path) -> None:
    original = "auth timeout answer\nbilling history omitted"
    spans = [
        EvidenceSpan(
            span_id="answer",
            source="incident.txt",
            text="auth timeout answer",
            token_count=3,
            ordinal=0,
            start_char=0,
            end_char=19,
        ),
        EvidenceSpan(
            span_id="history",
            source="incident.txt",
            text="billing history omitted",
            token_count=3,
            ordinal=1,
            start_char=19,
            end_char=len(original),
        ),
    ]
    query = "auth timeout"
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [1.0, 0.0],
            spans[1].text: [0.0, 1.0],
        }
    )
    result = select_neural_evidence(query, spans, budget_tokens=3, encoder=encoder)
    store = CompressionRetrievalStore(tmp_path / "atomic.json")

    with pytest.raises(ValueError, match="offsets do not match"):
        persist_neural_selection(
            store,
            original_text=original,
            spans=spans,
            result=result,
        )

    assert store.list_receipts() == []


def test_exact_span_persistence_rejects_partial_receipt_omissions(
    tmp_path: Path,
) -> None:
    original = "auth timeout answer\nbilling history omitted\nfuture decision"
    spans = [
        EvidenceSpan(
            span_id="answer",
            source="incident.txt",
            text="auth timeout answer",
            token_count=3,
            ordinal=0,
            start_char=0,
            end_char=19,
        ),
        EvidenceSpan(
            span_id="history",
            source="incident.txt",
            text="billing history omitted",
            token_count=3,
            ordinal=1,
            start_char=20,
            end_char=43,
        ),
        EvidenceSpan(
            span_id="future",
            source="incident.txt",
            text="future decision",
            token_count=2,
            ordinal=2,
            start_char=44,
            end_char=len(original),
        ),
    ]
    query = "auth timeout"
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [1.0, 0.0],
            spans[1].text: [0.0, 1.0],
            spans[2].text: [0.0, -1.0],
        }
    )
    result = select_neural_evidence(query, spans, budget_tokens=3, encoder=encoder)
    store = CompressionRetrievalStore(tmp_path / "partial.json")
    receipt_omissions = [
        record for record in result.receipt["spans"] if not record["selected"]
    ]

    with pytest.raises(ValueError, match="every receipt omission"):
        store.put_exact_spans(
            original_text=original,
            compressed_text=result.text,
            receipt=result.receipt,
            spans=[
                {
                    "span_id": receipt_omissions[0]["span_id"],
                    "source": receipt_omissions[0]["source"],
                    "start_char": receipt_omissions[0]["start_char"],
                    "end_char": receipt_omissions[0]["end_char"],
                    "content_sha256": receipt_omissions[0]["content_sha256"],
                }
            ],
        )

    assert store.list_receipts() == []


def test_exact_span_persistence_rolls_back_failed_disk_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = "auth timeout answer\nbilling history omitted"
    spans = [
        EvidenceSpan(
            span_id="answer",
            source="incident.txt",
            text="auth timeout answer",
            token_count=3,
            ordinal=0,
            start_char=0,
            end_char=19,
        ),
        EvidenceSpan(
            span_id="history",
            source="incident.txt",
            text="billing history omitted",
            token_count=3,
            ordinal=1,
            start_char=20,
            end_char=len(original),
        ),
    ]
    query = "auth timeout"
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [1.0, 0.0],
            spans[1].text: [0.0, 1.0],
        }
    )
    result = select_neural_evidence(query, spans, budget_tokens=3, encoder=encoder)
    store = CompressionRetrievalStore(tmp_path / "disk-failure.json")
    monkeypatch.setattr(
        store,
        "_persist",
        lambda: (_ for _ in ()).throw(OSError("simulated disk failure")),
    )

    with pytest.raises(OSError, match="simulated disk failure"):
        persist_neural_selection(
            store,
            original_text=original,
            spans=spans,
            result=result,
        )

    assert store.list_receipts() == []


def test_exact_span_store_detects_corruption_on_restart(tmp_path: Path) -> None:
    original = "auth timeout answer\nbilling history omitted"
    spans = [
        EvidenceSpan(
            span_id="answer",
            source="incident.txt",
            text="auth timeout answer",
            token_count=3,
            ordinal=0,
            start_char=0,
            end_char=19,
        ),
        EvidenceSpan(
            span_id="history",
            source="incident.txt",
            text="billing history omitted",
            token_count=3,
            ordinal=1,
            start_char=20,
            end_char=len(original),
        ),
    ]
    query = "auth timeout"
    encoder = FakeEncoder(
        {
            query: [1.0, 0.0],
            spans[0].text: [1.0, 0.0],
            spans[1].text: [0.0, 1.0],
        }
    )
    result = select_neural_evidence(query, spans, budget_tokens=3, encoder=encoder)
    path = tmp_path / "corrupt.json"
    persist_neural_selection(
        CompressionRetrievalStore(path),
        original_text=original,
        spans=spans,
        result=result,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["spans"][0]["content"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        CompressionRetrievalStore(path)
