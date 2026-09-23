from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pytest

from entroly.codec import RecoveryStore
from entroly.evidence_locator import (
    EncoderEvidenceRanker,
    LexicalEvidenceRanker,
    PassageJudgment,
    SourcePassage,
    locate_evidence,
    segment_source,
    sentence_spans,
    verify_evidence_location,
)


class FixedRanker:
    ranker_id = "test.fixed"
    fingerprint = hashlib.sha256(b"test.fixed").hexdigest()

    def __init__(self, rows: Sequence[PassageJudgment]) -> None:
        self.rows = tuple(rows)

    def rank(self, query: str, passages: Sequence[SourcePassage]):
        return self.rows


class ParaphraseEncoder:
    model_id = "test-paraphrase"
    fingerprint = hashlib.sha256(b"test-paraphrase").hexdigest()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            cancellation = float("cancel" in lower or "termination" in lower)
            pricing = float("price" in lower or "cost" in lower)
            neutral = 1.0 if not cancellation and not pricing else 0.0
            vectors.append([cancellation, pricing, neutral])
        return vectors


def test_sentence_spans_preserve_exact_unicode_and_offsets() -> None:
    text = "  Price is 1.3.  Kündigung gilt sofort!  尾文。  "
    spans = sentence_spans(text)
    assert [row.text for row in spans] == [
        "Price is 1.3.",
        "Kündigung gilt sofort!",
        "尾文。",
    ]
    assert all(text[row.start_char : row.end_char] == row.text for row in spans)


def test_segmentation_includes_final_paragraph_before_trailing_newline() -> None:
    source = "# Heading\n\nFirst paragraph.\n\nFinal evidence.\n"
    passages, complete = segment_source(source)
    assert complete is True
    assert passages[-1].text == "Final evidence."
    assert passages[-1].end_char == len(source.rstrip())


def test_lexical_locator_returns_only_exact_source_text() -> None:
    source = "Pricing is annual.\n\nRate limits apply per workspace. Retry later."
    result = locate_evidence(source, "workspace rate limits", min_score=0.2)
    assert result.status == "selected"
    assert result.matches[0].focus_text == "Rate limits apply per workspace."
    assert result.matches[0].verify(source)
    assert result.receipt()["selection"]["calibrated"] is False
    assert "not proof" in result.receipt()["claim_boundary"]


def test_semantic_ranker_can_find_paraphrase_without_generating_an_answer() -> None:
    source = (
        "The listed price excludes local taxes.\n\n"
        "Termination of service takes effect after thirty days."
    )
    result = locate_evidence(
        source,
        "what happens if I cancel?",
        ranker=EncoderEvidenceRanker(ParaphraseEncoder()),
        min_score=0.5,
        calibration_id="fixture-v1",
    )
    assert result.status == "selected"
    assert result.matches[0].focus_text == (
        "Termination of service takes effect after thirty days."
    )
    assert result.calibrated is True
    assert result.matches[0].focus_text in source


@pytest.mark.parametrize(
    "rows",
    [
        (),
        (PassageJudgment("invented", 1.0, 0),),
        (PassageJudgment("p0", 1.0, 99),),
        (PassageJudgment("p0", float("nan"), 0),),
    ],
)
def test_invalid_or_incomplete_ranker_output_abstains_without_fallback(rows) -> None:
    source = "The answer is present.\n\nAnother passage exists."
    result = locate_evidence(source, "answer", ranker=FixedRanker(rows))
    assert result.status == "abstained"
    assert result.reason.startswith("ranker_failed:")
    assert result.matches == ()


def test_threshold_no_match_and_budget_abstention_are_distinct() -> None:
    source = "Rate limits are documented here."
    no_match = locate_evidence(source, "unrelated", min_score=0.9)
    too_small = locate_evidence(source, "rate limits", budget_tokens=1)
    assert (no_match.status, no_match.reason) == (
        "no_match",
        "no_score_cleared_threshold",
    )
    assert (too_small.status, too_small.reason) == (
        "abstained",
        "budget_too_small",
    )


def test_recovery_is_exact_when_selection_or_candidate_scan_omits_source(
    tmp_path: Path,
) -> None:
    source = "alpha target.\n\nbeta.\n\ngamma beyond scan."
    store = RecoveryStore(tmp_path / "recovery.json")
    result = locate_evidence(
        source,
        "alpha target",
        max_source_chars=20,
        recovery_store=store,
    )
    assert result.status == "selected"
    assert result.source_coverage_complete is False
    assert result.recovery is not None
    assert store.recover(result.recovery) == source


def test_freshness_verification_detects_source_mutation() -> None:
    source = "Keep this exact sentence."
    result = locate_evidence(source, "exact sentence")
    assert verify_evidence_location(result, source)["valid"] is True
    changed = source.replace("exact", "changed")
    verification = verify_evidence_location(result, changed)
    assert verification["source_fresh"] is False
    assert verification["valid"] is False


def test_render_context_defaults_to_source_order() -> None:
    source = "first weak.\n\nsecond strong."
    passages, _ = segment_source(source)
    ranker = FixedRanker(
        (
            PassageJudgment(passages[0].passage_id, 0.4, 0),
            PassageJudgment(passages[1].passage_id, 0.9, 0),
        )
    )
    result = locate_evidence(source, "anything", ranker=ranker)
    assert result.render_context() == "first weak.\n\nsecond strong."
    assert result.render_context(relevance_order=True) == "second strong.\n\nfirst weak."


def test_public_mcp_locator_is_reachable_and_extracts_source_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENTROLY_MCP_PROFILE", "public")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("ENTROLY_SEMANTIC_MODEL_PATH", raising=False)
    from entroly.server import create_mcp_server

    server, _engine = create_mcp_server()
    raw = server._tool_manager._tools["locate_evidence"].fn(
        text="Price excludes taxes.\n\nRate limits apply per workspace.",
        query="workspace rate limits",
    )
    payload = json.loads(raw)
    assert payload["status"] == "selected"
    assert payload["matches"][0]["focus_text"] == "Rate limits apply per workspace."
    assert payload["ranker"]["id"] == "entroly.lexical-evidence.v1"
