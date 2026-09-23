from __future__ import annotations

from pathlib import Path

import pytest

from entroly.browser_context import _validate_url, compress_accessibility_snapshot
from entroly.codec import RecoveryStore
from entroly.evidence_locator import EncoderEvidenceRanker


class _ParaphraseEncoder:
    model_id = "fixture"
    fingerprint = "fixture-fingerprint"

    def encode(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            relevant = float("cancel" in lower or "termination" in lower)
            rows.append([relevant, 1.0 - relevant])
        return rows


def _snapshot() -> str:
    noise = "\n".join(f"    - paragraph: unrelated catalog item {index}" for index in range(300))
    return (
        '- navigation "Primary":\n'
        '  - link "Home"\n'
        '- main:\n'
        '  - heading "Billing settings" [level=1]\n'
        '    - textbox "Invoice email"\n'
        '    - button "Save billing settings"\n'
        f"{noise}\n"
    )


def test_browser_context_requires_complete_query_coverage_and_exact_recovery(tmp_path: Path) -> None:
    original = _snapshot()
    store = RecoveryStore(tmp_path / "recovery.json", scope_id="browser-test")
    result = compress_accessibility_snapshot(original, query="billing invoice", budget=80, store=store)
    assert result.mode == "compressed"
    assert result.receipt()["query_coverage"]["complete"] is True
    assert "Billing settings" in result.text
    assert "Invoice email" in result.text
    assert result.recovery is not None
    assert store.recover(result.recovery) == original


def test_browser_context_passes_through_on_query_miss_or_insufficient_budget(tmp_path: Path) -> None:
    original = _snapshot()
    store = RecoveryStore(tmp_path / "recovery.json", scope_id="browser-test")
    missing = compress_accessibility_snapshot(original, query="nonexistent evidence", budget=80, store=store)
    assert missing.mode == "passthrough-query-miss"
    assert missing.text == original
    cramped = compress_accessibility_snapshot(original, query="billing invoice", budget=1, store=store)
    assert cramped.mode == "passthrough-budget-insufficient"
    assert cramped.text == original


def test_browser_capture_rejects_private_targets_without_explicit_override() -> None:
    with pytest.raises(ValueError, match="private, loopback"):
        _validate_url("http://127.0.0.1:8000", allow_private_network=False)
    _validate_url("http://127.0.0.1:8000", allow_private_network=True)


def test_browser_can_select_semantic_paraphrase_with_exact_recovery(tmp_path: Path) -> None:
    noise = "\n".join(f"- paragraph: catalog {index}" for index in range(200))
    original = f"- paragraph: Termination occurs after thirty days.\n{noise}"
    store = RecoveryStore(tmp_path / "semantic-recovery.json", scope_id="browser-test")
    result = compress_accessibility_snapshot(
        original,
        query="what if I cancel?",
        budget=50,
        store=store,
        ranker=EncoderEvidenceRanker(_ParaphraseEncoder()),
        min_score=0.5,
        calibration_id="fixture-v1",
    )
    assert result.mode == "semantic-extractive"
    assert result.text == "- paragraph: Termination occurs after thirty days."
    assert result.evidence_location is not None
    assert result.evidence_location["status"] == "selected"
    assert result.recovery is not None
    assert store.recover(result.recovery) == original
