from __future__ import annotations

from dataclasses import dataclass

import pytest

from entroly import domain_assurance
from entroly.domain_assurance import DomainDecision, compress_domain_assured


def test_short_input_is_byte_identical() -> None:
    result = compress_domain_assured(
        '{"status":"ok"}', query="status", budget_tokens=100
    )
    assert result.text == '{"status":"ok"}'
    assert result.receipt.decision is DomainDecision.BYPASS_ALREADY_FITS
    assert result.receipt.input_sha256 == result.receipt.output_sha256


def test_json_output_is_valid_and_keeps_query_evidence() -> None:
    payload = {
        "records": [
            {"id": i, "status": "ok", "message": "ordinary"} for i in range(40)
        ]
        + [{"id": 99, "status": "failed", "message": "database timeout"}]
    }
    import json

    text = json.dumps(payload)
    result = compress_domain_assured(
        text,
        query="database timeout status",
        budget_tokens=120,
    )
    assert result.receipt.decision is DomainDecision.COMPRESSED_VALIDATED
    assert result.receipt.validation.valid
    parsed = json.loads(result.text)
    assert parsed["elc"]
    assert "database timeout" in result.text.lower()


@dataclass
class FakeReceipt:
    content_type: str
    original_tokens: int = 100
    compressed_tokens: int = 5

    def as_dict(self):
        return {
            "content_type": self.content_type,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
        }


@dataclass
class FakeResult:
    compressed: str
    receipt: FakeReceipt
    changed: bool = True


def test_missing_critical_log_line_falls_back_to_original(monkeypatch) -> None:
    original = "INFO start\nERROR database unavailable request_id=abc\nINFO end\n" * 20
    monkeypatch.setattr(
        domain_assurance,
        "compress_evidence_locked",
        lambda *_args, **_kwargs: FakeResult("INFO start\nINFO end", FakeReceipt("logs")),
    )
    result = compress_domain_assured(
        original, query="database error", budget_tokens=20
    )
    assert result.receipt.decision is DomainDecision.BYPASS_INVALID
    assert result.text == original
    assert not result.receipt.validation.valid


def test_hard_budget_mode_is_explicitly_uncertified(monkeypatch) -> None:
    original = "ERROR important failure\n" * 50
    monkeypatch.setattr(
        domain_assurance,
        "compress_evidence_locked",
        lambda *_args, **_kwargs: FakeResult("summary", FakeReceipt("logs")),
    )
    result = compress_domain_assured(
        original,
        query="important failure",
        budget_tokens=10,
        fallback="compressed",
    )
    assert result.text == "summary"
    assert result.receipt.decision is DomainDecision.UNCERTIFIED_BUDGET_ENFORCED
    assert not result.receipt.validation.valid


def test_code_validator_requires_query_relevant_symbol() -> None:
    original = "def resolve_payment():\n    return charge_card()\n\ndef unrelated():\n    return 1\n"
    validation = domain_assurance.validate_domain_output(
        original,
        "def unrelated():\n    return 1\n",
        content_type="code",
        query="resolve_payment",
    )
    assert not validation.valid
    assert validation.critical_items == 1
    assert validation.critical_items_retained == 0


def test_raise_mode_contains_auditable_failure(monkeypatch) -> None:
    original = "FATAL lost evidence\n" * 30
    monkeypatch.setattr(
        domain_assurance,
        "compress_evidence_locked",
        lambda *_args, **_kwargs: FakeResult("tiny", FakeReceipt("logs")),
    )
    with pytest.raises(RuntimeError, match="raise_invalid_domain_compression"):
        compress_domain_assured(
            original, query="fatal evidence", budget_tokens=10, fallback="raise"
        )


def test_cjk_query_coverage_is_not_ascii_only() -> None:
    original = "INFO start\nERROR 認証失敗 request_id=abc\nINFO end\n"
    emitted = "ERROR 認証失敗 request_id=abc\n"
    validation = domain_assurance.validate_domain_output(
        original,
        emitted,
        content_type="logs",
        query="認証失敗の原因",
    )
    assert validation.valid
    assert validation.query_coverage >= 0.5


def test_space_separated_json_documents_are_validated() -> None:
    original = (
        '{"id":1,"status":"ok"} '
        '{"id":2,"status":"failed","message":"認証失敗"}'
    )
    emitted = '{"elc":"json_summary","message":"認証失敗","status":"failed"}'
    validation = domain_assurance.validate_domain_output(
        original,
        emitted,
        content_type="json_text",
        query="認証失敗 status",
    )
    assert validation.valid
    assert validation.checks["original_json_valid"]
    assert validation.critical_items_retained >= 1


def test_partial_json_stream_fails_closed() -> None:
    validation = domain_assurance.validate_domain_output(
        '{"id":1} {broken}',
        '{"elc":"json_summary"}',
        content_type="json",
        query="id",
    )
    assert not validation.valid
