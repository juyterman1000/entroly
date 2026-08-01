from __future__ import annotations

from dataclasses import dataclass

from entroly import assurance_sdk
from entroly.assurance_sdk import compress_assured, compress_messages_assured
from entroly.guarded_selection import GuardDecision, GuardedSelectionReceipt
from entroly.sufficiency_contract import CertificateScope


@dataclass
class FakeSelection:
    selected: tuple[dict, ...]
    receipt: GuardedSelectionReceipt
    audits: tuple[dict, ...]


def receipt(decision: GuardDecision, raw: int, delivered: int) -> GuardedSelectionReceipt:
    return GuardedSelectionReceipt(
        decision=decision,
        requested_budget=10,
        final_budget=10,
        raw_tokens=raw,
        delivered_tokens=delivered,
        required_scope=CertificateScope.CANDIDATE_UNITS,
        exact_identity=decision.name.startswith("BYPASS"),
        budget_compliant=delivered <= 10,
        input_sha256="a" * 64,
        output_sha256="b" * 64,
    )


def test_text_structural_selection_returns_receipt(monkeypatch) -> None:
    monkeypatch.setattr(assurance_sdk, "detect_heavy_content_type", lambda _text: "text")
    monkeypatch.setattr(
        assurance_sdk,
        "select_assured",
        lambda *_args, **_kwargs: FakeSelection(
            ({"source": "sdk:text", "content": "answer", "token_count": 2},),
            receipt(GuardDecision.COMPRESSED_CERTIFIED, 100, 2),
            ({"metrics": {"scope": "candidate_units"}},),
        ),
    )
    result = compress_assured(
        "x" * 400,
        query="answer",
        budget=10,
        required_scope="candidate_units",
    )
    assert result.text == "answer"
    assert result.mode == "audited_qccr"
    assert result.receipt["decision"] == "COMPRESSED_CERTIFIED"


def test_messages_preserve_recent_turns(monkeypatch) -> None:
    monkeypatch.setattr(
        assurance_sdk,
        "select_assured",
        lambda *_args, **_kwargs: FakeSelection(
            ({"source": "message:0:system", "content": "short", "token_count": 2},),
            receipt(GuardDecision.COMPRESSED_CERTIFIED, 100, 2),
            (),
        ),
    )
    messages = [
        {"role": "system", "content": "x" * 400},
        {"role": "assistant", "content": "y" * 200},
        {"role": "user", "content": "fix answer"},
    ]
    result = compress_messages_assured(
        messages,
        budget=80,
        preserve_last_n=2,
        required_scope="candidate_units",
    )
    assert result.messages[0]["content"] == "short"
    assert list(result.messages[-2:]) == messages[-2:]
    assert result.receipt["overall_delivered_tokens"] == result.delivered_tokens


def test_semantic_scope_is_default_and_quality_first(monkeypatch) -> None:
    captured = {}

    def fake_select(*_args, **kwargs):
        captured.update(kwargs)
        return FakeSelection(
            ({"source": "sdk:text", "content": "x" * 400, "token_count": 100},),
            receipt(GuardDecision.BYPASS_UNCERTIFIED, 100, 100),
            (),
        )

    monkeypatch.setattr(assurance_sdk, "detect_heavy_content_type", lambda _text: "text")
    monkeypatch.setattr(assurance_sdk, "select_assured", fake_select)
    compress_assured("x" * 400, query="answer", budget=10)
    assert captured["required_scope"] is CertificateScope.SEMANTIC
    assert captured["fallback"] == "original"


def test_instruction_file_is_never_compressed(monkeypatch) -> None:
    monkeypatch.setattr(assurance_sdk, "detect_heavy_content_type", lambda _text: "text")

    def fail_selector(*_args, **_kwargs):
        raise AssertionError("selector must not run for instruction files")

    monkeypatch.setattr(assurance_sdk, "select_assured", fail_selector)
    text = "Always follow the repository security rules.\n" * 40
    result = compress_assured(
        text,
        query="security rules",
        budget=10,
        source_path="AGENTS.md",
        required_scope="candidate_units",
    )
    assert result.text == text
    assert result.mode == "identity"
    assert result.receipt["decision"] == "BYPASS_INSTRUCTION_FILE"
    assert not result.receipt["budget_compliant"]


def test_small_plain_output_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(assurance_sdk, "detect_heavy_content_type", lambda _text: "text")
    monkeypatch.setattr(
        assurance_sdk,
        "select_assured",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selector must not run for protected short input")
        ),
    )
    text = "ERROR: payment failed"
    result = compress_assured(
        text,
        query="payment failure",
        budget=2,
        required_scope="candidate_units",
    )
    assert result.text == text
    assert result.receipt["decision"] == "BYPASS_SHORT_INPUT"


def test_file_api_rejects_workspace_escape_and_preserves_rules(tmp_path) -> None:
    rules = tmp_path / "AGENTS.md"
    rules.write_text("Do not alter release evidence.\n" * 30, encoding="utf-8")
    result = assurance_sdk.compress_file_assured(
        "AGENTS.md",
        workspace=tmp_path,
        query="release evidence",
        budget=5,
        required_scope="candidate_units",
    )
    assert result.text == rules.read_text(encoding="utf-8")
    assert result.receipt["decision"] == "BYPASS_INSTRUCTION_FILE"

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="escapes workspace"):
        assurance_sdk.compress_file_assured(
            outside,
            workspace=tmp_path,
            query="outside",
            budget=10,
        )
