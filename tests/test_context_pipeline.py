from __future__ import annotations

import json

from entroly.context_pipeline import (
    ContentEnvelope,
    ContextTransformPipeline,
    TransformPolicy,
)


class _Span:
    def __init__(self, span_id: str) -> None:
        self.span_id = span_id


class _Stored:
    def __init__(self, receipt_id: str) -> None:
        self.receipt_id = receipt_id
        self.spans = [_Span("span-1")]


class _Store:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def put(self, **kwargs):
        self.calls.append(kwargs)
        return _Stored("recovery-1")


def _large_test_output() -> str:
    return "\n".join(f"test case_{index} ... ok" for index in range(400))


def test_exact_repeat_becomes_recoverable_reference() -> None:
    store = _Store()
    pipeline = ContextTransformPipeline(retrieval_store=store)
    envelope = ContentEnvelope(
        content=_large_test_output(),
        source="pytest",
        tool_name="pytest",
        command="pytest -q",
        workspace="repo-a",
        cwd="/workspace/repo-a",
    )
    policy = TransformPolicy(token_budget=100)

    first = pipeline.transform(envelope, policy)
    second = pipeline.transform(envelope, policy)

    assert first.receipt.recovery_receipt_id == "recovery-1"
    assert second.receipt.algorithm == "exact_reference"
    assert second.receipt.exact_reference_of == first.receipt.receipt_id
    assert second.content.startswith("[entroly-ref:recovery-1:span-1]")
    assert len(second.content) < len(envelope.content)


def test_redaction_precedes_recovery_persistence() -> None:
    store = _Store()
    pipeline = ContextTransformPipeline(retrieval_store=store)
    content = "api_key=sk-abcdefghijklmnopqrstuvwxyz123456 " + ("payload " * 300)

    result = pipeline.transform(
        ContentEnvelope(content=content, source="tool:config"),
        TransformPolicy(token_budget=100, redact_sensitive=True),
    )

    assert result.receipt.redacted is True
    assert result.receipt.redaction_counts["openai_api_key"] == 1
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(store.calls[0]["original_text"])


def test_exact_json_policy_preserves_parseable_payload() -> None:
    content = json.dumps({"rows": [{"id": index, "value": "x" * 40} for index in range(100)]})
    pipeline = ContextTransformPipeline()

    result = pipeline.transform(
        ContentEnvelope(content=content, source="tool:json_query"),
        TransformPolicy(token_budget=100, preserve_exact_json=True),
    )

    assert result.content == content
    assert result.receipt.algorithm == "identity"
    assert any(stage.name == "exact_json_policy" for stage in result.receipt.stages)


def test_existing_entroly_marker_prevents_double_transformation() -> None:
    content = "[entroly-elc: prior receipt]\n" + ("diagnostic\n" * 400)
    pipeline = ContextTransformPipeline()

    result = pipeline.transform(
        ContentEnvelope(content=content, source="tool:build"),
        TransformPolicy(token_budget=100),
    )

    assert result.content == content
    assert any(stage.name == "idempotency" for stage in result.receipt.stages)


def test_receipt_is_versioned_and_hashes_transmitted_content() -> None:
    pipeline = ContextTransformPipeline()
    result = pipeline.transform(
        ContentEnvelope(content=_large_test_output(), source="tool:pytest"),
        TransformPolicy(token_budget=100),
    )

    receipt = result.receipt.as_dict()
    assert receipt["version"] == "1"
    assert receipt["receipt_id"]
    assert receipt["input_sha256"]
    assert receipt["output_sha256"]
    assert receipt["transmitted_tokens"] <= receipt["original_tokens"]
    assert receipt["stages"]
