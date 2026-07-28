from __future__ import annotations

import json

from entroly.compression_retrieval_store import CompressionRetrievalStore
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


class _AdvancingClock:
    def __init__(self, step_seconds: float) -> None:
        self.value = -step_seconds
        self.step_seconds = step_seconds

    def __call__(self) -> float:
        self.value += self.step_seconds
        return self.value


def _large_test_output() -> str:
    return "\n".join(f"test case_{index} ... ok" for index in range(400))


def _policy(**kwargs) -> TransformPolicy:
    return TransformPolicy(deadline_ms=1000.0, **kwargs)


def _scoped_envelope(content: str, *, workspace: str = "repo-a") -> ContentEnvelope:
    return ContentEnvelope(
        content=content,
        source="pytest",
        tool_name="pytest",
        command="pytest -q",
        workspace=workspace,
        cwd=f"/workspace/{workspace}",
        metadata={"session_id": "session-1"},
    )


def test_exact_repeat_becomes_recoverable_reference() -> None:
    store = _Store()
    pipeline = ContextTransformPipeline(retrieval_store=store)
    envelope = _scoped_envelope(_large_test_output())
    policy = _policy(token_budget=100, allow_exact_reference=True)

    first = pipeline.transform(envelope, policy)
    second = pipeline.transform(envelope, policy)

    assert first.receipt.recovery_receipt_id == "recovery-1"
    assert second.receipt.algorithm == "exact_reference"
    assert second.receipt.exact_reference_of == first.receipt.receipt_id
    assert second.receipt.recovery_receipt_id == "recovery-1"
    assert second.receipt.recovery_span_ids == ("span-1",)
    assert second.content.startswith("[entroly-ref:recovery-1:span-1]")
    assert len(second.content) < len(envelope.content)


def test_exact_references_are_opt_in() -> None:
    store = _Store()
    pipeline = ContextTransformPipeline(retrieval_store=store)
    envelope = _scoped_envelope(_large_test_output())
    policy = _policy(token_budget=100)

    first = pipeline.transform(envelope, policy)
    second = pipeline.transform(envelope, policy)

    assert first.receipt.algorithm != "exact_reference"
    assert second.receipt.algorithm != "exact_reference"
    assert second.receipt.exact_reference_of is None


def test_unscoped_text_never_enters_exact_reuse_cache() -> None:
    store = _Store()
    pipeline = ContextTransformPipeline(retrieval_store=store)
    envelope = ContentEnvelope(content=_large_test_output(), source="generic")
    policy = _policy(token_budget=100, allow_exact_reference=True)

    first = pipeline.transform(envelope, policy)
    second = pipeline.transform(envelope, policy)

    assert first.receipt.algorithm != "exact_reference"
    assert second.receipt.algorithm != "exact_reference"
    assert second.receipt.exact_reference_of is None


def test_recovery_ids_are_isolated_by_workspace(tmp_path) -> None:
    store = CompressionRetrievalStore(tmp_path / "recovery.json")
    content = _large_test_output()
    policy = _policy(token_budget=100, allow_exact_reference=True)

    first = ContextTransformPipeline(retrieval_store=store).transform(
        _scoped_envelope(content, workspace="repo-a"),
        policy,
    )
    second = ContextTransformPipeline(retrieval_store=store).transform(
        _scoped_envelope(content, workspace="repo-b"),
        policy,
    )

    assert first.receipt.scope_sha256 != second.receipt.scope_sha256
    assert first.receipt.recovery_receipt_id
    assert second.receipt.recovery_receipt_id
    assert first.receipt.recovery_receipt_id != second.receipt.recovery_receipt_id


def test_redaction_precedes_recovery_persistence() -> None:
    store = _Store()
    pipeline = ContextTransformPipeline(retrieval_store=store)
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    content = f"api_key={secret} " + ("payload " * 300)
    envelope = ContentEnvelope(
        content=content,
        source="tool:config",
        tool_name="config_dump",
        command="config dump --verbose",
        workspace="repo-a",
        metadata={
            "authorization_scope": "tenant-a",
            "session_id": "session-a",
        },
    )

    result = pipeline.transform(
        envelope,
        _policy(token_budget=100, redact_sensitive=True),
    )

    assert result.receipt.redacted is True
    assert result.receipt.redaction_counts["openai_api_key"] == 1
    assert store.calls
    call = store.calls[0]
    assert secret not in str(call["original_text"])
    assert "command" not in call["metadata"]
    assert call["metadata"]["command_sha256"]
    assert call["metadata"]["workspace_sha256"]
    assert call["metadata"]["authorization_scope_sha256"]
    assert call["metadata"]["session_scope_sha256"]
    assert call["metadata"]["scope_sha256"] == result.receipt.scope_sha256
    assert "authorization_scope" not in result.receipt.metadata
    assert "session_id" not in result.receipt.metadata
    assert result.receipt.metadata["authorization_scope_sha256"]
    assert result.receipt.metadata["session_scope_sha256"]


def test_different_raw_secrets_are_not_exact_repeats_after_redaction() -> None:
    store = _Store()
    pipeline = ContextTransformPipeline(retrieval_store=store)
    policy = _policy(
        token_budget=100,
        redact_sensitive=True,
        allow_exact_reference=True,
    )

    first = pipeline.transform(
        _scoped_envelope(
            "api_key=sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaa " + ("payload " * 300)
        ),
        policy,
    )
    second = pipeline.transform(
        _scoped_envelope(
            "api_key=sk-bbbbbbbbbbbbbbbbbbbbbbbbbbbb " + ("payload " * 300)
        ),
        policy,
    )

    assert first.receipt.safe_input_sha256 == second.receipt.safe_input_sha256
    assert first.receipt.input_sha256 != second.receipt.input_sha256
    assert second.receipt.algorithm != "exact_reference"
    assert second.receipt.exact_reference_of is None


def test_exact_json_policy_preserves_parseable_payload_across_repeats() -> None:
    content = json.dumps(
        {"rows": [{"id": index, "value": "x" * 40} for index in range(100)]}
    )
    store = _Store()
    pipeline = ContextTransformPipeline(retrieval_store=store)
    envelope = ContentEnvelope(
        content=content,
        source="tool:json_query",
        tool_name="json_query",
        command="json query",
        workspace="repo-a",
        metadata={"session_id": "session-json"},
    )
    policy = _policy(
        token_budget=100,
        preserve_exact_json=True,
        allow_exact_reference=True,
    )

    first = pipeline.transform(envelope, policy)
    second = pipeline.transform(envelope, policy)

    assert first.content == content
    assert second.content == content
    assert first.receipt.algorithm == "identity"
    assert second.receipt.algorithm == "identity"
    assert first.receipt.exact_reference_of is None
    assert second.receipt.exact_reference_of is None
    assert store.calls == []
    assert any(stage.name == "exact_json_policy" for stage in first.receipt.stages)


def test_existing_entroly_marker_prevents_double_transformation() -> None:
    content = "[entroly-elc: prior receipt]\n" + ("diagnostic\n" * 400)
    pipeline = ContextTransformPipeline()

    result = pipeline.transform(
        ContentEnvelope(content=content, source="tool:build"),
        _policy(token_budget=100),
    )

    assert result.content == content
    assert any(stage.name == "idempotency" for stage in result.receipt.stages)


def test_cooperative_deadline_fails_open_before_compression() -> None:
    clock = _AdvancingClock(step_seconds=0.01)
    pipeline = ContextTransformPipeline(clock=clock)
    content = _large_test_output()

    result = pipeline.transform(
        ContentEnvelope(content=content, source="tool:pytest"),
        TransformPolicy(token_budget=100, deadline_ms=1.0),
    )

    assert result.content == content
    assert result.receipt.algorithm == "deadline_fail_open"
    assert result.receipt.deadline_exceeded is True
    assert any(stage.name == "deadline" for stage in result.receipt.stages)
    assert all(stage.name != "typed_formatter" for stage in result.receipt.stages)


def test_receipt_header_cannot_erase_token_gain(monkeypatch) -> None:
    class _Receipt:
        content_type = "text"
        savings_ratio = 0.9

    class _Result:
        changed = True
        receipt = _Receipt()

        @staticmethod
        def with_receipt_header() -> str:
            return "header-expansion " * 500

    monkeypatch.setattr(
        "entroly.context_pipeline.compress_evidence_locked",
        lambda *args, **kwargs: _Result(),
    )
    content = "evidence " * 100
    pipeline = ContextTransformPipeline()

    result = pipeline.transform(
        ContentEnvelope(content=content, source="tool:generic"),
        _policy(token_budget=100, prefer_typed_formatter=False),
    )

    assert result.content == content
    assert result.receipt.algorithm == "identity"
    stage = next(
        stage
        for stage in result.receipt.stages
        if stage.name == "evidence_locked_compression"
    )
    assert stage.outcome == "receipt_overhead_erased_gain"


def test_receipt_is_versioned_and_hashes_transmitted_content() -> None:
    pipeline = ContextTransformPipeline()
    result = pipeline.transform(
        ContentEnvelope(content=_large_test_output(), source="tool:pytest"),
        _policy(token_budget=100),
    )

    receipt = result.receipt.as_dict()
    assert receipt["version"] == "1"
    assert receipt["policy_sha256"]
    assert receipt["scope_sha256"]
    assert receipt["receipt_id"]
    assert receipt["input_sha256"]
    assert receipt["output_sha256"]
    assert receipt["transmitted_tokens"] <= receipt["original_tokens"]
    assert receipt["stages"]
