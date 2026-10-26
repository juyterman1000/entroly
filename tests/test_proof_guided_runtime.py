from __future__ import annotations

import json

import pytest

from entroly.proof_guided_runtime import (
    ProofGuidedRuntime,
    ProofGuidedSessionConflict,
)


def _documents():
    return [
        (
            "evidence.md",
            "Entroly selects relevant evidence under an explicit token budget.\n"
            "Every selected span retains its source fingerprint.\n"
            "Omitted spans remain recoverable from a content-addressed bundle.\n"
            "A signed receipt records the context delivered to the model.\n",
        ),
        (
            "operations.md",
            "The local service listens on a configurable port.\n"
            "Operators rotate credentials through their normal secret store.\n"
            "Deployment health checks report an actionable failure reason.\n"
            "Restart recovery replays durable events in their original order.\n",
        ),
    ]


def _runtime(tmp_path):
    return ProofGuidedRuntime(
        tmp_path,
        prefer_rust=False,
        context_risk_mode="audit",
    )


def _prepare(runtime):
    return runtime.prepare(
        _documents(),
        query="How does Entroly preserve and recover selected evidence?",
        token_budget=70,
        chunk_tokens=24,
        overlap_tokens=3,
        idempotency_key="prepare-request",
    )


def test_runtime_resumes_after_restart_and_recovers_exact_evidence(tmp_path):
    first_runtime = _runtime(tmp_path)
    prepared = _prepare(first_runtime)

    assert prepared["status"] == "awaiting_model"
    assert prepared["provider_call_performed"] is False
    assert prepared["request"]["appended_evidence"] == ""

    first = first_runtime.advance(
        prepared["session_id"],
        model_output="Restart recovery replays durable events in their original order.",
        idempotency_key="model-round-0",
        expected_revision=prepared["revision"],
        request_sha256=prepared["request_sha256"],
    )
    assert first["status"] == "awaiting_model"
    assert first["round"]["decision"] == "recover_and_retry"
    assert first["request"]["full_context"].startswith(
        first["request"]["stable_context_prefix"]
    )
    assert "Restart recovery replays durable events" in first["request"][
        "appended_evidence"
    ]

    restarted = _runtime(tmp_path)
    second = restarted.advance(
        prepared["session_id"],
        model_output="Restart recovery replays durable events in their original order.",
        idempotency_key="model-round-1",
        expected_revision=first["revision"],
        request_sha256=first["request_sha256"],
    )

    assert second["status"] == "supported"
    assert second["converged"] is True
    assert second["final_output"]
    assert second["request"] is None
    assert second["provider_call_performed"] is False
    assert restarted.inspect(prepared["session_id"]) == second


def test_prepare_and_advance_are_idempotent(tmp_path):
    runtime = _runtime(tmp_path)
    first_prepare = _prepare(runtime)
    second_prepare = _prepare(runtime)
    assert first_prepare == second_prepare

    kwargs = {
        "model_output": "Restart recovery replays durable events in their original order.",
        "idempotency_key": "round-0",
        "expected_revision": first_prepare["revision"],
        "request_sha256": first_prepare["request_sha256"],
    }
    first_advance = runtime.advance(first_prepare["session_id"], **kwargs)
    second_advance = runtime.advance(first_prepare["session_id"], **kwargs)
    assert first_advance == second_advance

    with pytest.raises(ProofGuidedSessionConflict, match="different model output"):
        runtime.advance(
            first_prepare["session_id"],
            model_output="different output",
            idempotency_key="round-0",
            expected_revision=first_prepare["revision"],
            request_sha256=first_prepare["request_sha256"],
        )


def test_older_advance_idempotency_key_replays_original_response(tmp_path):
    runtime = _runtime(tmp_path)
    prepared = _prepare(runtime)
    first = runtime.advance(
        prepared["session_id"],
        model_output="Restart recovery replays durable events in their original order.",
        idempotency_key="round-0",
        expected_revision=prepared["revision"],
        request_sha256=prepared["request_sha256"],
    )
    runtime.advance(
        prepared["session_id"],
        model_output="Restart recovery replays durable events in their original order.",
        idempotency_key="round-1",
        expected_revision=first["revision"],
        request_sha256=first["request_sha256"],
    )

    replayed = runtime.advance(
        prepared["session_id"],
        model_output="Restart recovery replays durable events in their original order.",
        idempotency_key="round-0",
        expected_revision=prepared["revision"],
        request_sha256=prepared["request_sha256"],
    )

    assert replayed == first


def test_prepare_idempotency_key_rejects_a_different_request(tmp_path):
    runtime = _runtime(tmp_path)
    _prepare(runtime)
    with pytest.raises(ProofGuidedSessionConflict, match="different request"):
        runtime.prepare(
            _documents(),
            query="A different question",
            token_budget=70,
            idempotency_key="prepare-request",
        )


def test_session_state_corruption_fails_closed(tmp_path):
    runtime = _runtime(tmp_path)
    prepared = _prepare(runtime)
    path = runtime.sessions_dir / f"{prepared['session_id']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["revision"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProofGuidedSessionConflict, match="state hash mismatch"):
        runtime.inspect(prepared["session_id"])


def test_issued_request_must_match_the_request_the_controller_verifies(tmp_path):
    """Even a resealed inconsistent snapshot must not bind the wrong request.

    This consistency check does not make the unkeyed local state seal an
    authentication boundary against a writer who controls the entire store.
    """
    from entroly.proof_guided_runtime import _request_digest

    runtime = _runtime(tmp_path)
    prepared = _prepare(runtime)
    path = runtime.sessions_dir / f"{prepared['session_id']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    response = payload["last_response"]
    response["request"]["query"] = "A different issued question"
    response["request_sha256"] = _request_digest(
        response,
        prepare_request_hash=payload["prepare_request_hash"],
        commit_id=payload["session"]["prepared"]["commit_id"],
    )
    runtime._save(payload)
    before = path.read_bytes()

    with pytest.raises(ProofGuidedSessionConflict, match="does not match the session"):
        runtime.advance(
            prepared["session_id"],
            model_output="Restart recovery replays durable events in their original order.",
            idempotency_key="inconsistent-request",
            expected_revision=response["revision"],
            request_sha256=response["request_sha256"],
        )
    assert path.read_bytes() == before
