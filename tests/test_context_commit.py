from __future__ import annotations

import copy

from entroly.context_commit import (
    CONTEXT_COMMIT_SCHEMA,
    create_context_commit,
    replay_context,
    verify_context_commit,
)


DOCUMENTS = [
    (
        "auth.py",
        "def validate_token(token):\n"
        "    return token.startswith('sk-')\n\n"
        "ROTATION_DAYS = 30\n",
    ),
    (
        "billing.py",
        "def charge(amount):\n"
        "    return gateway.capture(amount)\n\n"
        "RETRY_LIMIT = 3\n",
    ),
    (
        "runbook.md",
        "# Incident runbook\n"
        "Rotate credentials every 30 days.\n"
        "Retry failed charges three times.\n",
    ),
]


def _commit(*, prefer_rust: bool = False, parent: str | None = None):
    return create_context_commit(
        DOCUMENTS,
        query="How often should credentials rotate and charges retry?",
        token_budget=25,
        chunk_tokens=40,
        overlap_tokens=8,
        parent_commit_id=parent,
        prefer_rust=prefer_rust,
    )


def test_context_commit_is_deterministic_replayable_and_recoverable():
    first = _commit()
    second = _commit()
    verification = verify_context_commit(first)

    assert first == second
    assert first["schema_version"] == CONTEXT_COMMIT_SCHEMA
    assert first["commit_id"].startswith("ctx_")
    assert replay_context(first) == first["receipt"]["selected_context"]
    assert verification.valid is True
    assert verification.omitted_chunks > 0
    assert verification.recovered_omitted_chunks == verification.omitted_chunks


def test_context_commit_detects_receipt_and_recovery_tampering():
    original = _commit()
    receipt_tamper = copy.deepcopy(original)
    receipt_tamper["receipt"]["selected_context"][0]["text"] += " tampered"
    recovery_tamper = copy.deepcopy(original)
    chunk = next(iter(recovery_tamper["recovery_bundle"]["chunks"].values()))
    chunk["text"] += " tampered"

    receipt_result = verify_context_commit(receipt_tamper)
    recovery_result = verify_context_commit(recovery_tamper)

    assert receipt_result.valid is False
    assert "commit_id" in receipt_result.errors
    assert "receipt_hash" in receipt_result.errors
    assert "selected_context_integrity" in receipt_result.errors
    assert recovery_result.valid is False
    assert "recovery_bundle_digest" in recovery_result.errors
    assert "recovery_chunk_integrity" in recovery_result.errors


def test_context_commit_binds_parent_lineage_and_engine_identity():
    parent = _commit()
    child = _commit(parent=parent["commit_id"])

    assert child["parent_commit_id"] == parent["commit_id"]
    assert child["engine"]["implementation"] == "python"
    assert verify_context_commit(child).valid is True


def test_context_commit_uses_native_receipt_engine_when_available():
    commit = _commit(prefer_rust=True)

    assert commit["engine"]["implementation"] in {"python", "rust"}
    assert verify_context_commit(commit).valid is True


def test_context_commit_verification_fails_closed_on_non_mapping():
    result = verify_context_commit(None)

    assert result.valid is False
    assert result.errors == ("commit is not a mapping",)
