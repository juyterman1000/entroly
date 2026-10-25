"""Adversarial, offline acceptance tests for the v2 host request binding."""

from __future__ import annotations

import copy
import json
import multiprocessing
import re
import socket
import sys
import traceback
from pathlib import Path

import pytest

from entroly.context_receipts.models import stable_hash
from entroly.proof_guided_runtime import (
    ProofGuidedRuntime,
    ProofGuidedSessionConflict,
)


MODEL_OUTPUT = "Restart recovery replays durable events in their original order."
V1 = "entroly.proof-guided-runtime.v1"
V2 = "entroly.proof-guided-runtime.v2"


def _deny_network(*args, **kwargs):
    raise AssertionError("request-binding conformance must not access the network")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(socket, "create_connection", _deny_network)


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


def _runtime(path):
    return ProofGuidedRuntime(path, prefer_rust=False, context_risk_mode="audit")


def _prepare(runtime, *, key="binding-prepare"):
    return runtime.prepare(
        _documents(),
        query="How does Entroly preserve and recover evidence?",
        # Select a real evidence packet before testing the binding contract.
        token_budget=70,
        chunk_tokens=24,
        overlap_tokens=3,
        max_rounds=3,
        recovery_token_budget=100,
        idempotency_key=key,
    )


def _binding(issued):
    """Retain the binding actually issued for this invocation, including retries."""
    return {
        "expected_revision": issued["revision"],
        "request_sha256": issued["request_sha256"],
    }


def _deliver(runtime, issued, *, key="delivery-0", output=MODEL_OUTPUT):
    return runtime.advance(
        issued["session_id"],
        model_output=output,
        idempotency_key=key,
        **_binding(issued),
    )


def _session_path(runtime, session_id):
    return runtime.sessions_dir / f"{session_id}.json"


def _snapshot(runtime, session_id):
    """Session bytes and audit artifacts must be unchanged on reject or replay."""
    return (
        _session_path(runtime, session_id).read_bytes(),
        {
            str(path.relative_to(runtime.layer.audit_dir)): path.read_bytes()
            for path in runtime.layer.audit_dir.rglob("*.json")
        },
    )


def _assert_rejected(
    runtime, issued, fields, *, error=ProofGuidedSessionConflict,
    key="rejected-delivery", output=MODEL_OUTPUT,
):
    before = _snapshot(runtime, issued["session_id"])
    with pytest.raises(error):
        runtime.advance(
            issued["session_id"],
            model_output=output,
            idempotency_key=key,
            **fields,
        )
    assert _snapshot(runtime, issued["session_id"]) == before


def test_advance_rejects_missing_binding_without_changing_session(tmp_path):
    runtime = _runtime(tmp_path)
    prepared = _prepare(runtime)
    path = runtime.sessions_dir / f"{prepared['session_id']}.json"
    before = path.read_bytes()

    with pytest.raises(ValueError):
        runtime.advance(
            prepared["session_id"],
            model_output=MODEL_OUTPUT,
            idempotency_key="unbound-delivery",
        )

    assert path.read_bytes() == before
    assert runtime.inspect(prepared["session_id"]) == prepared


@pytest.mark.parametrize("missing", ["expected_revision", "request_sha256"])
def test_advance_rejects_either_missing_binding_field(tmp_path, missing):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    fields = _binding(issued)
    fields.pop(missing)
    _assert_rejected(runtime, issued, fields, error=ValueError)


@pytest.mark.parametrize(
    "revision",
    [None, False, True, 0.0, 0.5, "0", "1", -1, [], {}],
    ids=["null", "false", "true", "whole-float", "fraction", "str-zero",
         "str-one", "negative", "list", "mapping"],
)
def test_revision_is_strict_nonnegative_integer(tmp_path, revision):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    fields = {**_binding(issued), "expected_revision": revision}
    _assert_rejected(runtime, issued, fields, error=ValueError)


@pytest.mark.parametrize(
    "digest",
    [None, False, 0, b"a" * 64, "", "a" * 63, "a" * 65,
     "A" * 64, "g" * 64, " " + "a" * 64, "a" * 64 + "\n", [], {}],
    ids=["null", "bool", "integer", "bytes", "empty", "short", "long",
         "uppercase", "non-hex", "space", "newline", "list", "mapping"],
)
def test_digest_is_lowercase_64_hex_characters(tmp_path, digest):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    fields = {**_binding(issued), "request_sha256": digest}
    _assert_rejected(runtime, issued, fields, error=ValueError)


def test_new_session_issues_v2_binding_and_prepare_replays_without_writes(tmp_path):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    assert issued["schema_version"] == V2
    assert type(issued["revision"]) is int
    assert issued["revision"] == 0
    assert re.fullmatch(r"[0-9a-f]{64}", issued["request_sha256"])
    before = _snapshot(runtime, issued["session_id"])
    assert _prepare(runtime) == issued
    assert _snapshot(runtime, issued["session_id"]) == before


@pytest.mark.parametrize("mismatch", ["digest", "future-revision", "other-session"])
def test_well_formed_but_unissued_binding_conflicts_without_writes(tmp_path, mismatch):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    fields = _binding(issued)
    if mismatch == "digest":
        digest = issued["request_sha256"]
        fields["request_sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    elif mismatch == "future-revision":
        fields["expected_revision"] += 1
    else:
        other = _prepare(runtime, key="another-session")
        assert other["revision"] == issued["revision"]
        assert other["request_sha256"] != issued["request_sha256"]
        fields = _binding(other)
    _assert_rejected(runtime, issued, fields)


@pytest.mark.parametrize("stale_part", ["both", "revision", "digest"])
def test_delayed_reply_under_new_key_cannot_advance_new_request(tmp_path, stale_part):
    runtime = _runtime(tmp_path)
    original = _prepare(runtime)
    current = _deliver(runtime, original)
    assert current["status"] == "awaiting_model"
    assert current["revision"] == original["revision"] + 1
    fields = _binding(current)
    if stale_part in ("both", "revision"):
        fields["expected_revision"] = original["revision"]
    if stale_part in ("both", "digest"):
        fields["request_sha256"] = original["request_sha256"]

    # MODEL_OUTPUT is exactly the delayed first-round reply. A new delivery key
    # must not turn it into a response to the recovered-evidence request.
    _assert_rejected(runtime, current, fields, key="delayed-old-reply-new-key")
    assert runtime.inspect(original["session_id"]) == current


def test_fresh_recovery_binding_preserves_exact_evidence_and_prefix(tmp_path):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    recovered = _deliver(runtime, issued)
    assert recovered["status"] == "awaiting_model"
    assert recovered["round"]["decision"] == "recover_and_retry"
    assert recovered["revision"] == issued["revision"] + 1
    assert re.fullmatch(r"[0-9a-f]{64}", recovered["request_sha256"])
    assert recovered["request_sha256"] != issued["request_sha256"]
    first_request = issued["request"]
    next_request = recovered["request"]
    prefix = first_request["stable_context_prefix"].encode("utf-8")
    assert next_request["stable_context_prefix"].encode("utf-8") == prefix
    assert next_request["full_context"].encode("utf-8") == (
        prefix + next_request["appended_evidence"].encode("utf-8")
    )
    assert next_request["previous_output"] == MODEL_OUTPUT
    assert next_request["remaining_recovery_tokens"] < first_request[
        "remaining_recovery_tokens"
    ]

    # Compare actual recovered bytes with the immutable Context Commit bundle,
    # not just the verifier's boolean or a substring chosen to look supported.
    payload = json.loads(_session_path(runtime, issued["session_id"]).read_text())
    prepared = payload["session"]["prepared"]
    commit = json.loads(Path(prepared["commit_path"]).read_text(encoding="utf-8"))
    chunks = payload["session"]["recovered_chunks"]
    assert chunks
    assert [chunk["chunk_id"] for chunk in chunks] == recovered["recovered_chunk_ids"]
    for chunk in chunks:
        committed = commit["recovery_bundle"]["chunks"][chunk["chunk_id"]]
        assert chunk["text"].encode("utf-8") == committed["text"].encode("utf-8")
        assert chunk["fingerprint"] == committed["fingerprint"]
        assert chunk["verified"] is True
        assert chunk["text"] in next_request["appended_evidence"]

    terminal = _deliver(_runtime(tmp_path), recovered, key="delivery-1")
    assert terminal["status"] == "supported"
    assert terminal["revision"] == recovered["revision"] + 1
    assert terminal["request"] is None
    assert "request_sha256" in terminal
    assert terminal["request_sha256"] is None


def test_exact_old_tuple_replays_after_later_revision_and_restart(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    original_response = _deliver(runtime, issued, key="old-tuple")
    terminal = _deliver(runtime, original_response, key="later-tuple")
    assert terminal["revision"] == 2
    restarted = _runtime(tmp_path)
    before = _snapshot(restarted, issued["session_id"])

    def unexpected_work(*args, **kwargs):
        pytest.fail("exact replay performed new verification, recovery, or save")

    # These guards cannot accept a response: the real durable replay must serve
    # it, without calling the controller, verifier, recovery store, or writer.
    monkeypatch.setattr(
        "entroly.proof_guided_runtime.advance_proof_guided_fixed_point", unexpected_work
    )
    monkeypatch.setattr(restarted.layer, "_verify_output_with_context", unexpected_work)
    monkeypatch.setattr(restarted.layer, "recover", unexpected_work)
    monkeypatch.setattr(restarted, "_save", unexpected_work)
    replay = _deliver(restarted, issued, key="old-tuple")
    assert replay == original_response
    assert replay["revision"] == 1
    assert restarted.inspect(issued["session_id"]) == terminal
    assert _snapshot(restarted, issued["session_id"]) == before


@pytest.mark.parametrize("changed", ["output", "revision", "digest", "both"])
def test_same_key_rejects_changed_output_or_binding(tmp_path, changed):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    current = _deliver(runtime, issued, key="accepted-key")
    fields = _binding(issued)
    output = MODEL_OUTPUT
    if changed == "output":
        output += " Different reply."
    if changed in ("revision", "both"):
        fields["expected_revision"] = current["revision"]
    if changed in ("digest", "both"):
        fields["request_sha256"] = current["request_sha256"]
    _assert_rejected(
        _runtime(tmp_path), issued, fields, key="accepted-key", output=output
    )


def test_same_key_rejects_newline_only_change_in_output_bytes(tmp_path):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    output = MODEL_OUTPUT + "\n"
    _deliver(runtime, issued, key="newline-key", output=output)
    _assert_rejected(
        _runtime(tmp_path), issued, _binding(issued), key="newline-key",
        output=output.replace("\n", "\r\n"),
    )


def test_boolean_revision_cannot_bypass_validation_through_exact_key_replay(tmp_path):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    _deliver(runtime, issued, key="accepted-key")
    fields = {**_binding(issued), "expected_revision": False}
    _assert_rejected(runtime, issued, fields, error=ValueError, key="accepted-key")


def _write_v1_snapshot(runtime, issued):
    """Build the historical v1 wire layout with its real unkeyed state seal."""
    path = _session_path(runtime, issued["session_id"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = V1
    payload["last_response"]["schema_version"] = V1
    payload["last_response"].pop("request_sha256", None)
    for record in payload["advance_idempotency"].values():
        record.pop("expected_revision", None)
        record.pop("request_sha256", None)
        record["response"]["schema_version"] = V1
        record["response"].pop("request_sha256", None)
    payload.pop("state_sha256")
    payload["state_sha256"] = stable_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return copy.deepcopy(payload["last_response"])


@pytest.mark.parametrize("with_history", [False, True], ids=["prepared", "has-replay"])
def test_v1_is_inspect_only_and_prepare_key_requires_explicit_reprepare(
    tmp_path, with_history
):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    if with_history:
        _deliver(runtime, issued, key="historical-key")
    historical = _write_v1_snapshot(runtime, issued)
    restarted = _runtime(tmp_path)
    before = _snapshot(restarted, issued["session_id"])
    assert restarted.inspect(issued["session_id"]) == historical
    assert "request_sha256" not in historical
    with pytest.raises(ProofGuidedSessionConflict, match=r"(?i)legacy.*re.prepare"):
        restarted.advance(
            issued["session_id"],
            model_output=MODEL_OUTPUT,
            idempotency_key="historical-key",
        )
    with pytest.raises(ProofGuidedSessionConflict, match=r"(?i)legacy.*re.prepare"):
        _deliver(restarted, issued, key="historical-key")
    with pytest.raises(ProofGuidedSessionConflict, match=r"(?i)legacy.*re.prepare"):
        _prepare(restarted)
    assert _snapshot(restarted, issued["session_id"]) == before

    replacement = _prepare(restarted, key="explicit-reprepare")
    assert replacement["schema_version"] == V2
    assert replacement["session_id"] != issued["session_id"]
    assert re.fullmatch(r"[0-9a-f]{64}", replacement["request_sha256"])
    assert _session_path(restarted, issued["session_id"]).read_bytes() == before[0]


def _racing_delivery(root, issued, index, start_gate, loaded_events, results):
    """Spawn-safe worker: execute the real runtime, with a post-load race probe."""
    socket.socket.connect = _deny_network
    socket.create_connection = _deny_network
    try:
        runtime = _runtime(root)
        original_load = runtime._load

        def load_with_overlap(session_id):
            payload = original_load(session_id)
            if payload["revision"] == issued["revision"]:
                loaded_events[index].set()
                # With the real lock the peer cannot load until this writer
                # commits. Without it, both stale snapshots are now in flight.
                loaded_events[1 - index].wait(timeout=0.75)
            return payload

        runtime._load = load_with_overlap
        start_gate.wait(timeout=15)
        response = _deliver(runtime, issued, key=f"process-{index}")
        results.put({"index": index, "kind": "committed", "response": response})
    except ProofGuidedSessionConflict as exc:
        results.put({"index": index, "kind": "conflict", "error": str(exc)})
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        results.put({"index": index, "kind": "error", "error": error})


def test_independent_processes_commit_once_and_exact_key_replays(tmp_path):
    runtime = _runtime(tmp_path)
    issued = _prepare(runtime)
    context = multiprocessing.get_context("spawn")
    start_gate = context.Barrier(3)
    loaded_events = [context.Event(), context.Event()]
    results = context.Queue()
    processes = [
        context.Process(
            target=_racing_delivery,
            args=(str(tmp_path), issued, index, start_gate, loaded_events, results),
        )
        for index in range(2)
    ]
    try:
        for process in processes:
            process.start()
        # Both independently initialized workers must reach the gate before
        # either is allowed to submit, irrespective of import/startup latency.
        start_gate.wait(timeout=20)
        outcomes = [results.get(timeout=25) for _ in processes]
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0, outcomes
        assert sorted(item["kind"] for item in outcomes) == ["committed", "conflict"], outcomes
        winner = next(item for item in outcomes if item["kind"] == "committed")
        restarted = _runtime(tmp_path)
        current = restarted.inspect(issued["session_id"])
        assert current == winner["response"]
        assert current["revision"] == issued["revision"] + 1
        payload = json.loads(_session_path(restarted, issued["session_id"]).read_text())
        assert len(payload["session"]["rounds"]) == 1
        assert list(payload["advance_idempotency"]) == [f"process-{winner['index']}"]
        before = _snapshot(restarted, issued["session_id"])
        assert _deliver(restarted, issued, key=f"process-{winner['index']}") == current
        assert _snapshot(restarted, issued["session_id"]) == before
        loser = next(item for item in outcomes if item["kind"] == "conflict")
        _assert_rejected(restarted, issued, _binding(issued), key=f"process-{loser['index']}")
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)
        results.close()
        results.join_thread()
