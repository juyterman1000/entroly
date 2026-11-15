"""Host-authenticated acceptance of an existing Work Graph inbox receipt."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .context_receipts.models import stable_hash, stable_json
from .local_file_lock import exclusive_file_lock
from .receipt_attestation import AttestedReceipt, verify_attestation
from .work_graph_inbox import _inbox, _MAX_HANDOFF_BYTES
from .work_graph_resume import workspace_commitment
from .work_graph_review import (
    _graph_source_state,
    _handoff_graph,
    _observation,
    _source_state,
)
from .work_graph_store import WorkGraphStore

_AUDIENCE = "entroly.work-graph.handoff-acceptance.v1"
_MAX_ASSERTION_BYTES = 16 * 1024
_MAX_TTL_MS = 5 * 60 * 1000


def _read_object(path: Path, limit: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected a regular handoff record")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("handoff record exceeds its byte limit")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("handoff record must be a JSON object")
    return value


def _reference(ref: str) -> str:
    if not re.fullmatch(r"handoff:[0-9a-f]{64}", ref):
        raise ValueError("expected a handoff SHA-256 reference")
    return ref.removeprefix("handoff:")


def _live_binding(root: Path) -> tuple[dict[str, Any], str]:
    observation = _observation(root)
    disclosure = observation["path_disclosure"]
    branch = observation["branch"]
    if (
        not branch["head_sha"]
        or branch["merge_in_progress"]
        or branch["rebase_in_progress"]
        or disclosure["generated_omitted"]
        or disclosure["sensitive_withheld"]
        or any(not change.get("content_digest") for change in observation["changes"])
    ):
        raise ValueError(
            "workspace contains changes that cannot be bound to this handoff"
        )
    return observation, workspace_commitment(observation)


def challenge(root: Path, store: WorkGraphStore, ref: str) -> dict[str, Any]:
    """Return the exact fields a trusted host must sign for this workspace."""
    digest = _reference(ref)
    inbox = _read_object(_inbox(store) / f"{digest}.json", _MAX_HANDOFF_BYTES)
    handoff, proof = inbox.get("handoff"), inbox.get("continuation_proof")
    if (
        inbox.get("schema_version") != 1
        or not isinstance(handoff, dict)
        or not isinstance(proof, dict)
    ):
        raise ValueError("invalid Work Graph inbox record")
    ref_digest = hashlib.sha256(
        str(handoff.get("payload_commitment")).encode()
    ).hexdigest()
    if ref_digest != digest:
        raise ValueError("handoff reference does not match its receipt")
    graph = store.load()
    if handoff.get("graph_revision") != graph.revision:
        raise ValueError("handoff graph revision is no longer current")
    sealed = _handoff_graph(graph, handoff)
    from entroly_core import work_continuation_proof_verify_json

    verified = json.loads(work_continuation_proof_verify_json(stable_json(proof)))
    proof_binding = {
        "repository_id": "repo_id",
        "graph_revision": "graph_revision",
        "graph_commitment": "graph_commitment",
        "workstream_id": "workstream_id",
        "from_agent": "from_agent",
        "to_agent": "to_agent",
        "handoff_commitment": "payload_commitment",
    }
    if verified != proof or any(
        verified.get(field) != handoff[receipt_field]
        for field, receipt_field in proof_binding.items()
    ):
        raise ValueError("continuation proof does not match the handoff receipt")
    observation, live = _live_binding(root)
    if _source_state(store.repo_id, observation) != _graph_source_state(
        sealed, store.repo_id
    ):
        raise ValueError("handoff is stale against current repository source")
    return {
        "audience": _AUDIENCE,
        "handoff_ref": ref,
        "repo_id": store.repo_id,
        "workspace_id": stable_hash(str(root.resolve())),
        "workspace_commitment": live,
        "workstream_id": handoff["workstream_id"],
        "successor_id": handoff["to_agent"],
        "handoff_commitment": handoff["payload_commitment"],
        "continuation_sha256": stable_hash(proof),
    }


def accept(
    root: Path, store: WorkGraphStore, ref: str, assertion_path: Path
) -> dict[str, Any]:
    """Validate and record one acceptance. The OS account owns storage and config."""
    issuer = os.environ.get("ENTROLY_HANDOFF_ISSUER", "")
    key = os.environ.get("ENTROLY_HANDOFF_PUBLIC_KEY", "")
    if not issuer or not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("configure the trusted handoff issuer and Ed25519 public key")
    document = _read_object(assertion_path, _MAX_ASSERTION_BYTES)
    try:
        assertion = AttestedReceipt.from_dict(document)
        valid = (
            isinstance(assertion.receipt, dict)
            and isinstance(assertion.signature, str)
            and isinstance(assertion.public_key, str)
            and type(assertion.sequence) is int
            and type(assertion.signed_at) in {int, float}
            and verify_attestation(assertion, public_key=key)
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("host handoff assertion failed signature verification")

    digest = _reference(ref)
    folder = _inbox(store)
    target = folder / f"{digest}.accepted"
    with exclusive_file_lock(folder / "acceptance.lock", timeout=5):
        expected = challenge(root, store, ref)
        claims = assertion.receipt
        issued_at, expires_at = claims.get("issued_at_ms"), claims.get("expires_at_ms")
        if claims != {
            **expected,
            "issuer": issuer,
            "issued_at_ms": issued_at,
            "expires_at_ms": expires_at,
        }:
            raise ValueError(
                "host assertion does not match the handoff and current workspace"
            )
        if (
            type(issued_at) is not int
            or type(expires_at) is not int
            or not 0 < expires_at - issued_at <= _MAX_TTL_MS
            or not issued_at <= time.time_ns() // 1_000_000 < expires_at
        ):
            raise ValueError("host assertion is outside its validity window")
        result = {
            "schema_version": 1,
            **expected,
            "issuer": issuer,
            "assertion_sha256": stable_hash(document),
            "decision": "accepted",
            "verification_evidence": "revalidate_before_reuse",
        }
        if target.exists() or target.is_symlink():
            if _read_object(target, _MAX_ASSERTION_BYTES) != result:
                raise ValueError(
                    "a different acceptance already exists for this handoff"
                )
        else:
            if challenge(root, store, ref) != expected:
                raise ValueError("workspace changed during handoff acceptance")
            _write_receipt(target, result)
    return result


def _write_receipt(target: Path, result: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(stable_json(result).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
