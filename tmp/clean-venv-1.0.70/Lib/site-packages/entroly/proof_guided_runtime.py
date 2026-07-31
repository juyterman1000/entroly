"""Durable host adapter for Entroly's proof-guided context fixed point.

The core fixed-point controller accepts a Python model callback.  Product
surfaces such as MCP, HTTP sidecars, and OpenClaw instead own an asynchronous
provider transport.  This module bridges those worlds with an idempotent,
restart-safe prepare/advance protocol.  It stores context locally, never stores
provider credentials, and never performs a network call.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .context_fixed_point import (
    EvidenceCandidate,
    EvidenceObligation,
    EvidencePlan,
    FixedPointModelRequest,
    FixedPointRound,
    FixedPointSession,
    advance_proof_guided_fixed_point,
    start_proof_guided_fixed_point,
)
from .context_receipts.models import stable_hash, stable_json, text_fingerprint
from .verified_efficiency import (
    AuditArtifact,
    PreparedContext,
    VerifiedEfficiencyLayer,
    VerifiedOutput,
)

RUNTIME_SCHEMA = "entroly.proof-guided-runtime.v1"
_SESSION_RE = re.compile(r"pgs_[0-9a-f]{24}")
_TERMINAL_STATES = frozenset(
    {
        "supported",
        "no_verifiable_claims",
        "no_supporting_omitted_evidence",
        "no_omitted_evidence",
        "recovery_budget_exhausted",
        "max_rounds_reached",
    }
)


class ProofGuidedRuntimeError(RuntimeError):
    """Base class for visible and actionable runtime failures."""


class ProofGuidedSessionNotFound(ProofGuidedRuntimeError):
    """A requested session does not exist in the configured local store."""


class ProofGuidedSessionConflict(ProofGuidedRuntimeError):
    """An idempotency key or session state conflicts with prior durable state."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProofGuidedSessionConflict(f"invalid persisted {field}")
    return dict(value)


def _audit(value: Any) -> AuditArtifact:
    item = _mapping(value, "audit artifact")
    return AuditArtifact(
        artifact_id=str(item.get("artifact_id", "")),
        path=str(item.get("path", "")),
        attestation=_mapping(item.get("attestation"), "audit attestation"),
    )


def _prepared(value: Any) -> PreparedContext:
    item = _mapping(value, "prepared context")
    selected = item.get("selected_context")
    scans = item.get("security_scans")
    warnings = item.get("warnings")
    if not isinstance(selected, list) or not isinstance(scans, list) or not isinstance(
        warnings, list
    ):
        raise ProofGuidedSessionConflict("invalid persisted prepared-context arrays")
    return PreparedContext(
        schema_version=str(item.get("schema_version", "")),
        commit_id=str(item.get("commit_id", "")),
        context=str(item.get("context", "")),
        selected_context=tuple(_mapping(part, "selected context") for part in selected),
        receipt=_mapping(item.get("receipt"), "context receipt"),
        security_scans=tuple(_mapping(part, "security scan") for part in scans),
        integrity_chain=_mapping(item.get("integrity_chain"), "integrity chain"),
        warnings=tuple(str(part) for part in warnings),
        commit_path=str(item.get("commit_path", "")),
        audit=_audit(item.get("audit")),
    )


def _request(value: Any) -> FixedPointModelRequest:
    item = _mapping(value, "model request")
    recovered = item.get("recovered_chunk_ids")
    if not isinstance(recovered, list):
        raise ProofGuidedSessionConflict("invalid persisted recovered chunk IDs")
    return FixedPointModelRequest(
        schema_version=str(item.get("schema_version", "")),
        round_index=int(item.get("round_index", -1)),
        query=str(item.get("query", "")),
        stable_context_prefix=str(item.get("stable_context_prefix", "")),
        appended_evidence=str(item.get("appended_evidence", "")),
        full_context=str(item.get("full_context", "")),
        previous_output=(
            None if item.get("previous_output") is None else str(item["previous_output"])
        ),
        recovered_chunk_ids=tuple(str(part) for part in recovered),
        remaining_recovery_tokens=int(item.get("remaining_recovery_tokens", -1)),
        previous_round_artifact_id=(
            None
            if item.get("previous_round_artifact_id") is None
            else str(item["previous_round_artifact_id"])
        ),
    )


def _verified_output(value: Any) -> VerifiedOutput:
    item = _mapping(value, "verified output")
    recovered = item.get("recovered_chunk_ids", [])
    if not isinstance(recovered, list):
        raise ProofGuidedSessionConflict("invalid verified-output recovery IDs")
    return VerifiedOutput(
        commit_id=str(item.get("commit_id", "")),
        output=str(item.get("output", "")),
        original_output=str(item.get("original_output", "")),
        changed=bool(item.get("changed")),
        suppression=_mapping(item.get("suppression"), "suppression result"),
        security_scan=_mapping(item.get("security_scan"), "output security scan"),
        audit=_audit(item.get("audit")),
        grounding_context_hash=str(item.get("grounding_context_hash", "")),
        recovered_chunk_ids=tuple(str(part) for part in recovered),
    )


def _obligation(value: Any) -> EvidenceObligation:
    item = _mapping(value, "evidence obligation")
    terms = item.get("terms")
    if not isinstance(terms, list):
        raise ProofGuidedSessionConflict("invalid evidence-obligation terms")
    return EvidenceObligation(
        claim_id=str(item.get("claim_id", "")),
        claim_text=str(item.get("claim_text", "")),
        claim_hash=str(item.get("claim_hash", "")),
        decision=str(item.get("decision", "")),
        phi=float(item.get("phi", 0.0)),
        weight=float(item.get("weight", 0.0)),
        terms=tuple(str(part) for part in terms),
    )


def _candidate(value: Any) -> EvidenceCandidate:
    item = _mapping(value, "evidence candidate")
    claims = item.get("supported_claim_ids")
    if not isinstance(claims, list):
        raise ProofGuidedSessionConflict("invalid evidence-candidate claim IDs")
    return EvidenceCandidate(
        chunk_id=str(item.get("chunk_id", "")),
        source_path=str(item.get("source_path", "")),
        fingerprint=str(item.get("fingerprint", "")),
        token_count=int(item.get("token_count", 0)),
        expected_error_reduction=float(item.get("expected_error_reduction", 0.0)),
        utility_per_token=float(item.get("utility_per_token", 0.0)),
        supported_claim_ids=tuple(str(part) for part in claims),
    )


def _plan(value: Any) -> EvidencePlan | None:
    if value is None:
        return None
    item = _mapping(value, "evidence plan")
    obligations = item.get("obligations")
    candidates = item.get("candidates")
    selected = item.get("selected")
    if not all(isinstance(part, list) for part in (obligations, candidates, selected)):
        raise ProofGuidedSessionConflict("invalid persisted evidence plan")
    return EvidencePlan(
        obligations=tuple(_obligation(part) for part in obligations),
        candidates=tuple(_candidate(part) for part in candidates),
        selected=tuple(_candidate(part) for part in selected),
        eligible_candidate_count=int(item.get("eligible_candidate_count", 0)),
        candidate_limit_applied=bool(item.get("candidate_limit_applied")),
        token_budget=int(item.get("token_budget", 0)),
        selected_tokens=int(item.get("selected_tokens", 0)),
        expected_error_reduction=float(item.get("expected_error_reduction", 0.0)),
    )


def _round(value: Any) -> FixedPointRound:
    item = _mapping(value, "fixed-point round")
    obligations = item.get("obligations")
    recovered = item.get("recovered_for_next_round")
    if not isinstance(obligations, list) or not isinstance(recovered, list):
        raise ProofGuidedSessionConflict("invalid persisted fixed-point round")
    return FixedPointRound(
        round_index=int(item.get("round_index", -1)),
        request=_request(item.get("request")),
        verified_output=_verified_output(item.get("verified_output")),
        obligations=tuple(_obligation(part) for part in obligations),
        recovery_plan=_plan(item.get("recovery_plan")),
        recovered_for_next_round=tuple(
            _mapping(part, "recovered chunk") for part in recovered
        ),
        decision=str(item.get("decision", "")),
        audit=_audit(item.get("audit")),
    )


def _session(value: Any) -> FixedPointSession:
    item = _mapping(value, "fixed-point session")
    recovered_chunks = item.get("recovered_chunks")
    recovered_ids = item.get("recovered_chunk_ids")
    rounds = item.get("rounds")
    if not all(isinstance(part, list) for part in (recovered_chunks, recovered_ids, rounds)):
        raise ProofGuidedSessionConflict("invalid persisted fixed-point session arrays")
    return FixedPointSession(
        prepared=_prepared(item.get("prepared")),
        max_rounds=int(item.get("max_rounds", 0)),
        recovery_token_budget=int(item.get("recovery_token_budget", -1)),
        max_chunks_per_round=int(item.get("max_chunks_per_round", 0)),
        profile=str(item.get("profile", "")),
        round_index=int(item.get("round_index", -1)),
        recovered_chunks=tuple(
            _mapping(part, "recovered chunk") for part in recovered_chunks
        ),
        recovered_chunk_ids=tuple(str(part) for part in recovered_ids),
        recovery_tokens_used=int(item.get("recovery_tokens_used", -1)),
        previous_output=(
            None if item.get("previous_output") is None else str(item["previous_output"])
        ),
        previous_round_artifact_id=(
            None
            if item.get("previous_round_artifact_id") is None
            else str(item["previous_round_artifact_id"])
        ),
        rounds=tuple(_round(part) for part in rounds),
    )


def _json_value(value: Any) -> Any:
    """Normalize tuples and mapping subclasses to their durable JSON form."""
    return json.loads(stable_json(value))


def _request_payload(request: FixedPointModelRequest) -> dict[str, Any]:
    return _json_value(asdict(request))


def _round_payload(round_result: FixedPointRound) -> dict[str, Any]:
    plan = round_result.recovery_plan
    return _json_value({
        "round_index": round_result.round_index,
        "decision": round_result.decision,
        "verified_output": {
            "output": round_result.verified_output.output,
            "changed": round_result.verified_output.changed,
            "suppression": round_result.verified_output.suppression,
            "security_scan": round_result.verified_output.security_scan,
            "audit_artifact_id": round_result.verified_output.audit.artifact_id,
        },
        "obligations": [asdict(part) for part in round_result.obligations],
        "recovery_plan": (
            None
            if plan is None
            else {
                "selected": [asdict(part) for part in plan.selected],
                "selected_tokens": plan.selected_tokens,
                "expected_error_reduction": plan.expected_error_reduction,
                "eligible_candidate_count": plan.eligible_candidate_count,
                "candidate_limit_applied": plan.candidate_limit_applied,
            }
        ),
        "recovered_for_next_round": [
            {
                "chunk_id": part.get("chunk_id"),
                "source_path": part.get("source_path"),
                "fingerprint": part.get("fingerprint"),
                "verified": part.get("verified"),
                "recovery_artifact_id": part.get("recovery_artifact_id"),
            }
            for part in round_result.recovered_for_next_round
        ],
        "audit_artifact_id": round_result.audit.artifact_id,
    })


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    serialized = stable_json(payload) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ProofGuidedRuntime:
    """Durable prepare/advance facade shared by CLI, MCP, proxy, and plugins."""

    def __init__(
        self,
        state_dir: str | Path = ".entroly/proof-guided",
        *,
        security_mode: str = "block",
        context_risk_mode: str = "block_high",
        prefer_rust: bool = True,
    ) -> None:
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.sessions_dir = self.state_dir / "sessions"
        self.layer = VerifiedEfficiencyLayer(
            self.state_dir / "verified-efficiency",
            security_mode=security_mode,
            context_risk_mode=context_risk_mode,
            prefer_rust=prefer_rust,
        )
        self._lock = threading.RLock()

    @staticmethod
    def _idempotency_key(value: str | None, *, required: bool) -> str | None:
        if value is None:
            if required:
                raise ValueError("idempotency_key is required for a durable advance")
            return None
        normalized = str(value).strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("idempotency_key must contain 1 to 128 characters")
        return normalized

    def _path(self, session_id: str) -> Path:
        normalized = str(session_id)
        if _SESSION_RE.fullmatch(normalized) is None:
            raise ValueError("invalid proof-guided session ID")
        return self.sessions_dir / f"{normalized}.json"

    @staticmethod
    def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
        sealed = dict(payload)
        sealed.pop("state_sha256", None)
        sealed["state_sha256"] = stable_hash(sealed)
        return sealed

    def _load(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProofGuidedSessionNotFound(
                f"proof-guided session {session_id!r} was not found"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ProofGuidedSessionConflict(
                f"proof-guided session {session_id!r} cannot be read: {exc}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != RUNTIME_SCHEMA:
            raise ProofGuidedSessionConflict("proof-guided session schema is unsupported")
        expected = self._sealed(payload).get("state_sha256")
        if payload.get("state_sha256") != expected:
            raise ProofGuidedSessionConflict("proof-guided session state hash mismatch")
        if payload.get("session_id") != session_id:
            raise ProofGuidedSessionConflict("proof-guided session identity mismatch")
        return payload

    def _save(self, payload: Mapping[str, Any]) -> None:
        session_id = str(payload.get("session_id", ""))
        _atomic_write(self._path(session_id), self._sealed(payload))

    def prepare(
        self,
        documents: Iterable[tuple[str, str]],
        *,
        query: str,
        token_budget: int,
        max_rounds: int = 3,
        recovery_token_budget: int = 1_200,
        max_chunks_per_round: int = 3,
        profile: str = "rag",
        chunk_tokens: int = 360,
        overlap_tokens: int = 32,
        parent_commit_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        docs = [(str(path), str(text)) for path, text in documents]
        key = self._idempotency_key(idempotency_key, required=False)
        request_hash = stable_hash(
            {
                "documents": docs,
                "query": str(query),
                "token_budget": token_budget,
                "max_rounds": max_rounds,
                "recovery_token_budget": recovery_token_budget,
                "max_chunks_per_round": max_chunks_per_round,
                "profile": profile,
                "chunk_tokens": chunk_tokens,
                "overlap_tokens": overlap_tokens,
                "parent_commit_id": parent_commit_id,
            }
        )
        session_id = (
            "pgs_" + stable_hash({"idempotency_key": key})[:24]
            if key is not None
            else "pgs_" + uuid.uuid4().hex[:24]
        )
        with self._lock:
            path = self._path(session_id)
            if path.exists():
                existing = self._load(session_id)
                if existing.get("prepare_request_hash") != request_hash:
                    raise ProofGuidedSessionConflict(
                        "prepare idempotency key was already used for a different request"
                    )
                return dict(existing["last_response"])

            prepared = self.layer.prepare(
                docs,
                query=query,
                token_budget=token_budget,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                parent_commit_id=parent_commit_id,
            )
            session, request = start_proof_guided_fixed_point(
                self.layer,
                prepared,
                max_rounds=max_rounds,
                recovery_token_budget=recovery_token_budget,
                max_chunks_per_round=max_chunks_per_round,
                profile=profile,
            )
            response = _json_value({
                "schema_version": RUNTIME_SCHEMA,
                "session_id": session_id,
                "revision": 0,
                "status": "awaiting_model",
                "commit_id": prepared.commit_id,
                "prepared_audit_artifact_id": prepared.audit.artifact_id,
                "request": _request_payload(request),
                "warnings": list(prepared.warnings),
                "local_only": True,
                "provider_call_performed": False,
            })
            payload = {
                "schema_version": RUNTIME_SCHEMA,
                "session_id": session_id,
                "revision": 0,
                "prepare_request_hash": request_hash,
                "session": asdict(session),
                "advance_idempotency": {},
                "last_response": response,
            }
            self._save(payload)
            return response

    def advance(
        self,
        session_id: str,
        *,
        model_output: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        key = self._idempotency_key(idempotency_key, required=True)
        if not isinstance(model_output, str):
            raise ValueError("model_output must be a string")
        output_hash = text_fingerprint(model_output)
        with self._lock:
            payload = self._load(session_id)
            history = _mapping(
                payload.get("advance_idempotency", {}),
                "advance idempotency history",
            )
            replay = history.get(key)
            if replay is not None:
                replay_item = _mapping(replay, "advance idempotency record")
                if replay_item.get("output_sha256") != output_hash:
                    raise ProofGuidedSessionConflict(
                        "advance idempotency key was replayed with different model output"
                    )
                return _mapping(replay_item.get("response"), "idempotent response")
            previous_response = _mapping(payload.get("last_response"), "last response")
            if previous_response.get("status") in _TERMINAL_STATES:
                raise ProofGuidedSessionConflict(
                    f"proof-guided session is already terminal: {previous_response['status']}"
                )
            session = _session(payload.get("session"))
            step = advance_proof_guided_fixed_point(
                self.layer,
                session,
                model_output,
            )
            revision = int(payload.get("revision", 0)) + 1
            terminal = step.result is not None
            status = step.result.status if terminal else "awaiting_model"
            response = _json_value({
                "schema_version": RUNTIME_SCHEMA,
                "session_id": session_id,
                "revision": revision,
                "status": status,
                "converged": bool(step.result and step.result.converged),
                "round": _round_payload(step.round),
                "request": (
                    None
                    if step.next_request is None
                    else _request_payload(step.next_request)
                ),
                "final_output": (
                    None if step.result is None else step.result.final_output.output
                ),
                "recovered_chunk_ids": list(step.session.recovered_chunk_ids),
                "recovery_tokens_used": step.session.recovery_tokens_used,
                "recovery_token_budget": step.session.recovery_token_budget,
                "local_only": True,
                "provider_call_performed": False,
            })
            payload.update(
                {
                    "revision": revision,
                    "session": asdict(step.session),
                    "last_response": response,
                }
            )
            history[key] = {
                "output_sha256": output_hash,
                "revision": revision,
                "response": response,
            }
            # The controller permits at most eight rounds, but keep a larger
            # bounded window for forward-compatible retries and migration data.
            while len(history) > 32:
                history.pop(next(iter(history)))
            payload["advance_idempotency"] = history
            self._save(payload)
            return response

    def inspect(self, session_id: str) -> dict[str, Any]:
        """Return the last durable response without advancing the session."""
        with self._lock:
            return dict(self._load(session_id)["last_response"])


__all__ = [
    "ProofGuidedRuntime",
    "ProofGuidedRuntimeError",
    "ProofGuidedSessionConflict",
    "ProofGuidedSessionNotFound",
    "RUNTIME_SCHEMA",
]
