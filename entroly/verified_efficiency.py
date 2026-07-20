"""A fail-closed, proof-carrying efficiency layer for AI context.

``VerifiedEfficiencyLayer`` composes Entroly's existing context selection,
security, recovery, hallucination suppression, attestation, and verified world
model primitives behind one narrow contract. It does not make an LLM call and
does not treat model-generated or self-reported outcomes as learning evidence.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .context_commit import (
    create_context_commit,
    replay_context,
    verify_context_commit,
)
from .context_firewall import IntegrityChain, ScanResult, scan
from .context_receipts import recover_omitted
from .context_receipts.models import stable_hash, stable_json, text_fingerprint
from .eicv_suppressor import EICVSuppressor, SuppressionResult
from .receipt_attestation import (
    AttestationKey,
    AttestationLog,
    AttestedReceipt,
    verify_attestation,
)

_SCHEMA_VERSION = "entroly.verified-efficiency.v1"
_AUDIT_SCHEMA_VERSION = "entroly.verified-efficiency-audit.v1"
_SECURITY_MODES = frozenset({"block", "audit"})
_RISK_MODES = frozenset({"block_high", "audit"})
_OUTPUT_MODES = frozenset({"strict", "annotate", "audit"})
_REVIEW_RANK = {"low": 0.0, "medium": 0.5, "high": 1.0}


class EfficiencyLayerError(RuntimeError):
    """Base class for visible, actionable efficiency-layer failures."""


class AuditUnavailableError(EfficiencyLayerError):
    """The local signing or audit store could not be initialized safely."""


class UnsafeContextError(EfficiencyLayerError):
    """Untrusted content failed the pre-LLM security policy."""

    def __init__(self, message: str, scans: Sequence[dict[str, Any]]) -> None:
        super().__init__(message)
        self.scans = tuple(scans)


class ContextRiskError(EfficiencyLayerError):
    """Context selection needs more evidence or explicit human review."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


class VerificationFailureError(EfficiencyLayerError):
    """Output verification failed, so no unchecked output was returned."""


class RecoveryIntegrityError(EfficiencyLayerError):
    """Omitted context could not be recovered with matching integrity proof."""


class EvolutionEvidenceError(EfficiencyLayerError):
    """An outcome was not strong enough to train the verified world model."""


@dataclass(frozen=True)
class AuditArtifact:
    """A locally persisted, content-addressed, Ed25519-signed audit record."""

    artifact_id: str
    path: str
    attestation: dict[str, Any]


@dataclass(frozen=True)
class AuditVerification:
    valid: bool
    checks: dict[str, bool]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class PreparedContext:
    """Verified context ready to be sent to a model."""

    schema_version: str
    commit_id: str
    context: str
    selected_context: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]
    security_scans: tuple[dict[str, Any], ...]
    integrity_chain: dict[str, Any]
    warnings: tuple[str, ...]
    commit_path: str
    audit: AuditArtifact


@dataclass(frozen=True)
class VerifiedOutput:
    """An output checked against the exact context delivered to the model."""

    commit_id: str
    output: str
    original_output: str
    changed: bool
    suppression: dict[str, Any]
    security_scan: dict[str, Any]
    audit: AuditArtifact
    grounding_context_hash: str = ""
    recovered_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveredContext:
    """Fingerprint-verified omitted context plus a signed recovery record."""

    commit_id: str
    chunks: tuple[dict[str, Any], ...]
    audit: AuditArtifact


@dataclass(frozen=True)
class LearningReceipt:
    """A durable receipt for a real, externally verified learning update."""

    transition_id: str
    receipt_hash: str
    previous_hash: str
    ledger_path: str
    idempotent_replay: bool
    context_commit_id: str
    output_audit_id: str


def _scan_summary(result: ScanResult) -> dict[str, Any]:
    """Return deterministic scan evidence; runtime latency is diagnostic only."""
    return {
        "is_safe": result.is_safe,
        "threats": [asdict(threat) for threat in result.threats],
        "n_critical": result.n_critical,
        "n_high": result.n_high,
        "n_medium": result.n_medium,
        "n_low": result.n_low,
        "content_hash": result.content_hash,
    }


def _full_integrity_summary(chain: IntegrityChain) -> dict[str, Any]:
    return {
        "pipeline_id": chain.pipeline_id,
        "stages": [asdict(stage) for stage in chain.stages],
        "chain_valid": chain.verify(),
    }


def _suppression_result_consistent(result: SuppressionResult) -> bool:
    counts = (
        result.n_claims,
        result.n_supported,
        result.n_abstained,
        result.n_hallucinated,
        result.flagged_count,
        result.suppressed_count,
        result.warned_count,
    )
    return (
        all(isinstance(value, int) and value >= 0 for value in counts)
        and result.n_claims == len(result.certificates)
        and result.n_claims
        == result.n_supported + result.n_abstained + result.n_hallucinated
        and result.flagged_count == result.n_abstained + result.n_hallucinated
        and isinstance(result.rewritten_output, str)
        and isinstance(result.original_output, str)
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically persist JSON, returning False for an identical existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = stable_json(payload) + "\n"
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AuditUnavailableError(
                f"cannot read existing state file {path}: {exc}"
            ) from exc
        if existing == serialized:
            return False
        raise AuditUnavailableError(
            f"refusing to overwrite non-matching content-addressed state file: {path}"
        )

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditUnavailableError(
            f"cannot read verified state file {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AuditUnavailableError(f"verified state file is not a JSON object: {path}")
    return payload


class VerifiedEfficiencyLayer:
    """Secure, auditable context efficiency with verified-only evolution.

    The layer is local-first. ``prepare`` and ``verify_output`` never call a
    remote service. Learning occurs only when the caller explicitly supplies a
    strong external RAVS outcome to ``record_verified_outcome``.
    """

    def __init__(
        self,
        state_dir: str | Path = ".entroly/verified-efficiency",
        *,
        security_mode: str = "block",
        context_risk_mode: str = "block_high",
        prefer_rust: bool = True,
        world_model_controller: Any | None = None,
    ) -> None:
        if security_mode not in _SECURITY_MODES:
            raise ValueError(f"security_mode must be one of {sorted(_SECURITY_MODES)}")
        if context_risk_mode not in _RISK_MODES:
            raise ValueError(f"context_risk_mode must be one of {sorted(_RISK_MODES)}")
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.commits_dir = self.state_dir / "commits"
        self.audit_dir = self.state_dir / "audit"
        self.security_mode = security_mode
        self.context_risk_mode = context_risk_mode
        self.prefer_rust = prefer_rust
        self._lock = threading.RLock()
        self._world_model_controller = world_model_controller
        self._key = self._load_or_create_key()

    def _load_or_create_key(self) -> AttestationKey:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        key_path = self.state_dir / "attestation.key"
        if key_path.exists():
            try:
                return AttestationKey.from_private_hex(
                    key_path.read_text(encoding="ascii").strip()
                )
            except (OSError, ValueError, RuntimeError) as exc:
                raise AuditUnavailableError(
                    f"invalid local attestation key at {key_path}; restore the key "
                    "from backup instead of replacing it"
                ) from exc

        lock_path = self.state_dir / ".attestation-key.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            if key_path.exists():
                return self._load_or_create_key()
            raise AuditUnavailableError(
                f"attestation key initialization is already in progress ({lock_path}); "
                "retry after the other process finishes"
            ) from exc

        try:
            os.close(lock_fd)
            try:
                key = AttestationKey.generate()
            except RuntimeError as exc:
                raise AuditUnavailableError(
                    "signed audit receipts require the 'cryptography' package; "
                    "install Entroly with its full dependencies"
                ) from exc
            temporary = key_path.with_name(f".{key_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="ascii", newline="\n") as handle:
                    handle.write(key.private_hex() + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.chmod(temporary, 0o600)
                except OSError:
                    pass
                os.replace(temporary, key_path)
            finally:
                temporary.unlink(missing_ok=True)
            return key
        finally:
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _artifact_id(receipt: Mapping[str, Any]) -> str:
        payload = dict(receipt)
        payload.pop("artifact_id", None)
        return "vea_" + stable_hash(payload)[:24]

    def _persist_audit(self, payload: Mapping[str, Any]) -> AuditArtifact:
        receipt = dict(payload)
        receipt["schema_version"] = _AUDIT_SCHEMA_VERSION
        artifact_id = self._artifact_id(receipt)
        receipt["artifact_id"] = artifact_id
        path = self.audit_dir / f"{artifact_id}.json"

        with self._lock:
            if path.exists():
                return self._reuse_audit(path, receipt)

            self.audit_dir.mkdir(parents=True, exist_ok=True)
            lock_path = self.audit_dir / f".{artifact_id}.lock"
            try:
                lock_fd = os.open(
                    lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError as exc:
                if path.exists():
                    return self._reuse_audit(path, receipt)
                raise AuditUnavailableError(
                    f"audit artifact creation is already in progress ({lock_path}); "
                    "retry after the other process finishes"
                ) from exc

            try:
                os.close(lock_fd)
                if path.exists():
                    return self._reuse_audit(path, receipt)
                try:
                    entry = AttestationLog(self._key).append(receipt)
                except RuntimeError as exc:
                    raise AuditUnavailableError(
                        "could not sign the local audit receipt"
                    ) from exc
                artifact_payload = {
                    "schema_version": _AUDIT_SCHEMA_VERSION,
                    "artifact_id": artifact_id,
                    "attestation": entry.to_dict(),
                }
                _atomic_write_json(path, artifact_payload)
                return AuditArtifact(artifact_id, str(path), entry.to_dict())
            finally:
                lock_path.unlink(missing_ok=True)

    def _reuse_audit(
        self, path: Path, expected_receipt: Mapping[str, Any]
    ) -> AuditArtifact:
        artifact_payload = _read_json(path)
        verification = self.verify_audit_artifact(artifact_payload)
        if not verification.valid:
            raise AuditUnavailableError(
                f"existing audit artifact failed verification: {path}; "
                + ", ".join(verification.errors)
            )
        existing_entry = AttestedReceipt.from_dict(
            dict(artifact_payload["attestation"])
        )
        if stable_hash(existing_entry.receipt) != stable_hash(expected_receipt):
            raise AuditUnavailableError(
                f"audit artifact ID collision or tampering detected: {path}"
            )
        return AuditArtifact(
            str(artifact_payload["artifact_id"]), str(path), existing_entry.to_dict()
        )

    def verify_audit_artifact(
        self, artifact: AuditArtifact | Mapping[str, Any] | str | Path
    ) -> AuditVerification:
        artifact_snapshot: AuditArtifact | None = None
        if isinstance(artifact, AuditArtifact):
            artifact_snapshot = artifact
            payload = _read_json(Path(artifact.path))
        elif isinstance(artifact, (str, Path)):
            payload = _read_json(Path(artifact))
        else:
            payload = dict(artifact)

        attestation_raw = payload.get("attestation")
        if not isinstance(attestation_raw, Mapping):
            return AuditVerification(False, {"attestation": False}, ("attestation",))
        try:
            entry = AttestedReceipt.from_dict(dict(attestation_raw))
        except (TypeError, ValueError):
            return AuditVerification(False, {"attestation": False}, ("attestation",))

        expected_id = self._artifact_id(entry.receipt)
        expected_key = self._key.public_hex()
        checks = {
            "schema": payload.get("schema_version") == _AUDIT_SCHEMA_VERSION
            and entry.receipt.get("schema_version") == _AUDIT_SCHEMA_VERSION,
            "artifact_id": payload.get("artifact_id") == expected_id
            and entry.receipt.get("artifact_id") == expected_id,
            "signature": verify_attestation(entry, public_key=expected_key),
        }
        if artifact_snapshot is not None:
            checks["snapshot"] = (
                artifact_snapshot.artifact_id == expected_id
                and stable_hash(artifact_snapshot.attestation)
                == stable_hash(attestation_raw)
            )
        errors = tuple(name for name, passed in checks.items() if not passed)
        return AuditVerification(not errors, checks, errors)

    def _verified_audit_receipt(self, artifact: AuditArtifact) -> dict[str, Any]:
        payload = _read_json(Path(artifact.path))
        verification = self.verify_audit_artifact(artifact)
        if not verification.valid:
            raise VerificationFailureError(
                "audit artifact failed closed: " + ", ".join(verification.errors)
            )
        entry = AttestedReceipt.from_dict(dict(payload["attestation"]))
        if artifact.artifact_id != entry.receipt.get("artifact_id"):
            raise VerificationFailureError("audit artifact identity mismatch")
        return dict(entry.receipt)

    def prepare(
        self,
        documents: Iterable[tuple[str, str]],
        *,
        query: str,
        token_budget: int,
        chunk_tokens: int = 360,
        overlap_tokens: int = 32,
        parent_commit_id: str | None = None,
    ) -> PreparedContext:
        """Securely select, commit, verify, and sign model-ready context."""
        normalized_query = str(query).strip()
        if not normalized_query:
            raise ValueError(
                "query cannot be empty for intelligent context selection; "
                "use entroly.compress() for query-independent compression"
            )
        if not isinstance(token_budget, int) or isinstance(token_budget, bool):
            raise ValueError("token_budget must be a positive integer")
        if token_budget <= 0:
            raise ValueError("token_budget must be a positive integer")
        docs = [(str(path), str(text)) for path, text in documents]
        if not docs:
            raise ValueError("documents cannot be empty")
        if any(not path.strip() for path, _ in docs):
            raise ValueError("every document requires a non-empty source path")
        source_paths = [path for path, _ in docs]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError(
                "source paths must be unique so fingerprints and recovery "
                "remain unambiguous"
            )

        source_scans = tuple(
            _scan_summary(scan(text, source=path)) for path, text in docs
        )
        unsafe = [result for result in source_scans if not result["is_safe"]]
        if unsafe and self.security_mode == "block":
            raise UnsafeContextError(
                f"blocked {len(unsafe)} unsafe source(s) before compression; "
                "inspect scans and remove or explicitly review the detected content",
                source_scans,
            )

        commit = create_context_commit(
            docs,
            query=normalized_query,
            token_budget=token_budget,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            parent_commit_id=parent_commit_id,
            prefer_rust=self.prefer_rust,
        )
        commit_verification = verify_context_commit(commit)
        if not commit_verification.valid:
            raise VerificationFailureError(
                "context commit failed closed: " + ", ".join(commit_verification.errors)
            )

        selected = tuple(replay_context(commit))
        if not selected:
            raise ContextRiskError(
                "the token budget selected no context; increase the budget before "
                "sending this request to a model",
                commit.get("receipt", {}),
            )
        context = "\n\n".join(str(item.get("text", "")) for item in selected)
        selected_scan = _scan_summary(
            scan(context, source=f"{commit['commit_id']}:selected")
        )
        all_scans = (*source_scans, selected_scan)
        if not selected_scan["is_safe"] and self.security_mode == "block":
            raise UnsafeContextError(
                "selected context failed the post-compression security scan",
                all_scans,
            )

        receipt = dict(commit.get("receipt", {}))
        risk_summary = receipt.get("risk_summary", {})
        review_level = (
            str(risk_summary.get("review_level", "high"))
            if isinstance(risk_summary, Mapping)
            else "high"
        )
        if review_level == "high" and self.context_risk_mode == "block_high":
            raise ContextRiskError(
                "context receipt requires high review; increase the token budget, "
                "resolve missing dependencies, or explicitly use "
                "context_risk_mode='audit'",
                receipt,
            )

        source_manifest = {
            "documents": [
                {"source_path": path, "content_hash": result["content_hash"]}
                for (path, _), result in zip(docs, source_scans)
            ]
        }
        chain = IntegrityChain(pipeline_id=str(commit["commit_id"]))
        chain.record("source_manifest", stable_json(source_manifest))
        chain.record(
            "context_commit",
            stable_json(
                {
                    "commit_id": commit["commit_id"],
                    "selected_context_digest": commit["selected_context_digest"],
                    "recovery_bundle_digest": commit["recovery_bundle_digest"],
                }
            ),
        )
        chain.record("model_context", context)
        integrity = _full_integrity_summary(chain)
        if not integrity["chain_valid"]:
            raise VerificationFailureError("context integrity chain failed closed")

        commit_path = self.commits_dir / f"{commit['commit_id']}.json"
        created_commit = False
        try:
            created_commit = _atomic_write_json(commit_path, commit)
            audit = self._persist_audit(
                {
                    "artifact_type": "prepared_context",
                    "commit_id": commit["commit_id"],
                    "receipt_id": receipt.get("receipt_id"),
                    "receipt_hash": receipt.get("reproducibility_hash"),
                    "query_hash": text_fingerprint(normalized_query),
                    "security_mode": self.security_mode,
                    "context_risk_mode": self.context_risk_mode,
                    "security_scans": all_scans,
                    "integrity_chain": integrity,
                    "commit_verification": commit_verification.to_dict(),
                }
            )
        except Exception:
            if created_commit:
                commit_path.unlink(missing_ok=True)
            raise

        return PreparedContext(
            schema_version=_SCHEMA_VERSION,
            commit_id=str(commit["commit_id"]),
            context=context,
            selected_context=selected,
            receipt=receipt,
            security_scans=all_scans,
            integrity_chain=integrity,
            warnings=tuple(str(item) for item in receipt.get("warnings", [])),
            commit_path=str(commit_path),
            audit=audit,
        )

    def _load_prepared_commit(self, prepared: PreparedContext) -> dict[str, Any]:
        commit_path = Path(prepared.commit_path).resolve()
        if commit_path.parent != self.commits_dir.resolve():
            raise VerificationFailureError(
                "prepared context belongs to a different or untrusted state directory"
            )
        commit = _read_json(commit_path)
        verification = verify_context_commit(commit)
        if not verification.valid:
            raise VerificationFailureError(
                "persisted context commit failed closed: "
                + ", ".join(verification.errors)
            )
        if commit.get("commit_id") != prepared.commit_id:
            raise VerificationFailureError("prepared context commit identity mismatch")
        if commit_path.name != f"{prepared.commit_id}.json":
            raise VerificationFailureError("prepared context path identity mismatch")
        replayed = tuple(replay_context(commit))
        context = "\n\n".join(str(item.get("text", "")) for item in replayed)
        if context != prepared.context:
            raise VerificationFailureError(
                "prepared model context does not match its commit"
            )
        if stable_hash(prepared.receipt) != stable_hash(commit.get("receipt", {})):
            raise VerificationFailureError("prepared receipt does not match its commit")
        if stable_hash(prepared.selected_context) != stable_hash(replayed):
            raise VerificationFailureError(
                "prepared selected-context records do not match their commit"
            )
        if prepared.schema_version != _SCHEMA_VERSION:
            raise VerificationFailureError("prepared context schema is unsupported")
        if prepared.warnings != tuple(
            str(item) for item in prepared.receipt.get("warnings", [])
        ):
            raise VerificationFailureError(
                "prepared warnings do not match their signed receipt"
            )
        audit_receipt = self._verified_audit_receipt(prepared.audit)
        if (
            audit_receipt.get("artifact_type") != "prepared_context"
            or audit_receipt.get("commit_id") != prepared.commit_id
            or stable_hash(audit_receipt.get("security_scans", []))
            != stable_hash(prepared.security_scans)
            or stable_hash(audit_receipt.get("integrity_chain", {}))
            != stable_hash(prepared.integrity_chain)
        ):
            raise VerificationFailureError(
                "prepared context is not bound to its signed audit artifact"
            )
        return commit

    def verify_output(
        self,
        prepared: PreparedContext,
        output: str,
        *,
        profile: str = "rag",
        mode: str = "strict",
    ) -> VerifiedOutput:
        """Ground an output in the exact committed context and sign the result."""
        return self._verify_output_with_context(
            prepared,
            output,
            grounding_context=prepared.context,
            profile=profile,
            mode=mode,
        )

    def _verify_output_with_context(
        self,
        prepared: PreparedContext,
        output: str,
        *,
        grounding_context: str,
        profile: str = "rag",
        mode: str = "strict",
        recovered_chunk_ids: Sequence[str] = (),
        fixed_point_metadata: Mapping[str, Any] | None = None,
    ) -> VerifiedOutput:
        """Verify an output against a monotonic extension of committed context."""
        if mode not in _OUTPUT_MODES:
            raise ValueError(f"mode must be one of {sorted(_OUTPUT_MODES)}")
        self._load_prepared_commit(prepared)
        if not grounding_context.startswith(prepared.context):
            raise VerificationFailureError(
                "grounding context must preserve the committed context as a "
                "byte-identical prefix"
            )
        grounding_security = _scan_summary(
            scan(
                grounding_context,
                source=f"{prepared.commit_id}:grounding-context",
            )
        )
        if not grounding_security["is_safe"] and self.security_mode == "block":
            raise UnsafeContextError(
                "augmented grounding context failed the security scan",
                (grounding_security,),
            )
        original_output = str(output)
        try:
            result: SuppressionResult = EICVSuppressor(
                profile=profile, mode=mode
            ).suppress(grounding_context, original_output)
        except Exception as exc:
            raise VerificationFailureError(
                "output verification failed; unchecked output was withheld"
            ) from exc
        if not _suppression_result_consistent(result):
            raise VerificationFailureError(
                "output verifier returned an inconsistent certificate set; "
                "unchecked output was withheld"
            )

        security = _scan_summary(
            scan(
                result.rewritten_output,
                source=f"{prepared.commit_id}:model-output",
            )
        )
        if not security["is_safe"] and self.security_mode == "block":
            raise UnsafeContextError(
                "verified output still contains high-severity downstream instructions; "
                "the output was withheld",
                (security,),
            )

        suppression = result.as_dict()
        certificate_commitments = [
            {
                "claim_id": cert.claim_id,
                "claim_hash": text_fingerprint(cert.claim_text),
                "decision": cert.decision,
                "action": cert.action,
                "phi": cert.phi,
                "hallucination_score": cert.hallucination_score,
            }
            for cert in result.certificates
        ]
        normalized_chunk_ids = tuple(str(item) for item in recovered_chunk_ids)
        audit_payload: dict[str, Any] = {
            "artifact_type": "verified_output",
            "parent_artifact_id": prepared.audit.artifact_id,
            "commit_id": prepared.commit_id,
            "profile": result.profile,
            "mode": result.mode,
            "grounding_context_hash": text_fingerprint(grounding_context),
            "recovered_chunk_ids": normalized_chunk_ids,
            "grounding_security_scan": grounding_security,
            "original_output_hash": text_fingerprint(original_output),
            "verified_output_hash": text_fingerprint(result.rewritten_output),
            "changed": result.changed,
            "counts": {
                "claims": result.n_claims,
                "supported": result.n_supported,
                "abstained": result.n_abstained,
                "hallucinated": result.n_hallucinated,
                "suppressed": result.suppressed_count,
            },
            "certificate_commitments": certificate_commitments,
            "security_scan": security,
        }
        if fixed_point_metadata is not None:
            audit_payload["fixed_point"] = dict(fixed_point_metadata)
        audit = self._persist_audit(audit_payload)
        return VerifiedOutput(
            commit_id=prepared.commit_id,
            output=result.rewritten_output,
            original_output=original_output,
            changed=result.changed,
            suppression=suppression,
            security_scan=security,
            audit=audit,
            grounding_context_hash=text_fingerprint(grounding_context),
            recovered_chunk_ids=normalized_chunk_ids,
        )

    def recover(
        self, prepared: PreparedContext, chunk_id: str | None = None
    ) -> RecoveredContext:
        """Recover exact omitted evidence and sign the recovery decision."""
        commit = self._load_prepared_commit(prepared)
        receipt = dict(commit.get("receipt", {}))
        recovered = recover_omitted(
            receipt,
            chunk_id,
            bundle=dict(commit.get("recovery_bundle", {})),
        )
        if chunk_id is not None and not recovered:
            raise RecoveryIntegrityError(
                f"chunk {chunk_id!r} is not listed as omitted by {prepared.commit_id}"
            )
        failed = [item for item in recovered if not item.get("verified")]
        if failed:
            failed_ids = ", ".join(str(item.get("chunk_id")) for item in failed[:3])
            raise RecoveryIntegrityError(
                f"omitted context failed fingerprint verification: {failed_ids}"
            )

        audit = self._persist_audit(
            {
                "artifact_type": "recovered_context",
                "parent_artifact_id": prepared.audit.artifact_id,
                "commit_id": prepared.commit_id,
                "requested_chunk_id": chunk_id,
                "recovered_chunks": [
                    {
                        "chunk_id": item.get("chunk_id"),
                        "fingerprint": item.get("fingerprint"),
                        "content_hash": text_fingerprint(str(item.get("text", ""))),
                    }
                    for item in recovered
                ],
            }
        )
        return RecoveredContext(prepared.commit_id, tuple(recovered), audit)

    def run_fixed_point(
        self,
        prepared: PreparedContext,
        *,
        model_call: Any,
        max_rounds: int = 3,
        recovery_token_budget: int = 1_200,
        max_chunks_per_round: int = 3,
        profile: str = "rag",
    ) -> Any:
        """Run bounded proof-guided recovery with an explicit model callback."""
        from .context_fixed_point import run_proof_guided_fixed_point

        return run_proof_guided_fixed_point(
            self,
            prepared,
            model_call=model_call,
            max_rounds=max_rounds,
            recovery_token_budget=recovery_token_budget,
            max_chunks_per_round=max_chunks_per_round,
            profile=profile,
        )

    @staticmethod
    def _bounded_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return min(1.0, max(0.0, numerator / denominator))

    def _learning_vectors(
        self, prepared: PreparedContext, verified: VerifiedOutput
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        ratio = prepared.receipt.get("compression_ratio", {})
        risk = prepared.receipt.get("risk_summary", {})
        source_tokens = float(ratio.get("source_tokens", 0.0))
        selected_tokens = float(ratio.get("selected_tokens", 0.0))
        budget = float(prepared.receipt.get("token_budget", 0.0))
        selected_chunks = float(risk.get("selected_chunks", 0.0))
        total_chunks = float(risk.get("total_chunks", 0.0))
        omitted = float(risk.get("omitted_relevant_chunks", 0.0))
        source_unsafe = sum(
            not item.get("is_safe", False) for item in prepared.security_scans[:-1]
        )
        source_count = max(1, len(prepared.security_scans) - 1)
        review_level = _REVIEW_RANK.get(str(risk.get("review_level", "high")), 1.0)

        suppression = verified.suppression
        claims = float(suppression.get("n_claims", 0.0))
        supported = float(suppression.get("n_supported", 0.0))
        hallucinated = float(suppression.get("n_hallucinated", 0.0))
        hallucination_rate = self._bounded_ratio(hallucinated, claims)
        strictness = {
            "audit": 0.0,
            "annotate": 0.5,
            "strict": 1.0,
        }.get(str(suppression.get("mode", "strict")), 1.0)

        state = (
            self._bounded_ratio(budget, source_tokens),
            self._bounded_ratio(selected_chunks, total_chunks),
            min(1.0, max(0.0, float(risk.get("coverage_score", 0.0)))),
            self._bounded_ratio(omitted, selected_chunks + omitted),
            self._bounded_ratio(float(source_unsafe), float(source_count)),
            review_level,
        )
        action = (
            1.0 - self._bounded_ratio(selected_tokens, source_tokens),
            strictness,
            1.0 if self.security_mode == "block" else 0.0,
        )
        next_state = (
            self._bounded_ratio(selected_tokens, source_tokens),
            self._bounded_ratio(selected_tokens, budget),
            min(1.0, max(0.0, float(risk.get("coverage_score", 0.0)))),
            1.0 - min(1.0, max(0.0, hallucination_rate)),
            1.0 if verified.security_scan.get("is_safe") else 0.0,
            self._bounded_ratio(supported, claims) if claims else 1.0,
        )
        if not all(math.isfinite(value) for value in (*state, *action, *next_state)):
            raise EvolutionEvidenceError("learning features contain non-finite values")
        return state, action, next_state

    def _controller(self) -> Any:
        if self._world_model_controller is not None:
            return self._world_model_controller
        from .ravs.world_model import TransitionLedger, VerifiedDreamController

        with self._lock:
            if self._world_model_controller is None:
                ledger = TransitionLedger(self.state_dir / "evolution")
                self._world_model_controller = VerifiedDreamController(ledger)
        return self._world_model_controller

    def record_verified_outcome(
        self,
        prepared: PreparedContext,
        verified: VerifiedOutput,
        trace: Any,
        outcome: Any,
        *,
        environment: str = "entroly.verified-efficiency.v1",
    ) -> LearningReceipt:
        """Learn from one strong real outcome; reject synthetic or self-report data."""
        self._load_prepared_commit(prepared)
        if verified.commit_id != prepared.commit_id:
            raise EvolutionEvidenceError(
                "output does not belong to the prepared context"
            )
        try:
            output_receipt = self._verified_audit_receipt(verified.audit)
        except VerificationFailureError as exc:
            raise EvolutionEvidenceError("output audit artifact is invalid") from exc
        suppression = verified.suppression
        expected_counts = {
            "claims": suppression.get("n_claims"),
            "supported": suppression.get("n_supported"),
            "abstained": suppression.get("n_abstained"),
            "hallucinated": suppression.get("n_hallucinated"),
            "suppressed": suppression.get("suppressed_count"),
        }
        if (
            output_receipt.get("artifact_type") != "verified_output"
            or output_receipt.get("parent_artifact_id") != prepared.audit.artifact_id
            or output_receipt.get("commit_id") != prepared.commit_id
            or output_receipt.get("original_output_hash")
            != text_fingerprint(verified.original_output)
            or output_receipt.get("verified_output_hash")
            != text_fingerprint(verified.output)
            or output_receipt.get("changed") != verified.changed
            or stable_hash(output_receipt.get("counts", {}))
            != stable_hash(expected_counts)
            or output_receipt.get("mode") != suppression.get("mode")
            or output_receipt.get("profile") != suppression.get("profile")
            or output_receipt.get("grounding_context_hash")
            != verified.grounding_context_hash
            or tuple(output_receipt.get("recovered_chunk_ids", ()))
            != verified.recovered_chunk_ids
            or stable_hash(output_receipt.get("security_scan", {}))
            != stable_hash(verified.security_scan)
        ):
            raise EvolutionEvidenceError(
                "verified output is not bound to its signed audit artifact"
            )

        def read(value: Any, key: str) -> Any:
            if isinstance(value, Mapping):
                return value.get(key)
            return getattr(value, key, None)

        trace_request = str(read(trace, "request_id") or "")
        outcome_request = str(read(outcome, "request_id") or "")
        if not trace_request or trace_request != outcome_request:
            raise EvolutionEvidenceError(
                "trace and outcome must carry the same non-empty request_id"
            )

        state, action, next_state = self._learning_vectors(prepared, verified)
        try:
            from .ravs.world_model import transition_from_ravs

            transition = transition_from_ravs(
                trace,
                outcome,
                state=state,
                action=action,
                next_state=next_state,
                environment=environment,
            )
        except (TypeError, ValueError) as exc:
            raise EvolutionEvidenceError(
                "world-model learning requires a strong, externally verified "
                "RAVS outcome"
            ) from exc

        transition = replace(
            transition,
            source=(
                f"{transition.source};context_commit={prepared.commit_id};"
                f"output_audit={verified.audit.artifact_id}"
            ),
        )
        controller = self._controller()
        existing = controller.ledger.real_receipt(transition.transition_id)
        if existing is not None:
            if not controller.ledger.contains_exact_real(transition):
                raise EvolutionEvidenceError(
                    "transition ID already exists with different evidence"
                )
            receipt = existing
            idempotent = True
        else:
            try:
                receipt = controller.observe_real(transition)
                idempotent = False
            except ValueError as exc:
                # Another process may have committed this exact outcome after
                # our lookup. Re-read under the ledger's process lock and
                # accept only a byte-identical transition.
                existing = controller.ledger.real_receipt(transition.transition_id)
                if existing is None or not controller.ledger.contains_exact_real(
                    transition
                ):
                    raise EvolutionEvidenceError(
                        "verified outcome could not be committed without conflict"
                    ) from exc
                controller.model.fit(controller.ledger.read_real())
                receipt = existing
                idempotent = True
        return LearningReceipt(
            transition_id=receipt.transition_id,
            receipt_hash=receipt.receipt_hash,
            previous_hash=receipt.previous_hash,
            ledger_path=receipt.ledger_path,
            idempotent_replay=idempotent,
            context_commit_id=prepared.commit_id,
            output_audit_id=verified.audit.artifact_id,
        )


__all__ = [
    "AuditArtifact",
    "AuditUnavailableError",
    "AuditVerification",
    "ContextRiskError",
    "EfficiencyLayerError",
    "EvolutionEvidenceError",
    "LearningReceipt",
    "PreparedContext",
    "RecoveredContext",
    "RecoveryIntegrityError",
    "UnsafeContextError",
    "VerificationFailureError",
    "VerifiedEfficiencyLayer",
    "VerifiedOutput",
]
