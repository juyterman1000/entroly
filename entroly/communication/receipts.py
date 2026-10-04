"""Evidence-backed, durable communication receipts.

Communication receipts intentionally contain no raw message text. They bind
policy decisions to immutable event/content commitments and can be proven
against Entroly's existing signed Merkle receipt layer.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..receipt_attestation import AttestationKey
from ..receipt_merkle import ReceiptMerkleLog, verify_inclusion
from .models import (
    CommunicationActionProposal,
    CommunicationEvent,
    CommunicationStateConflict,
    CommunicationStateError,
    canonical_json,
    sha256_text,
)
from .store import default_store_path

if TYPE_CHECKING:
    from .store import CommunicationStore

RECEIPT_SCHEMA = "entroly.communication.receipt.v1"
_ROUTINE_NONFACTUAL = {
    "birthday_wish",
    "congratulations",
    "holiday_wish",
    "thanks",
}
_MAX_EICV_EVIDENCE_CHARS = 100_000


def default_receipt_state_dir() -> Path:
    return default_store_path().parent / "receipts"


def resolve_receipt_state_dir(
    path: str | os.PathLike[str] | None,
) -> Path:
    if path is None or not str(path).strip():
        return default_receipt_state_dir()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise CommunicationStateError(
            "communication receipt state override must be an absolute path"
        )
    return candidate.absolute()


def _scope_commitment(
    *,
    channel: str,
    account_id: str,
    conversation_id: str,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "channel": channel,
                "account_id": account_id,
                "conversation_id": conversation_id,
            }
        )
    )


def action_evidence_verification(
    proposal: CommunicationActionProposal,
    source_events: Sequence[CommunicationEvent],
    *,
    payload: str,
) -> dict[str, Any]:
    """Run EICV only when the proposed text makes evidence-dependent claims."""
    if proposal.action_type in {"react", "no_action"}:
        return {"status": "not_applicable", "reason": "non_text_action"}
    if proposal.payload_sha256 != (sha256_text(payload) if payload else ""):
        return {"status": "unavailable", "reason": "payload_commitment_mismatch"}
    if not payload.strip():
        return {"status": "not_applicable", "reason": "empty_payload"}
    if proposal.category in _ROUTINE_NONFACTUAL:
        return {
            "status": "not_applicable",
            "reason": "routine_nonfactual_acknowledgement",
        }

    evidence = "\n".join(
        event.content
        for event in source_events
        if event.direction == "inbound" and event.content.strip()
    )
    if not evidence:
        return {"status": "insufficient_context", "reason": "no_inbound_evidence"}
    if len(evidence) > _MAX_EICV_EVIDENCE_CHARS:
        return {
            "status": "insufficient_context",
            "reason": "evidence_exceeds_local_verification_bound",
            "evidence_chars": len(evidence),
        }

    try:
        from ..sdk import eicv_verify

        cert = eicv_verify(
            payload,
            evidence=evidence,
            profile="dialogue",
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": "eicv_verification_error",
            "error_type": type(exc).__name__,
        }

    return {
        "status": "verified",
        "decision": str(cert.get("decision") or "abstain"),
        "phi": cert.get("phi"),
        "hallucination_score": cert.get("hallucination_score"),
        "unsupported_fraction": cert.get("unsupported_fraction"),
        "contradiction_fraction": cert.get("contradiction_fraction"),
        "profile": cert.get("profile", "dialogue"),
    }


def eicv_supports_automatic_action(verification: Mapping[str, Any]) -> bool:
    status = str(verification.get("status") or "")
    if status == "not_applicable":
        return True
    return status == "verified" and verification.get("decision") == "supported"


def build_action_receipt(
    proposal: CommunicationActionProposal,
    *,
    decision: str,
    reasons: Sequence[str],
    source_events: Sequence[CommunicationEvent],
    evidence_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-free receipt binding one action to exact evidence."""
    by_id = {event.event_id: event for event in source_events}
    evidence = []
    for event_id in proposal.source_event_ids:
        event = by_id.get(event_id)
        if event is None:
            raise CommunicationStateError(
                f"receipt source event is unavailable: {event_id}"
            )
        evidence.append(
            {
                "event_id": event.event_id,
                "content_sha256": event.content_sha256,
                "commitment_sha256": event.commitment_sha256,
                "direction": event.direction,
                "message_identity_strength": event.identity_strength,
            }
        )

    verification = dict(evidence_verification or {})
    # Never serialize arbitrary exception text into a durable receipt.
    verification.pop("error", None)

    material = {
        "kind": "communication_action_assurance",
        "action_id": proposal.action_id,
        "action_type": proposal.action_type,
        "scope_commitment": _scope_commitment(
            channel=proposal.channel,
            account_id=proposal.account_id,
            conversation_id=proposal.conversation_id,
        ),
        "conversation_kind": proposal.conversation_kind,
        "payload_sha256": proposal.payload_sha256,
        "category": proposal.category,
        "risk_class": proposal.risk_class,
        "creates_commitment": proposal.creates_commitment,
        "decision": str(decision),
        "reasons": sorted({str(reason) for reason in reasons}),
        "source_evidence": evidence,
        "eicv": verification,
    }
    receipt_id = "commrcpt_" + sha256_text(canonical_json(material))[:40]
    return {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        **material,
    }


def build_learning_receipt(
    feedback: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    source_events: Sequence[CommunicationEvent],
) -> dict[str, Any]:
    """Bind one taste-learning episode to exact evidence without raw text."""
    feedback_id = str(feedback.get("feedback_id") or "").strip()
    source = str(feedback.get("source") or "").strip()
    scope_type = str(feedback.get("scope_type") or "").strip()
    scope_id_hash = str(feedback.get("scope_id_hash") or "").strip()
    raw_ids = feedback.get("evidence_event_ids")
    evidence_ids = tuple(
        str(item).strip()
        for item in (raw_ids if isinstance(raw_ids, (list, tuple)) else ())
        if str(item).strip()
    )
    if not feedback_id or not source or not scope_type or not scope_id_hash:
        raise CommunicationStateError(
            "learning receipt requires feedback identity, source, and scope"
        )
    if not evidence_ids:
        raise CommunicationStateError(
            "learning receipt requires exact evidence event ids"
        )

    by_id = {event.event_id: event for event in source_events}
    evidence: list[dict[str, Any]] = []
    for event_id in evidence_ids:
        event = by_id.get(event_id)
        if event is None:
            raise CommunicationStateError(
                f"learning receipt source event is unavailable: {event_id}"
            )
        evidence.append(
            {
                "event_id": event.event_id,
                "content_sha256": event.content_sha256,
                "commitment_sha256": event.commitment_sha256,
                "direction": event.direction,
                "source": event.source,
                "message_identity_strength": event.identity_strength,
            }
        )

    feature_vector = feedback.get("feature_vector")
    if not isinstance(feature_vector, (list, tuple)) or len(feature_vector) != 5:
        raise CommunicationStateError(
            "learning receipt requires a five-dimensional PRISM feature vector"
        )
    selected_ids = selection.get("event_ids")
    normalized_selected = sorted(
        {
            str(item).strip()
            for item in (
                selected_ids if isinstance(selected_ids, (list, tuple)) else ()
            )
            if str(item).strip()
        }
    )
    if not normalized_selected or not set(normalized_selected).issubset(set(evidence_ids)):
        raise CommunicationStateError(
            "learning receipt selection must be covered by exact evidence ids"
        )

    material = {
        "kind": "communication_taste_feedback",
        "feedback_id": feedback_id,
        "scope_type": scope_type,
        "scope_id_hash": scope_id_hash,
        "reward": float(feedback.get("reward", 0.0)),
        "source": source,
        "feature_vector": [float(value) for value in feature_vector],
        "selection": {
            "event_ids": normalized_selected,
            "feature_mean": dict(selection.get("feature_mean") or {}),
            "weights": dict(selection.get("weights") or {}),
        },
        "source_evidence": evidence,
        "authority_surface": "none",
    }
    receipt_id = "commrcpt_" + sha256_text(canonical_json(material))[:40]
    return {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        **material,
    }


class CommunicationReceiptLedger:
    """Rebuildable signed Merkle log over durable receipt rows."""

    def __init__(
        self,
        state_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.state_dir = resolve_receipt_state_dir(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            self.state_dir.chmod(0o700)
        self._key = self._load_or_create_key()

    def _load_or_create_key(self) -> AttestationKey:
        key_path = self.state_dir / "attestation.key"
        if key_path.exists():
            try:
                return AttestationKey.from_private_hex(
                    key_path.read_text(encoding="ascii").strip()
                )
            except (OSError, ValueError, RuntimeError) as exc:
                raise CommunicationStateError(
                    "invalid communication receipt attestation key; "
                    "restore the original key instead of replacing it"
                ) from exc

        lock_path = self.state_dir / ".attestation-key.lock"
        try:
            lock_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            if key_path.exists():
                return self._load_or_create_key()
            raise CommunicationStateError(
                "communication receipt key initialization is already in progress"
            ) from exc

        try:
            os.close(lock_fd)
            try:
                key = AttestationKey.generate()
            except RuntimeError as exc:
                raise CommunicationStateError(
                    "signed communication receipts require the cryptography dependency"
                ) from exc
            temporary = key_path.with_name(
                f".{key_path.name}.{uuid.uuid4().hex}.tmp"
            )
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
                try:
                    os.chmod(key_path, 0o600)
                except OSError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
            return key
        finally:
            lock_path.unlink(missing_ok=True)

    @property
    def public_key(self) -> str:
        return self._key.public_hex()

    def _rebuild(
        self,
        store: "CommunicationStore",
    ) -> tuple[ReceiptMerkleLog, list[dict[str, Any]]]:
        rows = store.receipt_rows()
        log = ReceiptMerkleLog(self._key)
        for row in rows:
            log.append(dict(row["receipt"]))
        return log, rows

    def record(
        self,
        store: "CommunicationStore",
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Prove first, then append durably so failed signing leaves no row."""
        payload = dict(receipt)
        receipt_id = str(payload.get("receipt_id") or "").strip()
        if not receipt_id:
            raise CommunicationStateError("communication receipt_id is required")

        existing_rows = store.receipt_rows()
        existing_index = next(
            (
                index
                for index, row in enumerate(existing_rows)
                if row["receipt_id"] == receipt_id
            ),
            None,
        )
        if existing_index is not None:
            existing = existing_rows[existing_index]["receipt"]
            if canonical_json(existing) != canonical_json(payload):
                raise CommunicationStateConflict(
                    "stable communication receipt identity was reused "
                    "with different content"
                )

        # Build and cryptographically self-verify the candidate tree before
        # mutating durable state.  This prevents an ALLOW receipt from being
        # left behind if key loading/signing/proof generation fails.
        candidate_log = ReceiptMerkleLog(self._key)
        for row in existing_rows:
            candidate_log.append(dict(row["receipt"]))
        if existing_index is None:
            candidate_index = candidate_log.append(payload)
        else:
            candidate_index = existing_index

        head = candidate_log.signed_tree_head()
        leaf = candidate_log.leaf_at(candidate_index)
        audit_path = candidate_log.prove_inclusion(candidate_index)
        if not verify_inclusion(
            candidate_index,
            head.tree_size,
            leaf,
            audit_path,
            bytes.fromhex(head.root_hash),
        ):
            raise CommunicationStateConflict(
                "communication receipt Merkle proof failed self-verification"
            )
        if not head.verify(public_key=self.public_key):
            raise CommunicationStateConflict(
                "communication receipt signed tree head failed self-verification"
            )

        stored_index = store.record_receipt(payload)
        if stored_index != candidate_index:
            raise CommunicationStateConflict(
                "communication receipt ledger index changed during append"
            )

        return {
            "schema_version": RECEIPT_SCHEMA,
            "receipt_id": receipt_id,
            "index": candidate_index,
            "leaf_hex": leaf.hex(),
            "audit_path": [item.hex() for item in audit_path],
            "tree_size": head.tree_size,
            "root_hash": head.root_hash,
            "signed_at": head.timestamp,
            "operator_signature": head.signature,
            "operator_public_key": head.public_key,
            "verified": True,
        }

    def get_verified(
        self,
        store: "CommunicationStore",
        receipt_id: str,
    ) -> dict[str, Any] | None:
        """Return receipt + a freshly verified signed-Merkle proof."""
        proof = self.prove(store, receipt_id)
        if proof is None:
            return None
        rows = store.receipt_rows()
        row = next(
            (item for item in rows if item["receipt_id"] == receipt_id),
            None,
        )
        if row is None:
            return None
        if proof.get("operator_public_key") != self.public_key:
            raise CommunicationStateConflict(
                "communication receipt proof changed operator key"
            )
        return {
            "receipt": dict(row["receipt"]),
            "proof": proof,
        }

    def prove(
        self,
        store: "CommunicationStore",
        receipt_id: str,
    ) -> dict[str, Any] | None:
        log, rows = self._rebuild(store)
        index = next(
            (
                idx
                for idx, row in enumerate(rows)
                if row["receipt_id"] == receipt_id
            ),
            None,
        )
        if index is None:
            return None
        head = log.signed_tree_head()
        return {
            "schema_version": RECEIPT_SCHEMA,
            "receipt_id": receipt_id,
            "index": index,
            "leaf_hex": log.leaf_at(index).hex(),
            "audit_path": [
                item.hex() for item in log.prove_inclusion(index)
            ],
            "tree_size": head.tree_size,
            "root_hash": head.root_hash,
            "signed_at": head.timestamp,
            "operator_signature": head.signature,
            "operator_public_key": head.public_key,
        }
