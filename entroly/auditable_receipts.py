from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from entroly.context_receipts import run_receipt_pipeline
from entroly.receipt_attestation import AttestationKey
from entroly.receipt_disclosure import ReceiptCommitment
from entroly.receipt_merkle import ReceiptMerkleLog, verify_inclusion


@dataclass(frozen=True)
class RecordedReceipt:
    receipt: dict[str, Any]
    index: int

    @property
    def receipt_id(self) -> str:
        return str(self.receipt.get("receipt_id", ""))


@dataclass(frozen=True)
class ReceiptProof:
    index: int
    leaf_hex: str
    audit_path: list[str]
    tree_size: int
    root_hash: str
    operator_signature: str
    operator_public_key: str

    def verify(self, *, operator_public_key: str | None = None) -> bool:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        if operator_public_key is not None and operator_public_key != self.operator_public_key:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.operator_public_key)).verify(
                bytes.fromhex(self.operator_signature),
                f"{self.tree_size}|{self.root_hash}".encode(),
            )
        except (InvalidSignature, ValueError):
            return False
        return verify_inclusion(
            self.index,
            self.tree_size,
            bytes.fromhex(self.leaf_hex),
            [bytes.fromhex(item) for item in self.audit_path],
            bytes.fromhex(self.root_hash),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "leaf_hex": self.leaf_hex,
            "audit_path": list(self.audit_path),
            "tree_size": self.tree_size,
            "root_hash": self.root_hash,
            "operator_signature": self.operator_signature,
            "operator_public_key": self.operator_public_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReceiptProof":
        return cls(
            index=int(data["index"]),
            leaf_hex=str(data["leaf_hex"]),
            audit_path=[str(item) for item in data.get("audit_path", [])],
            tree_size=int(data["tree_size"]),
            root_hash=str(data["root_hash"]),
            operator_signature=str(data["operator_signature"]),
            operator_public_key=str(data["operator_public_key"]),
        )


class AuditableReceiptLog:
    def __init__(self, key: AttestationKey | None = None, *, prefer_rust: bool = True) -> None:
        self._key = key or AttestationKey.generate()
        self._log = ReceiptMerkleLog(self._key)
        self._prefer_rust = prefer_rust

    @property
    def public_key(self) -> str:
        return self._key.public_hex()

    @property
    def size(self) -> int:
        return self._log.size

    def record(
        self,
        documents: Iterable[tuple[str, str]],
        *,
        query: str,
        token_budget: int,
        chunk_tokens: int = 360,
        overlap_tokens: int = 32,
    ) -> RecordedReceipt:
        receipt = run_receipt_pipeline(
            documents,
            query=query,
            token_budget=token_budget,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            prefer_rust=self._prefer_rust,
        )
        return self.record_receipt(receipt)

    def record_receipt(self, receipt: dict[str, Any]) -> RecordedReceipt:
        index = self._log.append(receipt)
        return RecordedReceipt(receipt=receipt, index=index)

    def record_commitment(self, receipt: dict[str, Any]) -> tuple[RecordedReceipt, ReceiptCommitment]:
        commitment = ReceiptCommitment.from_receipt(receipt)
        public_receipt = {
            "receipt_id": receipt.get("receipt_id", ""),
            "commitment_root": commitment.root_hex(),
            "schema_version": receipt.get("schema_version", ""),
            "token_budget": receipt.get("token_budget"),
            "selected_count": len(receipt.get("selected_context", []) or []),
            "omitted_count": len(receipt.get("omitted_context", []) or []),
        }
        return self.record_receipt(public_receipt), commitment

    def prove(self, index: int) -> ReceiptProof:
        head = self._log.signed_tree_head()
        return ReceiptProof(
            index=index,
            leaf_hex=self._log.leaf_at(index).hex(),
            audit_path=[item.hex() for item in self._log.prove_inclusion(index)],
            tree_size=head.tree_size,
            root_hash=head.root_hash,
            operator_signature=head.signature,
            operator_public_key=head.public_key,
        )

    def signed_tree_head(self):
        return self._log.signed_tree_head()

    def consistency_proof(self, old_size: int) -> list[bytes]:
        return self._log.prove_consistency(old_size)


__all__ = ["AuditableReceiptLog", "RecordedReceipt", "ReceiptProof"]
