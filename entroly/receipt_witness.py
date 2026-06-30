from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from entroly.receipt_attestation import AttestationKey
from entroly.receipt_merkle import SignedTreeHead, verify_consistency


class WitnessRejection(RuntimeError):
    pass


@dataclass(frozen=True)
class WitnessSignature:
    witness_id: str
    tree_size: int
    root_hash: str
    signature: str
    public_key: str

    def message(self) -> bytes:
        return f"{self.witness_id}|{self.tree_size}|{self.root_hash}".encode()

    def verify(self) -> bool:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.public_key)).verify(
                bytes.fromhex(self.signature), self.message()
            )
            return True
        except (InvalidSignature, ValueError):
            return False


class ReceiptWitness:
    def __init__(self, *, name: str = "", key: AttestationKey | None = None) -> None:
        self._key = key or AttestationKey.generate()
        self.name = name or self._key.public_hex()[:16]
        self.witness_id = self.name
        self._last_size = 0
        self._last_root = b""

    def cosign(
        self,
        head: SignedTreeHead,
        *,
        operator_key: str,
        consistency_proof: list[bytes] | None = None,
    ) -> WitnessSignature:
        if not head.verify(public_key=operator_key):
            raise WitnessRejection("operator signature did not verify")
        root = bytes.fromhex(head.root_hash)
        if self._last_size:
            proof = consistency_proof or []
            if not verify_consistency(self._last_size, head.tree_size, proof, self._last_root, root):
                raise WitnessRejection("tree head is not a verified extension")
        self._last_size = head.tree_size
        self._last_root = root
        msg = f"{self.witness_id}|{head.tree_size}|{head.root_hash}".encode()
        return WitnessSignature(
            witness_id=self.witness_id,
            tree_size=head.tree_size,
            root_hash=head.root_hash,
            signature=self._key.sign(msg),
            public_key=self._key.public_hex(),
        )


@dataclass(frozen=True)
class CosignedTreeHead:
    head: SignedTreeHead
    signatures: tuple[WitnessSignature, ...] = field(default_factory=tuple)

    def verify(self, *, operator_key: str, trusted_witnesses: Iterable[str], threshold: int) -> bool:
        if threshold < 0:
            return False
        if not self.head.verify(public_key=operator_key):
            return False
        trusted = set(trusted_witnesses)
        seen: set[str] = set()
        for signature in self.signatures:
            if signature.witness_id not in trusted or signature.witness_id in seen:
                continue
            if signature.tree_size != self.head.tree_size or signature.root_hash != self.head.root_hash:
                continue
            if signature.verify():
                seen.add(signature.witness_id)
        return len(seen) >= threshold


Witness = ReceiptWitness
Cosignature = WitnessSignature

__all__ = [
    "ReceiptWitness",
    "Witness",
    "WitnessSignature",
    "Cosignature",
    "CosignedTreeHead",
    "WitnessRejection",
]
