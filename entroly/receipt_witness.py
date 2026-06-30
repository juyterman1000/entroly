from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from entroly.context_receipts.models import stable_json
from entroly.receipt_attestation import AttestationKey
from entroly.receipt_merkle import SignedTreeHead, verify_consistency

_WITNESS_DOMAIN = "entroly.receipt-witness.v1"


def _witness_message(
    witness_id: str,
    tree_size: int,
    root_hash: str,
    operator_public_key: str,
    head_timestamp: float,
) -> bytes:
    return stable_json(
        {
            "domain": _WITNESS_DOMAIN,
            "head_timestamp": head_timestamp,
            "operator_public_key": operator_public_key,
            "root_hash": root_hash,
            "tree_size": tree_size,
            "witness_id": witness_id,
        }
    ).encode("utf-8")


class WitnessRejection(RuntimeError):
    pass


@dataclass(frozen=True)
class WitnessSignature:
    witness_id: str
    tree_size: int
    root_hash: str
    signature: str
    public_key: str
    operator_public_key: str
    head_timestamp: float

    def message(self) -> bytes:
        return _witness_message(
            self.witness_id,
            self.tree_size,
            self.root_hash,
            self.operator_public_key,
            self.head_timestamp,
        )

    def verify(self, *, public_key: str | None = None) -> bool:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        if public_key is not None and public_key != self.public_key:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.public_key)).verify(
                bytes.fromhex(self.signature), self.message()
            )
            return True
        except (InvalidSignature, TypeError, ValueError):
            return False


class ReceiptWitness:
    def __init__(self, *, name: str = "", key: AttestationKey | None = None) -> None:
        self._key = key or AttestationKey.generate()
        self.name = name or self._key.public_hex()[:16]
        self.witness_id = self.name
        self._last_size: int | None = None
        self._last_root = b""

    @property
    def public_key(self) -> str:
        return self._key.public_hex()

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
        if self._last_size is not None:
            proof = consistency_proof or []
            if not verify_consistency(self._last_size, head.tree_size, proof, self._last_root, root):
                raise WitnessRejection("tree head is not a verified extension")
        self._last_size = head.tree_size
        self._last_root = root
        msg = _witness_message(
            self.witness_id,
            head.tree_size,
            head.root_hash,
            operator_key,
            head.timestamp,
        )
        return WitnessSignature(
            witness_id=self.witness_id,
            tree_size=head.tree_size,
            root_hash=head.root_hash,
            signature=self._key.sign(msg),
            public_key=self._key.public_hex(),
            operator_public_key=operator_key,
            head_timestamp=head.timestamp,
        )


@dataclass(frozen=True)
class CosignedTreeHead:
    head: SignedTreeHead
    signatures: tuple[WitnessSignature, ...] = field(default_factory=tuple)

    def verify(
        self,
        *,
        operator_key: str,
        trusted_witnesses: Mapping[str, str],
        threshold: int,
    ) -> bool:
        if threshold <= 0 or not isinstance(trusted_witnesses, Mapping):
            return False
        if not self.head.verify(public_key=operator_key):
            return False
        seen: set[str] = set()
        for signature in self.signatures:
            trusted_key = trusted_witnesses.get(signature.witness_id)
            if trusted_key is None or signature.witness_id in seen:
                continue
            if signature.tree_size != self.head.tree_size or signature.root_hash != self.head.root_hash:
                continue
            if (
                signature.operator_public_key != operator_key
                or signature.head_timestamp != self.head.timestamp
            ):
                continue
            if signature.verify(public_key=trusted_key):
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
