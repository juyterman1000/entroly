"""Receipt attestation primitives for context receipts.

This module keeps the cryptography optional at import time. When the
``cryptography`` package is available, receipts can be signed with Ed25519 and
verified as an append-only hash chain. The chain is useful for local custody;
Merkle transparency proofs live in ``receipt_merkle``.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any

from entroly.context_receipts.models import stable_hash, stable_json

_GENESIS = "0" * 64
_ATTESTATION_DOMAIN = "entroly.receipt-attestation.v1"


def _attestation_message(
    sequence: int, prev_hash: str, content_hash: str, signed_at: float
) -> bytes:
    return stable_json(
        {
            "content_hash": content_hash,
            "domain": _ATTESTATION_DOMAIN,
            "prev_hash": prev_hash,
            "sequence": sequence,
            "signed_at": signed_at,
        }
    ).encode("utf-8")


def _crypto():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            PublicFormat,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("receipt attestation requires cryptography") from exc
    return (
        Ed25519PrivateKey,
        Ed25519PublicKey,
        InvalidSignature,
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )


class AttestationKey:
    """Ed25519 key wrapper used to sign receipt custody records."""

    def __init__(self, private_key: Any) -> None:
        self._private_key = private_key

    @classmethod
    def generate(cls) -> "AttestationKey":
        return cls(_crypto()[0].generate())

    @classmethod
    def from_private_hex(cls, value: str) -> "AttestationKey":
        return cls(_crypto()[0].from_private_bytes(bytes.fromhex(value)))

    def private_hex(self) -> str:
        _, _, _, encoding, no_encryption, private_format, _ = _crypto()
        return self._private_key.private_bytes(
            encoding.Raw, private_format.Raw, no_encryption()
        ).hex()

    def public_hex(self) -> str:
        _, _, _, encoding, _, _, public_format = _crypto()
        return self._private_key.public_key().public_bytes(
            encoding.Raw, public_format.Raw
        ).hex()

    def sign(self, message: bytes) -> str:
        return self._private_key.sign(message).hex()


@dataclass(frozen=True)
class AttestedReceipt:
    receipt: dict[str, Any]
    sequence: int
    prev_hash: str
    content_hash: str
    signature: str
    public_key: str
    signed_at: float

    def signing_message(self) -> bytes:
        return _attestation_message(
            self.sequence, self.prev_hash, self.content_hash, self.signed_at
        )

    def attestation_hash(self) -> str:
        return stable_hash(
            {
                "sequence": self.sequence,
                "prev_hash": self.prev_hash,
                "content_hash": self.content_hash,
                "signature": self.signature,
                "public_key": self.public_key,
                "signed_at": self.signed_at,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttestedReceipt":
        return cls(**data)


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    entries_checked: int
    broken_at: int | None = None
    reason: str = ""


class AttestationLog:
    """In-memory append-only hash chain for signed receipts."""

    def __init__(self, key: AttestationKey) -> None:
        self._key = key
        self._entries: list[AttestedReceipt] = []

    @property
    def entries(self) -> list[AttestedReceipt]:
        return copy.deepcopy(self._entries)

    @property
    def head_hash(self) -> str:
        return self._entries[-1].attestation_hash() if self._entries else _GENESIS

    def append(self, receipt: dict[str, Any]) -> AttestedReceipt:
        snapshot = copy.deepcopy(receipt)
        sequence = len(self._entries)
        content_hash = stable_hash(snapshot)
        prev_hash = self.head_hash
        signed_at = time.time()
        message = _attestation_message(sequence, prev_hash, content_hash, signed_at)
        entry = AttestedReceipt(
            receipt=snapshot,
            sequence=sequence,
            prev_hash=prev_hash,
            content_hash=content_hash,
            signature=self._key.sign(message),
            public_key=self._key.public_hex(),
            signed_at=signed_at,
        )
        self._entries.append(entry)
        return copy.deepcopy(entry)


def _signature_valid(entry: AttestedReceipt) -> bool:
    _, public_key_cls, invalid_signature, *_ = _crypto()
    try:
        public_key_cls.from_public_bytes(bytes.fromhex(entry.public_key)).verify(
            bytes.fromhex(entry.signature), entry.signing_message()
        )
        return True
    except (ValueError, invalid_signature):
        return False


def verify_attestation(
    entry: AttestedReceipt, *, public_key: str | None = None
) -> bool:
    if public_key is not None and entry.public_key != public_key:
        return False
    if stable_hash(entry.receipt) != entry.content_hash:
        return False
    return _signature_valid(entry)


def verify_chain(
    entries: list[AttestedReceipt], *, public_key: str | None = None
) -> VerificationResult:
    prev_hash = _GENESIS
    expected_key = public_key or (entries[0].public_key if entries else None)
    for index, entry in enumerate(entries):
        if entry.sequence != index:
            return VerificationResult(False, index, index, "sequence out of order")
        if entry.prev_hash != prev_hash:
            return VerificationResult(False, index, index, "broken previous hash")
        if not verify_attestation(entry, public_key=expected_key):
            return VerificationResult(False, index, index, "invalid attestation")
        prev_hash = entry.attestation_hash()
    return VerificationResult(True, len(entries))


__all__ = [
    "AttestationKey",
    "AttestedReceipt",
    "AttestationLog",
    "VerificationResult",
    "verify_attestation",
    "verify_chain",
]
