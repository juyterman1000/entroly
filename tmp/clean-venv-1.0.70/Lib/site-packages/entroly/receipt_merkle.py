from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from entroly.context_receipts.models import stable_json
from entroly.receipt_attestation import AttestationKey

_LEAF = b"\x00"
_NODE = b"\x01"
_TREE_HEAD_DOMAIN = "entroly.receipt-tree-head.v1"


def _sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def leaf_hash(data: bytes) -> bytes:
    return _sha(_LEAF + data)


def _node_hash(left: bytes, right: bytes) -> bytes:
    return _sha(_NODE + left + right)


def receipt_leaf(receipt: dict[str, Any]) -> bytes:
    return leaf_hash(stable_json(receipt).encode("utf-8"))


def _tree_head_message(tree_size: int, root_hash: str, timestamp: float) -> bytes:
    return stable_json(
        {
            "domain": _TREE_HEAD_DOMAIN,
            "root_hash": root_hash,
            "timestamp": timestamp,
            "tree_size": tree_size,
        }
    ).encode("utf-8")


def _split(size: int) -> int:
    if size <= 1:
        raise ValueError("size must be greater than one")
    k = 1
    while (k << 1) < size:
        k <<= 1
    return k


def merkle_root(leaves: list[bytes]) -> bytes:
    size = len(leaves)
    if size == 0:
        return _sha(b"")
    if size == 1:
        return leaves[0]
    pivot = _split(size)
    return _node_hash(merkle_root(leaves[:pivot]), merkle_root(leaves[pivot:]))


def inclusion_proof(leaves: list[bytes], index: int) -> list[bytes]:
    size = len(leaves)
    if not 0 <= index < size:
        raise IndexError("index out of range")
    if size == 1:
        return []
    pivot = _split(size)
    if index < pivot:
        return inclusion_proof(leaves[:pivot], index) + [merkle_root(leaves[pivot:])]
    return inclusion_proof(leaves[pivot:], index - pivot) + [merkle_root(leaves[:pivot])]


def _rebuild(index: int, size: int, leaf: bytes, proof: list[bytes]) -> bytes:
    if size == 1:
        if proof:
            raise ValueError("proof too long")
        return leaf
    pivot = _split(size)
    sibling = proof.pop()
    if index < pivot:
        return _node_hash(_rebuild(index, pivot, leaf, proof), sibling)
    return _node_hash(sibling, _rebuild(index - pivot, size - pivot, leaf, proof))


def verify_inclusion(index: int, size: int, leaf: bytes, proof: list[bytes], root: bytes) -> bool:
    if not 0 <= index < size:
        return False
    if len(leaf) != 32 or len(root) != 32 or any(len(item) != 32 for item in proof):
        return False
    try:
        return _rebuild(index, size, leaf, list(proof)) == root
    except (IndexError, ValueError):
        return False


def consistency_proof(leaves: list[bytes], old_size: int) -> list[bytes]:
    size = len(leaves)
    if not 0 <= old_size <= size:
        raise ValueError("old_size must be in range")
    if old_size in {0, size}:
        return []
    return _subproof(old_size, leaves, True)


def _subproof(old_size: int, leaves: list[bytes], on_old_path: bool) -> list[bytes]:
    size = len(leaves)
    if old_size == size:
        return [] if on_old_path else [merkle_root(leaves)]
    pivot = _split(size)
    if old_size <= pivot:
        return _subproof(old_size, leaves[:pivot], on_old_path) + [merkle_root(leaves[pivot:])]
    return _subproof(old_size - pivot, leaves[pivot:], False) + [merkle_root(leaves[:pivot])]


def verify_consistency(old_size: int, new_size: int, proof: list[bytes], old_root: bytes, new_root: bytes) -> bool:
    if old_size < 0 or new_size < 0 or old_size > new_size:
        return False
    if len(old_root) != 32 or len(new_root) != 32 or any(len(item) != 32 for item in proof):
        return False
    if old_size == 0:
        return (
            old_root == _sha(b"")
            and not proof
            and (new_size != 0 or new_root == old_root)
        )
    if old_size == new_size:
        return old_root == new_root and not proof
    path = list(proof)
    if old_size & (old_size - 1) == 0:
        path = [old_root] + path
    node = old_size - 1
    last = new_size - 1
    while node & 1:
        node >>= 1
        last >>= 1
    if not path:
        return False
    old_acc = path[0]
    new_acc = path[0]
    for sibling in path[1:]:
        if last == 0:
            return False
        if (node & 1) or node == last:
            old_acc = _node_hash(sibling, old_acc)
            new_acc = _node_hash(sibling, new_acc)
            if not (node & 1):
                while (node & 1) == 0 and node != 0:
                    node >>= 1
                    last >>= 1
        else:
            new_acc = _node_hash(new_acc, sibling)
        node >>= 1
        last >>= 1
    return old_acc == old_root and new_acc == new_root and last == 0


@dataclass(frozen=True)
class SignedTreeHead:
    tree_size: int
    root_hash: str
    signature: str
    public_key: str
    timestamp: float

    def signing_message(self) -> bytes:
        return _tree_head_message(self.tree_size, self.root_hash, self.timestamp)

    def verify(self, *, public_key: str | None = None) -> bool:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        if public_key is not None and public_key != self.public_key:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.public_key)).verify(bytes.fromhex(self.signature), self.signing_message())
            return True
        except (InvalidSignature, TypeError, ValueError):
            return False


class ReceiptMerkleLog:
    def __init__(self, key: AttestationKey | None = None) -> None:
        self._key = key
        self._leaves: list[bytes] = []

    @property
    def size(self) -> int:
        return len(self._leaves)

    def leaf_at(self, index: int) -> bytes:
        return self._leaves[index]

    def append(self, receipt: dict[str, Any]) -> int:
        self._leaves.append(receipt_leaf(receipt))
        return len(self._leaves) - 1

    def root(self) -> bytes:
        return merkle_root(self._leaves)

    def root_hex(self) -> str:
        return self.root().hex()

    def prove_inclusion(self, index: int) -> list[bytes]:
        return inclusion_proof(self._leaves, index)

    def prove_consistency(self, old_size: int) -> list[bytes]:
        return consistency_proof(self._leaves, old_size)

    def signed_tree_head(self) -> SignedTreeHead:
        if self._key is None:
            raise RuntimeError("a signing key is required")
        root_hex = self.root_hex()
        timestamp = time.time()
        message = _tree_head_message(self.size, root_hex, timestamp)
        return SignedTreeHead(
            self.size,
            root_hex,
            self._key.sign(message),
            self._key.public_hex(),
            timestamp,
        )


# Backward-friendly alias for the design package name.
MerkleTransparencyLog = ReceiptMerkleLog

__all__ = [
    "SignedTreeHead",
    "ReceiptMerkleLog",
    "MerkleTransparencyLog",
    "leaf_hash",
    "receipt_leaf",
    "merkle_root",
    "inclusion_proof",
    "verify_inclusion",
    "consistency_proof",
    "verify_consistency",
]
