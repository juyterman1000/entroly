from __future__ import annotations

import hashlib

from entroly.receipt_merkle import (
    consistency_proof,
    inclusion_proof,
    leaf_hash,
    merkle_root,
    verify_consistency,
    verify_inclusion,
)


def _leaves(count: int) -> list[bytes]:
    return [leaf_hash(f"leaf-{idx}".encode()) for idx in range(count)]


def test_roots_match_expected_hashes() -> None:
    assert merkle_root([]) == hashlib.sha256(b"").digest()
    one = leaf_hash(b"x")
    assert merkle_root([one]) == one
    two = [leaf_hash(b"a"), leaf_hash(b"b")]
    assert merkle_root(two) == hashlib.sha256(b"\x01" + two[0] + two[1]).digest()


def test_inclusion_proofs_for_small_trees() -> None:
    for count in range(1, 18):
        leaves = _leaves(count)
        root = merkle_root(leaves)
        for index, leaf in enumerate(leaves):
            proof = inclusion_proof(leaves, index)
            assert verify_inclusion(index, count, leaf, proof, root)
            assert not verify_inclusion(index, count, leaf_hash(b"other"), proof, root)


def test_consistency_proofs_for_prefixes() -> None:
    leaves = _leaves(20)
    new_root = merkle_root(leaves)
    for old_size in range(1, 20):
        old_root = merkle_root(leaves[:old_size])
        proof = consistency_proof(leaves, old_size)
        assert verify_consistency(old_size, 20, proof, old_root, new_root)


def test_consistency_false_for_different_old_root() -> None:
    leaves = _leaves(20)
    proof = consistency_proof(leaves, 7)
    assert not verify_consistency(7, 20, proof, leaf_hash(b"other"), merkle_root(leaves))


def test_empty_tree_consistency_requires_the_canonical_empty_root() -> None:
    leaves = _leaves(3)
    proof = consistency_proof(leaves, 0)
    assert proof == []
    assert verify_consistency(0, 3, proof, merkle_root([]), merkle_root(leaves))
    assert not verify_consistency(0, 3, proof, b"x" * 32, merkle_root(leaves))
    assert not verify_consistency(0, 0, [], b"x" * 32, b"x" * 32)


def test_inclusion_rejects_non_digest_proof_elements() -> None:
    leaf = leaf_hash(b"leaf")
    assert not verify_inclusion(0, 1, leaf, [b"short"], leaf)
