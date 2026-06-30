from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from entroly.context_receipts.models import stable_json
from entroly.receipt_merkle import inclusion_proof, leaf_hash, merkle_root, verify_inclusion


@dataclass(frozen=True)
class ProvenanceAtom:
    kind: str
    identifier: str
    content_hash: str
    status: str
    detail: str = ""

    def canonical(self) -> bytes:
        return stable_json(
            {
                "kind": self.kind,
                "identifier": self.identifier,
                "content_hash": self.content_hash,
                "status": self.status,
                "detail": self.detail,
            }
        ).encode("utf-8")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "content_hash": self.content_hash,
            "status": self.status,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ProvenanceAtom":
        return cls(
            str(data["kind"]),
            str(data["identifier"]),
            str(data["content_hash"]),
            str(data["status"]),
            str(data.get("detail", "")),
        )


def _atom_leaf(atom: ProvenanceAtom, salt_hex: str) -> bytes:
    return leaf_hash(bytes.fromhex(salt_hex) + atom.canonical())


def receipt_atoms(receipt: dict[str, Any]) -> list[ProvenanceAtom]:
    atoms: list[ProvenanceAtom] = []
    for chunk in receipt.get("selected_context", []) or []:
        if isinstance(chunk, dict):
            atoms.append(
                ProvenanceAtom(
                    kind="source",
                    identifier=str(chunk.get("source_path") or chunk.get("chunk_id") or ""),
                    content_hash=str(chunk.get("fingerprint") or chunk.get("content_hash") or ""),
                    status="selected",
                    detail="; ".join(str(v) for v in (chunk.get("reasons") or [])[:2])[:160],
                )
            )
    for chunk in receipt.get("omitted_context", []) or []:
        if isinstance(chunk, dict):
            atoms.append(
                ProvenanceAtom(
                    kind="source",
                    identifier=str(chunk.get("source_path") or chunk.get("chunk_id") or ""),
                    content_hash=str(chunk.get("fingerprint") or chunk.get("content_hash") or ""),
                    status="omitted",
                    detail=str(chunk.get("reason") or "")[:160],
                )
            )
    return atoms


@dataclass(frozen=True)
class AtomDisclosure:
    atom: ProvenanceAtom
    salt: str
    index: int
    tree_size: int
    audit_path: list[str]
    commitment_root: str

    def verify(self, *, commitment_root: str | None = None) -> bool:
        if commitment_root is not None and commitment_root != self.commitment_root:
            return False
        return verify_inclusion(
            self.index,
            self.tree_size,
            _atom_leaf(self.atom, self.salt),
            [bytes.fromhex(item) for item in self.audit_path],
            bytes.fromhex(self.commitment_root),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom": self.atom.to_dict(),
            "salt": self.salt,
            "index": self.index,
            "tree_size": self.tree_size,
            "audit_path": list(self.audit_path),
            "commitment_root": self.commitment_root,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomDisclosure":
        return cls(
            atom=ProvenanceAtom.from_dict(data["atom"]),
            salt=str(data["salt"]),
            index=int(data["index"]),
            tree_size=int(data["tree_size"]),
            audit_path=[str(item) for item in data.get("audit_path", [])],
            commitment_root=str(data["commitment_root"]),
        )


class ReceiptCommitment:
    def __init__(self, atoms: list[ProvenanceAtom], salts: list[str] | None = None) -> None:
        self._atoms = list(atoms)
        self._salts = list(salts) if salts is not None else [secrets.token_bytes(16).hex() for _ in self._atoms]
        if len(self._atoms) != len(self._salts):
            raise ValueError("atoms and salts must have the same length")
        self._leaves = [_atom_leaf(atom, salt) for atom, salt in zip(self._atoms, self._salts)]
        self._root = merkle_root(self._leaves)

    @classmethod
    def from_receipt(cls, receipt: dict[str, Any]) -> "ReceiptCommitment":
        return cls(receipt_atoms(receipt))

    @property
    def atoms(self) -> list[ProvenanceAtom]:
        return list(self._atoms)

    def root_hex(self) -> str:
        return self._root.hex()

    def disclose(self, index: int) -> AtomDisclosure:
        return AtomDisclosure(
            atom=self._atoms[index],
            salt=self._salts[index],
            index=index,
            tree_size=len(self._leaves),
            audit_path=[item.hex() for item in inclusion_proof(self._leaves, index)],
            commitment_root=self.root_hex(),
        )

    def disclose_where(self, *, identifier: str | None = None, status: str | None = None) -> list[AtomDisclosure]:
        result: list[AtomDisclosure] = []
        for index, atom in enumerate(self._atoms):
            if identifier is not None and atom.identifier != identifier:
                continue
            if status is not None and atom.status != status:
                continue
            result.append(self.disclose(index))
        return result


ReceiptDisclosure = ReceiptCommitment

__all__ = [
    "AtomDisclosure",
    "ProvenanceAtom",
    "ReceiptCommitment",
    "ReceiptDisclosure",
    "receipt_atoms",
]
