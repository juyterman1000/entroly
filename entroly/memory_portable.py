"""Cryptographically portable, provenance-preserving memory transfer across agents.

When work moves from one agent or model runtime to another (for example Claude
to Codex or Gemini), exporting plain text drops provenance and creates a path
for indirect prompt injection or authority laundering on import.

This module serializes a slice of ``MemoryOS`` into a deterministic,
content-addressed SHA-256 chain (with an optional Ed25519 signature over the
canonical header) and enforces three fail-closed properties on import:

1. **Tamper evidence:** every leaf digest and the chain root are recomputed
   before any memory is admitted; any modified, reordered, or dropped entry
   refuses the bundle.
2. **Pinned signature verification:** when ``require_signature=True`` (the
   default), the bundle must carry a valid Ed25519 signature from a key pinned
   for the declared ``issuer``.
3. **Non-amplifying, injection-resistant re-hydration:** every imported entry
   is scanned through ``MemoryOS.scan_safety`` and its authority is clamped to
   ``min(entry.authority, bundle.trust_ceiling, max_imported_authority)``.
   Roles incompatible with the clamped authority (such as ``directive`` or
   ``evidence``) are downgraded to ``observation`` or ``inference`` instead of
   silently inheriting high trust.

No network call is made.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .memory import (
    AUTHORITY_ORDER,
    MemoryAuthority,
    MemoryOS,
    MemoryRole,
    SafetyPolicy,
    min_authority_level,
)

PORTABLE_BUNDLE_SCHEMA = "entroly.portable-memory.v1"
_SIGNATURE_DOMAIN = b"entroly.portable-memory.v1\x00"
_GENESIS_DIGEST = "sha256:" + ("0" * 64)


class PortableMemoryError(RuntimeError):
    """Raised when a portable memory bundle fails integrity, signature, or schema checks."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(slots=True)
class BundleVerificationResult:
    valid: bool
    merkle_root: str
    entry_count: int
    issuer: str
    signature_verified: bool
    kid: str | None = None
    reason: str = "ok"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PortableImportReceipt:
    merkle_root: str
    issuer: str
    signature_verified: bool
    imported_ids: list[str] = field(default_factory=list)
    id_map: dict[str, str] = field(default_factory=dict)
    blocked_entries: list[dict[str, str]] = field(default_factory=list)
    clamped_count: int = 0
    role_downgraded_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _crypto():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )
    except ImportError as exc:
        raise PortableMemoryError(
            "crypto-unavailable",
            "signing/verifying portable memory bundles requires the cryptography package",
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature, Encoding, PublicFormat


def _key_id(public_key_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()[:16]


def _downgrade_role_for_authority(role: MemoryRole, authority: MemoryAuthority) -> tuple[MemoryRole, bool]:
    rank = AUTHORITY_ORDER[authority]
    if role == "directive" and rank < AUTHORITY_ORDER["user_direct"]:
        return ("inference" if rank >= AUTHORITY_ORDER["agent_inferred"] else "observation"), True
    if role == "evidence" and rank < AUTHORITY_ORDER["verified_evidence"]:
        return ("inference" if rank >= AUTHORITY_ORDER["agent_inferred"] else "observation"), True
    if role == "preference" and rank < AUTHORITY_ORDER["agent_inferred"]:
        return "observation", True
    return role, False


def _header_for_signing(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_count": bundle["entry_count"],
        "exported_at_ms": bundle["exported_at_ms"],
        "issuer": bundle["issuer"],
        "merkle_root": bundle["merkle_root"],
        "schema": bundle["schema"],
        "source_agent": bundle["source_agent"],
        "trust_ceiling": bundle["trust_ceiling"],
    }


def export_portable_bundle(
    memory_os: MemoryOS,
    *,
    agent_id: str = "default",
    include_shared: bool = False,
    min_authority: MemoryAuthority | None = None,
    issuer: str = "local",
    trust_ceiling: MemoryAuthority = "agent_inferred",
    signing_key_hex: str | None = None,
) -> dict[str, Any]:
    """Export a content-addressed, optionally Ed25519-signed memory bundle."""
    if trust_ceiling not in AUTHORITY_ORDER:
        raise ValueError(f"unknown trust_ceiling: {trust_ceiling!r}")
    min_rank = AUTHORITY_ORDER[min_authority] if min_authority is not None else -1
    snap = memory_os.snapshot()
    raw_entries = snap.get("entries", [])
    if not isinstance(raw_entries, list):
        raw_entries = []

    selected: list[dict[str, Any]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        entry_agent = str(raw.get("agent_id", "default"))
        if entry_agent != agent_id and not (include_shared and entry_agent == "shared"):
            continue
        auth = str(raw.get("authority", "agent_inferred"))
        if AUTHORITY_ORDER.get(auth, 0) < min_rank:
            continue
        selected.append(raw)

    selected.sort(key=lambda item: str(item.get("id", "")))

    chain: list[dict[str, Any]] = []
    prev_digest = _GENESIS_DIGEST
    for raw in selected:
        entry_payload = {
            "authority": str(raw.get("authority", "agent_inferred")),
            "content": str(raw.get("content", "")),
            "derived_from": list(raw.get("derived_from", [])),
            "id": str(raw.get("id", "")),
            "importance": float(raw.get("importance", 0.5)),
            "memory_role": str(raw.get("memory_role", "observation")),
            "origin_digest": str(raw.get("origin_digest", "")),
            "source": str(raw.get("source", "manual")),
            "supersedes": raw.get("supersedes"),
            "tags": list(raw.get("tags", [])),
            "temporal_class": str(raw.get("temporal_class", "unclassified")),
            "tier": str(raw.get("tier", "working")),
        }
        leaf_digest = _sha256(_canonical({"entry": entry_payload, "prev": prev_digest}))
        chain.append({
            "entry": entry_payload,
            "leaf_digest": leaf_digest,
            "prev_digest": prev_digest,
        })
        prev_digest = leaf_digest

    merkle_root = prev_digest
    bundle: dict[str, Any] = {
        "schema": PORTABLE_BUNDLE_SCHEMA,
        "issuer": str(issuer),
        "source_agent": str(agent_id),
        "trust_ceiling": trust_ceiling,
        "exported_at_ms": int(time.time() * 1000),
        "entry_count": len(chain),
        "merkle_root": merkle_root,
        "entries": chain,
        "signature": None,
    }

    if signing_key_hex is not None:
        Ed25519PrivateKey, _, _, Encoding, PublicFormat = _crypto()
        try:
            priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(signing_key_hex))
        except Exception as exc:
            raise PortableMemoryError("invalid-signing-key", str(exc)) from exc
        pub_hex = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        sig = priv.sign(_SIGNATURE_DOMAIN + _canonical(_header_for_signing(bundle)))
        bundle["signature"] = {
            "algorithm": "ed25519",
            "kid": _key_id(pub_hex),
            "sig": sig.hex(),
        }

    return bundle


def verify_portable_bundle(
    bundle: Mapping[str, Any],
    *,
    pinned_public_keys: Mapping[str, str | list[str] | tuple[str, ...]] | None = None,
    require_signature: bool = True,
) -> BundleVerificationResult:
    """Verify hash-chain integrity and (when present or required) Ed25519 signature."""
    if not isinstance(bundle, Mapping):
        raise PortableMemoryError("malformed-bundle", "bundle must be a JSON object")
    if bundle.get("schema") != PORTABLE_BUNDLE_SCHEMA:
        raise PortableMemoryError("unsupported-schema", f"unexpected schema: {bundle.get('schema')!r}")
    if bundle.get("trust_ceiling") not in AUTHORITY_ORDER:
        raise PortableMemoryError("malformed-bundle", f"invalid trust_ceiling: {bundle.get('trust_ceiling')!r}")

    entries = bundle.get("entries")
    if not isinstance(entries, list):
        raise PortableMemoryError("malformed-bundle", "entries must be a list")
    if bundle.get("entry_count") != len(entries):
        raise PortableMemoryError("entry-count-mismatch", "entry_count does not match entries length")

    prev_digest = _GENESIS_DIGEST
    for idx, node in enumerate(entries):
        if not isinstance(node, Mapping) or "entry" not in node:
            raise PortableMemoryError("malformed-entry", f"entry {idx} is malformed")
        if node.get("prev_digest") != prev_digest:
            raise PortableMemoryError("broken-chain", f"entry {idx} prev_digest does not chain")
        expected_leaf = _sha256(_canonical({"entry": node["entry"], "prev": prev_digest}))
        if node.get("leaf_digest") != expected_leaf:
            raise PortableMemoryError("tampered-entry", f"entry {idx} digest mismatch")
        prev_digest = expected_leaf

    if bundle.get("merkle_root") != prev_digest:
        raise PortableMemoryError("root-mismatch", "merkle_root does not match recomputed chain root")

    issuer = str(bundle.get("issuer", ""))
    sig_obj = bundle.get("signature")
    if sig_obj is None:
        if require_signature:
            raise PortableMemoryError("missing-signature", "bundle is unsigned and require_signature=True")
        return BundleVerificationResult(
            valid=True,
            merkle_root=prev_digest,
            entry_count=len(entries),
            issuer=issuer,
            signature_verified=False,
        )

    if not isinstance(sig_obj, Mapping) or sig_obj.get("algorithm") != "ed25519":
        raise PortableMemoryError("unsupported-signature", "only ed25519 signatures are accepted")
    if not pinned_public_keys or issuer not in pinned_public_keys:
        raise PortableMemoryError("untrusted-issuer", f"no pinned public key for issuer {issuer!r}")

    raw_keys = pinned_public_keys[issuer]
    key_list = [raw_keys] if isinstance(raw_keys, str) else list(raw_keys)
    kid = str(sig_obj.get("kid", ""))
    matched_key = next((k for k in key_list if _key_id(k) == kid), None)
    if matched_key is None:
        raise PortableMemoryError("unpinned-key", f"key id {kid!r} is not pinned for issuer {issuer!r}")

    _, Ed25519PublicKey, InvalidSignature, _, _ = _crypto()
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(matched_key))
        sig_bytes = bytes.fromhex(str(sig_obj.get("sig", "")))
        pub.verify(sig_bytes, _SIGNATURE_DOMAIN + _canonical(_header_for_signing(bundle)))
    except (ValueError, InvalidSignature) as exc:
        raise PortableMemoryError("invalid-signature", f"signature verification failed: {exc}") from exc

    return BundleVerificationResult(
        valid=True,
        merkle_root=prev_digest,
        entry_count=len(entries),
        issuer=issuer,
        signature_verified=True,
        kid=kid,
    )


def import_portable_bundle(
    memory_os: MemoryOS,
    bundle: Mapping[str, Any],
    *,
    target_agent_id: str = "default",
    pinned_public_keys: Mapping[str, str | list[str] | tuple[str, ...]] | None = None,
    require_signature: bool = True,
    max_imported_authority: MemoryAuthority = "agent_inferred",
    safety_policy: SafetyPolicy | None = None,
) -> PortableImportReceipt:
    """Verify and re-hydrate a portable memory bundle with strict non-amplification."""
    if max_imported_authority not in AUTHORITY_ORDER:
        raise ValueError(f"unknown max_imported_authority: {max_imported_authority!r}")

    verification = verify_portable_bundle(
        bundle,
        pinned_public_keys=pinned_public_keys,
        require_signature=require_signature,
    )
    bundle_ceiling: MemoryAuthority = bundle["trust_ceiling"]
    effective_ceiling = min_authority_level(bundle_ceiling, max_imported_authority)
    bundle_tag = f"portable_bundle:{verification.merkle_root.split(':', 1)[-1][:16]}"

    receipt = PortableImportReceipt(
        merkle_root=verification.merkle_root,
        issuer=verification.issuer,
        signature_verified=verification.signature_verified,
    )

    for node in bundle["entries"]:
        entry = node["entry"]
        orig_id = str(entry.get("id", ""))
        content = str(entry.get("content", "")).strip()
        if not content:
            receipt.blocked_entries.append({"orig_id": orig_id, "reason": "empty_content"})
            continue

        raw_auth = str(entry.get("authority", "untrusted"))
        entry_auth: MemoryAuthority = raw_auth if raw_auth in AUTHORITY_ORDER else "untrusted"  # type: ignore[assignment]
        clamped_auth = min_authority_level(entry_auth, effective_ceiling)
        if clamped_auth != entry_auth:
            receipt.clamped_count += 1

        raw_role = str(entry.get("memory_role", "observation"))
        entry_role: MemoryRole = raw_role if raw_role in ("observation", "inference", "evidence", "directive", "preference") else "observation"  # type: ignore[assignment]
        safe_role, downgraded = _downgrade_role_for_authority(entry_role, clamped_auth)
        if downgraded:
            receipt.role_downgraded_count += 1

        raw_derived = entry.get("derived_from", [])
        raw_parent_ids = [str(p) for p in raw_derived] if isinstance(raw_derived, list) else []
        mapped_parents = [receipt.id_map[p] for p in raw_parent_ids if p in receipt.id_map]
        # Fail-closed non-amplification: if an entry claims derivation from a
        # parent that was blocked or omitted from the bundle, clamp to untrusted.
        if len(mapped_parents) < len(raw_parent_ids):
            if clamped_auth != "untrusted":
                clamped_auth = "untrusted"
                receipt.clamped_count += 1
            safe_role, extra_downgraded = _downgrade_role_for_authority(safe_role, clamped_auth)
            if extra_downgraded and not downgraded:
                receipt.role_downgraded_count += 1

        tags = [str(t) for t in entry.get("tags", []) if isinstance(t, str)]
        if bundle_tag not in tags:
            tags.append(bundle_tag)
        issuer_tag = f"issuer:{verification.issuer}"
        if issuer_tag not in tags:
            tags.append(issuer_tag)

        # Injection-resistant re-hydration: scan_safety runs inside remember().
        try:
            new_id = memory_os.remember(
                content,
                agent_id=target_agent_id,
                importance=float(entry.get("importance", 0.5)),
                tier=str(entry.get("tier", "working")),  # type: ignore[arg-type]
                source=f"portable:{verification.issuer}:{entry.get('source', 'unknown')}",
                tags=tags,
                safety_policy=safety_policy or "block",
                temporal_class=entry.get("temporal_class"),
                authority=clamped_auth,
                memory_role=safe_role,
                derived_from=mapped_parents or None,
            )
        except ValueError as exc:
            receipt.blocked_entries.append({"orig_id": orig_id, "reason": str(exc)})
            continue

        receipt.imported_ids.append(new_id)
        if orig_id:
            receipt.id_map[orig_id] = new_id

    return receipt


__all__ = [
    "PORTABLE_BUNDLE_SCHEMA",
    "BundleVerificationResult",
    "PortableImportReceipt",
    "PortableMemoryError",
    "export_portable_bundle",
    "import_portable_bundle",
    "verify_portable_bundle",
]
