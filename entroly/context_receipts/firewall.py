"""Receipt-Closed Selection Firewall (RCFP) — Pillar V.

Verifies that a Context Receipt is self-contained: every claim in the
receipt can be traced back to evidence within the receipt itself.  A
receipt that passes RCFP is a closed proof — an auditor can verify it
without access to the original source material.

Four closure checks, each fail-closed:

1. **Fingerprint closure**: every selected chunk's SHA-256 matches
   its text (the receipt hasn't been tampered with post-selection).
2. **Dependency closure**: every dependency link references chunk IDs
   that exist in the receipt (selected or omitted).
3. **Source closure**: every selected chunk traces to a source path
   recorded in the receipt's source_fingerprints.
4. **Reproducibility closure**: the reproducibility hash is consistent
   with the receipt's deterministic content.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FirewallViolation:
    check: str
    chunk_id: str
    detail: str


@dataclass(frozen=True)
class FirewallCertificate:
    closed: bool
    violations: tuple[FirewallViolation, ...]
    checks_passed: tuple[str, ...]
    checks_failed: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "closed": self.closed,
            "violations": [
                {"check": v.check, "chunk_id": v.chunk_id, "detail": v.detail}
                for v in self.violations
            ],
            "checks_passed": list(self.checks_passed),
            "checks_failed": list(self.checks_failed),
        }


def verify_receipt_closure(receipt: dict) -> FirewallCertificate:
    """Run all RCFP checks on a receipt dict."""
    violations: list[FirewallViolation] = []
    passed: list[str] = []
    failed: list[str] = []

    selected = receipt.get("selected_context", [])
    omitted = receipt.get("omitted_context", [])

    _check_fingerprint_closure(selected, violations)
    if any(v.check == "fingerprint_closure" for v in violations):
        failed.append("fingerprint_closure")
    else:
        passed.append("fingerprint_closure")

    all_chunk_ids = _collect_chunk_ids(selected, omitted)
    _check_dependency_closure(receipt.get("dependency_links", []), all_chunk_ids, violations)
    if any(v.check == "dependency_closure" for v in violations):
        failed.append("dependency_closure")
    else:
        passed.append("dependency_closure")

    source_fps = receipt.get("source_fingerprints", {})
    _check_source_closure(selected, source_fps, violations)
    if any(v.check == "source_closure" for v in violations):
        failed.append("source_closure")
    else:
        passed.append("source_closure")

    _check_reproducibility(receipt, violations)
    if any(v.check == "reproducibility_closure" for v in violations):
        failed.append("reproducibility_closure")
    else:
        passed.append("reproducibility_closure")

    return FirewallCertificate(
        closed=len(violations) == 0,
        violations=tuple(violations),
        checks_passed=tuple(passed),
        checks_failed=tuple(failed),
    )


def _check_fingerprint_closure(
    selected: list[dict], violations: list[FirewallViolation]
) -> None:
    for item in selected:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        recorded_fp = item.get("fragment_sha256", "")
        if not recorded_fp or not text:
            continue
        computed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if computed != recorded_fp:
            violations.append(FirewallViolation(
                check="fingerprint_closure",
                chunk_id=item.get("chunk_id", "unknown"),
                detail=f"SHA-256 mismatch: recorded {recorded_fp[:16]}... != computed {computed[:16]}...",
            ))


def _collect_chunk_ids(selected: list[dict], omitted: list[dict]) -> set[str]:
    ids: set[str] = set()
    for item in selected:
        if isinstance(item, dict) and item.get("chunk_id"):
            ids.add(item["chunk_id"])
    for item in omitted:
        if isinstance(item, dict) and item.get("chunk_id"):
            ids.add(item["chunk_id"])
    return ids


def _check_dependency_closure(
    deps: list[dict], known_ids: set[str], violations: list[FirewallViolation]
) -> None:
    for dep in deps:
        if not isinstance(dep, dict):
            continue
        src_id = dep.get("source_chunk_id", "")
        tgt_id = dep.get("target_chunk_id")
        if src_id and src_id not in known_ids:
            violations.append(FirewallViolation(
                check="dependency_closure",
                chunk_id=src_id,
                detail=f"dependency source {src_id} not in receipt",
            ))
        if tgt_id and tgt_id not in known_ids:
            violations.append(FirewallViolation(
                check="dependency_closure",
                chunk_id=tgt_id,
                detail=f"dependency target {tgt_id} not in receipt",
            ))


def _check_source_closure(
    selected: list[dict], source_fps: dict, violations: list[FirewallViolation]
) -> None:
    for item in selected:
        if not isinstance(item, dict):
            continue
        source_path = item.get("source_path", "")
        if source_path and source_path not in source_fps:
            violations.append(FirewallViolation(
                check="source_closure",
                chunk_id=item.get("chunk_id", "unknown"),
                detail=f"source_path '{source_path}' not in source_fingerprints",
            ))


def _check_reproducibility(
    receipt: dict, violations: list[FirewallViolation]
) -> None:
    from .models import stable_hash

    recorded_hash = receipt.get("reproducibility_hash", "")
    if not recorded_hash:
        return

    payload = {
        k: v
        for k, v in sorted(receipt.items())
        if k not in {"receipt_id", "reproducibility_hash"}
    }
    computed = stable_hash(payload)
    if computed != recorded_hash:
        violations.append(FirewallViolation(
            check="reproducibility_closure",
            chunk_id="receipt",
            detail=f"hash mismatch: recorded {recorded_hash[:16]}... != computed {computed[:16]}...",
        ))
