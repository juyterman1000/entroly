from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.70"


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_audited_qccr_manifest_and_locks_match_release() -> None:
    manifest = text("entroly-qccr-audit/Cargo.toml")
    assert re.search(rf'^version\s*=\s*"{re.escape(VERSION)}"$', manifest, re.MULTILINE)

    for lock_path in ("entroly-core/Cargo.lock", "entroly-wasm/Cargo.lock"):
        lock = text(lock_path)
        assert re.search(
            rf'\[\[package\]\]\s*\nname = "entroly-qccr-audit"\s*\nversion = "{re.escape(VERSION)}"',
            lock,
        )


def test_both_runtime_manifests_depend_on_audited_qccr() -> None:
    assert 'entroly-qccr-audit = { path = "../entroly-qccr-audit"' in text(
        "entroly-core/Cargo.toml"
    )
    assert 'entroly-qccr-audit = { path = "../entroly-qccr-audit"' in text(
        "entroly-wasm/Cargo.toml"
    )


def test_both_version_synchronizers_include_audit_surfaces() -> None:
    for script in ("scripts/bump_version.py", "scripts/sync_release_version.py"):
        source = text(script)
        assert "entroly-qccr-audit/Cargo.toml" in source
        assert "entroly-qccr-audit" in source
