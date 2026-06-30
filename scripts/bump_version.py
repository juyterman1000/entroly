#!/usr/bin/env python3
"""Bump version across all Entroly manifests.

Usage: python scripts/bump_version.py 1.0.39
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    ("pyproject.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("pyproject.toml", r'entroly-core>=[0-9]+\.[0-9]+\.[0-9]+,<2', 'entroly-core>={v},<2'),
    ("entroly/pyproject.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly/pyproject.toml", r'entroly-core>=[0-9]+\.[0-9]+\.[0-9]+,<2', 'entroly-core>={v},<2'),
    ("entroly-core/pyproject.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-core/Cargo.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-core/Cargo.lock",
        r'(name\s*=\s*"entroly-core"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-core/Cargo.lock",
        r'(name\s*=\s*"entroly-qccr"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-core/README.md", r'entroly-core>=[0-9]+\.[0-9]+\.[0-9]+', 'entroly-core>={v}'),
    ("entroly-qccr/Cargo.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-qccr/Cargo.lock",
        r'(name\s*=\s*"entroly-qccr"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-wasm/Cargo.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-wasm/Cargo.lock",
        r'(name\s*=\s*"entroly-wasm"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-wasm/Cargo.lock",
        r'(name\s*=\s*"entroly-qccr"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-wasm/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly-wasm/pkg/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/npm/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/npm-alias/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/npm-alias/package.json", r'"entroly-wasm"\s*:\s*"[^"]+"', '"entroly-wasm": "{v}"'),
    ("entroly/__init__.py", r'__version__\s*=\s*"[^"]+"', '__version__ = "{v}"'),
    ("entroly/native_status.py",
        r'MIN_ENTROLY_CORE_VERSION\s*=\s*"[^"]+"',
        'MIN_ENTROLY_CORE_VERSION = "{v}"'),
    ("entroly/cli.py", r'__version__\s*=\s*"[^"]+"', '__version__ = "{v}"'),
    ("entroly/cli.py", r'entroly-core>=[0-9]+\.[0-9]+\.[0-9]+', 'entroly-core>={v}'),
    ("entroly/server.py", r'_version\s*=\s*"[^"]+"', '_version = "{v}"'),
    (".claude-plugin/manifest.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/daemon.py", r'version:\s*str\s*=\s*"[^"]+"', 'version: str = "{v}"'),
    # Homebrew: URL pin only. The sha256 changes per-release and must be
    # updated manually after the PyPI tarball is published.
    ("packaging/homebrew/entroly.rb",
        r'entroly-[0-9]+\.[0-9]+\.[0-9]+\.tar\.gz', 'entroly-{v}.tar.gz'),
    # Homebrew release runbook (example bash). Two `VER=...` lines.
    ("packaging/homebrew/README.md",
        r'VER=[0-9]+\.[0-9]+\.[0-9]+', 'VER={v}'),
]

SEMVER = re.compile(r"^\d+\.\d+\.\d+([-+].+)?$")


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not SEMVER.match(argv[1]):
        print("usage: bump_version.py <semver>", file=sys.stderr)
        return 2
    new = argv[1]
    pending: dict[Path, str] = {}
    changed: list[str] = []
    for rel, pattern, template in TARGETS:
        path = ROOT / rel
        if not path.exists():
            print(f"  {rel} missing; skipping generated artifact")
            continue
        text = pending.get(path)
        if text is None:
            text = path.read_text(encoding="utf-8")
        updated, n = re.subn(pattern, template.format(v=new), text, flags=re.MULTILINE)
        if n == 0:
            print(f"!! no match in {rel}", file=sys.stderr)
            return 1
        pending[path] = updated
        changed.append(rel)

    for path, updated in pending.items():
        path.write_text(updated, encoding="utf-8")
    for rel in changed:
        print(f"  {rel} -> {new}")
    print(f"bumped {len(TARGETS)} files to {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))