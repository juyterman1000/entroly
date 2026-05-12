#!/usr/bin/env python3
"""Bump version across all Entroly manifests.

Usage: python scripts/bump_version.py 0.8.0
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    ("pyproject.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly/pyproject.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-core/pyproject.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-core/Cargo.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-wasm/Cargo.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-wasm/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly-wasm/pkg/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/npm/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/__init__.py", r'__version__\s*=\s*"[^"]+"', '__version__ = "{v}"'),
    ("entroly/cli.py", r'__version__\s*=\s*"[^"]+"', '__version__ = "{v}"'),
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
    for rel, pattern, template in TARGETS:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        updated, n = re.subn(pattern, template.format(v=new), text, flags=re.MULTILINE)
        if n == 0:
            print(f"!! no match in {rel}", file=sys.stderr)
            return 1
        path.write_text(updated, encoding="utf-8")
        print(f"  {rel} -> {new}")
    print(f"bumped {len(TARGETS)} files to {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
