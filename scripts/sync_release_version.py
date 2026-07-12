#!/usr/bin/env python3
"""Synchronize Entroly release versions across approved release surfaces.

The release version is security-sensitive metadata. This tool updates only the
explicit allowlist below; it never rewrites workflow definitions, historical
release notes, arbitrary documentation, or unreviewed files. The narrow scope
prevents an Actions token from accidentally modifying its own workflow and
makes release drift reviewable.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
PROJECT_VERSION_RE = re.compile(r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"')

# Every active file that intentionally embeds the current public release.
# Keep this list explicit: adding a new package or manifest must be reviewed.
RELEASE_SURFACES: tuple[str, ...] = (
    ".claude-plugin/manifest.json",
    ".mcpb-build/manifest.json",
    "entroly-core/Cargo.lock",
    "entroly-core/Cargo.toml",
    "entroly-core/README.md",
    "entroly-core/pyproject.toml",
    "entroly-qccr/Cargo.lock",
    "entroly-qccr/Cargo.toml",
    "entroly-wasm/Cargo.lock",
    "entroly-wasm/Cargo.toml",
    "entroly-wasm/package.json",
    "entroly/__init__.py",
    "entroly/cli.py",
    "entroly/daemon.py",
    "entroly/native_status.py",
    "entroly/npm-alias/package.json",
    "entroly/npm/package.json",
    "entroly/pyproject.toml",
    "entroly/server.py",
    "integrations/openclaw/package.json",
    "packaging/homebrew/README.md",
    "packaging/homebrew/entroly.rb",
    "pyproject.toml",
    "server.json",
    "tests/test_release_surface.py",
)


def _release_note(version: str) -> str:
    return f"""# Entroly {version}

Release metadata for Entroly {version} is synchronized across Python, Rust,
npm, MCP, OpenClaw, WASM, plugin, Docker, Homebrew, and GitHub release
surfaces.

See the GitHub release and merged pull requests for the complete feature,
security, compatibility, and migration notes.
"""


def synchronize(root: Path, target: str) -> list[str]:
    if not SEMVER_RE.fullmatch(target):
        raise ValueError(f"invalid semantic version: {target!r}")

    pyproject_path = root / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    match = PROJECT_VERSION_RE.search(pyproject)
    if not match:
        raise RuntimeError("unable to resolve current version from pyproject.toml")
    current = match.group(1)

    if current == target:
        return []

    changed: list[str] = []
    missing: list[str] = []
    stale: list[str] = []
    old_identifier = current.replace(".", "_")
    new_identifier = target.replace(".", "_")

    for relative_name in RELEASE_SURFACES:
        path = root / relative_name
        if not path.is_file():
            missing.append(relative_name)
            continue

        data = path.read_bytes()
        if current.encode() not in data and old_identifier.encode() not in data:
            stale.append(relative_name)
            continue

        updated = data.replace(current.encode(), target.encode())
        updated = updated.replace(old_identifier.encode(), new_identifier.encode())
        if updated != data:
            path.write_bytes(updated)
            changed.append(relative_name)

    if missing:
        raise RuntimeError("release surfaces are missing: " + ", ".join(missing))
    if stale:
        raise RuntimeError(
            "release surfaces no longer contain the current version; review the allowlist: "
            + ", ".join(stale)
        )

    note = root / "docs" / "releases" / f"v{target}.md"
    if not note.exists():
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(_release_note(target), encoding="utf-8")
        changed.append(note.relative_to(root).as_posix())

    return sorted(changed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    changed = synchronize(root, args.version)
    if changed:
        print("Synchronized release surfaces:")
        for path in changed:
            print(f"  {path}")
    else:
        print(f"Release surfaces already synchronized at {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
