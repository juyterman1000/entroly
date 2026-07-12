#!/usr/bin/env python3
"""Synchronize Entroly release versions across tracked release surfaces.

The script intentionally preserves historical release notes while updating every
other tracked text file that advertises the current release. It is idempotent,
validates the requested semantic version, and fails if stale current-version
references remain outside ``docs/releases``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
PROJECT_VERSION_RE = re.compile(r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"')


def _tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _release_note(version: str) -> str:
    return f"""# Entroly {version}

Entroly {version} introduces the Frontier Context Quality Layer and completes
its first production release across the Python, Rust, npm, MCP, OpenClaw,
WASM, plugin, Docker, and GitHub release surfaces.

Highlights:

- deterministic, content-addressed Context Checks linked to verified Context
  Commits;
- retrospective changed-file recall, indexed recall, and precision evidence;
- conservative risk escalation for sensitive, unsupported, truncated, and
  unmeasured context;
- replayable JSON and Markdown artifacts with a strict schema contract;
- reusable GitHub Action integration with artifact publication and CI risk
  gating;
- synchronized release metadata across Python, Rust, QCCR, WASM, npm, MCP,
  OpenClaw, plugins, manifests, lockfiles, Homebrew, and release verification
  tests.

Changed-file coverage is evidence about context selection. It is not presented
as proof of model correctness or complete task knowledge.
"""


def synchronize(root: Path, target: str) -> list[str]:
    if not SEMVER_RE.fullmatch(target):
        raise ValueError(f"invalid semantic version: {target!r}")

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = PROJECT_VERSION_RE.search(pyproject)
    if not match:
        raise RuntimeError("unable to resolve current version from pyproject.toml")
    current = match.group(1)

    changed: list[str] = []
    old_identifier = current.replace(".", "_")
    new_identifier = target.replace(".", "_")

    for path in _tracked_files(root):
        relative = path.relative_to(root)
        if relative.parts[:2] == ("docs", "releases"):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        updated = data.replace(current.encode(), target.encode())
        updated = updated.replace(old_identifier.encode(), new_identifier.encode())
        if updated != data:
            path.write_bytes(updated)
            changed.append(relative.as_posix())

    note = root / "docs" / "releases" / f"v{target}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    rendered_note = _release_note(target)
    if not note.exists() or note.read_text(encoding="utf-8") != rendered_note:
        note.write_text(rendered_note, encoding="utf-8")
        changed.append(note.relative_to(root).as_posix())

    stale: list[str] = []
    for path in _tracked_files(root):
        relative = path.relative_to(root)
        if relative.parts[:2] == ("docs", "releases"):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" not in data and current.encode() in data:
            stale.append(relative.as_posix())
    if current != target and stale:
        raise RuntimeError("stale release references remain: " + ", ".join(stale))

    return sorted(set(changed))


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
