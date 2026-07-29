#!/usr/bin/env python3
"""Synchronize Entroly release versions across approved release surfaces.

Release metadata is security-sensitive. This tool updates only the explicit
allowlist below and applies a typed rule for every file. It never performs a
repository-wide string replacement, rewrites workflow definitions, mutates
historical release notes, or advances the live Homebrew formula before its
post-release workflow has a real artifact URL and checksum.

The synchronizer is convergent: running it when ``pyproject.toml`` already has
the target version still validates and repairs every other approved surface.
All transforms are computed and validated before any file is written, so a
malformed surface cannot leave a partially rewritten working tree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _release_artifacts import MCPB_MANIFEST, rebuild_mcpb  # noqa: E402

SEMVER_TEXT = r"[0-9]+\.[0-9]+\.[0-9]+"
SEMVER_RE = re.compile(rf"^{SEMVER_TEXT}$")

# Every active file that intentionally embeds the current public package
# release. Keep this list explicit: adding a package or manifest requires a
# reviewed transform below. Workflow definitions and historical release notes
# are deliberately excluded.
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
    "pyproject.toml",
    "server.json",
    "tests/test_release_surface.py",
)

TOML_VERSION_SURFACES = {
    "entroly-core/Cargo.toml",
    "entroly-core/pyproject.toml",
    "entroly-qccr/Cargo.toml",
    "entroly-wasm/Cargo.toml",
    "entroly/pyproject.toml",
    "pyproject.toml",
}

JSON_TOP_LEVEL_VERSION_SURFACES = {
    ".claude-plugin/manifest.json",
    ".mcpb-build/manifest.json",
    "entroly-wasm/package.json",
    "entroly/npm-alias/package.json",
    "entroly/npm/package.json",
    "integrations/openclaw/package.json",
}

CARGO_LOCK_PACKAGES: dict[str, tuple[str, ...]] = {
    "entroly-core/Cargo.lock": ("entroly-core", "entroly-qccr"),
    "entroly-qccr/Cargo.lock": ("entroly-qccr",),
    "entroly-wasm/Cargo.lock": ("entroly-wasm", "entroly-qccr"),
}

PYTHON_VERSION_PATTERNS: dict[str, str] = {
    "entroly/__init__.py": rf'^__version__\s*=\s*"(?P<version>{SEMVER_TEXT})"',
    "entroly/cli.py": rf'__version__[^\r\n]*?(?P<version>{SEMVER_TEXT})',
    "entroly/daemon.py": rf'^\s*version:\s*str\s*=\s*"(?P<version>{SEMVER_TEXT})"',
    "entroly/native_status.py": (
        rf'^MIN_ENTROLY_CORE_VERSION\s*=\s*"(?P<version>{SEMVER_TEXT})"'
    ),
    "entroly/server.py": rf'^\s*_version\s*=\s*"(?P<version>{SEMVER_TEXT})"',
    "tests/test_release_surface.py": (
        rf'^RELEASE_VERSION\s*=\s*"(?P<version>{SEMVER_TEXT})"'
    ),
}


def _release_note(version: str) -> str:
    return f"""# Entroly {version}

Release metadata for Entroly {version} is synchronized across Python, Rust,
npm, MCP, OpenClaw, WASM, plugin, Docker, Homebrew, and GitHub release
surfaces.

See the GitHub release and merged pull requests for the complete feature,
security, compatibility, and migration notes.
"""


def _replace_versions(
    text: str,
    pattern: str,
    target: str,
    *,
    surface: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> str:
    """Replace only the named ``version`` group in validated matches."""

    expression = re.compile(pattern, flags=re.MULTILINE)
    matches = list(expression.finditer(text))
    if len(matches) < minimum or (maximum is not None and len(matches) > maximum):
        expected = (
            str(minimum)
            if maximum == minimum
            else f"between {minimum} and {maximum or 'any'}"
        )
        raise RuntimeError(
            f"{surface}: expected {expected} version field(s), found {len(matches)}"
        )

    def replacement(match: re.Match[str]) -> str:
        start, end = match.span("version")
        relative_start = start - match.start()
        relative_end = end - match.start()
        value = match.group(0)
        return value[:relative_start] + target + value[relative_end:]

    return expression.sub(replacement, text)


def _replace_cli_fallback_version(text: str, target: str, surface: str) -> str:
    """Replace exactly one CLI fallback version without relying on line regex anchors."""

    lines = text.splitlines(keepends=True)
    matches: list[tuple[int, re.Match[str], int]] = []
    for index, line in enumerate(lines):
        if "__version__" not in line or "=" not in line:
            continue
        left, right = line.split("=", 1)
        if "__version__" not in left:
            continue
        match = re.search(rf"(?P<version>{SEMVER_TEXT})", right)
        if match:
            matches.append((index, match, len(left) + 1))

    if len(matches) != 1:
        raise RuntimeError(
            f"{surface}: expected exactly one CLI fallback version, found {len(matches)}"
        )

    index, match, offset = matches[0]
    line = lines[index]
    start, end = match.span("version")
    start += offset
    end += offset
    lines[index] = line[:start] + target + line[end:]
    return "".join(lines)


def _replace_toml_project_version(text: str, target: str, surface: str) -> str:
    return _replace_versions(
        text,
        rf'^version\s*=\s*"(?P<version>{SEMVER_TEXT})"\s*$',
        target,
        surface=surface,
        minimum=1,
        maximum=1,
    )


def _replace_json_top_level_version(text: str, target: str, surface: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{surface}: invalid JSON: {exc}") from exc
    current = payload.get("version") if isinstance(payload, dict) else None
    if not isinstance(current, str) or not SEMVER_RE.fullmatch(current):
        raise RuntimeError(f"{surface}: missing semantic top-level version")

    updated = _replace_versions(
        text,
        rf'"version"\s*:\s*"(?P<version>{re.escape(current)})"',
        target,
        surface=surface,
        minimum=1,
        maximum=None,
    )
    parsed = json.loads(updated)
    if parsed.get("version") != target:
        raise RuntimeError(f"{surface}: top-level version did not converge to {target}")
    return updated


def _replace_server_json_versions(text: str, target: str, surface: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{surface}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("packages"), list):
        raise RuntimeError(f"{surface}: missing package registry structure")

    updated = _replace_versions(
        text,
        rf'"version"\s*:\s*"(?P<version>{SEMVER_TEXT})"',
        target,
        surface=surface,
        minimum=1 + len(payload["packages"]),
        maximum=None,
    )
    parsed = json.loads(updated)
    if parsed.get("version") != target:
        raise RuntimeError(f"{surface}: registry version did not converge to {target}")
    package_versions = {
        package.get("version")
        for package in parsed.get("packages", [])
        if isinstance(package, dict)
    }
    if package_versions != {target}:
        raise RuntimeError(
            f"{surface}: package versions did not converge to {target}: "
            f"{sorted(str(value) for value in package_versions)}"
        )
    return updated


def _replace_cargo_lock_versions(text: str, target: str, surface: str) -> str:
    updated = text
    for package in CARGO_LOCK_PACKAGES[surface]:
        updated = _replace_versions(
            updated,
            (
                rf'\[\[package\]\]\s*\n'
                rf'name\s*=\s*"{re.escape(package)}"\s*\n'
                rf'version\s*=\s*"(?P<version>{SEMVER_TEXT})"'
            ),
            target,
            surface=f"{surface}:{package}",
            minimum=1,
            maximum=1,
        )
    return updated


def _replace_native_dependency_minimums(
    text: str,
    target: str,
    surface: str,
) -> str:
    return _replace_versions(
        text,
        rf'entroly-core>=(?P<version>{SEMVER_TEXT}),<2',
        target,
        surface=surface,
        minimum=1,
        maximum=None,
    )


def _transform_surface(surface: str, text: str, target: str) -> str:
    if surface in TOML_VERSION_SURFACES:
        updated = _replace_toml_project_version(text, target, surface)
        if surface in {"pyproject.toml", "entroly/pyproject.toml"}:
            updated = _replace_native_dependency_minimums(updated, target, surface)
        return updated

    if surface in JSON_TOP_LEVEL_VERSION_SURFACES:
        return _replace_json_top_level_version(text, target, surface)

    if surface == "server.json":
        return _replace_server_json_versions(text, target, surface)

    if surface in CARGO_LOCK_PACKAGES:
        return _replace_cargo_lock_versions(text, target, surface)

    if surface == "entroly/cli.py":
        updated = _replace_cli_fallback_version(text, target, surface)
        return _replace_native_dependency_minimums(updated, target, surface)

    if surface in PYTHON_VERSION_PATTERNS:
        updated = _replace_versions(
            text,
            PYTHON_VERSION_PATTERNS[surface],
            target,
            surface=surface,
            minimum=1,
            maximum=1,
        )
        if surface == "entroly/cli.py":
            updated = _replace_native_dependency_minimums(updated, target, surface)
        return updated

    if surface == "entroly-core/README.md":
        return _replace_versions(
            text,
            rf'entroly-core>=(?P<version>{SEMVER_TEXT})',
            target,
            surface=surface,
            minimum=1,
            maximum=None,
        )

    if surface == "packaging/homebrew/README.md":
        updated = _replace_versions(
            text,
            rf'Current release example version:\s*`(?P<version>{SEMVER_TEXT})`',
            target,
            surface=surface,
            minimum=1,
            maximum=1,
        )
        updated = _replace_versions(
            updated,
            rf'\bVER=(?P<version>{SEMVER_TEXT})\b',
            target,
            surface=surface,
            minimum=1,
            maximum=1,
        )
        return _replace_versions(
            updated,
            rf'entroly-(?P<version>{SEMVER_TEXT})\.tar\.gz',
            target,
            surface=surface,
            minimum=1,
            maximum=1,
        )

    raise RuntimeError(f"{surface}: no typed release-surface transform is defined")


def synchronize(root: Path, target: str) -> list[str]:
    if not SEMVER_RE.fullmatch(target):
        raise ValueError(f"invalid semantic version: {target!r}")

    missing: list[str] = []
    invalid: list[str] = []
    planned: dict[str, str] = {}

    # Preflight every surface before writing anything. This prevents a malformed
    # late file from leaving an earlier file partially rewritten.
    for relative_name in RELEASE_SURFACES:
        path = root / relative_name
        if not path.is_file():
            missing.append(relative_name)
            continue
        try:
            original = path.read_text(encoding="utf-8")
            updated = _transform_surface(relative_name, original, target)
        except (OSError, UnicodeError, RuntimeError) as exc:
            invalid.append(f"{relative_name}: {exc}")
            continue
        if updated != original:
            planned[relative_name] = updated

    if missing or invalid:
        parts: list[str] = []
        if missing:
            parts.append("missing release surfaces: " + ", ".join(sorted(missing)))
        if invalid:
            parts.append("invalid release surfaces: " + "; ".join(sorted(invalid)))
        raise RuntimeError(" | ".join(parts))

    changed: list[str] = []
    for relative_name in RELEASE_SURFACES:
        updated = planned.get(relative_name)
        if updated is None:
            continue
        (root / relative_name).write_text(updated, encoding="utf-8")
        changed.append(relative_name)

    if MCPB_MANIFEST.as_posix() in changed:
        bundle = rebuild_mcpb(root)
        changed.append(bundle.relative_to(root).as_posix())

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
