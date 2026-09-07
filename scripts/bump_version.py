#!/usr/bin/env python3
"""Bump version across all Entroly manifests.

Usage: python scripts/bump_version.py <semver>
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _release_artifacts import MCPB_MANIFEST, rebuild_mcpb  # noqa: E402

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
    ("entroly-core/Cargo.lock",
        r'(name\s*=\s*"entroly-engine"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-core/README.md", r'entroly-core>=[0-9]+\.[0-9]+\.[0-9]+', 'entroly-core>={v}'),
    ("entroly-engine/Cargo.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-engine/Cargo.lock",
        r'(name\s*=\s*"entroly-engine"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-qccr/Cargo.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-qccr/Cargo.lock",
        r'(name\s*=\s*"entroly-qccr"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-wasm/Cargo.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
    ("entroly-wasm/Cargo.lock",
        r'(name\s*=\s*"entroly-wasm"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-wasm/Cargo.lock",
        r'(name\s*=\s*"entroly-qccr"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-wasm/Cargo.lock",
        r'(name\s*=\s*"entroly-engine"\s*\nversion\s*=\s*)"[^"]+"', r'\g<1>"{v}"'),
    ("entroly-wasm/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly-wasm/pkg/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/npm/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/npm-alias/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/npm-alias/package.json", r'"entroly-wasm"\s*:\s*"[^"]+"', '"entroly-wasm": "{v}"'),
    ("integrations/openclaw/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    # Surfaces the 1.0.82 bump had to fix by hand. They were corrected in that
    # release but never added here, so the next bump would have left them
    # behind again -- which is exactly what the post-bump sweep reported.
    ("integrations/openclaw/bridge-client.js",
        r'entroly>=[0-9]+\.[0-9]+\.[0-9]+', 'entroly>={v}'),
    ("entroly/integrations/hermes_context_engine/plugin.yaml",
        r'^version:\s*[0-9]+\.[0-9]+\.[0-9]+', 'version: {v}'),
    ("entroly/integrations/hermes_context_engine/plugin.yaml",
        r'entroly>=[0-9]+\.[0-9]+\.[0-9]+', 'entroly>={v}'),
    ("BENCHMARKS.md",
        r'entroly-core [0-9]+\.[0-9]+\.[0-9]+', 'entroly-core {v}'),
    ("deploy/cloudflare-community-savings/package.json",
        r'"version"\s*:\s*"[0-9]+\.[0-9]+\.[0-9]+"', '"version": "{v}"'),
    # Version examples shown to a human filling in a manual-dispatch field or an
    # issue form. They are not internal comments -- an operator reads them while
    # deciding what to type, and one had been offering 1.0.25.
    (".github/workflows/entroly-publish.yml",
        r'for example [0-9]+\.[0-9]+\.[0-9]+', 'for example {v}'),
    (".github/workflows/publish-mcp-registry.yml",
        r'for example [0-9]+\.[0-9]+\.[0-9]+', 'for example {v}'),
    (".github/workflows/publish-openclaw-clawhub.yml",
        r'for example [0-9]+\.[0-9]+\.[0-9]+', 'for example {v}'),
    (".github/ISSUE_TEMPLATE/independent-review.yml",
        r'Example: [0-9]+\.[0-9]+\.[0-9]+', 'Example: {v}'),
    (".github/workflows/round7-runtime-repair.yml",
        r'entroly-core>=[0-9]+\.[0-9]+\.[0-9]+', 'entroly-core>={v}'),
    ("integrations/openclaw/README.md",
        r'pip install "entroly>=[0-9]+\.[0-9]+\.[0-9]+"',
        'pip install "entroly>={v}"'),
    ("entroly/__init__.py", r'__version__\s*=\s*"[^"]+"', '__version__ = "{v}"'),
    ("entroly/native_status.py",
        r'MIN_ENTROLY_CORE_VERSION\s*=\s*"[^"]+"',
        'MIN_ENTROLY_CORE_VERSION = "{v}"'),
    ("entroly/cli.py", r'__version__\s*=\s*"[^"]+"', '__version__ = "{v}"'),
    ("entroly/cli.py", r'entroly-core>=[0-9]+\.[0-9]+\.[0-9]+', 'entroly-core>={v}'),
    ("entroly/server.py", r'_version\s*=\s*"[^"]+"', '_version = "{v}"'),
    (".claude-plugin/manifest.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    # `plugin.json` sits beside `manifest.json` and was never listed, so it
    # lagged a release behind every bump that touched its neighbour.
    (".claude-plugin/plugin.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    # The marketplace entry declares the plugin version to Claude Code's
    # install UI. It is a third `.claude-plugin` surface; the sweep below
    # would catch it, but catching it here means the bump never emits a
    # warning in the first place.
    (".claude-plugin/marketplace.json",
        r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    (".mcpb-build/manifest.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    # Agent bundles and per-host extension manifests. Each declares the product
    # version to its host, and none of them was in this list -- a bump left
    # seven surfaces behind, which `tests/test_version_surfaces_are_complete.py`
    # now catches rather than the next release doing it.
    ("skills/entroly-evidence-operations/entroly-bundle.json",
        r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("integrations/codex/entroly/.codex-plugin/plugin.json",
        r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("integrations/codex/entroly/entroly-bundle.json",
        r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("integrations/codex/entroly/skills/entroly-evidence-operations/entroly-bundle.json",
        r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("integrations/gemini/entroly/entroly-bundle.json",
        r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("integrations/gemini/entroly/gemini-extension.json",
        r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("server.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("CITATION.cff", r'^version:\s*[^\s]+\s*$', 'version: {v}'),
    ("codemeta.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ("entroly/daemon.py", r'version:\s*str\s*=\s*"[^"]+"', 'version: str = "{v}"'),
    ("tests/test_release_surface.py",
        r'RELEASE_VERSION\s*=\s*"[^"]+"',
        'RELEASE_VERSION = "{v}"'),
    ("tests/test_release_surface.py",
        r'def test_public_package_versions_are_[0-9]+_[0-9]+_[0-9]+\(\)',
        'def test_public_package_versions_are_{v_ident}()'),
    # Keep the canonical Homebrew formula on the last verified sdist until the
    # post-PyPI workflow can update its URL and checksum atomically. The release
    # runbook may still point at the version being prepared.
    ("packaging/homebrew/README.md",
        r'Current release example version: `[0-9]+\.[0-9]+\.[0-9]+`',
        'Current release example version: `{v}`'),
    ("packaging/homebrew/README.md",
        r'VER=[0-9]+\.[0-9]+\.[0-9]+', 'VER={v}'),
    ("packaging/homebrew/README.md",
        r'entroly-[0-9]+\.[0-9]+\.[0-9]+\.tar\.gz', 'entroly-{v}.tar.gz'),
]

_NUMERIC_ID = r"(?:0|[1-9][0-9]*)"
_PRERELEASE_ID = r"(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
_BUILD_ID = r"[0-9A-Za-z-]+"
SEMVER = re.compile(
    rf"^{_NUMERIC_ID}\.{_NUMERIC_ID}\.{_NUMERIC_ID}"
    rf"(?:-{_PRERELEASE_ID}(?:\.{_PRERELEASE_ID})*)?"
    rf"(?:\+{_BUILD_ID}(?:\.{_BUILD_ID})*)?$"
)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not SEMVER.match(argv[1]):
        print("usage: bump_version.py <semver>", file=sys.stderr)
        return 2
    new = argv[1]
    pending: dict[Path, str] = {}
    changed: list[str] = []
    replacement_count = 0
    for rel, pattern, template in TARGETS:
        path = ROOT / rel
        if not path.exists():
            print(f"  {rel} missing; skipping generated artifact")
            continue
        text = pending.get(path)
        if text is None:
            text = path.read_text(encoding="utf-8")
        updated, n = re.subn(
            pattern,
            template.format(v=new, v_ident=new.replace(".", "_")),
            text,
            flags=re.MULTILINE,
        )
        if n == 0:
            print(f"!! no match in {rel}", file=sys.stderr)
            return 1
        pending[path] = updated
        changed.append(rel)
        replacement_count += n

    for path, updated in pending.items():
        path.write_text(updated, encoding="utf-8")
    artifacts: list[str] = []
    if ROOT / MCPB_MANIFEST in pending:
        bundle = rebuild_mcpb(ROOT)
        artifacts.append(bundle.relative_to(ROOT).as_posix())
    for rel in dict.fromkeys(changed):
        print(f"  {rel} -> {new}")
    for rel in artifacts:
        print(f"  {rel} -> rebuilt")
    file_count = len(pending) + len(artifacts)
    print(f"bumped {replacement_count} target(s) across {file_count} file(s) to {new}")

    # TARGETS above is a hand-maintained list, so it cannot know about a version
    # surface added after it was last edited -- that is exactly how seven
    # manifests kept 1.0.81 through the 1.0.82 bump while everything beside them
    # moved. Sweeping the tree here means the person running the bump learns
    # immediately, instead of a release shipping with a manifest that tells its
    # host the wrong version.
    checker = ROOT / "scripts" / "check_version_staleness.py"
    if not (ROOT / ".git").exists():
        # A synthetic tree, not the repository -- `tests/test_bump_version.py`
        # points ROOT at a fixture directory to exercise the rewrite logic.
        # Sweeping it would report every unrelated fixture string. The sweep is
        # a property of the real repository, so it is skipped here rather than
        # failing a test that is not about it.
        return 0
    print()
    if not checker.exists():
        # Inside the real repository the sweep is not optional: without it a
        # bump can silently leave a manifest behind, which is the defect this
        # step exists to catch.
        print("!! check_version_staleness.py is missing; bump is UNVERIFIED",
              file=sys.stderr)
        return 1
    result = subprocess.run(
        [sys.executable, str(checker)], cwd=str(ROOT),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(result.stdout.rstrip())
    if result.returncode != 0:
        print(
            "\n!! The bump left a version behind. Add each file above to TARGETS "
            "in this script, then re-run.",
            file=sys.stderr,
        )
        return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
