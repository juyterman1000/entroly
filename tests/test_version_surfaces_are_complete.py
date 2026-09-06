"""No tracked file may declare a version older than the runtime master.

`scripts/bump_version.py` carries an explicit list of files to rewrite. A
version surface added later is not in that list, so it silently keeps the
previous release's number while everything around it moves — and nothing
notices, because a stale version is valid JSON that parses and loads fine.

Bumping 1.0.81 to 1.0.82 left seven behind: `.claude-plugin/plugin.json` (which
sits directly beside the `manifest.json` the script *does* rewrite), the
Codex and Gemini agent bundles, `gemini-extension.json`, and the
evidence-operations skill bundle. Each declares the product version to a
different host, so the hosts would have been told 1.0.81 by a 1.0.82 release.

`tests/test_release_surface_consistency.py` pins a hand-maintained list of
manifests. This asserts the property instead: whatever a version surface *is*,
it may not lag. A test that enumerates files has the same blind spot as the
script it is checking.
"""
from __future__ import annotations

import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER = re.search(
    r'__version__\s*=\s*"([^"]+)"',
    (REPO_ROOT / "entroly" / "__init__.py").read_text(encoding="utf-8"),
).group(1)

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

# Files whose version is bound to something other than the current product.
EXEMPT = {
    # Pins a *published* release asset together with its verified SHA-256.
    # Moving it before that release exists would point at a 404 and a hash that
    # matches nothing. It advances after the release is cut and re-verified.
    "packaging/scoop/entroly.json",
}

# Directories that quote or vendor versions rather than declaring their own.
SKIP_PARTS = (
    "node_modules", "/target/", "benchmarks/results", "docs/releases/",
    "docs/research", ".entroly/", "tmp/", "entroly-wasm/pkg/",
    # A lock file mirrors the manifest beside it; checking both reports the
    # same drift twice and says nothing new.
    "package-lock.json",
)


@lru_cache(maxsize=1)
def _tracked_json() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "*.json"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"git ls-files failed: {result.stderr.strip()[:200]}")
    return tuple(
        REPO_ROOT / line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not any(part in line for part in SKIP_PARTS)
    )


def _declared_version(path: Path) -> str | None:
    """A top-level `version` that this product actually ships.

    `"private": true` is the discriminator, not a name pattern. A private
    package is never published, so it keeps its own lifecycle and has no reason
    to track the product version -- `deploy/cloudflare-community-savings` is a
    Worker still at 1.0.0 whose version has never moved in its history. Every
    published manifest in this repo is non-private, so the rule separates them
    exactly without an allow-list that would need maintaining.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(document, dict) or document.get("private") is True:
        return None
    value = document.get("version")
    return value if isinstance(value, str) and _SEMVER.match(value) else None


def test_the_sweep_sees_the_manifests():
    """Guard the guard: an empty sweep would make the assertion vacuous."""
    found = [p for p in _tracked_json() if _declared_version(p)]
    assert len(found) >= 10, (
        f"only {len(found)} versioned manifests found; the sweep is not reading "
        f"the repository and this file would pass without checking anything"
    )


def test_no_tracked_manifest_lags_the_master_version():
    master = tuple(int(part) for part in MASTER.split("."))
    stale: list[str] = []
    for path in _tracked_json():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in EXEMPT:
            continue
        declared = _declared_version(path)
        if declared and tuple(int(p) for p in declared.split(".")) < master:
            stale.append(f"{rel}: declares {declared}, master is {MASTER}")

    assert not stale, (
        "these manifests were left behind by the version bump:\n  "
        + "\n  ".join(stale) + "\n"
        "Add each to TARGETS in scripts/bump_version.py, or to EXEMPT here if "
        "its version is bound to a published artifact rather than the product."
    )


def test_the_exempt_list_is_not_a_dumping_ground():
    """An exemption must be justified by the file still existing and being pinned.

    Without this, the easy fix for a failing bump is to add the file to EXEMPT
    and move on, which converts the gate into a list of things it ignores.
    """
    for rel in EXEMPT:
        path = REPO_ROOT / rel
        assert path.exists(), f"EXEMPT names {rel}, which no longer exists"
        document = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(document)
        assert "releases/download/" in blob or "sha256" in blob.lower(), (
            f"{rel} is exempt from the version sweep but does not pin a "
            f"published artifact; the exemption no longer has a reason"
        )
