"""The Docker build context must not carry every crate's target directory.

`.dockerignore` named `entroly-core/target/` alone. This repository has four
crates, so `entroly-engine/target` (2.9G), `entroly-wasm/target` (1.5G) and
`entroly-qccr/target` (757M) stayed in the context -- roughly 5.2G uploaded to
the daemon on every build, for artifacts the builder stage recompiles from
source anyway.

Pinned as a test rather than left to review because the failure is invisible
locally: `docker build` still succeeds, just slowly, and nobody reads an upload
progress bar as a defect.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _patterns() -> list[str]:
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    return [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _excluded(path: str) -> bool:
    for pattern in _patterns():
        bare = pattern.rstrip("/")
        if fnmatch.fnmatch(path, bare) or fnmatch.fnmatch(path, pattern):
            return True
        if bare.startswith("**/") and path.endswith(bare[3:]):
            return True
        if path.startswith(bare.replace("**/", "")):
            return True
    return False


@pytest.mark.parametrize(
    "crate", ["entroly-core", "entroly-engine", "entroly-qccr", "entroly-wasm"]
)
def test_every_crate_target_is_excluded(crate):
    assert _excluded(f"{crate}/target"), (
        f"{crate}/target would be uploaded to the Docker daemon; naming crates "
        "one at a time is how three of the four came to be included"
    )


def test_the_pattern_is_not_crate_specific():
    # A per-crate line passes the test above today and breaks the moment a
    # fifth crate is added, which is exactly how this regressed.
    assert any(p.rstrip("/") == "**/target" for p in _patterns()), (
        "use a wildcard so a new crate is covered on the day it is created"
    )


@pytest.mark.parametrize("path", ["entroly-wasm/node_modules", "entroly/npm/node_modules"])
def test_vendored_node_trees_are_excluded(path):
    assert _excluded(path)


@pytest.mark.parametrize(
    "name", ["id_rsa", "id_ed25519", "server.pem", "private.key", ".npmrc", ".env"]
)
def test_credential_shaped_files_are_excluded(name):
    # Not a security boundary -- a build context is the wrong place to catch
    # secrets -- but `COPY . /app` takes whatever is in the tree, and a key
    # dropped there for one test would be baked into a layer that outlives it.
    assert _excluded(name), f"{name} would be copied into the image"


def test_source_is_still_included():
    # The exclusions must not be so broad that the build loses its input.
    for needed in ("entroly", "pyproject.toml", "entroly-core/src"):
        assert not _excluded(needed), f"{needed} must reach the build"
