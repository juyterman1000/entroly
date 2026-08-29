from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pure_python_gate_cannot_self_install_the_native_engine() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^  python-fallback:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
    )

    assert match is not None, "pure-Python fallback job is missing"
    assert re.search(
        r'(?m)^      ENTROLY_NO_SELF_HEAL:\s*["\']1["\']\s*$',
        match.group("body"),
    ), (
        "the engine-less gate must disable Entroly self-heal or a CLI test can "
        "install entroly-core midway through the suite"
    )


def test_docker_publish_job_has_timeout() -> None:
    workflow = (ROOT / ".github/workflows/entroly-publish.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(?ms)^  build-and-push:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
    )

    assert match is not None
    assert re.search(r"(?m)^    timeout-minutes:\s*[1-9][0-9]*\s*$", match.group("body"))


def test_docker_quality_gate_exposes_installed_console_scripts() -> None:
    workflow = (ROOT / ".github/workflows/entroly-publish.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(?ms)^  quality-gate:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
    )

    assert match is not None
    body = match.group("body")
    path_export = 'echo "$PWD/.venv/bin" >> "$GITHUB_PATH"'
    assert path_export in body
    assert body.index(path_export) < body.index(".venv/bin/pytest tests/")


def test_github_release_is_created_as_a_draft() -> None:
    """A release must not be user-visible before its binaries are attached.

    v1.0.79 shipped as Latest with zero assets: the publish run was cancelled
    after github-release succeeded and before publish-binaries uploaded
    anything, so the download page for the current version was empty while the
    previous release still carried all eight archives. Creating the release as
    a draft makes that intermediate state invisible instead of broken.
    """
    workflow = (ROOT / ".github/workflows/entroly-publish.yml").read_text(
        encoding="utf-8"
    )
    create = re.search(
        r"(?ms)^\s+gh release create \"\$\{RELEASE_TAG\}\"(?P<flags>.*?)\n\s*fi$",
        workflow,
    )

    assert create is not None, "gh release create invocation not found"
    assert "--draft" in create.group("flags"), (
        "gh release create must pass --draft; finalize-release publishes it "
        "only once every platform archive is attached"
    )


def test_finalize_release_publishes_only_after_binaries() -> None:
    workflow = (ROOT / ".github/workflows/entroly-publish.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(?ms)^  finalize-release:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
    )

    assert match is not None, "finalize-release job is missing"
    body = match.group("body")
    # Gating on publish-binaries is the whole contract: without it the job would
    # publish a release whose upload never ran.
    assert "publish-binaries" in body
    assert "--draft=false" in body
    # `always()` would publish regardless of the upload outcome.
    assert "always()" not in body


def test_binary_upload_does_not_publish_the_draft() -> None:
    """softprops/action-gh-release defaults `draft` to false.

    The action PATCHes the release with whatever it is given, so omitting the
    input would let whichever matrix target finished first flip the draft to
    published -- re-exposing the empty release the draft-first flow prevents.
    """
    workflow = (ROOT / ".github/workflows/release-binary.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(?ms)^      - name: Attach to release\n(?P<body>.*?)(?=^      - |\Z)",
        workflow,
    )

    assert match is not None, "Attach to release step not found"
    assert re.search(r"(?m)^\s+draft:\s*true\s*$", match.group("body")), (
        "the upload step must set draft: true so only finalize-release publishes"
    )
