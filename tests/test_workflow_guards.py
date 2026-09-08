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


def test_every_security_scan_is_enforced() -> None:
    """A scan that writes a status file nobody reads is decorative.

    The security job deliberately runs each scanner under `set +e` with a
    trailing `exit 0`, so its log uploads as evidence even on failure. That
    makes the enforcement step the only thing standing between a finding and a
    green check -- and adding a scanner without adding it there produces a job
    that runs, reports, and passes regardless.

    Asserting the property rather than a fixed list, so a fourth scanner fails
    here instead of silently not counting.
    """
    workflow = (ROOT / ".github/workflows/deep-dogfood.yml").read_text(encoding="utf-8")
    job = re.search(
        r"(?ms)^  security-static:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)", workflow
    )
    assert job is not None, "security-static job is missing"
    body = job.group("body")

    # Every scanner that records a verdict, taken from the writes themselves.
    produced = set(re.findall(r'\$RUNNER_TEMP/([\w.-]+)\.status"', body))
    assert produced, "no scanner writes a status file; the evidence pattern is gone"

    enforce = re.search(
        r"(?ms)^      - name: Enforce security gates\n(?P<body>.*?)(?=^      - |\Z)", body
    )
    assert enforce is not None, "the security gates are no longer enforced at all"
    enforced_block = enforce.group("body")

    unenforced = sorted(name for name in produced if name not in enforced_block)
    assert not unenforced, (
        f"these scanners run but no gate reads their result: {unenforced}. "
        "They would report findings and still pass the job."
    )


def test_codeql_declares_languages_not_language() -> None:
    """`language:` is silently ignored by github/codeql-action/init.

    The action warns about the unknown input and carries on with language
    auto-detection, which in this repository selected Java and then failed
    finalizing a database for a language it does not contain. The failure
    surfaced minutes later as a build error with exit code 32, nowhere near
    the typo that caused it.

    Pinned because the two spellings differ by one character and the wrong one
    fails in a way that does not name itself.
    """
    workflow = (ROOT / ".github/workflows/codeql.yml").read_text(encoding="utf-8")
    init = re.search(
        r"(?ms)^      - name: Initialize CodeQL\n(?P<body>.*?)(?=^      - |\Z)", workflow
    )
    assert init is not None, "the CodeQL init step is missing"
    body = init.group("body")

    assert re.search(r"(?m)^\s+languages:\s*\$\{\{\s*matrix\.language\s*\}\}", body), (
        "the CodeQL init step must pass `languages:` (plural)"
    )
    # Scoped to the step body on purpose: `language:` is also the legitimate
    # name of the matrix key, so searching the whole file would fail on a
    # correct configuration.
    assert not re.search(r"(?m)^\s+language:\s", body), (
        "`language:` is not an input to codeql-action/init; it is ignored and "
        "the action falls back to auto-detection"
    )
