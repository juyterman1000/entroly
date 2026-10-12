"""A published benchmark digest must name the artifact that is on disk.

An evidence artifact commits to its own bytes: ``payload_sha256`` is the
canonical digest of everything else in the file. That makes a lexical rewrite
over such a file categorically different from an ordinary edit. A
repository-wide pass that neutralised named external products rewrote a system
key inside ``model_recovery_v7_holdout.json`` and re-sorted its keys. The
embedded digest was recomputed; the digest quoted in
``docs/benchmarks/model-triggered-recovery.md`` was not. For several releases
the published hash matched nothing, so a reader checking it would conclude the
evidence had been tampered with.

Nothing measured changed — only a key's name and the key ordering. The damage
was entirely to the integrity chain, which is exactly the part a reader cannot
check any other way.

These tests recompute the digest from the bytes rather than trusting the
embedded field, so they fail whether the artifact drifts from the document or
the document drifts from the artifact. The same pass also rewrote a command-line
flag inside the reproduction command, so the published command could not run;
the flags are checked against their declaration site too.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "benchmarks" / "results" / "model_recovery_v7_holdout.json"
DOC = REPO_ROOT / "docs" / "benchmarks" / "model-triggered-recovery.md"
RUNNER = REPO_ROOT / "benchmarks" / "model_recovery.py"

_DIGEST_CLAIM = re.compile(r"payload_sha256\s*=\s*`?([0-9a-f]{64})`?")
_FENCED = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)
_LONG_FLAG = re.compile(r"--[A-Za-z][A-Za-z0-9_-]*")
_DECLARED_FLAG = re.compile(r"""add_argument\(\s*["'](--[A-Za-z][A-Za-z0-9_-]*)["']""")


def _canonical_digest(payload: dict) -> str:
    """Recompute the artifact's digest without importing the runner.

    Importing the module under test to borrow its hashing helper would make a
    change to that helper invisible here. The canonicalisation is restated:
    sorted keys, compact separators, no ASCII escaping, excluding the digest
    field itself.
    """
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip(f"evidence artifact is absent: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def doc_text() -> str:
    if not DOC.is_file():
        pytest.skip(f"benchmark page is absent: {DOC}")
    return DOC.read_text(encoding="utf-8")


def test_the_artifact_commits_to_its_own_bytes(artifact) -> None:
    """A self-inconsistent artifact cannot be evidence for anything."""
    recomputed = _canonical_digest(artifact)
    assert recomputed == artifact["payload_sha256"], (
        "the artifact's contents do not hash to its own payload_sha256.\n"
        f"  recomputed: {recomputed}\n"
        f"  embedded:   {artifact['payload_sha256']}\n"
        "Either the contents were edited without recomputing, or the digest was "
        "edited without the contents."
    )


def test_the_published_digest_names_the_artifact_on_disk(artifact, doc_text) -> None:
    claimed = _DIGEST_CLAIM.findall(doc_text)
    assert claimed, "the page must publish a payload_sha256 for the artifact"

    actual = _canonical_digest(artifact)
    stale = sorted({digest for digest in claimed if digest != actual})
    assert not stale, (
        "the page publishes a digest that matches no artifact content.\n"
        f"  recomputed from bytes: {actual}\n"
        f"  published:             {', '.join(stale)}\n"
        "If a rewrite pass changed the artifact, recompute and update this page "
        "together, or version the artifact with a recorded provenance chain. Do "
        "not change the bytes and leave the published digest behind."
    )


def test_the_reproduction_commands_use_flags_the_runner_declares(doc_text) -> None:
    """A command that cannot run is not a reproduction procedure."""
    declared = set(_DECLARED_FLAG.findall(RUNNER.read_text(encoding="utf-8")))
    assert {"--input", "--output", "--protocol", "--phase"} <= declared, (
        "flag declarations were not found where expected; this guard would "
        "otherwise pass without checking anything"
    )

    invoked: set[str] = set()
    for block in _FENCED.findall(doc_text):
        if "benchmarks.model_recovery" not in block:
            continue
        invoked.update(_LONG_FLAG.findall(block))

    assert invoked, "the page must show how to run the benchmark"
    unknown = sorted(invoked - declared)
    assert not unknown, (
        f"the page invokes flags the runner does not accept: {unknown}. "
        f"Declared flags: {sorted(declared)}"
    )


def test_the_headline_numbers_come_from_the_artifact(artifact, doc_text) -> None:
    """Prose may not drift from the matrix it summarises."""
    systems = artifact["aggregates"]["systems"]
    entroly = systems["entroly"]
    baseline = next(value for key, value in systems.items() if key != "entroly")

    assert entroly["trials"] == baseline["trials"] == 24
    assert f"{entroly['final_exact']} / {entroly['trials']}" in doc_text
    assert f"{baseline['final_exact']} / {baseline['trials']}" in doc_text

    mcnemar = artifact["aggregates"]["paired"]["mcnemar_exact_p"]
    assert str(mcnemar) in doc_text

    for ratio in (entroly["mean_active_token_ratio"], baseline["mean_active_token_ratio"]):
        assert f"{ratio * 100:.2f}%" in doc_text
