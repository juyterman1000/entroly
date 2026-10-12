"""The guard must catch a rewrite that leaves the artifact self-consistent.

``a917fc29`` rewrote a key inside a hash-committed benchmark artifact and
recomputed its embedded digest. The artifact remained internally consistent, so
a self-consistency check would have passed it. What broke was the digest
published in the page that cites it, and the ``.sha256`` sidecar of a second
artifact the same pass touched.

So these tests drive the failure mode rather than the happy path: each one
builds a temporary tree in one of the real commitment formats, applies the
rewrite as that commit applied it, and asserts the guard objects. The
self-consistent-rewrite case is the important one -- it is the regression a
cheaper check would miss.

One test also pins the bug this guard shipped with: its first draft hashed raw
bytes, and on a CRLF checkout that disagreed with every sidecar in the tree and
reported all seven as corrupt. A guard that cries wolf is worse than none,
because the next real failure is read as noise.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

guard = pytest.importorskip("scripts.check_committed_evidence")


def _seal(payload: dict, field: str = "payload_sha256") -> dict:
    """Return ``payload`` carrying a correct digest of its own contents."""
    sealed = dict(payload)
    sealed.pop(field, None)
    sealed[field] = guard.canonical_digest(sealed, field)
    return sealed


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """Point the guard at a temporary tree instead of the repository."""
    monkeypatch.setattr(guard, "ROOT", tmp_path)

    def _tracked(pattern: str) -> list[Path]:
        if pattern == "*":
            return [p for p in tmp_path.rglob("*") if p.is_file()]
        return sorted(tmp_path.rglob(pattern))

    monkeypatch.setattr(guard, "_tracked", _tracked)
    return tmp_path


def test_discovery_needs_no_list_and_ignores_other_digests(tree) -> None:
    """A ``*_sha256`` field that digests something else is not a commitment."""
    artifact = tree / "results.json"
    payload = _seal(
        {
            "trials": 24,
            # Digests the harness, not this file. Must not be mistaken for a
            # self-commitment, or every artifact would look corrupt.
            "harness_sha256": "b" * 64,
            "source_sha256": "c" * 64,
        }
    )
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    found = guard.discover_committed()

    assert [(item.path.name, item.field) for item in found] == [
        ("results.json", "payload_sha256")
    ]


def test_a_self_consistent_rewrite_is_still_caught(tree) -> None:
    """The actual failure mode: bytes and embedded digest moved, citation did not."""
    artifact = tree / "holdout.json"
    original = _seal({"baseline_label": "before", "final_exact": 24})
    artifact.write_text(json.dumps(original, indent=2), encoding="utf-8")

    doc = tree / "page.md"
    doc.write_text(
        f"See [holdout.json](holdout.json) "
        f"(`payload_sha256 = {original['payload_sha256']}`).\n",
        encoding="utf-8",
    )
    assert guard.check_citations(guard.discover_committed()) == [], (
        "a correctly published digest must pass"
    )

    # The rename pass: rewrite a label, recompute the embedded digest, leave the
    # published citation alone.
    rewritten = _seal({"baseline_label": "after", "final_exact": 24})
    artifact.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")

    found = guard.discover_committed()
    assert len(found) == 1
    assert guard.canonical_digest(
        json.loads(artifact.read_text(encoding="utf-8")), "payload_sha256"
    ) == found[0].digest, "the rewritten artifact is still self-consistent"

    problems = guard.check_citations(found)
    assert problems, "a stale published digest must fail even when the file is consistent"
    assert "page.md" in problems[0]


def test_a_superseded_digest_may_be_recorded_as_history(tree) -> None:
    """Deleting the old digest would destroy the audit trail."""
    artifact = tree / "holdout.json"
    payload = _seal({"final_exact": 24})
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (tree / "page.md").write_text(
        f"[holdout.json](holdout.json) (`payload_sha256 = {payload['payload_sha256']}`).\n"
        "\n### Provenance\n\n"
        f"Superseded: `payload_sha256 = {'a' * 64}` before the relabelling.\n",
        encoding="utf-8",
    )

    assert guard.check_citations(guard.discover_committed()) == []


def test_a_stale_sidecar_is_caught(tree) -> None:
    artifact = tree / "sealed.json"
    artifact.write_text(json.dumps({"ok": True}, indent=2), encoding="utf-8")
    sidecar = tree / "sealed.json.sha256"

    sidecar.write_text(f"{guard.file_digest(artifact)}  sealed.json\n", encoding="ascii")
    assert guard.check_sidecars() == [], "a matching sidecar must pass"

    artifact.write_text(json.dumps({"ok": False}, indent=2), encoding="utf-8")
    problems = guard.check_sidecars()
    assert problems and "sealed.json" in problems[0]


def test_crlf_checkouts_do_not_report_false_corruption(tree) -> None:
    """Regression: the first draft hashed raw bytes and failed every sidecar."""
    artifact = tree / "sealed.json"
    body = json.dumps({"ok": True}, indent=2)
    artifact.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))

    normalized = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (tree / "sealed.json.sha256").write_text(
        f"{normalized}  sealed.json\n", encoding="ascii"
    )

    assert b"\r\n" in artifact.read_bytes(), "the fixture must actually be CRLF"
    assert guard.check_sidecars() == [], (
        "a CRLF checkout must not be reported as corrupt; the sealing "
        "convention normalises line endings"
    )


def test_a_sidecar_without_its_artifact_is_reported(tree) -> None:
    (tree / "gone.json.sha256").write_text(f"{'a' * 64}  gone.json\n", encoding="ascii")
    problems = guard.check_sidecars()
    assert problems and "no artifact beside it" in problems[0]
