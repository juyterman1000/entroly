"""Public evidence must stay traceable, and the debt must only shrink.

verify_readme_claims.py enforces provenance only for artifacts cited in the
first ~130 lines of README.md. Of 22 artifacts cited as public evidence across
README and docs/, 2 carried a harness hash and checksum sidecar and 20 carried
neither -- including every headline accuracy number.

Unsealed means unverifiable, not known-wrong. But it is the gap a fabricated
harness exploits: emit a plausible artifact, cite it below the first screen,
and nothing objects. That failure mode was live in this repository.

Sealing the existing twenty honestly requires re-running their benchmarks;
hand-writing a sidecar would assert provenance nobody verified. So the check is
a ratchet: existing gaps are recorded, new ones are refused.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_evidence_provenance.py"
DEBT_FILE = REPO_ROOT / "docs" / "evidence-provenance-debt.json"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=180,
    )


def test_current_tree_passes() -> None:
    """Every cited artifact is either sealed or recorded as known debt."""
    result = _run()
    assert result.returncode == 0, (
        "public evidence provenance check failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_debt_file_exists_and_is_well_formed() -> None:
    assert DEBT_FILE.is_file(), (
        f"{DEBT_FILE.name} is missing; the ratchet has no baseline and every "
        "unsealed citation would fail"
    )
    payload = json.loads(DEBT_FILE.read_text(encoding="utf-8"))
    assert isinstance(payload.get("unsealed"), list)
    assert payload.get("enforced_by", "").endswith("verify_evidence_provenance.py")
    # The purpose text is what stops a future reader treating the list as a
    # permanent exemption rather than a backlog.
    assert "only shrink" in payload.get("purpose", "").lower()


def test_every_debt_entry_still_exists_on_disk() -> None:
    """A stale entry would silently widen the exemption."""
    payload = json.loads(DEBT_FILE.read_text(encoding="utf-8"))
    missing = [
        entry for entry in payload["unsealed"]
        if not (REPO_ROOT / entry).is_file()
    ]
    assert not missing, f"debt entries reference missing artifacts: {missing}"


def test_a_newly_cited_unsealed_artifact_is_refused(tmp_path: Path) -> None:
    """The ratchet must bite; otherwise it is decoration.

    Reproduces the exact shape of the defect: a plausible-looking artifact
    dropped into benchmarks/results/ and cited from a docs page.
    """
    doc = REPO_ROOT / "docs" / "BENCHMARKS.md"
    artifact = REPO_ROOT / "benchmarks" / "results" / "_ratchet_probe.json"
    original = doc.read_text(encoding="utf-8")

    try:
        artifact.write_text(
            json.dumps({"headline": "Entroly wins by 40%"}), encoding="utf-8"
        )
        doc.write_text(
            original + "\n[probe](benchmarks/results/_ratchet_probe.json)\n",
            encoding="utf-8",
        )
        result = _run()
        assert result.returncode == 1, (
            "a newly cited artifact with no provenance was accepted:\n"
            f"{result.stdout}"
        )
        assert "_ratchet_probe.json" in result.stdout
    finally:
        doc.write_text(original, encoding="utf-8")
        artifact.unlink(missing_ok=True)


def test_update_debt_refuses_to_grow_the_ledger() -> None:
    """--update-debt must not become a way to launder new unsealed evidence."""
    doc = REPO_ROOT / "docs" / "BENCHMARKS.md"
    artifact = REPO_ROOT / "benchmarks" / "results" / "_ratchet_probe2.json"
    original = doc.read_text(encoding="utf-8")

    try:
        artifact.write_text(json.dumps({"headline": "x"}), encoding="utf-8")
        doc.write_text(
            original + "\n[probe](benchmarks/results/_ratchet_probe2.json)\n",
            encoding="utf-8",
        )
        result = _run("--update-debt")
        assert result.returncode == 1, (
            "--update-debt grew the exemption list instead of refusing:\n"
            f"{result.stdout}"
        )
        assert "refusing to grow" in result.stdout
        # The ledger on disk must be untouched by the refused call.
        payload = json.loads(DEBT_FILE.read_text(encoding="utf-8"))
        assert "benchmarks/results/_ratchet_probe2.json" not in payload["unsealed"]
    finally:
        doc.write_text(original, encoding="utf-8")
        artifact.unlink(missing_ok=True)


def test_the_two_sealed_artifacts_really_verify() -> None:
    """Guard the sealed set: a broken seal must not pass as sealed."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from verify_evidence_provenance import find_citations, seal_status
    finally:
        sys.path.pop(0)

    sealed = [rel for rel in find_citations() if seal_status(rel)[0]]
    assert sealed, "no cited artifact is fully traceable any more"
    for relative in sealed:
        ok, missing = seal_status(relative)
        assert ok, f"{relative} reported sealed but is missing {missing}"


@pytest.mark.parametrize("doc", ["README.md", "docs/BENCHMARKS.md"])
def test_evidence_docs_are_scanned(doc: str) -> None:
    """A doc dropping out of the scan list would silently reopen the hole."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert f'"{doc}"' in source, f"{doc} is not scanned for evidence citations"
