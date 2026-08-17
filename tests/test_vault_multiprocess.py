"""Vault behaviour when separate processes touch it at once.

The threaded tests cover one process with several workers. The real case is
different: `entroly serve` holds a vault open while `entroly compile` writes to
it, in a separate process with no shared locks, no shared memory and no shared
GIL. These spawn real interpreters rather than threads.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from entroly.vault import _parse_frontmatter
from entroly.vault_time import BeliefLedger

REPO_ROOT = Path(__file__).resolve().parents[1]

_WRITER = """
import sys
sys.path.insert(0, {repo!r})
from entroly.vault import BeliefArtifact, VaultConfig, VaultManager

vault = VaultManager(VaultConfig(base_path={base!r}))
failures = 0
for index in range({count}):
    try:
        vault.write_belief(
            BeliefArtifact(
                entity="{prefix}-%d" % index,
                title="t",
                body="body %d" % index,
                sources=["a.py:1"],
            )
        )
    except Exception as exc:
        failures += 1
        print("WRITE_FAILURE %s" % type(exc).__name__, file=sys.stderr)
print("done %d failures=%d" % ({count}, failures))
"""

_READER = """
import sys, time
sys.path.insert(0, {repo!r})
from pathlib import Path
from entroly.vault import _parse_frontmatter

beliefs = Path({base!r}) / "beliefs"
torn = 0
deadline = time.time() + {seconds}
while time.time() < deadline:
    for path in list(beliefs.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text and _parse_frontmatter(text) is None:
            torn += 1
    time.sleep(0.002)
print("torn=%d" % torn)
"""


def _run(script: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.timeout(300)
def test_separate_processes_can_write_the_same_vault(tmp_path):
    """No lost writes and no crashes when two interpreters write concurrently."""

    base = str(tmp_path / "vault")
    per_process = 25

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    _WRITER.format(
                        repo=str(REPO_ROOT), base=base, count=per_process, prefix=prefix
                    )
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for prefix in ("alpha", "beta", "gamma")
    ]
    outputs = [proc.communicate(timeout=240) for proc in procs]

    for (stdout, stderr), proc in zip(outputs, procs, strict=True):
        assert proc.returncode == 0, stderr
        assert "failures=0" in stdout, stderr

    beliefs = list((Path(base) / "beliefs").glob("*.md"))
    entities = {
        (_parse_frontmatter(path.read_text(encoding="utf-8")) or {}).get("entity")
        for path in beliefs
    }
    expected = {
        f"{prefix}-{index}"
        for prefix in ("alpha", "beta", "gamma")
        for index in range(per_process)
    }
    assert expected <= entities, "a concurrently written belief was lost"


@pytest.mark.timeout(300)
def test_a_reader_process_never_sees_a_half_written_belief(tmp_path):
    """Windows will not rename over an open handle; the retry must absorb that."""

    base = str(tmp_path / "vault")
    from entroly.vault import BeliefArtifact, VaultConfig, VaultManager

    seed = VaultManager(VaultConfig(base_path=base))
    for index in range(5):
        seed.write_belief(
            BeliefArtifact(entity=f"seed-{index}", title="t", body="seed",
                           sources=["a.py:1"])
        )

    reader = subprocess.Popen(
        [sys.executable, "-c",
         textwrap.dedent(_READER.format(repo=str(REPO_ROOT), base=base, seconds=8))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    writer = _run(
        _WRITER.format(repo=str(REPO_ROOT), base=base, count=40, prefix="seed"),
        timeout=240,
    )
    reader_out, reader_err = reader.communicate(timeout=120)

    assert writer.returncode == 0, writer.stderr
    assert "failures=0" in writer.stdout, writer.stderr
    assert "torn=0" in reader_out, reader_err


@pytest.mark.timeout(300)
def test_the_ledger_survives_concurrent_processes(tmp_path):
    """Appends from separate processes must not interleave into a broken chain.

    A torn or interleaved append would show up as an unparseable record or a
    prev_sha256 that no record produced, which is what verify_chain reports.
    """

    base = str(tmp_path / "vault")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c",
             textwrap.dedent(_WRITER.format(repo=str(REPO_ROOT), base=base,
                                            count=20, prefix=prefix))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for prefix in ("p1", "p2")
    ]
    for proc in procs:
        proc.communicate(timeout=240)

    log = Path(base) / "ledger" / "beliefs.jsonl"
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in lines:
        json.loads(line)  # every record is a complete, parseable JSON object

    # Every write is present and the chain verifies. Before the append was
    # serialized, two processes read the same tail and chained onto it, so one
    # record overwrote the other: three processes writing 90 beliefs left 65-71
    # records and a prev_sha256 mismatch on every run.
    assert len(lines) == 40, f"expected 40 ledger records, found {len(lines)}"

    report = BeliefLedger(Path(base)).verify_chain()
    assert report["status"] == "intact", report
