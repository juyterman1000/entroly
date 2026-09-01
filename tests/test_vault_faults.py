"""What the vault does when the filesystem or the process fails mid-operation.

A belief store is only as trustworthy as its behaviour when a write does not
complete. These inject the failures rather than reasoning about them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import entroly.vault as vault_module
import entroly.vault_time as vault_time
from entroly.vault import (
    BeliefArtifact,
    VaultConfig,
    VaultManager,
    _parse_frontmatter,
)
from entroly.vault_time import BeliefLedger, LedgerIntegrityError


def _vault(tmp_path) -> VaultManager:
    return VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))


def test_a_full_disk_during_a_belief_write_keeps_the_previous_version(tmp_path, monkeypatch):
    """The replacement is atomic, so a failed write leaves the old belief."""

    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="stable", title="v1", body="ORIGINAL", sources=["a.py:1"])
    )

    def full_disk(path, text):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(vault_module, "_atomic_write_text", full_disk)
    with pytest.raises(OSError):
        vault.write_belief(
            BeliefArtifact(entity="stable", title="v2", body="REPLACEMENT",
                           sources=["a.py:1"])
        )

    monkeypatch.undo()
    survivor = vault.read_belief("stable")
    assert survivor is not None
    assert "ORIGINAL" in survivor["body"]
    assert not list((vault._base / "beliefs").glob("*.tmp"))


def test_a_failed_ledger_append_is_reported_not_swallowed(tmp_path, monkeypatch):
    """The belief write must not be lost, and the ledger gap must be visible.

    Silently succeeding would leave the vault holding a belief the audit trail
    never recorded -- the same divergence a filename collision produced, from
    the other direction.
    """

    vault = _vault(tmp_path)

    def broken_ledger(self, artifact, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(BeliefLedger, "record", broken_ledger)
    result = vault.write_belief(
        BeliefArtifact(entity="orphan", title="t", body="b", sources=["a.py:1"])
    )

    assert result["status"] == "written"
    assert result["ledger"]["status"] == "failed"
    assert "No space left" in result["ledger"]["error"]

    monkeypatch.undo()
    assert vault.read_belief("orphan") is not None


def test_a_torn_final_ledger_line_fails_closed(tmp_path):
    """A half-written record must raise rather than be treated as the tail.

    Chaining onto a truncated record would bake the damage into every later
    record's prev_sha256.
    """

    vault = _vault(tmp_path)
    for index in range(3):
        vault.write_belief(
            BeliefArtifact(entity=f"e{index}", title="t", body=f"b{index}",
                           sources=["a.py:1"])
        )

    log = vault._base / "ledger" / "beliefs.jsonl"
    text = log.read_text(encoding="utf-8")
    log.write_text(text[: -len(text) // 4], encoding="utf-8")  # cut mid-record

    ledger = BeliefLedger(vault._base)
    with pytest.raises(LedgerIntegrityError):
        ledger._last_record()
    assert ledger.verify_chain()["status"] == "broken"


def test_a_head_left_behind_by_a_crash_is_not_reported_as_tampering(tmp_path):
    """The log is appended, then the head advances. A crash between the two
    leaves the head one record behind on an entirely intact chain.

    Reporting that as truncation would raise a permanent false alarm on a
    healthy ledger, and an alarm that cries wolf is the reason the real one
    gets ignored.
    """

    vault = _vault(tmp_path)
    for index in range(4):
        vault.write_belief(
            BeliefArtifact(entity=f"e{index}", title="t", body=f"b{index}",
                           sources=["a.py:1"])
        )

    log = vault._base / "ledger" / "beliefs.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    head = vault._base / "ledger" / "head.json"
    stored = json.loads(head.read_text(encoding="utf-8"))
    head.write_text(
        json.dumps(
            {
                "schema": stored["schema"],
                "seq": rows[-2]["seq"],
                "record_sha256": rows[-2]["record_sha256"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = BeliefLedger(vault._base).verify_chain()
    assert report["status"] == "intact"
    assert "lagged" in report.get("note", "")


def test_a_lagging_head_is_repaired_by_the_next_append(tmp_path):
    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="a", title="t", body="b", sources=["a.py:1"])
    )
    head = vault._base / "ledger" / "head.json"
    stored = json.loads(head.read_text(encoding="utf-8"))
    head.write_text(
        json.dumps({"schema": stored["schema"], "seq": 0, "record_sha256": ""},
                   sort_keys=True),
        encoding="utf-8",
    )

    vault.write_belief(
        BeliefArtifact(entity="b", title="t", body="b", sources=["a.py:1"])
    )

    assert BeliefLedger(vault._base).verify_chain()["status"] == "intact"


def test_a_failed_head_write_does_not_lose_the_appended_record(tmp_path, monkeypatch):
    """The record is durable even when the head cannot be updated."""

    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="first", title="t", body="b", sources=["a.py:1"])
    )

    def broken_head(self, seq, record_hash):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(BeliefLedger, "_write_head", broken_head)
    with pytest.raises(OSError):
        vault_time.BeliefLedger(vault._base).record(
            BeliefArtifact(entity="second", title="t", body="b", sources=["a.py:1"])
        )
    monkeypatch.undo()

    log = vault._base / "ledger" / "beliefs.jsonl"
    rows = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2, "the appended record must survive a failed head write"
    assert BeliefLedger(vault._base).verify_chain()["status"] == "intact"


def test_an_unreadable_belief_does_not_break_a_vault_scan(tmp_path):
    """One corrupt file must not stop retraction or hygiene from running."""

    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="good", title="t", body="b", sources=["a.py:1"])
    )
    (vault._base / "beliefs" / "corrupt.md").write_text(
        "not frontmatter at all", encoding="utf-8"
    )

    result = vault.mark_beliefs_ungrounded([str(tmp_path)])

    assert "good" in result["retracted_entities"] or result["retracted_entities"] == []
    survivor = vault.read_belief("good")
    assert survivor is not None
    assert _parse_frontmatter(
        Path(survivor["path"]).read_text(encoding="utf-8")
    ) is not None


# A soak, not a unit test: two writers race a verifier that re-reads the whole
# ledger on every iteration, so the work is quadratic in the number of writes
# and the wall time is dominated by the runner's disk and core count. It takes
# ~39s on a developer machine and exceeded the suite's 60s ceiling twice on a
# CI runner -- once on PR #391 and again on main at e4d2b049 -- both times on
# Python 3.12 alone while the other four versions passed the same commit.
#
# The ceiling was wrong, not the test. Lowering the write count would cut the
# cost, but the number of verifications landing inside the race window is
# exactly what gives this test its power to catch the false alarm it guards, so
# the budget is raised instead of the exposure reduced.
@pytest.mark.timeout(180)
def test_verification_during_concurrent_writes_does_not_false_alarm(tmp_path):
    """The log and the head are two files.

    Reading them at different instants let a concurrent append land between:
    the log counted before it, the head read after it, and a healthy ledger
    reported "truncated: head expects 299, found 298". A soak of three writers
    against a maintainer produced six such alarms in a minute on a chain that
    was intact throughout. A tamper alarm that fires on healthy data is worse
    than none.
    """

    import threading

    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="seed", title="t", body="b", sources=["a.py:1"])
    )
    ledger = BeliefLedger(vault._base)
    stop = False
    false_alarms: list[str] = []

    def verify_loop() -> None:
        while not stop:
            report = ledger.verify_chain()
            if report["status"] != "intact":
                false_alarms.append(str(report.get("reason", report)))

    def write_loop() -> None:
        for index in range(60):
            vault.write_belief(
                BeliefArtifact(entity=f"w{index}", title="t", body=f"b{index}",
                               sources=["a.py:1"])
            )

    verifier = threading.Thread(target=verify_loop)
    verifier.start()
    writers = [threading.Thread(target=write_loop) for _ in range(2)]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop = True
    verifier.join()

    assert false_alarms == [], f"verification false-alarmed: {false_alarms[:3]}"
