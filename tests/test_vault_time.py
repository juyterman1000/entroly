"""Tests for the bitemporal belief ledger (entroly/vault_time.py)."""

from __future__ import annotations

import json

import pytest

from entroly.vault import BeliefArtifact, VaultConfig, VaultManager
from entroly.vault_time import BeliefLedger, BeliefRedactedError, LedgerIntegrityError


def _belief(entity: str, body: str, confidence: float = 0.8,
            status: str = "verified", last_checked: str = "") -> BeliefArtifact:
    art = BeliefArtifact(entity=entity, status=status, confidence=confidence,
                         sources=["test"], title=entity, body=body)
    if last_checked:
        art.last_checked = last_checked
    return art


@pytest.fixture()
def vault(tmp_path):
    return VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))


@pytest.fixture()
def ledger(vault):
    return BeliefLedger(vault._base)


def test_write_belief_records_versions_and_file_stays_latest(vault, ledger):
    r1 = vault.write_belief(_belief("auth-flow", "Auth uses JWT tokens."))
    r2 = vault.write_belief(_belief("auth-flow", "Auth uses session cookies."))
    assert r1["ledger"]["status"] == "recorded"
    assert r2["ledger"]["seq"] == 2

    # Per-entity file keeps only the latest version (unchanged behavior)...
    current = vault.read_belief("auth-flow")
    assert "session cookies" in current["body"]
    # ...but the ledger preserves both.
    history = ledger.timeline("auth-flow")
    assert [ledger.body_of(v) for v in history] == [
        "Auth uses JWT tokens.", "Auth uses session cookies.",
    ]


def test_as_of_transaction_time_returns_what_was_known(ledger):
    ledger.record(_belief("api", "v1 endpoints"), tx_time="2026-07-01T00:00:00+00:00")
    ledger.record(_belief("api", "v2 endpoints"), tx_time="2026-07-10T00:00:00+00:00")
    ledger.record(_belief("db", "postgres 15"), tx_time="2026-07-12T00:00:00+00:00")

    tuesday = ledger.as_of("2026-07-07T00:00:00+00:00")
    assert set(tuesday) == {"api"}
    assert ledger.body_of(tuesday["api"]) == "v1 endpoints"

    later = ledger.as_of("2026-07-15T00:00:00+00:00")
    assert set(later) == {"api", "db"}
    assert ledger.body_of(later["api"]) == "v2 endpoints"


def test_valid_time_axis_is_distinct_from_transaction_time(ledger):
    # Learned late (tx July 10) about something verified early (valid July 1).
    ledger.record(
        _belief("cache", "LRU eviction", last_checked="2026-07-01T00:00:00+00:00"),
        tx_time="2026-07-10T00:00:00+00:00",
    )
    # On July 5 the vault had NOT yet learned it...
    assert ledger.as_of("2026-07-05T00:00:00+00:00") == {}
    # ...but on the valid-time axis it had been verified by then.
    assert set(ledger.as_of("2026-07-05T00:00:00+00:00", time_axis="valid")) == {"cache"}


def test_diff_reports_added_and_changed(ledger):
    ledger.record(_belief("api", "v1", confidence=0.5),
                  tx_time="2026-07-01T00:00:00+00:00")
    ledger.record(_belief("api", "v2", confidence=0.9),
                  tx_time="2026-07-10T00:00:00+00:00")
    ledger.record(_belief("db", "postgres"),
                  tx_time="2026-07-12T00:00:00+00:00")

    d = ledger.diff("2026-07-05T00:00:00+00:00", "2026-07-15T00:00:00+00:00")
    assert d["added"] == ["db"]
    assert len(d["changed"]) == 1
    change = d["changed"][0]
    assert change["entity"] == "api"
    assert change["confidence"] == [0.5, 0.9]
    assert change["body_changed"] is True


def test_chain_verifies_and_detects_tampering(ledger):
    ledger.record(_belief("a", "one"))
    ledger.record(_belief("b", "two"))
    assert ledger.verify_chain() == {"status": "intact", "records": 2}

    lines = ledger._log.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["confidence"] = 0.99  # post-hoc edit
    lines[0] = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    ledger._log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = ledger.verify_chain()
    assert result["status"] == "broken"
    assert result["line"] == 1


def test_body_object_tampering_is_detected(ledger):
    ledger.record(_belief("a", "original body"))
    version = ledger.timeline("a")[0]
    obj = ledger._objects / f"{version.body_sha256}.md"
    obj.write_text("forged body", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="tampered"):
        ledger.body_of(version)


def test_unparseable_ledger_fails_closed(ledger):
    ledger.record(_belief("a", "one"))
    with ledger._log.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(LedgerIntegrityError, match="line 2"):
        ledger.as_of("2099-01-01T00:00:00+00:00")


def test_redact_entity_erases_content_but_chain_still_verifies(ledger):
    ledger.record(_belief("secret-user", "Alice's API key rotation schedule."))
    ledger.record(_belief("public", "The build uses vite."))

    result = ledger.redact(entity="secret-user", reason="gdpr_erasure")
    assert result["status"] == "redacted"
    assert result["objects_deleted"] == 1

    # Content is gone from disk and refused with a redaction (not tamper) error.
    v = ledger.timeline("secret-user")[0]
    assert v.redacted is True
    with pytest.raises(BeliefRedactedError, match="gdpr_erasure"):
        ledger.body_of(v)
    # The chain — including the tombstone — still verifies.
    assert ledger.verify_chain()["status"] == "intact"
    # Unrelated beliefs are untouched.
    assert ledger.body_of(ledger.timeline("public")[0]) == "The build uses vite."
    # Snapshots flag the redacted version instead of hiding history.
    snap = ledger.as_of("2099-01-01T00:00:00+00:00")
    assert snap["secret-user"].redacted is True


def test_redact_preserves_body_objects_shared_with_other_beliefs(ledger):
    ledger.record(_belief("a", "shared body text"))
    ledger.record(_belief("b", "shared body text"))  # same content-addressed object

    result = ledger.redact(entity="a")
    assert result["objects_deleted"] == 0
    assert result["objects_retained_shared"] == 1

    # 'a' refuses by policy; 'b' still reads the shared object.
    with pytest.raises(BeliefRedactedError):
        ledger.body_of(ledger.timeline("a")[0])
    assert ledger.body_of(ledger.timeline("b")[0]) == "shared body text"


def test_redact_requires_exactly_one_selector_and_reports_no_match(ledger):
    ledger.record(_belief("a", "one"))
    with pytest.raises(ValueError, match="exactly one"):
        ledger.redact()
    with pytest.raises(ValueError, match="exactly one"):
        ledger.redact(entity="a", claim_id="x")
    assert ledger.redact(entity="ghost")["status"] == "no_match"


def test_tombstone_tampering_breaks_the_chain(ledger):
    ledger.record(_belief("a", "one"))
    ledger.redact(entity="a")
    lines = ledger._log.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["reason"] = "rewritten history"
    lines[1] = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    ledger._log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert ledger.verify_chain()["status"] == "broken"


def test_seed_from_current_backfills_once_and_flags(vault, ledger):
    beliefs_dir = vault._base / "beliefs"
    vault.ensure_structure()
    (beliefs_dir / "legacy.md").write_text(
        "---\nclaim_id: c1\nentity: legacy\nstatus: verified\nconfidence: 0.7\n"
        "sources:\n  - old\nlast_checked: 2026-01-01T00:00:00+00:00\n---\n\n"
        "# legacy\n\nOld knowledge.\n",
        encoding="utf-8",
    )
    assert ledger.seed_from_current(beliefs_dir)["entities"] == 1
    v = ledger.timeline("legacy")[0]
    assert v.backfilled is True
    # Backfill preserves the legacy file's post-frontmatter content verbatim
    # (house _extract_body keeps the markdown heading).
    assert ledger.body_of(v) == "# legacy\n\nOld knowledge."
    # Idempotent: seeding again adds nothing.
    assert ledger.seed_from_current(beliefs_dir)["entities"] == 0
