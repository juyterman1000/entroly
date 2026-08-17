"""Integrity of the belief store (entroly/vault.py).

Beliefs are the vault's audit surface: every write carries a claim_id, an
entity and its sources, and an append-only ledger records each version. These
tests cover the ways that record could quietly stop being true.
"""

from __future__ import annotations

import json
from pathlib import Path

from entroly.vault import (
    BeliefArtifact,
    VaultConfig,
    VaultManager,
    _parse_frontmatter,
)


def _vault(tmp_path) -> VaultManager:
    return VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))


def _frontmatter(path: str) -> dict[str, str]:
    return _parse_frontmatter(Path(path).read_text(encoding="utf-8")) or {}


def test_entities_that_sanitise_alike_keep_separate_beliefs(tmp_path):
    """`_safe_filename` is many-to-one, and it used to lose a belief.

    `foo::bar` and `foo_bar` both sanitise to `foo_bar`, so the second write
    overwrote the first: the vault held one belief while the append-only ledger
    recorded two, the audit trail asserting something the vault had destroyed.
    """

    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="foo::bar", title="one", body="FIRST", sources=["a.py:1"])
    )
    vault.write_belief(
        BeliefArtifact(entity="foo_bar", title="two", body="SECOND", sources=["b.py:1"])
    )

    beliefs = list((vault._base / "beliefs").glob("*.md"))
    assert len(beliefs) == 2

    first = vault.read_belief("foo::bar")
    second = vault.read_belief("foo_bar")
    assert first is not None and second is not None
    assert first["frontmatter"]["entity"] == "foo::bar"
    assert second["frontmatter"]["entity"] == "foo_bar"
    assert "FIRST" in first["body"]
    assert "SECOND" in second["body"]


def test_vault_and_ledger_agree_on_how_many_beliefs_exist(tmp_path):
    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="a::b", title="one", body="x", sources=["a.py:1"])
    )
    vault.write_belief(
        BeliefArtifact(entity="a_b", title="two", body="y", sources=["b.py:1"])
    )

    ledger_path = vault._base / "ledger" / "beliefs.jsonl"
    recorded = {
        json.loads(line)["entity"]
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    on_disk = {
        (_parse_frontmatter(p.read_text(encoding="utf-8")) or {}).get("entity")
        for p in (vault._base / "beliefs").glob("*.md")
    }

    assert recorded == on_disk


def test_read_belief_never_answers_with_a_different_entity(tmp_path):
    """The lookup used to fall back to a substring match on the filename.

    Asking for `cache` returned `cache_aligner`'s belief under
    `cache_aligner`'s name. In a store whose purpose is auditable claims that
    is a wrong answer, not a near miss.
    """

    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="cache_aligner", title="ca", body="ALIGNER",
                       sources=["a.py:1"])
    )

    assert vault.read_belief("cache") is None
    found = vault.read_belief("cache_aligner")
    assert found is not None and found["frontmatter"]["entity"] == "cache_aligner"


def test_entity_cannot_forge_other_frontmatter_fields(tmp_path):
    """Entity names come from indexed source code, which is untrusted input.

    Frontmatter is parsed line by line, so a newline in a value starts a new
    key: an entity of "x\nclaim_id: FORGED" replaced the claim_id the ledger
    is cross-referenced by.
    """

    vault = _vault(tmp_path)
    artifact = BeliefArtifact(
        entity="x\nclaim_id: FORGED-0000\nstatus: verified",
        title="t", body="b", sources=["a.py:1"],
        status="hypothesis", confidence=0.1,
    )
    result = vault.write_belief(artifact)
    frontmatter = _frontmatter(result["path"])

    assert frontmatter["claim_id"] == artifact.claim_id
    assert frontmatter["status"] == "hypothesis"
    assert "\n" not in frontmatter["entity"]


def test_a_source_cannot_forge_other_frontmatter_fields(tmp_path):
    vault = _vault(tmp_path)
    result = vault.write_belief(
        BeliefArtifact(entity="y", title="t", body="b", status="hypothesis",
                       sources=["a.py:1\nstatus: verified"])
    )

    assert _frontmatter(result["path"])["status"] == "hypothesis"


def test_an_interrupted_write_leaves_the_previous_belief_readable(tmp_path, monkeypatch):
    """Writes replace by rename, so a reader never sees a half-written belief."""

    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="stable", title="v1", body="ORIGINAL", sources=["a.py:1"])
    )

    import entroly.vault as vault_module

    def explode(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(vault_module, "_atomic_write_text", explode)
    try:
        vault.write_belief(
            BeliefArtifact(entity="stable", title="v2", body="REPLACEMENT",
                           sources=["a.py:1"])
        )
    except OSError:
        pass

    survivor = vault.read_belief("stable")
    assert survivor is not None
    assert "ORIGINAL" in survivor["body"]
    assert not list((vault._base / "beliefs").glob("*.tmp"))
