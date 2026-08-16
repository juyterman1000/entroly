"""Integrity of the belief store (entroly/vault.py).

Beliefs are the vault's audit surface: every write carries a claim_id, an
entity and its sources, and an append-only ledger records each version. These
tests cover the ways that record could quietly stop being true.
"""

from __future__ import annotations

import json
import time
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


def test_confidence_update_does_not_rewrite_the_belief_body(tmp_path):
    """A belief's body is prose compiled from source docstrings.

    It can legitimately contain the text `confidence: 0.5` or
    `status: inferred`. An unanchored replace over the whole document rewrote
    those too, silently changing what the belief claims about the code.
    """

    vault = _vault(tmp_path)
    body = "Docs mention: confidence: 0.5 and status: inferred and last_checked: never."
    artifact = BeliefArtifact(entity="m", title="m", body=body, sources=["a.py:1"],
                              confidence=0.5, status="inferred")
    path = vault.write_belief(artifact)["path"]

    vault._update_belief_confidence(artifact.claim_id, 0.2)

    content = Path(path).read_text(encoding="utf-8")
    assert body in content
    frontmatter = _parse_frontmatter(content) or {}
    assert float(frontmatter["confidence"]) == 0.7
    assert frontmatter["status"] == "verified"


def test_confidence_stays_within_bounds(tmp_path):
    vault = _vault(tmp_path)
    high = BeliefArtifact(entity="high", title="h", body="b", sources=["a.py:1"],
                          confidence=0.9)
    low = BeliefArtifact(entity="low", title="l", body="b", sources=["a.py:1"],
                         confidence=0.1)
    high_path = vault.write_belief(high)["path"]
    low_path = vault.write_belief(low)["path"]

    vault._update_belief_confidence(high.claim_id, 5.0)
    vault._update_belief_confidence(low.claim_id, -5.0)

    assert float((_parse_frontmatter(Path(high_path).read_text(encoding="utf-8")) or {})["confidence"]) == 1.0
    assert float((_parse_frontmatter(Path(low_path).read_text(encoding="utf-8")) or {})["confidence"]) == 0.0


def test_changing_one_file_does_not_mark_unrelated_beliefs_stale(tmp_path):
    """The entity fallback was a substring test.

    Editing `auth.py` also marked `authentication_service` and `oauth_client`
    stale. Staleness is what gates trust in a belief, so marking fresh ones
    stale erodes the signal rather than erring safely.
    """

    vault = _vault(tmp_path)
    for entity in ("auth", "authentication_service", "oauth_client"):
        vault.write_belief(
            BeliefArtifact(entity=entity, title=entity, body="b",
                           sources=[f"{entity}.py:1"])
        )

    result = vault.mark_beliefs_stale_for_files(["auth.py"])

    assert sorted(result["updated_entities"]) == ["auth"]


def test_concurrent_writers_never_tear_a_belief_or_lose_a_write(tmp_path):
    """Windows refuses to rename over a file another handle has open.

    Without a jittered retry the durability fix traded torn reads for lost
    writes: measured at 61 of 240 writes failing with no retry and 2 with a
    fixed backoff.
    """

    import threading

    vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="hot", title="h", body="seed", sources=["a.py:1"])
    )
    target = vault._base / "beliefs" / "hot.md"
    stop = False
    torn: list[int] = []
    errors: list[str] = []

    def read_loop() -> None:
        while not stop:
            try:
                if _parse_frontmatter(target.read_text(encoding="utf-8", errors="replace")) is None:
                    torn.append(1)
            except OSError:
                pass
            time.sleep(0.001)

    def write_loop(count: int) -> None:
        for index in range(count):
            try:
                vault.write_belief(
                    BeliefArtifact(entity="hot", title="h", body=f"b{index}",
                                   sources=["a.py:1"])
                )
            except Exception as exc:  # noqa: BLE001 - recorded, then asserted on
                errors.append(type(exc).__name__)
            time.sleep(0.001)

    readers = [threading.Thread(target=read_loop) for _ in range(2)]
    writers = [threading.Thread(target=write_loop, args=(25,)) for _ in range(3)]
    for thread in readers:
        thread.start()
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop = True
    for thread in readers:
        thread.join()

    assert errors == []
    assert torn == []
    assert _parse_frontmatter(target.read_text(encoding="utf-8")) is not None
    assert not list((vault._base / "beliefs").glob("*.tmp"))
