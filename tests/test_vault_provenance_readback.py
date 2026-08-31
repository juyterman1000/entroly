"""A belief must read back with the evidence it was written with.

Two defects found by dogfooding the vault, both in the gap between what is
written and what is read.

1. `_parse_frontmatter` kept only `key: value` lines and skipped anything
   starting with `-`, so YAML block sequences were dropped. `sources` and
   `derived_from` are block sequences. They were written to disk correctly and
   then silently lost on the way back in: `read_belief` -- returned verbatim by
   the `vault_query` MCP tool -- reported `status: verified, confidence: 0.95`
   with no citations, so an agent could not check what the claim had been
   verified against. CLAUDE.md requires beliefs to be machine-auditable and
   omitted evidence to be inspectable; evidence that survives the write and not
   the read is neither.

   Every caller that actually needed provenance had worked around this by
   re-scanning the raw text with `_extract_sources` beside its parse, so the
   store had two extractors that could disagree about what a belief cites.

2. Recency outranked evidence. A `verified` belief at 0.95 citing
   `src/auth.py:42` was replaced by a `hypothesis` at 0.1 citing nothing, and
   `read_belief` returned the hypothesis. The ledger kept both, so provenance
   survived -- but nothing reads the ledger by default, so what every agent
   actually read was the unsourced guess.

   This is the dominant state-update failure reported for long-horizon agents
   (SKILL.state, arXiv 2608.26263: premature state overwrite/deletion), and the
   answer is the same -- a weaker patch must not corrupt current state. The
   claim is still kept and still recorded as a competing claim; it just does
   not become what agents read.

The guard is deliberately narrow. It fires only when an update is weaker in
status *and* in confidence *and* cites nothing new -- weaker on every axis at
once, which is a regression rather than a correction. Anything that carries new
evidence, raises confidence, or reports staleness goes through untouched. The
`TestCorrectionsStillGoThrough` cases below are the ones that matter: a guard
that also blocked real corrections would be worse than the bug.
"""

from __future__ import annotations

import json

import pytest

from entroly.vault import (
    BeliefArtifact,
    VaultConfig,
    VaultManager,
    _extract_sources,
    _parse_frontmatter,
)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    manager = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    manager.ensure_structure()
    return manager


def _write(vault, **kw):
    fields = dict(
        entity="src/auth",
        title="auth",
        body="passwords are hashed with a per-user salt",
        status="verified",
        confidence=0.95,
        sources=["src/auth.py:42"],
    )
    fields.update(kw)
    return vault.write_belief(BeliefArtifact(**fields))


def _frontmatter(vault, entity="src/auth"):
    return (vault.read_belief(entity) or {}).get("frontmatter", {})


class TestBlockSequencesSurviveTheRoundTrip:
    def test_sources_are_readable_after_being_written(self, vault):
        _write(vault, sources=["src/auth.py:42", "src/hash.py:7"])
        assert _frontmatter(vault)["sources"] == ["src/auth.py:42", "src/hash.py:7"]

    def test_derived_from_is_readable_after_being_written(self, vault):
        _write(vault, derived_from=["query-1", "query-2"])
        assert _frontmatter(vault)["derived_from"] == ["query-1", "query-2"]

    def test_a_verified_belief_never_reads_back_without_its_evidence(self, vault):
        """The defect exactly: a claim asserting evidence, presented with none."""
        _write(vault)
        fm = _frontmatter(vault)
        assert fm["status"] == "verified"
        assert fm.get("sources"), (
            "a belief that reads back as verified while citing nothing is "
            "unauditable -- there is no way to check what it was verified against"
        )

    def test_a_belief_citing_nothing_says_so_explicitly(self, vault):
        """`sources: []` is a recorded answer, not a missing field."""
        _write(vault, status="inferred", sources=[])
        assert _frontmatter(vault)["sources"] == []

    def test_scalars_are_unchanged(self, vault):
        _write(vault)
        fm = _frontmatter(vault)
        assert fm["entity"] == "src/auth"
        assert fm["status"] == "verified"
        assert fm["claim_id"]
        assert fm["last_checked"]

    def test_a_colon_in_a_list_item_is_preserved(self):
        """`path:line` citations contain the delimiter the parser splits on."""
        parsed = _parse_frontmatter(
            "---\nentity: e\nsources:\n  - src/auth.py:42\n---\n\nbody\n"
        )
        assert parsed["sources"] == ["src/auth.py:42"]

    def test_a_key_with_no_value_and_no_items_is_not_invented_as_a_list(self):
        parsed = _parse_frontmatter("---\nentity: e\nempty:\nstatus: inferred\n---\n\nb\n")
        assert parsed["status"] == "inferred"
        assert "empty" not in parsed

    def test_the_two_source_extractors_agree(self, vault):
        """They disagreed before; a store with two answers has no audit trail."""
        _write(vault, sources=["src/auth.py:42", "src/hash.py:7"])
        content = open(vault.read_belief("src/auth")["path"], encoding="utf-8").read()
        assert _extract_sources(content) == _parse_frontmatter(content)["sources"]


class TestTheMcpSurfaceExposesProvenance:
    def test_the_payload_an_agent_receives_carries_the_citations(self, vault):
        """`vault_query` returns `read_belief` verbatim, so this is what agents see."""
        _write(vault, sources=["src/auth.py:42"])
        payload = json.loads(json.dumps(vault.read_belief("src/auth"), default=str))
        assert payload["frontmatter"]["sources"] == ["src/auth.py:42"]


class TestEvidenceOutranksRecency:
    def test_an_unsourced_guess_does_not_replace_a_sourced_verified_belief(self, vault):
        _write(vault)
        result = _write(
            vault,
            body="passwords are stored unsalted",
            status="hypothesis",
            confidence=0.1,
            sources=[],
        )

        assert result["status"] == "kept_stronger_claim"
        fm = _frontmatter(vault)
        assert fm["status"] == "verified"
        assert float(fm["confidence"]) == pytest.approx(0.95)
        assert "salt" in (vault.read_belief("src/auth") or {})["body"]

    def test_the_refused_claim_is_still_recorded(self, vault):
        """Refusing to promote a claim is not the same as discarding it."""
        from entroly.vault_time import BeliefLedger

        _write(vault)
        _write(vault, status="hypothesis", confidence=0.1, sources=[])

        timeline = BeliefLedger(vault._base).timeline("src/auth")
        assert len(timeline) == 2, (
            "the competing claim must remain auditable even though it did not win"
        )


class TestCorrectionsStillGoThrough:
    """A guard that blocked real corrections would be worse than the bug."""

    def test_new_evidence_overwrites_even_at_lower_confidence(self, vault):
        _write(vault)
        result = _write(
            vault,
            body="actually unsalted",
            status="inferred",
            confidence=0.4,
            sources=["src/auth.py:99"],
        )
        assert result["status"] != "kept_stronger_claim"
        assert _frontmatter(vault)["status"] == "inferred"

    def test_a_stronger_claim_overwrites(self, vault):
        _write(vault)
        _write(vault, confidence=0.99)
        assert float(_frontmatter(vault)["confidence"]) == pytest.approx(0.99)

    def test_marking_a_belief_stale_is_never_refused(self, vault):
        """Staleness reports freshness; it is not a competing evidence claim."""
        _write(vault)
        result = _write(vault, status="stale", confidence=0.3)
        assert result["status"] != "kept_stronger_claim"
        assert _frontmatter(vault)["status"] == "stale"

    def test_a_lower_status_with_higher_confidence_overwrites(self, vault):
        _write(vault)
        result = _write(vault, status="inferred", confidence=0.99)
        assert result["status"] != "kept_stronger_claim"

    def test_the_first_belief_about_an_entity_is_never_refused(self, vault):
        result = _write(vault, entity="brand/new", status="hypothesis", confidence=0.1)
        assert result["status"] != "kept_stronger_claim"
        assert _frontmatter(vault, "brand/new")["status"] == "hypothesis"

    def test_a_weak_claim_may_replace_another_weak_unsourced_claim(self, vault):
        """With no evidence on either side there is nothing to protect."""
        _write(vault, entity="e2", status="inferred", confidence=0.6, sources=[])
        result = _write(
            vault, entity="e2", status="hypothesis", confidence=0.2, sources=[]
        )
        assert result["status"] != "kept_stronger_claim"
