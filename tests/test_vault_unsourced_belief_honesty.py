"""A belief citing nothing cannot be verified.

`write_belief` accepted `status="verified"` with an empty `sources` list, and
`coverage_index` counted it toward `verified`. `mark_beliefs_ungrounded` could
not correct it either: that pass retracts a belief whose every cited source is
gone, and deliberately skips one citing nothing, because there is nothing to
resolve against.

So an evidence-free claim was written as verified, never retracted, and folded
into a number the vault asserts about its own trustworthiness.
"""

from __future__ import annotations

import pytest

from entroly.vault import BeliefArtifact, VaultConfig, VaultManager


@pytest.fixture
def vault(tmp_path):
    manager = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    manager.ensure_structure()
    return manager


def _write(vault, entity, sources, status="verified"):
    return vault.write_belief(BeliefArtifact(
        entity=entity, status=status, confidence=0.99,
        sources=sources, title=entity, body="body",
    ))


class TestUnsourcedBeliefsCannotClaimVerified:
    def test_status_is_downgraded_at_write(self, vault):
        _write(vault, "ghost.entity", sources=[])
        stored = {b["entity"]: b for b in vault.list_beliefs()}
        assert stored["ghost.entity"]["status"] == "unsupported"

    def test_the_belief_is_kept_not_rejected(self, vault):
        # Losing a claim because its provenance was never attached would be
        # worse than holding it at a lower status.
        result = _write(vault, "ghost.entity", sources=[])
        assert result["status"] == "written"
        assert vault.read_belief("ghost.entity") is not None

    def test_coverage_index_excludes_it_from_verified(self, vault):
        _write(vault, "grounded.entity", sources=["src/auth.py:1-2"])
        _write(vault, "ghost.entity", sources=[])

        index = vault.coverage_index()
        assert index["total_beliefs"] == 2
        assert index["verified"] == 1, (
            "an evidence-free claim was counted inside the vault's own "
            "trustworthiness number"
        )

    def test_a_sourced_belief_keeps_verified(self, vault):
        _write(vault, "grounded.entity", sources=["src/auth.py:1-2"])
        stored = {b["entity"]: b for b in vault.list_beliefs()}
        assert stored["grounded.entity"]["status"] == "verified"

    def test_non_verified_statuses_are_left_alone(self, vault):
        # Only the verified claim is unsupportable without evidence. An
        # explicitly inferred belief is already honest about what it is.
        _write(vault, "guess.entity", sources=[], status="inferred")
        stored = {b["entity"]: b for b in vault.list_beliefs()}
        assert stored["guess.entity"]["status"] == "inferred"
