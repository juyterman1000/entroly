from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from entroly.verified_handoff import chain_handoff, handoff, receive


@dataclass
class _Cert:
    claim: str
    label: str
    risk: float = 0.1
    proof_steps: list[object] | None = None


class _Analyzer:
    def __init__(self, certificates: list[_Cert] | None = None):
        self.certificates = certificates or [
            _Cert(
                claim="auth uses bcrypt",
                label="grounded",
                proof_steps=[SimpleNamespace(evidence="auth.py uses bcrypt")],
            )
        ]

    def analyze(self, output: str, evidence: str):
        del output, evidence
        return SimpleNamespace(certificates=self.certificates, summary_score=0.9)


def _bundle():
    return handoff(
        "auth uses bcrypt",
        "auth.py uses bcrypt",
        from_agent="agent-a",
        to_agent="agent-b",
        analyzer=_Analyzer(),
    )


def test_receive_accepts_unmodified_sealed_bundle():
    bundle = _bundle()
    assert bundle.integrity_hash
    assert receive(bundle) == "auth uses bcrypt"


def test_receive_rejects_verified_context_mutation():
    bundle = _bundle()
    sealed = bundle.integrity_hash
    bundle.verified_context = "tampered"
    assert bundle.integrity_hash == sealed
    with pytest.raises(ValueError, match="Integrity check failed"):
        receive(bundle)


def test_receive_rejects_routing_metadata_mutation():
    bundle = _bundle()
    bundle.to_agent = "attacker-controlled-target"
    with pytest.raises(ValueError, match="Integrity check failed"):
        receive(bundle)


def test_receive_rejects_unsealed_bundle_when_verification_requested():
    bundle = _bundle()
    bundle.sealed_integrity_hash = ""
    with pytest.raises(ValueError, match="no creation-time integrity seal"):
        receive(bundle)
    assert receive(bundle, verify_integrity=False) == bundle.verified_context


def test_handoff_rejects_unknown_mode_and_invalid_chain_metadata():
    with pytest.raises(ValueError, match="Unsupported WVH mode"):
        handoff("x", "x", mode="typo", analyzer=_Analyzer())
    with pytest.raises(ValueError, match="chain_position"):
        handoff("x", "x", chain_position=-1, analyzer=_Analyzer())
    with pytest.raises(ValueError, match="upstream_bundles"):
        handoff("x", "x", upstream_bundles=[""], analyzer=_Analyzer())


def test_chain_handoff_produces_independently_sealed_bundles():
    bundles = chain_handoff(
        ["agent-a", "agent-b", "agent-c"],
        "auth uses bcrypt",
        "auth.py uses bcrypt",
        analyzer=_Analyzer(),
    )
    assert len(bundles) == 2
    assert all(bundle.integrity_hash for bundle in bundles)
    assert bundles[1].upstream_bundle_ids == [bundles[0].bundle_id]
    assert receive(bundles[0])
    assert receive(bundles[1])
