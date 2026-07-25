"""Tests for the fail-closed verified world-state kernel."""

from __future__ import annotations

import json

import pytest

from entroly.world_state import (
    ClaimStatus,
    EvidenceRef,
    InvalidClaimTransition,
    SnapshotIntegrityError,
    TransitionPrediction,
    VerificationReceipt,
    VerificationRequired,
    VerifiedWorldState,
    WorldClaim,
)


def _evidence(name: str, *, repo_sha: str = "abc123") -> EvidenceRef:
    return EvidenceRef.from_text(
        source_id=name,
        text=f"evidence:{name}",
        repo_sha=repo_sha,
        locator=f"tests/{name}.py:1",
    )


def _verified(state: VerifiedWorldState, claim_id: str, *, depends_on=()) -> None:
    evidence = _evidence(claim_id)
    state.add_claim(
        WorldClaim(
            claim_id=claim_id,
            subject=claim_id,
            predicate="is_true",
            object_value="yes",
            depends_on=tuple(depends_on),
            repo_sha="abc123",
        )
    )
    state.transition(claim_id, ClaimStatus.SUPPORTED, evidence=[evidence])
    state.transition(
        claim_id,
        ClaimStatus.VERIFIED,
        receipt=VerificationReceipt(
            verifier="pytest",
            passed=True,
            evidence=(evidence,),
            repo_sha="abc123",
        ),
    )


def test_claim_cannot_enter_verified_state_directly():
    state = VerifiedWorldState()
    with pytest.raises(VerificationRequired, match="trusted states"):
        state.add_claim(
            WorldClaim(
                claim_id="c1",
                subject="cache",
                predicate="causes",
                object_value="loop",
                status=ClaimStatus.VERIFIED,
            )
        )


def test_observed_claim_and_transition_reject_stale_evidence():
    state = VerifiedWorldState()
    stale = _evidence("stale", repo_sha="old")
    with pytest.raises(VerificationRequired, match="stale or unversioned"):
        state.add_claim(
            WorldClaim(
                claim_id="observed",
                subject="cache",
                predicate="failed",
                object_value="yes",
                status=ClaimStatus.OBSERVED,
                evidence=(stale,),
                repo_sha="new",
            )
        )

    state.add_claim(
        WorldClaim(
            claim_id="proposed",
            subject="cache",
            predicate="failed",
            object_value="yes",
            repo_sha="new",
        )
    )
    with pytest.raises(VerificationRequired, match="stale or unversioned"):
        state.transition("proposed", ClaimStatus.OBSERVED, evidence=[stale])


def test_verified_transition_requires_positive_current_receipt():
    state = VerifiedWorldState()
    evidence = _evidence("cache")
    state.add_claim(
        WorldClaim(
            claim_id="c1",
            subject="cache",
            predicate="causes",
            object_value="loop",
            repo_sha="abc123",
        )
    )
    state.transition("c1", ClaimStatus.SUPPORTED, evidence=[evidence])

    with pytest.raises(VerificationRequired, match="passing receipt"):
        state.transition("c1", ClaimStatus.VERIFIED)
    with pytest.raises(VerificationRequired, match="stale"):
        state.transition(
            "c1",
            ClaimStatus.VERIFIED,
            receipt=VerificationReceipt(
                verifier="pytest",
                passed=True,
                evidence=(evidence,),
                repo_sha="wrong-sha",
            ),
        )


def test_illegal_status_reversal_fails_closed():
    state = VerifiedWorldState()
    state.add_claim(
        WorldClaim(
            claim_id="c1",
            subject="cache",
            predicate="exists",
            object_value="yes",
        )
    )
    state.transition("c1", ClaimStatus.OBSERVED, evidence=[_evidence("cache")])
    with pytest.raises(InvalidClaimTransition, match="observed -> proposed"):
        state.transition("c1", ClaimStatus.PROPOSED)


def test_invalidation_propagates_only_to_downstream_dependents():
    state = VerifiedWorldState()
    _verified(state, "root")
    _verified(state, "child", depends_on=("root",))
    _verified(state, "grandchild", depends_on=("child",))
    _verified(state, "independent")

    invalidation = _evidence("root-invalid")
    state.transition(
        "root",
        ClaimStatus.INVALIDATED,
        evidence=[invalidation],
        reason="test_failed",
    )

    assert state.claim("root").invalidated_by == "test_failed"
    assert state.claim("child").invalidated_by == "dependency:root"
    assert state.claim("grandchild").invalidated_by == "dependency:child"
    assert state.claim("independent").status is ClaimStatus.VERIFIED
    assert [claim.claim_id for claim in state.verified_frontier()] == ["independent"]


def test_digest_is_independent_of_claim_insertion_order():
    left = VerifiedWorldState()
    right = VerifiedWorldState()
    claims = [
        WorldClaim("b", "b", "exists", "yes"),
        WorldClaim("a", "a", "exists", "yes"),
    ]
    for claim in claims:
        left.add_claim(claim)
    for claim in reversed(claims):
        right.add_claim(claim)
    assert left.digest() == right.digest()


def test_prediction_error_reduces_model_authority():
    state = VerifiedWorldState()
    wrong = TransitionPrediction.create(
        "p1",
        "world-model-a",
        "edit-cache",
        {"tests_pass": 0.95, "regression": 0.05},
    )
    result = state.reconcile_prediction(wrong, {"regression"})
    assert result.brier_error > 0.8
    assert result.new_authority < 0.5

    correct = TransitionPrediction.create(
        "p2",
        "world-model-b",
        "edit-cache",
        {"tests_pass": 1.0, "regression": 0.0},
    )
    result = state.reconcile_prediction(correct, {"tests_pass"})
    assert result.brier_error == 0.0
    assert result.new_authority == 1.0


def test_snapshot_round_trip_and_tamper_detection(tmp_path):
    state = VerifiedWorldState()
    _verified(state, "claim")
    path = state.save(tmp_path / "state.json")

    loaded = VerifiedWorldState.load(path)
    assert loaded.digest() == state.digest()
    assert [claim.claim_id for claim in loaded.verified_frontier()] == ["claim"]

    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["claims"][0]["object_value"] = "forged"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="digest mismatch"):
        VerifiedWorldState.load(path)


def test_snapshot_rejects_missing_dependency_even_with_recomputed_digest():
    state = VerifiedWorldState()
    state.add_claim(WorldClaim("root", "root", "exists", "yes"))
    state.add_claim(
        WorldClaim("child", "child", "requires", "root", depends_on=("root",))
    )
    snapshot = state.snapshot()
    snapshot["claims"] = [
        item for item in snapshot["claims"] if item["claim_id"] != "root"
    ]

    import hashlib

    payload = {
        "schema": snapshot["schema"],
        "claims": snapshot["claims"],
        "model_authority": snapshot["model_authority"],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    snapshot["state_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    with pytest.raises(SnapshotIntegrityError, match="missing dependencies"):
        VerifiedWorldState.from_snapshot(snapshot)
