from __future__ import annotations

from pathlib import Path

import pytest

from entroly.context_fixed_point import (
    FixedPointModelError,
    FixedPointRecoveryError,
    FixedPointVerificationError,
    ProofGuidedRecoveryPlanner,
)
from entroly.context_receipts.models import text_fingerprint
from entroly.eicv_suppressor import ClaimVerdict, SuppressionResult
from entroly.ravs.events import OutcomeEvent, TraceEvent
from entroly.verified_efficiency import (
    AuditArtifact,
    VerificationFailureError,
    VerifiedEfficiencyLayer,
    VerifiedOutput,
)


def _documents() -> list[tuple[str, str]]:
    return [
        (
            "evidence.md",
            "Entroly selects relevant evidence under an explicit token budget.\n"
            "Every selected span retains its source fingerprint.\n"
            "Omitted spans remain recoverable from a content-addressed bundle.\n"
            "A signed receipt records the context delivered to the model.\n",
        ),
        (
            "operations.md",
            "The local service listens on a configurable port.\n"
            "Operators rotate credentials through their normal secret store.\n"
            "Deployment health checks report an actionable failure reason.\n"
            "Restart recovery replays durable events in their original order.\n",
        ),
    ]


def _prepared(tmp_path: Path):
    layer = VerifiedEfficiencyLayer(
        tmp_path,
        prefer_rust=False,
        context_risk_mode="audit",
    )
    prepared = layer.prepare(
        _documents(),
        query="How does Entroly preserve and recover selected evidence?",
        token_budget=50,
        chunk_tokens=24,
        overlap_tokens=3,
    )
    assert prepared.selected_context[0]["source_path"] == "evidence.md"
    assert prepared.receipt["omitted_context"][0]["source_path"] == "operations.md"
    return layer, prepared


def test_real_fixed_point_recovers_exact_evidence_and_converges(tmp_path):
    layer, prepared = _prepared(tmp_path)
    requests = []

    def model_call(request):
        requests.append(request)
        return "Restart recovery replays durable events in their original order."

    result = layer.run_fixed_point(
        prepared,
        model_call=model_call,
        max_rounds=3,
        recovery_token_budget=100,
    )

    assert result.status == "supported"
    assert result.converged is True
    assert len(result.rounds) == 2
    assert len(requests) == 2
    assert result.recovery_tokens_used == 37
    assert result.final_output.changed is False
    assert len(result.recovered_chunk_ids) == 1

    first, second = requests
    assert first.stable_context_prefix == prepared.context
    assert first.appended_evidence == ""
    assert first.recovered_chunk_ids == ()
    assert second.stable_context_prefix == prepared.context
    assert second.full_context.startswith(first.full_context)
    assert second.full_context.startswith(prepared.context)
    assert second.appended_evidence
    assert second.previous_output == (
        "Restart recovery replays durable events in their original order."
    )
    assert second.recovered_chunk_ids == result.recovered_chunk_ids
    assert "Restart recovery replays durable events" in second.appended_evidence

    first_round, second_round = result.rounds
    assert first_round.decision == "recover_and_retry"
    assert second_round.decision == "stop_supported"
    assert first_round.recovered_for_next_round[0]["verified"] is True
    assert layer.verify_audit_artifact(first_round.audit).valid
    assert layer.verify_audit_artifact(second_round.audit).valid
    assert layer.verify_audit_artifact(result.final_output.audit).valid
    assert (
        second_round.audit.attestation["receipt"]["previous_round_artifact_id"]
        == first_round.audit.artifact_id
    )

    trace = TraceEvent(request_id="fixed-point-outcome")
    outcome = OutcomeEvent(
        request_id="fixed-point-outcome",
        event_type="test_result",
        value="passed",
        strength="strong",
        source="fixed-point-test",
        include_in_default_training=True,
    )
    learning = layer.record_verified_outcome(
        prepared, result.final_output, trace, outcome
    )
    assert learning.output_audit_id == result.final_output.audit.artifact_id


def test_fixed_point_stops_when_no_omitted_evidence_supports_claim(tmp_path):
    layer, prepared = _prepared(tmp_path)
    result = layer.run_fixed_point(
        prepared,
        model_call=lambda _request: "A quasar is hidden beneath the ocean.",
        recovery_token_budget=100,
    )

    assert result.status == "no_supporting_omitted_evidence"
    assert result.converged is False
    assert result.recovered_chunk_ids == ()
    assert len(result.rounds) == 1
    assert result.rounds[0].recovery_plan is not None
    assert result.rounds[0].recovery_plan.candidates == ()


def test_fixed_point_respects_hard_recovery_budget(tmp_path):
    layer, prepared = _prepared(tmp_path)
    result = layer.run_fixed_point(
        prepared,
        model_call=lambda _request: (
            "Restart recovery replays durable events in their original order."
        ),
        recovery_token_budget=20,
    )

    assert result.status == "recovery_budget_exhausted"
    assert result.recovery_tokens_used == 0
    assert result.recovered_chunk_ids == ()
    plan = result.rounds[0].recovery_plan
    assert plan is not None
    assert plan.candidates
    assert plan.selected == ()


class _AlwaysUnsupported:
    def __init__(self, profile: str, mode: str) -> None:
        self.profile = profile
        self.mode = mode

    def suppress(self, _context: str, output: str) -> SuppressionResult:
        certificate = ClaimVerdict(
            claim_id="claim-1",
            claim_text=output,
            decision="hallucinated",
            action="suppress",
            phi=0.1,
            hallucination_score=0.9,
            e_product=3.0,
            n_claim_atoms=1,
            n_ev_atoms=1,
        )
        return SuppressionResult(
            rewritten_output="",
            original_output=output,
            changed=True,
            mode=self.mode,
            profile=self.profile,
            n_claims=1,
            n_supported=0,
            n_abstained=0,
            n_hallucinated=1,
            flagged_count=1,
            suppressed_count=1,
            warned_count=0,
            certificates=[certificate],
            latency_ms=1.0,
            calibrated=False,
        )


def test_fixed_point_has_a_hard_round_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "entroly.verified_efficiency.EICVSuppressor", _AlwaysUnsupported
    )
    layer, prepared = _prepared(tmp_path)
    calls = 0

    def model_call(_request):
        nonlocal calls
        calls += 1
        return "Restart recovery replays durable events in their original order."

    result = layer.run_fixed_point(
        prepared,
        model_call=model_call,
        max_rounds=2,
        recovery_token_budget=100,
    )

    assert result.status == "max_rounds_reached"
    assert result.converged is False
    assert calls == 2
    assert len(result.rounds) == 2


def test_model_failure_is_signed_and_actionable(tmp_path):
    layer, prepared = _prepared(tmp_path)

    def fail(_request):
        raise ConnectionError("provider unavailable")

    with pytest.raises(FixedPointModelError, match="provider unavailable") as error:
        layer.run_fixed_point(prepared, model_call=fail)

    assert error.value.round_index == 0
    assert layer.verify_audit_artifact(error.value.audit).valid
    receipt = error.value.audit.attestation["receipt"]
    assert receipt["status"] == "model_error"
    assert receipt["error_type"] == "ConnectionError"


def test_non_string_model_result_fails_with_a_signed_receipt(tmp_path):
    layer, prepared = _prepared(tmp_path)
    with pytest.raises(FixedPointModelError, match="must return a string") as error:
        layer.run_fixed_point(prepared, model_call=lambda _request: {"text": "bad"})
    assert layer.verify_audit_artifact(error.value.audit).valid


def test_claimless_output_does_not_count_as_convergence(tmp_path):
    layer, prepared = _prepared(tmp_path)
    result = layer.run_fixed_point(
        prepared,
        model_call=lambda _request: "",
    )

    assert result.status == "no_verifiable_claims"
    assert result.converged is False
    assert result.rounds[0].decision == "stop_no_verifiable_claims"


def test_inconsistent_verifier_result_is_withheld_and_audited(tmp_path, monkeypatch):
    class InconsistentSuppressor(_AlwaysUnsupported):
        def suppress(self, context: str, output: str) -> SuppressionResult:
            result = super().suppress(context, output)
            result.n_claims = 2
            return result

    monkeypatch.setattr(
        "entroly.verified_efficiency.EICVSuppressor", InconsistentSuppressor
    )
    layer, prepared = _prepared(tmp_path)
    with pytest.raises(
        FixedPointVerificationError, match="inconsistent certificate"
    ) as error:
        layer.run_fixed_point(prepared, model_call=lambda _request: "claim")

    assert layer.verify_audit_artifact(error.value.audit).valid
    assert error.value.audit.attestation["receipt"]["status"] == "verification_error"


def test_recovery_failure_records_partial_state_before_raising(tmp_path, monkeypatch):
    layer, prepared = _prepared(tmp_path)

    def fail_recovery(_prepared, _chunk_id=None):
        raise OSError("simulated recovery read failure")

    monkeypatch.setattr(layer, "recover", fail_recovery)
    with pytest.raises(FixedPointRecoveryError, match="recovery read failure") as error:
        layer.run_fixed_point(
            prepared,
            model_call=lambda _request: (
                "Restart recovery replays durable events in their original order."
            ),
            recovery_token_budget=100,
        )

    assert layer.verify_audit_artifact(error.value.audit).valid
    receipt = error.value.audit.attestation["receipt"]
    assert receipt["status"] == "recovery_error"
    assert receipt["selected_chunk_ids"]
    assert receipt["recovered_before_error"] == []


def test_augmented_verification_rejects_non_monotonic_context(tmp_path):
    layer, prepared = _prepared(tmp_path)
    with pytest.raises(VerificationFailureError, match="byte-identical prefix"):
        layer._verify_output_with_context(
            prepared,
            "output",
            grounding_context="replacement context",
        )


def test_planner_uses_global_bounded_knapsack_not_greedy_density():
    claim = "alpha beta gamma delta"
    verified = VerifiedOutput(
        commit_id="ctx_test",
        output="",
        original_output=claim,
        changed=True,
        suppression={
            "certificates": [
                {
                    "claim_id": "claim-1",
                    "claim_text": claim,
                    "decision": "hallucinated",
                    "action": "suppress",
                    "phi": 0.1,
                }
            ]
        },
        security_scan={"is_safe": True},
        audit=AuditArtifact("", "", {}),
    )
    texts = {
        "a": ("alpha beta gamma", 5),
        "b": ("alpha beta", 4),
        "c": ("gamma delta", 4),
    }
    commit = {
        "receipt": {
            "query": "",
            "omitted_context": [
                {
                    "chunk_id": chunk_id,
                    "source_path": f"{chunk_id}.md",
                    "fingerprint": f"fp-{chunk_id}",
                    "token_count": tokens,
                }
                for chunk_id, (_text, tokens) in texts.items()
            ],
        },
        "recovery_bundle": {
            "chunks": {
                chunk_id: {
                    "text": text,
                    "content_sha": text_fingerprint(text),
                }
                for chunk_id, (text, _tokens) in texts.items()
            }
        },
    }

    plan = ProofGuidedRecoveryPlanner().plan(
        verified,
        commit,
        token_budget=8,
        max_chunks=2,
    )

    assert [item.chunk_id for item in plan.selected] == ["b", "c"]
    assert plan.selected_tokens == 8


def test_planner_candidate_work_is_explicitly_bounded():
    verified = VerifiedOutput(
        commit_id="ctx_test",
        output="",
        original_output="alpha",
        changed=True,
        suppression={
            "certificates": [
                {
                    "claim_id": "claim-1",
                    "claim_text": "alpha",
                    "decision": "hallucinated",
                    "action": "suppress",
                    "phi": 0.1,
                }
            ]
        },
        security_scan={"is_safe": True},
        audit=AuditArtifact("", "", {}),
    )
    omitted = []
    chunks = {}
    for index in range(140):
        chunk_id = f"chunk-{index:03d}"
        text = f"alpha evidence {index}"
        omitted.append(
            {
                "chunk_id": chunk_id,
                "source_path": f"{chunk_id}.md",
                "fingerprint": f"fp-{index}",
                "token_count": 1,
            }
        )
        chunks[chunk_id] = {
            "text": text,
            "content_sha": text_fingerprint(text),
        }
    commit = {
        "receipt": {"query": "", "omitted_context": omitted},
        "recovery_bundle": {"chunks": chunks},
    }

    plan = ProofGuidedRecoveryPlanner().plan(
        verified,
        commit,
        token_budget=10,
        max_chunks=1,
    )

    assert plan.eligible_candidate_count == 140
    assert plan.candidate_limit_applied is True
    assert len(plan.candidates) == 128
    assert len(plan.selected) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_rounds": 0}, "max_rounds"),
        ({"max_rounds": 9}, "max_rounds"),
        ({"recovery_token_budget": -1}, "recovery_token_budget"),
        ({"recovery_token_budget": 100_001}, "recovery_token_budget"),
        ({"max_chunks_per_round": 0}, "max_chunks_per_round"),
    ],
)
def test_fixed_point_configuration_is_bounded(tmp_path, kwargs, message):
    layer, prepared = _prepared(tmp_path)
    with pytest.raises(ValueError, match=message):
        layer.run_fixed_point(
            prepared,
            model_call=lambda _request: "output",
            **kwargs,
        )
