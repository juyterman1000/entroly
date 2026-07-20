#!/usr/bin/env python3
"""No-network conformance contract for the proof-guided context fixed point."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from entroly.context_fixed_point import FixedPointModelError  # noqa: E402
from entroly.ravs.events import OutcomeEvent, TraceEvent  # noqa: E402
from entroly.verified_efficiency import VerifiedEfficiencyLayer  # noqa: E402


def main() -> int:
    selected_source = (
        "Entroly selects relevant evidence under an explicit token budget.\n"
        "Every selected span retains its source fingerprint.\n"
        "Omitted spans remain recoverable from a content-addressed bundle.\n"
        "A signed receipt records the context delivered to the model.\n"
    )
    omitted_source = (
        "The local service listens on a configurable port.\n"
        "Operators rotate credentials through their normal secret store.\n"
        "Deployment health checks report an actionable failure reason.\n"
        "Restart recovery replays durable events in their original order.\n"
    )

    with tempfile.TemporaryDirectory(prefix="entroly-fixed-point-contract-") as tmp:
        layer = VerifiedEfficiencyLayer(
            Path(tmp),
            prefer_rust=False,
            context_risk_mode="audit",
        )
        prepared = layer.prepare(
            [
                ("evidence.md", selected_source),
                ("operations.md", omitted_source),
            ],
            query="How does Entroly preserve and recover selected evidence?",
            token_budget=50,
            chunk_tokens=24,
            overlap_tokens=3,
        )
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

        failure_audited = False
        try:
            layer.run_fixed_point(
                prepared,
                model_call=lambda _request: (_ for _ in ()).throw(
                    ConnectionError("offline contract failure")
                ),
            )
        except FixedPointModelError as exc:
            failure_audited = layer.verify_audit_artifact(exc.audit).valid

        trace = TraceEvent(request_id="fixed-point-contract")
        outcome = OutcomeEvent(
            request_id="fixed-point-contract",
            event_type="test_result",
            value="passed",
            strength="strong",
            source="fixed-point-contract",
            include_in_default_training=True,
        )
        learning = layer.record_verified_outcome(
            prepared, result.final_output, trace, outcome
        )

        checks = {
            "converged_after_exact_recovery": result.status == "supported",
            "used_two_bounded_model_calls": len(requests) == len(result.rounds) == 2,
            "committed_prefix_is_byte_stable": all(
                request.full_context.startswith(prepared.context)
                and request.stable_context_prefix == prepared.context
                for request in requests
            ),
            "context_expansion_is_monotonic": requests[1].full_context.startswith(
                requests[0].full_context
            ),
            "recovered_one_verified_chunk": (
                len(result.recovered_chunk_ids) == 1
                and result.rounds[0].recovered_for_next_round[0]["verified"]
            ),
            "round_lineage_is_signed": all(
                layer.verify_audit_artifact(round_result.audit).valid
                for round_result in result.rounds
            ),
            "unsupported_first_round_opened_obligation": bool(
                result.rounds[0].obligations
            ),
            "supported_second_round_closed_obligations": not result.rounds[
                1
            ].obligations,
            "model_failure_is_audited": failure_audited,
            "final_output_accepts_strong_external_learning": bool(
                learning.receipt_hash
            ),
            "performance_claim_is_false": True,
        }
        passed = all(value is True for value in checks.values())
        print(
            json.dumps(
                {
                    "schema": "entroly.context-fixed-point-contract.v1",
                    "passed": passed,
                    "performance_claim": False,
                    "checks": checks,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
