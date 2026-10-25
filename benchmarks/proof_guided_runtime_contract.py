#!/usr/bin/env python3
"""No-network conformance contract for the host-driven proof protocol.

This verifies durability, exact recovery, and idempotency. It is not a model
quality, performance, cost-savings, or superiority benchmark.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from entroly.proof_guided_runtime import (  # noqa: E402
    ProofGuidedRuntime,
    ProofGuidedSessionConflict,
)


def main() -> int:
    documents = [
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
    model_output = "Restart recovery replays durable events in their original order."

    with tempfile.TemporaryDirectory(prefix="entroly-proof-runtime-") as tmp:
        runtime = ProofGuidedRuntime(
            tmp,
            prefer_rust=False,
            context_risk_mode="audit",
        )
        prepared = runtime.prepare(
            documents,
            query="How does Entroly preserve and recover evidence?",
            # Select a real evidence packet before testing the binding contract.
            token_budget=70,
            chunk_tokens=24,
            overlap_tokens=3,
            max_rounds=3,
            recovery_token_budget=100,
            idempotency_key="contract-prepare",
        )
        first = runtime.advance(
            prepared["session_id"],
            model_output=model_output,
            idempotency_key="contract-round-0",
            expected_revision=prepared["revision"],
            request_sha256=prepared["request_sha256"],
        )
        restarted = ProofGuidedRuntime(
            tmp,
            prefer_rust=False,
            context_risk_mode="audit",
        )
        path = restarted.sessions_dir / f"{prepared['session_id']}.json"
        before_stale = path.read_bytes()
        stale_error = None
        try:
            restarted.advance(
                prepared["session_id"],
                model_output=model_output,
                idempotency_key="contract-delayed-old-reply-new-key",
                expected_revision=prepared["revision"],
                request_sha256=prepared["request_sha256"],
            )
        except ProofGuidedSessionConflict as exc:
            stale_error = type(exc).__name__
        stale_unchanged = path.read_bytes() == before_stale
        stale_rejected = stale_error == "ProofGuidedSessionConflict"

        final = restarted.advance(
            prepared["session_id"],
            model_output=model_output,
            idempotency_key="contract-round-1",
            expected_revision=first["revision"],
            request_sha256=first["request_sha256"],
        )
        after_later_round_restart = ProofGuidedRuntime(
            tmp,
            prefer_rust=False,
            context_risk_mode="audit",
        )
        before_replay = path.read_bytes()
        old_replay = after_later_round_restart.advance(
            prepared["session_id"],
            model_output=model_output,
            idempotency_key="contract-round-0",
            expected_revision=prepared["revision"],
            request_sha256=prepared["request_sha256"],
        )

        checks = {
            "prepare_is_local_only": prepared["provider_call_performed"] is False,
            "new_runtime_schema_is_v2": (
                prepared["schema_version"] == "entroly.proof-guided-runtime.v2"
            ),
            "first_round_recovers_exact_evidence": (
                first["status"] == "awaiting_model"
                and first["round"]["decision"] == "recover_and_retry"
                and bool(first["recovered_chunk_ids"])
            ),
            "committed_prefix_is_byte_stable": (
                prepared["request"]["stable_context_prefix"].encode("utf-8")
                == first["request"]["stable_context_prefix"].encode("utf-8")
                and first["request"]["full_context"].encode("utf-8")
                == prepared["request"]["stable_context_prefix"].encode("utf-8")
                + first["request"]["appended_evidence"].encode("utf-8")
            ),
            "recovery_issues_fresh_request_binding": (
                first["revision"] == prepared["revision"] + 1
                and first["request_sha256"] != prepared["request_sha256"]
            ),
            "stale_request_binding_is_rejected": stale_rejected,
            "stale_rejection_preserves_session_bytes": stale_unchanged,
            "restart_recovery_converges": (
                final["status"] == "supported" and final["converged"] is True
            ),
            "terminal_response_has_no_request_digest": (
                final["request"] is None and final["request_sha256"] is None
            ),
            "older_idempotency_key_replays_original_revision": old_replay == first,
            "old_tuple_replay_preserves_latest_session_bytes": (
                path.read_bytes() == before_replay
            ),
            "inspection_returns_last_durable_revision": (
                after_later_round_restart.inspect(prepared["session_id"]) == final
            ),
            "provider_transport_remains_host_owned": all(
                response["provider_call_performed"] is False
                for response in (prepared, first, final)
            ),
            "performance_claim_is_false": True,
        }
        passed = all(value is True for value in checks.values())
        print(
            json.dumps(
                {
                    "schema": "entroly.proof-guided-runtime-contract.v2",
                    "passed": passed,
                    "agent_benchmark": False,
                    "cost_claim": False,
                    "performance_claim": False,
                    "stale_rejection": {
                        "rejected": stale_rejected,
                        "error_type": stale_error,
                        "submitted_revision": prepared["revision"],
                        "current_revision": first["revision"],
                        "session_bytes_unchanged": stale_unchanged,
                    },
                    "checks": checks,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
