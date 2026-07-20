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

from entroly.proof_guided_runtime import ProofGuidedRuntime  # noqa: E402


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
            token_budget=50,
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
        )
        restarted = ProofGuidedRuntime(
            tmp,
            prefer_rust=False,
            context_risk_mode="audit",
        )
        final = restarted.advance(
            prepared["session_id"],
            model_output=model_output,
            idempotency_key="contract-round-1",
        )
        old_replay = restarted.advance(
            prepared["session_id"],
            model_output=model_output,
            idempotency_key="contract-round-0",
        )

        checks = {
            "prepare_is_local_only": prepared["provider_call_performed"] is False,
            "first_round_recovers_exact_evidence": (
                first["status"] == "awaiting_model"
                and first["round"]["decision"] == "recover_and_retry"
                and bool(first["recovered_chunk_ids"])
            ),
            "committed_prefix_is_byte_stable": first["request"][
                "full_context"
            ].startswith(first["request"]["stable_context_prefix"]),
            "restart_recovery_converges": (
                final["status"] == "supported" and final["converged"] is True
            ),
            "older_idempotency_key_replays_original_revision": old_replay == first,
            "inspection_returns_last_durable_revision": (
                restarted.inspect(prepared["session_id"]) == final
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
                    "schema": "entroly.proof-guided-runtime-contract.v1",
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
