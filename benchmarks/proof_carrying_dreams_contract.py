"""Executable conformance check for the Proof-Carrying Dreams firewall.

This is not a performance benchmark. It verifies that simulator output can
rank a proposal while promotion remains restricted to exact, ledger-committed
real evidence under an anytime-valid boundary.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from entroly.ravs.world_model import (  # noqa: E402
    EmpiricalWorldModel,
    TransitionLedger,
    VerifiedDreamController,
    VerifiedTransition,
)


def _transition(
    index: int,
    *,
    reward: float,
    policy_version: str,
    environment: str,
    action: float = 0.1,
) -> VerifiedTransition:
    state = (index / 100.0, 0.25)
    return VerifiedTransition(
        transition_id=f"{environment}:{policy_version}:{index}",
        state=state,
        action=(action, 0.0),
        next_state=(state[0] + action, state[1]),
        reward=reward,
        environment=environment,
        source="proof_carrying_dreams_contract",
        verifier="deterministic_contract",
        strength="strong",
        policy_version=policy_version,
    )


def run_contract(root: str | Path) -> dict[str, object]:
    ledger = TransitionLedger(root)
    controller = VerifiedDreamController(
        ledger,
        EmpiricalWorldModel(min_samples=4, neighbors=4),
        min_confidence=0.0,
        max_horizon=1,
    )

    support = [
        _transition(
            index,
            reward=0.8,
            policy_version="support-positive",
            environment="dream-contract",
        )
        for index in range(4)
    ]
    support.extend(
        _transition(
            index + 20,
            reward=-0.8,
            policy_version="support-negative",
            environment="dream-contract",
            action=-0.1,
        )
        for index in range(4)
    )
    for transition in support:
        controller.observe_real(transition)

    dream = controller.dream(
        support[1].state,
        lambda _state: [(0.1, 0.0), (-0.1, 0.0)],
        horizon=1,
        environment="dream-contract",
        policy_version="proposal-pool:contract",
    )

    candidate = [
        _transition(
            index + 100,
            reward=1.0,
            policy_version="candidate-v1",
            environment="promotion-contract",
        )
        for index in range(64)
    ]
    incumbent = [
        _transition(
            index + 200,
            reward=-1.0,
            policy_version="incumbent-v1",
            environment="promotion-contract",
        )
        for index in range(64)
    ]
    for transition in [*candidate, *incumbent]:
        ledger.record_real(transition)

    accepted = controller.assess_promotion(
        candidate,
        incumbent,
        min_real_samples=10,
    )
    forged = replace(candidate[0], reward=0.0)
    rejected = controller.assess_promotion(
        [forged, *candidate[1:]],
        incumbent,
        min_real_samples=10,
    )

    passed = bool(
        dream.transitions
        and all(t.parent_receipt_hashes for t in dream.transitions)
        and all(t.influence_scope == "proposal_only" for t in dream.transitions)
        and accepted.promote
        and accepted.anytime_valid
        and not rejected.promote
        and "exactly committed" in rejected.reason
    )
    return {
        "schema": "entroly.proof-carrying-dreams.contract.v1",
        "passed": passed,
        "performance_claim": False,
        "dream_scope": dream.promotion_status,
        "dream_parent_receipts_closed": all(
            bool(t.parent_receipt_hashes) for t in dream.transitions
        ),
        "promotion_boundary": accepted.boundary_type,
        "promotion_anytime_valid": accepted.anytime_valid,
        "valid_real_promotion": accepted.promote,
        "forged_real_payload_rejected": not rejected.promote,
        "forgery_reason": rejected.reason,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="entroly-pcd-") as root:
        result = run_contract(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
