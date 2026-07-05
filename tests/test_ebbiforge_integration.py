from __future__ import annotations

from dataclasses import dataclass

from entroly.integrations.ebbiforge import (
    EbbiforgeEntrolyBridge,
    run_swarm_with_entroly,
    summarize_ebbiforge_anomalies,
)
from entroly.session_intelligence import HallucinationTaintTracker, SessionReceiptChain


@dataclass
class FakeRecord:
    agent_name: str
    claim: dict
    source: str
    confidence: float

    @property
    def verified(self) -> bool:
        return self.source not in ("internal_knowledge", "unknown", "")


@dataclass
class FakeProvenance:
    records: list[FakeRecord]


class FakeResult:
    def __init__(self, value: str, provenance: FakeProvenance) -> None:
        self.value = value
        self.provenance = provenance

    def __str__(self) -> str:
        return self.value


class FakeTask:
    prompt = "Investigate multi-agent hallucination propagation"
    pipeline = True


class FakeSwarm:
    def run(self, task: FakeTask) -> FakeResult:
        provenance = FakeProvenance(
            records=[
                FakeRecord(
                    agent_name="Researcher",
                    claim={"claim": "Use imaginary_api_client before deployment"},
                    source="internal_knowledge",
                    confidence=0.1,
                ),
                FakeRecord(
                    agent_name="Analyst",
                    claim={
                        "claim": (
                            "Continue the implementation with "
                            "imaginary_api_client in the gateway"
                        )
                    },
                    source="derived:Researcher",
                    confidence=0.8,
                ),
            ]
        )
        return FakeResult("done", provenance)

    def get_belief_provenance(self, result: FakeResult) -> FakeProvenance:
        return result.provenance


def test_ebbiforge_bridge_builds_receipt_chain_and_taint(tmp_path) -> None:
    bridge = EbbiforgeEntrolyBridge(
        session_id="ebbiforge-session",
        total_budget=10_000,
        risk_half_life_turns=1000.0,
    )

    audited = bridge.run_swarm(FakeSwarm(), FakeTask(), output_dir=tmp_path)

    assert str(audited.result) == "done"
    assert len(audited.turns) == 2
    assert audited.session_chain.verify_integrity()["valid"] is True
    assert audited.session_chain.links[0].budget_decision["policy"] == "decay"
    assert audited.taint_reports[0].propagated_entities == ()
    assert "imaginary_api_client" in audited.taint_reports[1].propagated_entities
    assert audited.taint_reports[1].origin_turns["imaginary_api_client"] == 0
    assert audited.artifacts["session_chain"].endswith("session_chain.json")
    assert audited.artifacts["session_taint"].endswith("session_taint.json")

    loaded_chain = SessionReceiptChain.read_json(tmp_path / "session_chain.json")
    loaded_taint = HallucinationTaintTracker.read_json(tmp_path / "session_taint.json")

    assert loaded_chain.verify_integrity()["valid"] is True
    assert "imaginary_api_client" in loaded_taint.suspects


def test_ebbiforge_convenience_wrapper_is_dependency_free() -> None:
    audited = run_swarm_with_entroly(
        FakeSwarm(),
        FakeTask(),
        session_id="wrapper-session",
    )

    assert audited.session_chain.session_id == "wrapper-session"
    assert audited.turns[0].agent_name == "Researcher"
    assert audited.turns[0].verified is False
    assert audited.turns[1].verified is True


def test_ebbiforge_anomaly_summary_matches_selective_reasoning_gate() -> None:
    quiet = summarize_ebbiforge_anomalies([0.1, 0.2, 0.29, 0.1])
    noisy = summarize_ebbiforge_anomalies([0.9, 0.8, 0.7, 0.1])

    assert quiet["should_escalate"] is False
    assert quiet["reason"] == "ebbiforge:rust_only"
    assert noisy["should_escalate"] is True
    assert noisy["surprised_agents"] == 3
    assert noisy["mean_surprise"] == 0.625
