"""Entroly bridge for Ebbiforge swarms.

The integration is intentionally duck-typed: it works with Ebbiforge's public
Python surface without making ``ebbiforge`` a required Entroly dependency.

What it does:
  - runs an Ebbiforge-style ``swarm.run(task)``;
  - reads ``swarm.get_belief_provenance(result)``;
  - turns provenance records into a tamper-evident Entroly session chain;
  - tracks Ebbiforge-reported unverified claims as taint when they propagate;
  - optionally writes ``session_chain.json`` and ``session_taint.json``.

This bridge records Ebbiforge's own provenance/confidence reports. It does not
independently run Entroly WITNESS over the swarm output.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from entroly.session_intelligence import (
    HallucinationTaintTracker,
    SessionReceiptChain,
    SessionBudgetDecision,
    TaintPropagationReport,
    allocate_session_turn_budget,
)


_UNVERIFIED_SOURCES = {"", "unknown", "internal_knowledge"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class EbbiforgeProvenanceTurn:
    """Normalized view of one Ebbiforge provenance record."""

    turn_index: int
    agent_name: str
    claim_text: str
    source: str
    confidence: float
    verified: bool

    def as_receipt_payload(self, task_prompt: str) -> dict[str, Any]:
        return {
            "receipt_id": f"ebbiforge_{self.turn_index}_{_safe_id(self.agent_name)}",
            "turn_index": self.turn_index,
            "agent_name": self.agent_name,
            "query": task_prompt,
            "claim_text": self.claim_text,
            "source": self.source,
            "confidence": round(self.confidence, 6),
            "verified": self.verified,
            "risk_summary": {
                "framework": "ebbiforge",
                "verification": "verified" if self.verified else "unverified",
                "source": self.source,
                "risk": round(max(0.0, 1.0 - self.confidence), 6),
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "agent_name": self.agent_name,
            "claim_text": self.claim_text,
            "source": self.source,
            "confidence": round(self.confidence, 6),
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class EbbiforgeAuditResult:
    """Entroly audit artifact for one Ebbiforge swarm run."""

    result: Any
    session_chain: SessionReceiptChain
    taint_tracker: HallucinationTaintTracker
    turns: tuple[EbbiforgeProvenanceTurn, ...]
    taint_reports: tuple[TaintPropagationReport, ...]
    verification_source: str = "ebbiforge_provenance"
    witness_mode: str = "not_run"
    propagation_context_window: int = 8
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "result": str(self.result),
            "turns": [turn.as_dict() for turn in self.turns],
            "session_chain": self.session_chain.as_dict(),
            "taint_reports": [report.as_dict() for report in self.taint_reports],
            "verification_source": self.verification_source,
            "witness_mode": self.witness_mode,
            "propagation_context_window": self.propagation_context_window,
            "artifacts": dict(self.artifacts),
        }


class EbbiforgeEntrolyBridge:
    """Audit Ebbiforge pipelines with Entroly receipt and taint primitives."""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        total_budget: int | None = None,
        budget_policy: str = "decay",
        risk_half_life_turns: float = 8.0,
        propagation_context_window: int = 8,
    ) -> None:
        if propagation_context_window < 0:
            raise ValueError("propagation_context_window cannot be negative")
        self.session_chain = SessionReceiptChain(session_id=session_id)
        self.taint_tracker = HallucinationTaintTracker(
            risk_half_life_turns=risk_half_life_turns
        )
        self.total_budget = total_budget
        self.budget_policy = budget_policy
        self.propagation_context_window = int(propagation_context_window)

    def run_swarm(
        self,
        swarm: Any,
        task: Any,
        *,
        timeout: float | None = None,
        output_dir: str | Path | None = None,
    ) -> EbbiforgeAuditResult:
        """Run ``swarm`` and emit Entroly audit artifacts for its provenance."""

        result = _run_swarm(swarm, task, timeout=timeout)
        provenance = _get_provenance(swarm, result)
        turns = tuple(
            _renumber_turns(
                _normalize_provenance_records(provenance),
                start=len(self.session_chain.links),
            )
        )
        task_prompt = str(getattr(task, "prompt", "") or "")

        taint_reports: list[TaintPropagationReport] = []
        # The tracker carries cross-run suspect state. This bounded per-run
        # window is only immediate textual propagation evidence, avoiding O(T^2)
        # string reconstruction on long Ebbiforge tick/session histories.
        prior_claims: deque[str] = deque(maxlen=self.propagation_context_window)
        spent_tokens = 0
        for turn in turns:
            decision = self._budget_decision(turn.turn_index, spent_tokens)
            if decision is not None:
                spent_tokens += max(0, decision.allocated_budget)
            link = self.session_chain.append(
                turn.as_receipt_payload(task_prompt),
                turn_index=turn.turn_index,
                budget_decision=decision,
            )
            report = self.taint_tracker.observe(
                turn_index=turn.turn_index,
                receipt_id=link.receipt_id,
                context="\n".join(prior_claims),
                response=turn.claim_text,
                witness_result=_witness_like_result(turn),
            )
            taint_reports.append(report)
            prior_claims.append(turn.claim_text)

        artifacts: dict[str, str] = {}
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            artifacts["session_chain"] = str(
                self.session_chain.write_json(out / "session_chain.json")
            )
            artifacts["session_taint"] = str(
                self.taint_tracker.write_json(out / "session_taint.json")
            )

        return EbbiforgeAuditResult(
            result=result,
            session_chain=self.session_chain,
            taint_tracker=self.taint_tracker,
            turns=turns,
            taint_reports=tuple(taint_reports),
            propagation_context_window=self.propagation_context_window,
            artifacts=artifacts,
        )

    def _budget_decision(
        self,
        turn_index: int,
        spent_tokens: int,
    ) -> SessionBudgetDecision | None:
        if self.total_budget is None:
            return None
        return allocate_session_turn_budget(
            total_budget=self.total_budget,
            turn_index=turn_index,
            spent_tokens=spent_tokens,
            policy=self.budget_policy,
        )


def run_swarm_with_entroly(
    swarm: Any,
    task: Any,
    *,
    session_id: str | None = None,
    total_budget: int | None = None,
    budget_policy: str = "decay",
    timeout: float | None = None,
    propagation_context_window: int = 8,
    output_dir: str | Path | None = None,
) -> EbbiforgeAuditResult:
    """Convenience wrapper for Ebbiforge users.

    Example:
        ``run_swarm_with_entroly(swarm, Task("Investigate", pipeline=True))``
    """

    bridge = EbbiforgeEntrolyBridge(
        session_id=session_id,
        total_budget=total_budget,
        budget_policy=budget_policy,
        propagation_context_window=propagation_context_window,
    )
    return bridge.run_swarm(swarm, task, timeout=timeout, output_dir=output_dir)


def summarize_ebbiforge_anomalies(
    surprise_scores: Iterable[float],
    *,
    agent_threshold: float = 0.3,
    escalation_ratio: float = 0.05,
) -> dict[str, Any]:
    """Summarize Ebbiforge Rust-core surprise scores for selective reasoning.

    The function mirrors Ebbiforge's README pattern: most ticks stay Rust-only;
    a tick is escalated only when enough agents exceed a surprise threshold.
    """

    scores = [max(0.0, min(1.0, float(score))) for score in surprise_scores]
    total = len(scores)
    surprised = sum(1 for score in scores if score > agent_threshold)
    ratio = surprised / total if total else 0.0
    mean_surprise = sum(scores) / total if total else 0.0
    return {
        "agents": total,
        "surprised_agents": surprised,
        "surprise_ratio": round(ratio, 6),
        "mean_surprise": round(mean_surprise, 6),
        "agent_threshold": agent_threshold,
        "escalation_ratio": escalation_ratio,
        "should_escalate": ratio > escalation_ratio,
        "reason": (
            "ebbiforge:anomaly_escalation"
            if ratio > escalation_ratio
            else "ebbiforge:rust_only"
        ),
    }


def _run_swarm(swarm: Any, task: Any, *, timeout: float | None) -> Any:
    if timeout is None:
        return swarm.run(task)
    return swarm.run(task, timeout=timeout)


def _get_provenance(swarm: Any, result: Any) -> Any:
    getter = getattr(swarm, "get_belief_provenance", None)
    if not callable(getter):
        raise TypeError("swarm must expose get_belief_provenance(result)")
    provenance = getter(result)
    if provenance is None:
        raise ValueError("Ebbiforge run did not expose belief provenance")
    return provenance


def _normalize_provenance_records(provenance: Any) -> Iterable[EbbiforgeProvenanceTurn]:
    records = getattr(provenance, "records", None)
    if records is None and isinstance(provenance, Mapping):
        records = provenance.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Ebbiforge provenance must expose a records sequence")

    for index, record in enumerate(records):
        source = _record_value(record, "source", "")
        confidence = _safe_float(_record_value(record, "confidence", 0.0), 0.0)
        verified = _record_verified(record, source)
        yield EbbiforgeProvenanceTurn(
            turn_index=index,
            agent_name=str(_record_value(record, "agent_name", f"agent_{index}") or ""),
            claim_text=_claim_text(_record_value(record, "claim", "")),
            source=str(source or ""),
            confidence=max(0.0, min(1.0, confidence)),
            verified=verified,
        )


def _renumber_turns(
    turns: Iterable[EbbiforgeProvenanceTurn],
    *,
    start: int,
) -> Iterable[EbbiforgeProvenanceTurn]:
    for offset, turn in enumerate(turns):
        yield EbbiforgeProvenanceTurn(
            turn_index=start + offset,
            agent_name=turn.agent_name,
            claim_text=turn.claim_text,
            source=turn.source,
            confidence=turn.confidence,
            verified=turn.verified,
        )


def _record_verified(record: Any, source: str) -> bool:
    explicit = _record_value(record, "verified", None)
    if explicit is not None:
        return bool(explicit)
    return str(source or "").strip().lower() not in _UNVERIFIED_SOURCES


def _record_value(record: Any, key: str, default: Any) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _claim_text(claim: Any) -> str:
    if isinstance(claim, Mapping):
        for key in ("claim", "claim_text", "text", "value"):
            if claim.get(key):
                return str(claim[key])
        return json.dumps(claim, sort_keys=True, default=str)
    return str(claim or "")


def _witness_like_result(turn: EbbiforgeProvenanceTurn) -> dict[str, Any] | None:
    if turn.verified:
        return None
    return {
        "certificates": [
            {
                "claim_text": turn.claim_text,
                "label": "unsupported",
                "risk": max(0.05, min(1.0, 1.0 - turn.confidence)),
            }
        ]
    }


def _safe_id(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip()).strip("_").lower()
    return cleaned or "agent"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "EbbiforgeAuditResult",
    "EbbiforgeEntrolyBridge",
    "EbbiforgeProvenanceTurn",
    "run_swarm_with_entroly",
    "summarize_ebbiforge_anomalies",
]
