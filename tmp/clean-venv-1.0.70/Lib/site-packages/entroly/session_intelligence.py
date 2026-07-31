"""Session intelligence primitives for realized savings and continuity.

This module separates exact savings from estimates, nets omitted-span retrievals
against gross compression savings, scores checkpoints by query relevance, extracts
compact decision state before compaction, forecasts cache-retention value without
issuing network calls, and detects repeated waste loops in a bounded window.
"""

from __future__ import annotations

import hashlib
import json
import keyword
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]{3,}")
_DECISION_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:decision|decided|choose|chosen|selected|use|ship|fix|root cause|remaining|next|todo|failure|failed|blocked)\b[:\-]?\s*(.+)$"
)
_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
_TAINT_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9_./:-])"
    r"(?:[A-Za-z_][A-Za-z0-9_]*(?:[./:-][A-Za-z0-9_.:-]+)+|"
    r"[A-Za-z_][A-Za-z0-9_]{4,}|"
    r"[A-Z][A-Za-z0-9_]{2,})"
    r"(?![A-Za-z0-9_/:-])"
)
_TAINT_STOPWORDS = frozenset(
    """
    about after again against already also although among another before being
    between both can cannot code could does doing done each either every first
    from have into just last like made make many more most much need needs only
    other over should some such than that their them then there these they this
    those turn turns very were what when where which while with would response
    context claim claims evidence unsupported unknown contradicted grounded
    use using used file path function class method variable value return
    api url uri http https json yaml true false none null class import export
    """.split()
)
_PYTHON_KEYWORDS = frozenset(word.lower() for word in keyword.kwlist)


class SavingsConfidence(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    OPPORTUNITY = "opportunity"


@dataclass(frozen=True, slots=True)
class SessionReceiptLink:
    """One receipt in an auditable multi-turn agent session chain."""

    turn_index: int
    receipt_id: str
    receipt_hash: str
    source_receipt_id: str = ""
    parent_receipt_id: str | None = None
    parent_receipt_hash: str | None = None
    query: str = ""
    token_budget: int = 0
    budget_decision: Mapping[str, Any] = field(default_factory=dict)
    risk_summary: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "receipt_id": self.receipt_id,
            "receipt_hash": self.receipt_hash,
            "source_receipt_id": self.source_receipt_id,
            "parent_receipt_id": self.parent_receipt_id,
            "parent_receipt_hash": self.parent_receipt_hash,
            "query": self.query,
            "token_budget": self.token_budget,
            "budget_decision": dict(self.budget_decision),
            "risk_summary": dict(self.risk_summary),
            "created_at": round(self.created_at, 6),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionReceiptLink":
        return cls(
            turn_index=_safe_int(data.get("turn_index"), 0),
            receipt_id=str(data.get("receipt_id") or ""),
            receipt_hash=str(data.get("receipt_hash") or ""),
            source_receipt_id=str(data.get("source_receipt_id") or ""),
            parent_receipt_id=(
                str(data["parent_receipt_id"])
                if data.get("parent_receipt_id") is not None
                else None
            ),
            parent_receipt_hash=(
                str(data["parent_receipt_hash"])
                if data.get("parent_receipt_hash") is not None
                else None
            ),
            query=str(data.get("query") or ""),
            token_budget=_safe_int(data.get("token_budget"), 0),
            budget_decision=_dict_or_empty(data.get("budget_decision")),
            risk_summary=_dict_or_empty(data.get("risk_summary")),
            created_at=float(data.get("created_at") or time.time()),
        )


class SessionReceiptChain:
    """Append-only receipt chain for reconstructing long agent runs."""

    schema_version = "entroly.session_chain.v1"

    def __init__(
        self,
        session_id: str | None = None,
        links: Iterable[SessionReceiptLink] | None = None,
        expected_chain_hash: str | None = None,
    ) -> None:
        self.session_id = session_id or f"session_{int(time.time() * 1000)}"
        self._links: list[SessionReceiptLink] = list(links or ())
        self._expected_chain_hash = expected_chain_hash

    @property
    def links(self) -> tuple[SessionReceiptLink, ...]:
        return tuple(self._links)

    @property
    def head(self) -> SessionReceiptLink | None:
        return self._links[-1] if self._links else None

    def append(
        self,
        receipt: Mapping[str, Any] | object,
        *,
        turn_index: int | None = None,
        budget_decision: "SessionBudgetDecision | Mapping[str, Any] | None" = None,
        created_at: float | None = None,
    ) -> SessionReceiptLink:
        payload = _object_payload(receipt)
        index = len(self._links) if turn_index is None else int(turn_index)
        if index < 0:
            raise ValueError("turn_index cannot be negative")
        if self._links and index <= self._links[-1].turn_index:
            raise ValueError("turn_index must increase monotonically")

        receipt_hash = _stable_payload_hash(_receipt_hash_payload(payload))
        # Chain ids are content-derived. Caller ids are kept only as metadata so
        # two equal receipts cannot carry different authoritative chain ids.
        receipt_id = f"cr_{receipt_hash[:12]}"
        source_receipt_id = str(payload.get("receipt_id") or "")
        decision_payload = _budget_decision_payload(
            budget_decision
            if budget_decision is not None
            else payload.get("budget_decision")
            or payload.get("session_budget_decision")
        )
        parent = self.head
        link = SessionReceiptLink(
            turn_index=index,
            receipt_id=receipt_id,
            receipt_hash=receipt_hash,
            source_receipt_id=source_receipt_id,
            parent_receipt_id=parent.receipt_id if parent else None,
            parent_receipt_hash=parent.receipt_hash if parent else None,
            query=str(payload.get("query") or ""),
            token_budget=_safe_int(payload.get("token_budget"), 0),
            budget_decision=decision_payload,
            risk_summary=_dict_or_empty(payload.get("risk_summary")),
            created_at=time.time() if created_at is None else float(created_at),
        )
        self._links.append(link)
        return link

    def verify_integrity(self) -> dict[str, Any]:
        issues: list[str] = []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        if (
            self._expected_chain_hash is not None
            and self._expected_chain_hash != self.chain_hash()
        ):
            issues.append("chain_hash mismatch")
        previous: SessionReceiptLink | None = None
        for link in self._links:
            expected_id = f"cr_{link.receipt_hash[:12]}"
            if link.receipt_id != expected_id:
                issues.append(f"turn {link.turn_index} receipt_id is not hash-derived")
            if link.receipt_id in seen_ids:
                issues.append(f"duplicate receipt_id: {link.receipt_id}")
            seen_ids.add(link.receipt_id)
            if link.receipt_hash in seen_hashes:
                issues.append(f"duplicate receipt_hash: {link.receipt_hash}")
            seen_hashes.add(link.receipt_hash)
            if previous is None:
                if link.parent_receipt_id or link.parent_receipt_hash:
                    issues.append("first receipt must not have a parent")
            else:
                if link.parent_receipt_id != previous.receipt_id:
                    issues.append(
                        f"turn {link.turn_index} parent_receipt_id mismatch"
                    )
                if link.parent_receipt_hash != previous.receipt_hash:
                    issues.append(
                        f"turn {link.turn_index} parent_receipt_hash mismatch"
                    )
            previous = link
        return {
            "valid": not issues,
            "issues": issues,
            "links": len(self._links),
            "head_receipt_id": self.head.receipt_id if self.head else None,
            "head_receipt_hash": self.head.receipt_hash if self.head else None,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self._chain_hash_payload()
        payload["chain_hash"] = self.chain_hash()
        return payload

    def chain_hash(self) -> str:
        return _strict_stable_payload_hash(self._chain_hash_payload())

    def _chain_hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "head_receipt_id": self.head.receipt_id if self.head else None,
            "head_receipt_hash": self.head.receipt_hash if self.head else None,
            "links": [link.as_dict() for link in self._links],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionReceiptChain":
        links = [
            SessionReceiptLink.from_dict(item)
            for item in _mapping_items(data.get("links", []))
        ]
        return cls(
            session_id=str(data.get("session_id") or ""),
            links=links,
            expected_chain_hash=(
                str(data.get("chain_hash")) if data.get("chain_hash") else None
            ),
        )

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output

    @classmethod
    def read_json(cls, path: str | Path) -> "SessionReceiptChain":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("session chain JSON must contain an object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class SessionBudgetDecision:
    policy: str
    turn_index: int
    total_budget: int
    spent_tokens: int
    allocated_budget: int
    remaining_budget: int
    reserved_closing_budget: int
    expected_remaining_turns: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "turn_index": self.turn_index,
            "total_budget": self.total_budget,
            "spent_tokens": self.spent_tokens,
            "allocated_budget": self.allocated_budget,
            "remaining_budget": self.remaining_budget,
            "reserved_closing_budget": self.reserved_closing_budget,
            "expected_remaining_turns": self.expected_remaining_turns,
            "reason": self.reason,
        }


def allocate_session_turn_budget(
    *,
    total_budget: int,
    turn_index: int,
    spent_tokens: int = 0,
    expected_total_turns: int | None = 8,
    policy: str = "decay",
    min_turn_budget: int = 256,
    max_turn_budget: int | None = None,
    closing_reserve_ratio: float = 0.15,
    closing_reserve_tokens: int | None = None,
    decay_rate: float = 0.82,
    is_closing_turn: bool = False,
) -> SessionBudgetDecision:
    """Allocate a per-turn context budget from a session-level envelope."""

    total = int(total_budget)
    turn = int(turn_index)
    spent = int(spent_tokens)
    if total < 0 or turn < 0 or spent < 0:
        raise ValueError("budgets, turn_index, and spent_tokens must be non-negative")
    if min_turn_budget < 0:
        raise ValueError("min_turn_budget cannot be negative")
    if max_turn_budget is not None and max_turn_budget < 0:
        raise ValueError("max_turn_budget cannot be negative")
    if not 0 <= closing_reserve_ratio < 1:
        raise ValueError("closing_reserve_ratio must be in [0, 1)")
    if closing_reserve_tokens is not None and closing_reserve_tokens < 0:
        raise ValueError("closing_reserve_tokens cannot be negative")
    if decay_rate <= 0 or decay_rate >= 1:
        raise ValueError("decay_rate must be in (0, 1)")

    expected_total = 8 if expected_total_turns is None else int(expected_total_turns)
    if expected_total <= 0:
        raise ValueError("expected_total_turns must be positive")
    expected_remaining = max(1, expected_total - turn)
    remaining = max(0, total - spent)

    if is_closing_turn:
        reserve = 0
    elif closing_reserve_tokens is None:
        reserve = int(round(total * closing_reserve_ratio))
    else:
        reserve = int(closing_reserve_tokens)
    reserve = min(reserve, remaining)
    allocatable = max(0, remaining - reserve)

    selected_policy = policy.lower().strip()
    if allocatable <= 0:
        raw = 0
        reason = "session_budget:exhausted_or_reserved"
    elif selected_policy == "flat":
        raw = math.ceil(allocatable / expected_remaining)
        reason = "session_budget:flat_share"
    elif selected_policy == "decay":
        weights = [decay_rate**i for i in range(expected_remaining)]
        raw = math.ceil(allocatable * weights[0] / sum(weights))
        reason = "session_budget:decay_exploration"
    elif selected_policy == "front_heavy":
        front_cutoff = max(1, math.ceil(expected_total * 0.35))
        weights = [
            1.45 if (turn + offset) < front_cutoff else 0.75
            for offset in range(expected_remaining)
        ]
        raw = math.ceil(allocatable * weights[0] / sum(weights))
        reason = "session_budget:front_heavy_exploration"
    else:
        raise ValueError("policy must be one of: flat, decay, front_heavy")

    allocation = min(allocatable, max(min_turn_budget, raw)) if allocatable else 0
    if max_turn_budget is not None:
        allocation = min(allocation, max_turn_budget)
    return SessionBudgetDecision(
        policy=selected_policy,
        turn_index=turn,
        total_budget=total,
        spent_tokens=spent,
        allocated_budget=allocation,
        remaining_budget=remaining,
        reserved_closing_budget=reserve,
        expected_remaining_turns=expected_remaining,
        reason=reason,
    )


@dataclass(slots=True)
class SuspectEntity:
    entity: str
    origin_turn: int
    origin_receipt_id: str
    origin_claim: str
    risk: float
    labels: set[str] = field(default_factory=set)
    last_flagged_turn: int = 0
    last_risk_update_turn: int = 0
    propagated_turns: tuple[int, ...] = ()

    def mark_propagated(self, turn_index: int) -> None:
        if turn_index == self.origin_turn or turn_index in self.propagated_turns:
            return
        self.propagated_turns = tuple(sorted((*self.propagated_turns, turn_index)))

    def decay_to_turn(self, turn_index: int, half_life_turns: float) -> None:
        """Decay stale suspicion until new WITNESS evidence resets the clock."""
        if half_life_turns <= 0 or turn_index <= self.last_risk_update_turn:
            return
        elapsed = max(0, int(turn_index) - self.last_risk_update_turn)
        self.risk = min(1.0, max(0.0, self.risk * (0.5 ** (elapsed / half_life_turns))))
        self.last_risk_update_turn = int(turn_index)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "origin_turn": self.origin_turn,
            "origin_receipt_id": self.origin_receipt_id,
            "origin_claim": self.origin_claim,
            "labels": sorted(self.labels),
            "risk": round(self.risk, 6),
            "last_flagged_turn": self.last_flagged_turn,
            "last_risk_update_turn": self.last_risk_update_turn,
            "propagated_turns": list(self.propagated_turns),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuspectEntity":
        labels = data.get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
        return cls(
            entity=str(data.get("entity") or ""),
            origin_turn=_safe_int(data.get("origin_turn"), 0),
            origin_receipt_id=str(data.get("origin_receipt_id") or ""),
            origin_claim=str(data.get("origin_claim") or ""),
            labels={str(label) for label in labels if str(label)},
            risk=_safe_float(data.get("risk"), 0.0),
            last_flagged_turn=_safe_int(
                data.get("last_flagged_turn"),
                _safe_int(data.get("origin_turn"), 0),
            ),
            last_risk_update_turn=_safe_int(
                data.get("last_risk_update_turn"),
                _safe_int(data.get("last_flagged_turn"), _safe_int(data.get("origin_turn"), 0)),
            ),
            propagated_turns=tuple(
                sorted(_safe_int(item, 0) for item in data.get("propagated_turns", []))
            ),
        )


@dataclass(frozen=True, slots=True)
class TaintPropagationReport:
    turn_index: int
    receipt_id: str
    propagated_entities: tuple[str, ...]
    origin_turns: Mapping[str, int]
    taint_score: float
    propagation_pressure: int
    risk_level: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "receipt_id": self.receipt_id,
            "propagated_entities": list(self.propagated_entities),
            "origin_turns": dict(self.origin_turns),
            "taint_score": round(self.taint_score, 6),
            "propagation_pressure": self.propagation_pressure,
            "risk_level": self.risk_level,
        }


class HallucinationTaintTracker:
    """Track WITNESS-flagged entities as they propagate through a loop."""

    schema_version = "entroly.hallucination_taint.v1"

    def __init__(self, *, risk_half_life_turns: float = 8.0) -> None:
        if risk_half_life_turns <= 0:
            raise ValueError("risk_half_life_turns must be positive")
        self.risk_half_life_turns = float(risk_half_life_turns)
        self._suspects: dict[str, SuspectEntity] = {}

    @property
    def suspects(self) -> Mapping[str, SuspectEntity]:
        return dict(self._suspects)

    def observe_witness(
        self,
        *,
        turn_index: int,
        receipt_id: str,
        witness_result: Any,
    ) -> tuple[SuspectEntity, ...]:
        created: list[SuspectEntity] = []
        for cert in _flagged_certificates(witness_result):
            claim = _certificate_claim(cert)
            if not claim:
                continue
            label = _certificate_label(cert)
            risk = _certificate_risk(cert)
            for entity in extract_taint_entities(claim):
                existing = self._suspects.get(entity)
                if existing is not None:
                    # Same-call duplicates update the just-created suspect here,
                    # so observe_witness returns each new entity only once.
                    existing.decay_to_turn(int(turn_index), self.risk_half_life_turns)
                    existing.risk = max(existing.risk, risk)
                    existing.last_flagged_turn = int(turn_index)
                    existing.last_risk_update_turn = int(turn_index)
                    if label:
                        existing.labels.add(label)
                    continue
                suspect = SuspectEntity(
                    entity=entity,
                    origin_turn=int(turn_index),
                    origin_receipt_id=str(receipt_id),
                    origin_claim=claim,
                    labels={label} if label else set(),
                    risk=risk,
                    last_flagged_turn=int(turn_index),
                    last_risk_update_turn=int(turn_index),
                )
                self._suspects[entity] = suspect
                created.append(suspect)
        return tuple(created)

    def observe_turn(
        self,
        *,
        turn_index: int,
        receipt_id: str,
        context: str = "",
        response: str = "",
    ) -> TaintPropagationReport:
        observed = set(extract_taint_entities(f"{context}\n{response}"))
        propagated: list[SuspectEntity] = []
        for entity in sorted(observed):
            suspect = self._suspects.get(entity)
            if suspect is None or int(turn_index) <= suspect.origin_turn:
                continue
            suspect.decay_to_turn(int(turn_index), self.risk_half_life_turns)
            suspect.mark_propagated(int(turn_index))
            propagated.append(suspect)

        taint_score = _noisy_or([item.risk for item in propagated])
        propagation_pressure = sum(len(item.propagated_turns) for item in propagated)
        return TaintPropagationReport(
            turn_index=int(turn_index),
            receipt_id=str(receipt_id),
            propagated_entities=tuple(item.entity for item in propagated),
            origin_turns={item.entity: item.origin_turn for item in propagated},
            taint_score=round(taint_score, 6),
            propagation_pressure=propagation_pressure,
            risk_level=_taint_risk_level(taint_score),
        )

    def observe(
        self,
        *,
        turn_index: int,
        receipt_id: str,
        context: str = "",
        response: str = "",
        witness_result: Any | None = None,
    ) -> TaintPropagationReport:
        if witness_result is not None:
            self.observe_witness(
                turn_index=turn_index,
                receipt_id=receipt_id,
                witness_result=witness_result,
            )
        return self.observe_turn(
            turn_index=turn_index,
            receipt_id=receipt_id,
            context=context,
            response=response,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "risk_half_life_turns": self.risk_half_life_turns,
            "suspects": {
                entity: suspect.as_dict()
                for entity, suspect in sorted(self._suspects.items())
            }
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HallucinationTaintTracker":
        tracker = cls(
            risk_half_life_turns=_safe_float(
                data.get("risk_half_life_turns"),
                8.0,
            )
        )
        suspects = data.get("suspects", {})
        if isinstance(suspects, Mapping):
            for entity, payload in suspects.items():
                if isinstance(payload, Mapping):
                    suspect = SuspectEntity.from_dict(payload)
                    tracker._suspects[str(entity)] = suspect
        return tracker

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output

    @classmethod
    def read_json(cls, path: str | Path) -> "HallucinationTaintTracker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("taint tracker JSON must contain an object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class RealizedSavingsRecord:
    receipt_id: str
    original_tokens: int
    compressed_tokens: int
    retrieved_tokens: int = 0
    repeated_expansion_tokens: int = 0
    confidence: SavingsConfidence = SavingsConfidence.MEASURED

    @property
    def gross_saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def net_realized_saved_tokens(self) -> int:
        return max(
            0,
            self.gross_saved_tokens
            - max(0, self.retrieved_tokens)
            - max(0, self.repeated_expansion_tokens),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "gross_saved_tokens": self.gross_saved_tokens,
            "retrieved_tokens": self.retrieved_tokens,
            "repeated_expansion_tokens": self.repeated_expansion_tokens,
            "net_realized_saved_tokens": self.net_realized_saved_tokens,
            "confidence": self.confidence.value,
        }


@dataclass(slots=True)
class SavingsTierLedger:
    _records: list[RealizedSavingsRecord] = field(default_factory=list)

    def add(self, record: RealizedSavingsRecord) -> None:
        self._records.append(record)

    def summary(self) -> dict[str, Any]:
        by_tier = {
            tier.value: {
                "records": 0,
                "gross_saved_tokens": 0,
                "retrieved_tokens": 0,
                "repeated_expansion_tokens": 0,
                "net_realized_saved_tokens": 0,
            }
            for tier in SavingsConfidence
        }
        for record in self._records:
            bucket = by_tier[record.confidence.value]
            bucket["records"] += 1
            bucket["gross_saved_tokens"] += record.gross_saved_tokens
            bucket["retrieved_tokens"] += max(0, record.retrieved_tokens)
            bucket["repeated_expansion_tokens"] += max(0, record.repeated_expansion_tokens)
            bucket["net_realized_saved_tokens"] += record.net_realized_saved_tokens
        return {"records": len(self._records), "by_confidence": by_tier}


@dataclass(frozen=True, slots=True)
class DecisionDigest:
    decisions: tuple[str, ...] = ()
    modified_paths: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    remaining_tasks: tuple[str, ...] = ()

    def as_metadata(self) -> dict[str, list[str]]:
        return {
            "decisions": list(self.decisions),
            "modified_paths": list(self.modified_paths),
            "failures": list(self.failures),
            "remaining_tasks": list(self.remaining_tasks),
        }


def extract_decision_digest(text: str, *, max_items: int = 12) -> DecisionDigest:
    decisions: list[str] = []
    failures: list[str] = []
    remaining: list[str] = []
    for match in _DECISION_RE.finditer(text or ""):
        raw = _clean_line(match.group(0))
        lower = raw.lower()
        if any(word in lower for word in ("failure", "failed", "blocked", "root cause")):
            failures.append(raw)
        elif any(word in lower for word in ("remaining", "next", "todo")):
            remaining.append(raw)
        else:
            decisions.append(raw)
    paths = sorted(set(_PATH_RE.findall(text or "")))
    return DecisionDigest(
        decisions=tuple(_dedupe(decisions)[:max_items]),
        modified_paths=tuple(paths[:max_items]),
        failures=tuple(_dedupe(failures)[:max_items]),
        remaining_tasks=tuple(_dedupe(remaining)[:max_items]),
    )


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    checkpoint_id: str
    score: float
    matched_terms: tuple[str, ...]
    age_seconds: float
    decision_hits: int
    path_hits: int
    is_trusted: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "score": round(self.score, 6),
            "matched_terms": list(self.matched_terms),
            "age_seconds": round(self.age_seconds, 3),
            "decision_hits": self.decision_hits,
            "path_hits": self.path_hits,
            "is_trusted": self.is_trusted,
        }


class CheckpointRelevanceScorer:
    def __init__(self, *, half_life_seconds: float = 86400.0) -> None:
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        self.half_life_seconds = half_life_seconds

    def score(self, checkpoint: Any, query: str, *, now: float | None = None) -> CheckpointScore:
        timestamp = time.time() if now is None else float(now)
        created = float(getattr(checkpoint, "timestamp", timestamp) or timestamp)
        age = max(0.0, timestamp - created)
        terms = _terms(query)
        metadata = getattr(checkpoint, "metadata", {}) or {}
        stats = getattr(checkpoint, "stats", {}) or {}
        fragments = getattr(checkpoint, "fragments", []) or []
        text_parts: list[str] = []
        for key in ("task", "query", "summary", "current_step"):
            value = metadata.get(key)
            if value:
                text_parts.append(str(value))
        continuity = metadata.get("continuity")
        if isinstance(continuity, Mapping):
            for value in continuity.values():
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    text_parts.extend(str(item) for item in value)
                else:
                    text_parts.append(str(value))
        for fragment in fragments[:50]:
            if isinstance(fragment, Mapping):
                text_parts.append(str(fragment.get("source", "")))
                text_parts.append(str(fragment.get("content", ""))[:500])
        haystack = "\n".join(text_parts).lower()
        matched = tuple(sorted(term for term in terms if term in haystack))
        decision_hits = _count_hits(continuity, "decisions", terms)
        path_hits = _count_hits(continuity, "modified_paths", terms)
        freshness = 0.5 ** (age / self.half_life_seconds)
        fragment_strength = min(1.0, math.log1p(len(fragments)) / math.log(101))
        try:
            stats_strength = min(1.0, float(stats.get("coverage_pct", 0.0)) / 100.0) if isinstance(stats, Mapping) else 0.0
        except (TypeError, ValueError):
            stats_strength = 0.0
        is_trusted = not bool(metadata.get("untrusted", False))
        trust = 1.0 if is_trusted else 0.35
        term_score = len(matched) / max(1, len(terms))
        continuity_score = min(1.0, 0.2 * decision_hits + 0.25 * path_hits)
        score = trust * (0.55 * term_score + 0.20 * continuity_score + 0.15 * freshness + 0.05 * fragment_strength + 0.05 * stats_strength)
        return CheckpointScore(str(getattr(checkpoint, "checkpoint_id", "")), score, matched, age, decision_hits, path_hits, is_trusted)

    def rank(self, checkpoints: Iterable[Any], query: str, *, now: float | None = None, limit: int = 5) -> list[CheckpointScore]:
        scored = [self.score(checkpoint, query, now=now) for checkpoint in checkpoints]
        scored.sort(key=lambda item: (item.score, -item.age_seconds, item.checkpoint_id), reverse=True)
        return scored[: max(0, limit)]


@dataclass(frozen=True, slots=True)
class CacheRetentionOption:
    name: str
    ttl_seconds: float
    write_multiplier: float
    read_multiplier: float


@dataclass(frozen=True, slots=True)
class CacheRetentionDecision:
    selected: str
    expected_cost_usd: Mapping[str, float]
    expected_savings_usd: Mapping[str, float]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "expected_cost_usd": {k: round(v, 9) for k, v in self.expected_cost_usd.items()},
            "expected_savings_usd": {k: round(v, 9) for k, v in self.expected_savings_usd.items()},
            "reason": self.reason,
        }


class CacheRetentionForecaster:
    def __init__(self, options: Iterable[CacheRetentionOption] | None = None) -> None:
        self.options = tuple(options or (
            CacheRetentionOption("none", 0.0, 1.0, 1.0),
            CacheRetentionOption("short", 300.0, 1.0, 0.1),
            CacheRetentionOption("long", 3600.0, 1.25, 0.1),
        ))
        if not self.options:
            raise ValueError("at least one cache option is required")

    def decide(self, *, prefix_tokens: int, input_price_per_million: float, pause_samples_seconds: Sequence[float], expected_future_turns: int = 1, min_savings_usd: float = 0.0) -> CacheRetentionDecision:
        if prefix_tokens < 0 or input_price_per_million < 0:
            raise ValueError("prefix tokens and price must be non-negative")
        turns = max(1, int(expected_future_turns))
        pauses = [max(0.0, float(pause)) for pause in pause_samples_seconds] or [float("inf")]
        base_write = prefix_tokens * input_price_per_million / 1_000_000.0
        expected_cost: dict[str, float] = {}
        expected_savings: dict[str, float] = {}
        for option in self.options:
            hit_probability = sum(1 for pause in pauses if pause <= option.ttl_seconds) / len(pauses)
            write_cost = base_write * option.write_multiplier
            read_cost = base_write * option.read_multiplier * hit_probability * turns
            miss_cost = base_write * (1.0 - hit_probability) * turns
            total = write_cost + read_cost + miss_cost
            no_cache_total = base_write * (1 + turns)
            expected_cost[option.name] = total
            expected_savings[option.name] = max(0.0, no_cache_total - total)
        selected = min(expected_cost, key=lambda name: (expected_cost[name], name))
        if expected_savings.get(selected, 0.0) < min_savings_usd:
            selected = "none" if "none" in expected_cost else selected
            reason = "forecast:no_material_savings"
        else:
            reason = "forecast:lowest_expected_cost"
        return CacheRetentionDecision(selected, expected_cost, expected_savings, reason)


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    kind: str
    key: str
    timestamp: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BehaviorWasteReport:
    repeated_errors: int
    repeated_tool_calls: int
    retry_loops: int
    model_switch_churn: int
    total_events: int

    @property
    def waste_score(self) -> float:
        weighted = 1.5 * self.repeated_errors + self.repeated_tool_calls + 1.25 * self.retry_loops + 1.25 * self.model_switch_churn
        return round(min(1.0, weighted / max(1, self.total_events)), 6)

    def as_dict(self) -> dict[str, Any]:
        return {
            "repeated_errors": self.repeated_errors,
            "repeated_tool_calls": self.repeated_tool_calls,
            "retry_loops": self.retry_loops,
            "model_switch_churn": self.model_switch_churn,
            "total_events": self.total_events,
            "waste_score": self.waste_score,
        }


class BehavioralWasteDetector:
    def __init__(self, *, window_seconds: float = 300.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._events: list[BehaviorEvent] = []

    def record(self, kind: str, key: str, *, timestamp: float | None = None, metadata: Mapping[str, Any] | None = None) -> BehaviorWasteReport:
        ts = time.time() if timestamp is None else float(timestamp)
        self._events.append(BehaviorEvent(kind, key, ts, dict(metadata or {})))
        self._evict(ts)
        return self.report(now=ts)

    def report(self, *, now: float | None = None) -> BehaviorWasteReport:
        ts = time.time() if now is None else float(now)
        self._evict(ts)
        repeated_errors = _repeat_count(self._events, "error")
        repeated_tool_calls = _repeat_count(self._events, "tool_call")
        retry_loops = _repeat_count(self._events, "retry")
        model_switch_churn = _churn_count([event for event in self._events if event.kind == "model"])
        return BehaviorWasteReport(repeated_errors, repeated_tool_calls, retry_loops, model_switch_churn, len(self._events))

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._events = [event for event in self._events if event.timestamp >= cutoff]


def _terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text or "")}


def extract_taint_entities(text: str, *, max_entities: int = 32) -> tuple[str, ...]:
    entities: list[str] = []
    for match in _TAINT_ENTITY_RE.finditer(text or ""):
        raw = match.group(0).strip(".,;:()[]{}<>`'\"")
        if not raw:
            continue
        key = raw.lower()
        if key in _TAINT_STOPWORDS or key in _PYTHON_KEYWORDS or key.isdigit():
            continue
        has_identifier_signal = (
            any(ch in raw for ch in "_./:-")
            or any(ch.isdigit() for ch in raw)
            or raw[:1].isupper()
            or any(ch.isupper() for ch in raw[1:])
        )
        if not has_identifier_signal:
            continue
        if len(key) < 5 and not any(ch in raw for ch in "_./:-"):
            continue
        entities.append(key)
    # _dedupe is first-occurrence preserving, which keeps audit output stable
    # while retaining the earliest textual position of each suspect entity.
    return tuple(_dedupe(entities)[:max(0, max_entities)])


def _stable_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_stable_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receipt_hash_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if key
        not in {
            "receipt_id",
            "receipt_hash",
            "reproducibility_hash",
            "parent_receipt_id",
            "parent_receipt_hash",
        }
    }


def _budget_decision_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, SessionBudgetDecision):
        return value.as_dict()
    if isinstance(value, Mapping):
        return dict(value)
    method = getattr(value, "as_dict", None)
    if callable(method):
        payload = method()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _object_payload(value: Mapping[str, Any] | object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            maybe_payload = method()
            if isinstance(maybe_payload, Mapping):
                return dict(maybe_payload)
    return {}


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _noisy_or(risks: Sequence[float]) -> float:
    """Aggregate independent suspect probabilities from a concrete sequence."""
    complement = 1.0
    for risk in risks:
        bounded = min(1.0, max(0.0, float(risk)))
        complement *= 1.0 - bounded
    return round(1.0 - complement, 6) if risks else 0.0


def _flagged_certificates(witness_result: Any) -> list[Any]:
    if witness_result is None:
        return []
    flagged = getattr(witness_result, "flagged", None)
    if callable(flagged):
        return list(flagged() or [])
    if isinstance(witness_result, Mapping):
        explicit = witness_result.get("flagged")
        if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
            return list(explicit)
        certificates = witness_result.get("certificates", [])
        if isinstance(certificates, Sequence) and not isinstance(
            certificates, (str, bytes)
        ):
            return [
                cert
                for cert in certificates
                if _certificate_label(cert) != "grounded"
            ]
    if isinstance(witness_result, Sequence) and not isinstance(
        witness_result, (str, bytes)
    ):
        return [
            cert
            for cert in witness_result
            if _certificate_label(cert) != "grounded"
        ]
    return []


def _certificate_claim(cert: Any) -> str:
    if isinstance(cert, Mapping):
        return _clean_line(str(cert.get("claim_text") or cert.get("text") or ""))
    return _clean_line(str(getattr(cert, "claim_text", getattr(cert, "text", ""))))


def _certificate_label(cert: Any) -> str:
    if isinstance(cert, Mapping):
        return str(cert.get("label") or "").lower()
    return str(getattr(cert, "label", "") or "").lower()


def _certificate_risk(cert: Any) -> float:
    raw = cert.get("risk") if isinstance(cert, Mapping) else getattr(cert, "risk", 0.5)
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return 0.5


def _taint_risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _clean_line(text: str) -> str:
    return " ".join((text or "").strip().split())[:300]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _count_hits(continuity: Any, key: str, terms: set[str]) -> int:
    if not isinstance(continuity, Mapping):
        return 0
    values = continuity.get(key, ())
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence):
        return 0
    count = 0
    for value in values:
        lowered = str(value).lower()
        if any(term in lowered for term in terms):
            count += 1
    return count


def _repeat_count(events: Sequence[BehaviorEvent], kind: str) -> int:
    counts: dict[str, int] = {}
    for event in events:
        if event.kind == kind:
            counts[event.key] = counts.get(event.key, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def _churn_count(events: Sequence[BehaviorEvent]) -> int:
    if len(events) < 2:
        return 0
    switches = 0
    previous = events[0].key
    for event in events[1:]:
        if event.key != previous:
            switches += 1
            previous = event.key
    return switches


__all__ = [
    "BehaviorEvent",
    "BehaviorWasteReport",
    "BehavioralWasteDetector",
    "CacheRetentionDecision",
    "CacheRetentionForecaster",
    "CacheRetentionOption",
    "CheckpointRelevanceScorer",
    "CheckpointScore",
    "DecisionDigest",
    "HallucinationTaintTracker",
    "RealizedSavingsRecord",
    "SavingsConfidence",
    "SavingsTierLedger",
    "SessionBudgetDecision",
    "SessionReceiptChain",
    "SessionReceiptLink",
    "SuspectEntity",
    "TaintPropagationReport",
    "allocate_session_turn_budget",
    "extract_decision_digest",
    "extract_taint_entities",
]
