"""Shared AI Traffic Value accounting seams.

The session compatibility functions remain lazy wrappers around
:mod:`entroly.proxy_traffic_value`. Keeping those imports lazy also makes this
small module a cycle-free home for the bounded request-local value vocabulary
used by Traffic Receipts and Traffic Value.
"""

from __future__ import annotations

import contextvars
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


_SOURCE_RE = re.compile(r"[^a-z0-9_.-]+")
_MAX_SOURCE_LEN = 64
_MAX_DETAIL_ITEMS = 12
_MAX_CONTRIBUTIONS = 32
_MAX_RECEIPT_META = 1_000
ATTRIBUTION_SCHEMA = "entroly.value-attribution.v1"


class ValueTier(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    OPPORTUNITY = "opportunity"


class AccountingRole(str, Enum):
    ADDITIVE = "additive"
    ADJUSTMENT = "adjustment"
    EXPLANATORY = "explanatory"
    PROTECTED = "protected"


class EvidenceSource(str, Enum):
    LOCAL_OBSERVATION = "local_observation"
    PROVIDER_USAGE = "provider_usage"
    MEASURED_COUNTERFACTUAL = "measured_counterfactual"
    MODELLED = "modelled"


def _source_name(value: object) -> str:
    name = _SOURCE_RE.sub("_", str(value or "other").strip().lower()).strip("_.-")
    return (name or "other")[:_MAX_SOURCE_LEN]


def _bounded_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep attribution details content-blind: scalar facts only."""
    if not details:
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in list(details.items())[:_MAX_DETAIL_ITEMS]:
        if raw_value is None or isinstance(raw_value, (bool, int, float)):
            result[_source_name(raw_key)] = raw_value
    return result


@dataclass(frozen=True, slots=True)
class ValueContribution:
    """Bounded, content-blind explanation of value or observed cost."""

    source: str
    tier: ValueTier
    role: AccountingRole
    tokens: int = 0
    micro_usd: int | None = None
    evidence_source: EvidenceSource = EvidenceSource.LOCAL_OBSERVATION
    headline_included: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _source_name(self.source))
        object.__setattr__(self, "tokens", int(self.tokens or 0))
        if self.micro_usd is not None:
            object.__setattr__(self, "micro_usd", int(self.micro_usd))
        object.__setattr__(self, "details", _bounded_details(self.details))
        if self.headline_included and self.role not in {
            AccountingRole.ADDITIVE,
            AccountingRole.ADJUSTMENT,
        }:
            raise ValueError("headline rows must be additive or adjustment rows")
        if self.role is AccountingRole.ADDITIVE and self.tokens < 0:
            raise ValueError("additive tokens cannot be negative")
        if self.role is AccountingRole.ADJUSTMENT and self.tokens > 0:
            raise ValueError("adjustment tokens cannot be positive")

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["tier"] = self.tier.value
        value["role"] = self.role.value
        value["evidence_source"] = self.evidence_source.value
        value["details"] = dict(self.details)
        return value


def aggregate_contributions(
    contributions: Iterable[ValueContribution | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate without collapsing evidence tier or accounting role."""
    grouped: dict[tuple[str, str, str, str, bool], dict[str, Any]] = {}
    for raw in contributions:
        if isinstance(raw, ValueContribution):
            item = raw.payload()
        elif isinstance(raw, Mapping):
            item = dict(raw)
        else:
            continue
        source = _source_name(item.get("source"))
        tier = str(item.get("tier") or ValueTier.ESTIMATED.value)
        role = str(item.get("role") or AccountingRole.EXPLANATORY.value)
        evidence = str(
            item.get("evidence_source") or EvidenceSource.LOCAL_OBSERVATION.value
        )
        included = bool(item.get("headline_included", False))
        key = (source, tier, role, evidence, included)
        row = grouped.setdefault(
            key,
            {
                "source": source,
                "tier": tier,
                "role": role,
                "evidence_source": evidence,
                "headline_included": included,
                "events": 0,
                "tokens": 0,
                "micro_usd": 0,
                "priced_events": 0,
            },
        )
        row["events"] += 1
        try:
            row["tokens"] += int(item.get("tokens", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            pass
        if item.get("micro_usd") is not None:
            try:
                row["micro_usd"] += int(item["micro_usd"])
                row["priced_events"] += 1
            except (TypeError, ValueError, OverflowError):
                pass
    return sorted(
        grouped.values(),
        key=lambda row: (
            -abs(int(row["tokens"])),
            -abs(int(row["micro_usd"])),
            str(row["source"]),
        ),
    )


@dataclass(slots=True)
class AttributionState:
    """Exactly one bounded contribution ledger for one admitted proxy request."""

    request_id: str
    lifecycle: str = "admitted"
    contributions: list[ValueContribution] = field(default_factory=list)
    extra_provider_calls: int = 0
    extra_provider_tokens: int = 0
    extra_provider_cost_micro_usd: int = 0
    extra_provider_priced_calls: int = 0
    finalized: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    def record(
        self,
        contribution: ValueContribution,
        *,
        replace_headline: bool = False,
    ) -> bool:
        with self.lock:
            if self.finalized:
                return False
            if replace_headline:
                self.contributions[:] = [
                    item
                    for item in self.contributions
                    if not (item.source == contribution.source and item.headline_included)
                ]
            if len(self.contributions) >= _MAX_CONTRIBUTIONS:
                return False
            self.contributions.append(contribution)
            return True


CURRENT_ATTRIBUTION: contextvars.ContextVar[AttributionState | None] = (
    contextvars.ContextVar("entroly_value_attribution", default=None)
)
_STATE_LOCK = threading.RLock()
_ACTIVE_BY_REQUEST: dict[str, AttributionState] = {}
_RECEIPT_META: OrderedDict[str, dict[str, Any]] = OrderedDict()


def remember_active(state: AttributionState) -> None:
    with _STATE_LOCK:
        _ACTIVE_BY_REQUEST[state.request_id] = state


def active_state(request_id: str) -> AttributionState | None:
    with _STATE_LOCK:
        return _ACTIVE_BY_REQUEST.get(str(request_id or ""))


def forget_active(request_id: str) -> None:
    with _STATE_LOCK:
        _ACTIVE_BY_REQUEST.pop(str(request_id or ""), None)


def remember_receipt_meta(receipt_id: str, meta: Mapping[str, Any]) -> None:
    with _STATE_LOCK:
        key = str(receipt_id or "")
        if not key:
            return
        _RECEIPT_META[key] = dict(meta)
        _RECEIPT_META.move_to_end(key)
        while len(_RECEIPT_META) > _MAX_RECEIPT_META:
            _RECEIPT_META.popitem(last=False)


def receipt_meta(receipt_id: str) -> dict[str, Any]:
    with _STATE_LOCK:
        return dict(_RECEIPT_META.get(str(receipt_id or ""), {}))


def clear_attribution_state() -> None:
    with _STATE_LOCK:
        _ACTIVE_BY_REQUEST.clear()
        _RECEIPT_META.clear()


def record_internal(
    source: object,
    *,
    tier: ValueTier,
    role: AccountingRole,
    tokens: int = 0,
    micro_usd: int | None = None,
    evidence_source: EvidenceSource = EvidenceSource.LOCAL_OBSERVATION,
    headline_included: bool = False,
    details: Mapping[str, Any] | None = None,
    state: AttributionState | None = None,
    replace_headline: bool = False,
) -> bool:
    target = state or CURRENT_ATTRIBUTION.get()
    if target is None:
        return False
    try:
        item = ValueContribution(
            source=_source_name(source),
            tier=tier,
            role=role,
            tokens=tokens,
            micro_usd=micro_usd,
            evidence_source=evidence_source,
            headline_included=headline_included,
            details=details or {},
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return target.record(item, replace_headline=replace_headline)


def set_canonical_context_delta(state: AttributionState, tokens: int) -> None:
    record_internal(
        "context_optimization",
        tier=ValueTier.MEASURED,
        role=AccountingRole.ADDITIVE,
        tokens=max(0, int(tokens or 0)),
        headline_included=True,
        state=state,
        replace_headline=True,
    )


def record_value_contribution(
    source: object,
    *,
    tokens: int = 0,
    micro_usd: int | None = None,
    tier: str = "estimated",
    details: Mapping[str, Any] | None = None,
) -> bool:
    """Public observer seam; callers cannot self-promote to measured value."""
    requested = str(tier or "estimated").strip().lower()
    safe_tier = (
        ValueTier.OPPORTUNITY
        if requested == ValueTier.OPPORTUNITY.value
        else ValueTier.ESTIMATED
    )
    return record_internal(
        source,
        tier=safe_tier,
        role=AccountingRole.EXPLANATORY,
        tokens=max(0, int(tokens or 0)),
        micro_usd=micro_usd,
        evidence_source=EvidenceSource.MODELLED,
        details=details,
    )


def _value_module():
    from . import proxy_traffic_value

    return proxy_traffic_value


def _record_session_receipt(receipt: Any) -> bool:
    return _value_module()._record_session_receipt(receipt)


def _record_with_session(receipt: Any, *, tracker: Any | None = None) -> bool:
    return _value_module().record_traffic_value_receipt(receipt, tracker=tracker)


def _session_rollup(*, now: float | None = None) -> dict[str, Any]:
    return _value_module()._session_rollup(now=now)


def _snapshot_with_session(
    tracker: Any | None = None,
    *,
    today: Any | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    return _value_module().build_traffic_value_snapshot(tracker, today=today, now=now)


def _reset_session_state_for_tests(*, started_at: float | None = None) -> None:
    _value_module()._reset_session_state_for_tests(started_at=started_at)
    clear_attribution_state()


def install_session_value() -> None:
    """Compatibility no-op: session value is native in the value module."""
    return None


__all__ = [
    "ATTRIBUTION_SCHEMA",
    "AccountingRole",
    "AttributionState",
    "CURRENT_ATTRIBUTION",
    "EvidenceSource",
    "ValueContribution",
    "ValueTier",
    "active_state",
    "aggregate_contributions",
    "clear_attribution_state",
    "forget_active",
    "receipt_meta",
    "record_internal",
    "record_value_contribution",
    "remember_active",
    "remember_receipt_meta",
    "set_canonical_context_delta",
    "_record_session_receipt",
    "_record_with_session",
    "_reset_session_state_for_tests",
    "_session_rollup",
    "_snapshot_with_session",
    "install_session_value",
]
