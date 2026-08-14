"""Request-local AI Traffic Value attribution and session compatibility."""

from __future__ import annotations

import contextvars
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

_SOURCE_RE = re.compile(r"[^a-z0-9_.-]+")
_MAX_ITEMS = 32
_MAX_META = 1000
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


def _name(value: object) -> str:
    text = _SOURCE_RE.sub("_", str(value or "other").strip().lower()).strip("_.-")
    return (text or "other")[:64]


def _details(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return {
        _name(key): item
        for key, item in list(value.items())[:12]
        if item is None or isinstance(item, (bool, int, float))
    }


@dataclass(frozen=True, slots=True)
class ValueContribution:
    source: str
    tier: ValueTier
    role: AccountingRole
    tokens: int = 0
    micro_usd: int | None = None
    evidence_source: EvidenceSource = EvidenceSource.LOCAL_OBSERVATION
    headline_included: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _name(self.source))
        object.__setattr__(self, "tokens", int(self.tokens or 0))
        if self.micro_usd is not None:
            object.__setattr__(self, "micro_usd", int(self.micro_usd))
        object.__setattr__(self, "details", _details(self.details))
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
        out = asdict(self)
        out["tier"] = self.tier.value
        out["role"] = self.role.value
        out["evidence_source"] = self.evidence_source.value
        out["details"] = dict(self.details)
        return out


def aggregate_contributions(values: Iterable[ValueContribution | Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, bool], dict[str, Any]] = {}
    for raw in values:
        item = raw.payload() if isinstance(raw, ValueContribution) else dict(raw)
        key = (
            _name(item.get("source")),
            str(item.get("tier") or "estimated"),
            str(item.get("role") or "explanatory"),
            str(item.get("evidence_source") or "local_observation"),
            bool(item.get("headline_included", False)),
        )
        row = grouped.setdefault(
            key,
            {"source": key[0], "tier": key[1], "role": key[2],
             "evidence_source": key[3], "headline_included": key[4],
             "events": 0, "tokens": 0, "micro_usd": 0, "priced_events": 0},
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
    return sorted(grouped.values(), key=lambda row: (-abs(int(row["tokens"])), str(row["source"])))


@dataclass(slots=True)
class AttributionState:
    request_id: str
    lifecycle: str = "admitted"
    contributions: list[ValueContribution] = field(default_factory=list)
    extra_provider_calls: int = 0
    extra_provider_tokens: int = 0
    extra_provider_cost_micro_usd: int = 0
    extra_provider_priced_calls: int = 0
    finalized: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    def add(self, item: ValueContribution, *, replace_headline: bool = False) -> bool:
        with self.lock:
            if self.finalized:
                return False
            if replace_headline:
                self.contributions[:] = [
                    old for old in self.contributions
                    if not (old.source == item.source and old.headline_included)
                ]
            if len(self.contributions) >= _MAX_ITEMS:
                return False
            self.contributions.append(item)
            return True


CURRENT_ATTRIBUTION: contextvars.ContextVar[AttributionState | None] = contextvars.ContextVar(
    "entroly_value_attribution", default=None
)
_LOCK = threading.RLock()
_ACTIVE: dict[str, AttributionState] = {}
_META: OrderedDict[str, dict[str, Any]] = OrderedDict()


def remember_active(state: AttributionState) -> None:
    with _LOCK:
        _ACTIVE[state.request_id] = state


def active_state(request_id: str) -> AttributionState | None:
    with _LOCK:
        return _ACTIVE.get(str(request_id or ""))


def forget_active(request_id: str) -> None:
    with _LOCK:
        _ACTIVE.pop(str(request_id or ""), None)


def remember_receipt_meta(receipt_id: str, meta: Mapping[str, Any]) -> None:
    key = str(receipt_id or "")
    if not key:
        return
    with _LOCK:
        _META[key] = dict(meta)
        _META.move_to_end(key)
        while len(_META) > _MAX_META:
            _META.popitem(last=False)


def receipt_meta(receipt_id: str) -> dict[str, Any]:
    with _LOCK:
        return dict(_META.get(str(receipt_id or ""), {}))


def clear_attribution_state() -> None:
    with _LOCK:
        _ACTIVE.clear()
        _META.clear()


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
            _name(source), tier, role, tokens, micro_usd,
            evidence_source, headline_included, details or {}
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return target.add(item, replace_headline=replace_headline)


def set_canonical_context_delta(state: AttributionState, tokens: int) -> None:
    record_internal(
        "context_optimization", tier=ValueTier.MEASURED,
        role=AccountingRole.ADDITIVE, tokens=max(0, int(tokens or 0)),
        headline_included=True, state=state, replace_headline=True,
    )


def record_value_contribution(
    source: object, *, tokens: int = 0, micro_usd: int | None = None,
    tier: str = "estimated", details: Mapping[str, Any] | None = None,
) -> bool:
    """Public seam: callers cannot self-promote to measured/headline value."""
    safe = ValueTier.OPPORTUNITY if str(tier).lower() == "opportunity" else ValueTier.ESTIMATED
    return record_internal(
        source, tier=safe, role=AccountingRole.EXPLANATORY,
        tokens=max(0, int(tokens or 0)), micro_usd=micro_usd,
        evidence_source=EvidenceSource.MODELLED, details=details,
    )


def _measured(source: str, tokens: int, **details: Any) -> None:
    if tokens > 0:
        record_internal(
            source, tier=ValueTier.MEASURED, role=AccountingRole.EXPLANATORY,
            tokens=tokens, details=details,
        )


def _install_observers() -> None:
    """Observe existing optimizer/provider accounting without changing policy."""
    try:
        from . import proxy as proxy_module, proxy_transform
        from .optimization_ledger import OptimizationLedger
        from .session_rescue import SessionRescueController
        from .usage_ledger import UsageLedger
    except Exception:
        return

    current = proxy_transform.compress_tool_messages
    if not hasattr(current, "__entroly_value_observer_original__"):
        original = current
        def tool(*args: Any, **kwargs: Any):
            messages, saved = original(*args, **kwargs)
            _measured("tool_output_compression", max(0, int(saved or 0)))
            return messages, saved
        tool.__entroly_value_observer_original__ = original
        proxy_transform.compress_tool_messages = tool

    current_conv = proxy_module.compress_conversation_messages
    if not hasattr(current_conv, "__entroly_value_observer_original__"):
        original_conv = current_conv
        def conversation(messages: list[dict], *args: Any, **kwargs: Any):
            before = proxy_module._estimate_message_tokens(messages)
            result = original_conv(messages, *args, **kwargs)
            _measured("conversation_compression", max(0, before - proxy_module._estimate_message_tokens(result)))
            return result
        conversation.__entroly_value_observer_original__ = original_conv
        proxy_module.compress_conversation_messages = conversation

    current_rescue = SessionRescueController.rescue
    if not hasattr(current_rescue, "__entroly_value_observer_original__"):
        original_rescue = current_rescue
        def rescue(self: Any, *args: Any, **kwargs: Any):
            result = original_rescue(self, *args, **kwargs)
            _measured(
                "session_rescue", max(0, int(getattr(result, "tokens_saved", 0) or 0)),
                recovery_receipts=len(getattr(result, "recovery_receipts", ()) or ()),
                blocked=bool(getattr(result, "blocked", False)),
            )
            return result
        rescue.__entroly_value_observer_original__ = original_rescue
        SessionRescueController.rescue = rescue

    current_opt = OptimizationLedger.record
    if not hasattr(current_opt, "__entroly_value_observer_original__"):
        original_opt = current_opt
        def opt(self: Any, event: Any) -> bool:
            inserted = original_opt(self, event)
            state = CURRENT_ATTRIBUTION.get()
            if inserted and state is not None:
                tiers = {item.value: item for item in ValueTier}
                raw = str(getattr(getattr(event, "tier", None), "value", "estimated"))
                gross = int(getattr(event, "gross_micro_usd", 0) or 0)
                cost = int(getattr(event, "cost_micro_usd", 0) or 0)
                record_internal(
                    getattr(event, "feature", "optimizer"),
                    tier=tiers.get(raw, ValueTier.ESTIMATED),
                    role=AccountingRole.EXPLANATORY,
                    tokens=max(0, int(getattr(event, "gross_tokens_saved", 0) or 0)),
                    micro_usd=(gross - cost if gross or cost else None), state=state,
                )
            return inserted
        opt.__entroly_value_observer_original__ = original_opt
        OptimizationLedger.record = opt

    current_adjust = OptimizationLedger.adjust
    if not hasattr(current_adjust, "__entroly_value_observer_original__"):
        original_adjust = current_adjust
        def adjust(self: Any, adjustment: Any) -> bool:
            inserted = original_adjust(self, adjustment)
            if inserted and CURRENT_ATTRIBUTION.get() is not None:
                cost = max(0, int(getattr(adjustment, "cost_micro_usd", 0) or 0))
                record_internal(
                    "recovery_adjustment", tier=ValueTier.MEASURED,
                    role=AccountingRole.ADJUSTMENT,
                    tokens=-max(0, int(getattr(adjustment, "tokens_reexpanded", 0) or 0)),
                    micro_usd=(-cost if cost else None),
                )
            return inserted
        adjust.__entroly_value_observer_original__ = original_adjust
        OptimizationLedger.adjust = adjust

    current_usage = UsageLedger.record
    if not hasattr(current_usage, "__entroly_value_observer_original__"):
        original_usage = current_usage
        def usage(self: Any, event: Any) -> bool:
            inserted = original_usage(self, event)
            state = CURRENT_ATTRIBUTION.get()
            if not inserted or state is None or str(event.request_id) == state.request_id:
                return inserted
            total = max(0, int(getattr(event.usage, "total_tokens", 0) or 0))
            priced = not str(getattr(event, "pricing_source", "")).startswith("unpriced:")
            cost = max(0, int(getattr(event, "cost_micro_usd", 0) or 0))
            with state.lock:
                state.extra_provider_calls += 1
                state.extra_provider_tokens += total
                if priced:
                    state.extra_provider_cost_micro_usd += cost
                    state.extra_provider_priced_calls += 1
            record_internal(
                "extra_provider_call", tier=ValueTier.MEASURED,
                role=AccountingRole.ADJUSTMENT,
                micro_usd=(-cost if priced else None),
                evidence_source=EvidenceSource.PROVIDER_USAGE,
                details={"tokens": total, "priced": priced}, state=state,
            )
            return inserted
        usage.__entroly_value_observer_original__ = original_usage
        UsageLedger.record = usage


def _value_module():
    from . import proxy_traffic_value
    return proxy_traffic_value


def _record_session_receipt(receipt: Any) -> bool:
    return _value_module()._record_session_receipt(receipt)


def _record_with_session(receipt: Any, *, tracker: Any | None = None) -> bool:
    return _value_module().record_traffic_value_receipt(receipt, tracker=tracker)


def _session_rollup(*, now: float | None = None) -> dict[str, Any]:
    return _value_module()._session_rollup(now=now)


def _snapshot_with_session(tracker: Any | None = None, *, today: Any | None = None, now: float | None = None) -> dict[str, Any]:
    return _value_module().build_traffic_value_snapshot(tracker, today=today, now=now)


def _reset_session_state_for_tests(*, started_at: float | None = None) -> None:
    _value_module()._reset_session_state_for_tests(started_at=started_at)
    clear_attribution_state()


def install_session_value() -> None:
    _install_observers()


_install_observers()
try:
    from . import proxy_value_projection as _proxy_value_projection  # noqa: F401
except ImportError:
    pass


__all__ = [
    "ATTRIBUTION_SCHEMA", "AccountingRole", "AttributionState",
    "CURRENT_ATTRIBUTION", "EvidenceSource", "ValueContribution", "ValueTier",
    "active_state", "aggregate_contributions", "clear_attribution_state",
    "forget_active", "receipt_meta", "record_internal", "record_value_contribution",
    "remember_active", "remember_receipt_meta", "set_canonical_context_delta",
    "_record_session_receipt", "_record_with_session", "_reset_session_state_for_tests",
    "_session_rollup", "_snapshot_with_session", "install_session_value",
]
