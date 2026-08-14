"""Optional OTEL emitter for canonical value-attribution rows."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

_SEEN: OrderedDict[str, None] = OrderedDict()
_INSTRUMENTS: tuple[Any, Any, Any] | None | bool = None
_SOURCES = frozenset(
    {
        "context_optimization",
        "tool_output_compression",
        "conversation_compression",
        "session_rescue",
        "provider_cache",
        "warm_prefix_protection",
        "recovery_evidence",
        "recovery_adjustment",
        "extra_provider_call",
    }
)


def _source(value: object) -> str:
    raw = str(value or "other")[:64]
    return raw if raw in _SOURCES else "other"


def emit_value_otel(receipt_id: str, rows: list[Mapping[str, Any]]) -> None:
    """Emit a receipt once; OTEL failures never affect proxy accounting."""
    global _INSTRUMENTS
    key = str(receipt_id or "")
    if not key or key in _SEEN:
        return
    _SEEN[key] = None
    _SEEN.move_to_end(key)
    while len(_SEEN) > 1000:
        _SEEN.popitem(last=False)
    try:
        if _INSTRUMENTS is None:
            from opentelemetry import metrics

            meter = metrics.get_meter("entroly.value_attribution")
            _INSTRUMENTS = (
                meter.create_counter("entroly.value.attribution.events", unit="1"),
                meter.create_up_down_counter("entroly.value.attribution.tokens", unit="1"),
                meter.create_up_down_counter("entroly.value.attribution.usd", unit="USD"),
            )
        if _INSTRUMENTS is False:
            return
        events, tokens, usd = _INSTRUMENTS
        for row in rows:
            attrs = {
                "source": _source(row.get("source")),
                "tier": str(row.get("tier") or "estimated")[:24],
                "role": str(row.get("role") or "explanatory")[:24],
                "evidence_source": str(
                    row.get("evidence_source") or "local_observation"
                )[:32],
            }
            events.add(int(row.get("events", 0) or 0), attrs)
            signed_tokens = int(row.get("tokens", 0) or 0)
            if signed_tokens:
                tokens.add(signed_tokens, attrs)
            if int(row.get("priced_events", 0) or 0):
                usd.add(
                    int(row.get("micro_usd", 0) or 0) / 1_000_000.0,
                    attrs,
                )
    except Exception:
        _INSTRUMENTS = False


__all__ = ["emit_value_otel"]
