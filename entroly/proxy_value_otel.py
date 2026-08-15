"""Optional OTEL emitter for canonical value-attribution rows."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

_SEEN: OrderedDict[str, None] = OrderedDict()
_INSTRUMENTS: tuple[Any, ...] | None | bool = None
_SOURCES = frozenset(
    {
        "context_optimization",
        "tool_output_compression",
        "tool_schema_deferral",
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
                meter.create_counter("entroly.proxy.tokens.saved", unit="1"),
                meter.create_counter(
                    "entroly.proxy.tokens.compression_saved", unit="1"
                ),
                meter.create_counter(
                    "entroly.proxy.tokens.tool_schema_saved", unit="1"
                ),
            )
        if _INSTRUMENTS is False:
            return
        events, tokens, usd, proxy_saved, compression_saved, tool_schema_saved = (
            _INSTRUMENTS
        )
        headline_tokens = 0
        raw_tool_schema_tokens = 0
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
            if (
                bool(row.get("headline_included"))
                and str(row.get("tier") or "") == "measured"
                and signed_tokens > 0
            ):
                headline_tokens += signed_tokens
            if (
                str(row.get("source") or "") == "tool_schema_deferral"
                and str(row.get("tier") or "") == "measured"
                and signed_tokens > 0
            ):
                raw_tool_schema_tokens += signed_tokens
            if signed_tokens:
                tokens.add(signed_tokens, attrs)
            if int(row.get("priced_events", 0) or 0):
                usd.add(
                    int(row.get("micro_usd", 0) or 0) / 1_000_000.0,
                    attrs,
                )
        # The canonical headline already measures the whole outbound request.
        # Components partition that total; they are never added on top of it.
        schema_tokens = min(headline_tokens, raw_tool_schema_tokens)
        compression_tokens = max(0, headline_tokens - schema_tokens)
        if headline_tokens:
            proxy_saved.add(headline_tokens)
        if compression_tokens:
            compression_saved.add(compression_tokens)
        if schema_tokens:
            tool_schema_saved.add(schema_tokens)
    except Exception:
        _INSTRUMENTS = False


__all__ = ["emit_value_otel"]
