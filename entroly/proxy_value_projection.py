"""Project canonical request value attribution into product surfaces."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from starlette.responses import JSONResponse

from . import proxy as _proxy
from . import proxy_traffic_session as _state
from . import proxy_traffic_value as _value
from .value_tracker import ValueTracker

_INSTALLED = False
_METRIC_SOURCES = frozenset(
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


def _metric_source(value: object) -> str:
    source = str(value or "other")[:64]
    return source if source in _METRIC_SOURCES else "other"


def _row_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row.get("source") or "other")[:64],
            str(row.get("tier") or "estimated")[:24],
            str(row.get("role") or "explanatory")[:24],
            str(row.get("evidence_source") or "local_observation")[:32],
            "1" if bool(row.get("headline_included")) else "0",
        )
    )


def _row_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_row_key(row): dict(row) for row in rows}


def _merge_rows(target: dict[str, Any], incoming: Any) -> None:
    rows = (
        [row for row in incoming.values() if isinstance(row, Mapping)]
        if isinstance(incoming, Mapping)
        else [row for row in incoming if isinstance(row, Mapping)]
        if isinstance(incoming, list)
        else []
    )
    current = target.setdefault("_value_attribution", {})
    if not isinstance(current, dict):
        current = {}
        target["_value_attribution"] = current
    for key, row in _row_map(rows).items():
        if key not in current:
            current[key] = row
            continue
        for field in ("events", "tokens", "micro_usd", "priced_events"):
            current[key][field] = int(current[key].get(field, 0) or 0) + int(
                row.get(field, 0) or 0
            )


def _install_cli_and_stats() -> None:
    current_receipt = ValueTracker.get_value_receipt
    if not hasattr(current_receipt, "__entroly_value_projection_original__"):
        original_receipt = current_receipt

        def value_receipt(self: Any, *args: Any, **kwargs: Any):
            out = original_receipt(self, *args, **kwargs)
            if isinstance(out, dict):
                lifetime = _value.build_traffic_value_snapshot(self).get(
                    "windows", {}
                ).get("lifetime", {})
                out["value_by_source"] = lifetime.get("value_by_source", [])
                out["extra_provider_cost_usd"] = lifetime.get(
                    "extra_provider_cost_usd", 0.0
                )
                out["net_value_after_observed_extra_provider_cost_usd"] = lifetime.get(
                    "net_value_after_observed_extra_provider_cost_usd", 0.0
                )
            return out

        value_receipt.__entroly_value_projection_original__ = original_receipt
        ValueTracker.get_value_receipt = value_receipt

    current_stats = _proxy._proxy_stats
    if not hasattr(current_stats, "__entroly_value_projection_original__"):
        original_stats = current_stats

        async def stats(request: Any):
            response = await original_stats(request)
            body = getattr(response, "body", None)
            if not isinstance(body, (bytes, bytearray)):
                return response
            try:
                payload = json.loads(bytes(body))
            except Exception:
                return response
            if not isinstance(payload, dict):
                return response
            lifetime = _value.build_traffic_value_snapshot().get(
                "windows", {}
            ).get("lifetime", {})
            payload["value_attribution"] = {
                "schema_version": _state.ATTRIBUTION_SCHEMA,
                "by_source": lifetime.get("value_by_source", []),
                "extra_provider_cost_usd": lifetime.get(
                    "extra_provider_cost_usd", 0.0
                ),
            }
            return JSONResponse(payload, status_code=response.status_code)

        stats.__entroly_value_projection_original__ = original_stats
        _proxy._proxy_stats = stats


def _prometheus_rows() -> str:
    lifetime = _value.build_traffic_value_snapshot().get("windows", {}).get("lifetime", {})
    lines: list[str] = []
    for row in lifetime.get("value_by_source", []) or []:
        labels = (
            f'source="{_metric_source(row.get("source"))}",'
            f'tier="{str(row.get("tier") or "estimated")[:24]}",'
            f'role="{str(row.get("role") or "explanatory")[:24]}",'
            f'evidence_source="{str(row.get("evidence_source") or "local_observation")[:32]}"'
        )
        lines.append(
            f"entroly_value_attribution_events_total{{{labels}}} {int(row.get('events', 0) or 0)}"
        )
        lines.append(
            f"entroly_value_attributed_tokens{{{labels}}} {int(row.get('tokens', 0) or 0)}"
        )
        if int(row.get("priced_events", 0) or 0):
            lines.append(
                f"entroly_value_attributed_usd{{{labels}}} {int(row.get('micro_usd', 0) or 0) / 1_000_000.0:.6f}"
            )
    lines.append(
        f"entroly_value_extra_provider_cost_usd {float(lifetime.get('extra_provider_cost_usd', 0.0) or 0.0):.6f}"
    )
    return "\n".join(lines) + "\n"


def _install_prometheus() -> None:
    current = _proxy._metrics_prometheus
    if hasattr(current, "__entroly_value_projection_original__"):
        return
    original = current

    async def metrics(request: Any):
        response = await original(request)
        iterator = getattr(response, "body_iterator", None)
        if iterator is None:
            return response

        async def combined():
            async for chunk in iterator:
                yield chunk
            yield _prometheus_rows()

        response.body_iterator = combined()
        return response

    metrics.__entroly_value_projection_original__ = original
    _proxy._metrics_prometheus = metrics


def install_value_projection() -> None:
    """Install one accounting projection without changing the headline math."""
    global _INSTALLED
    if _INSTALLED:
        return
    for field in (
        "attribution_events",
        "attribution_reconciled_requests",
        "extra_provider_calls",
        "extra_provider_tokens",
        "extra_provider_priced_requests",
    ):
        if field not in _value._COUNTER_FIELDS:
            _value._COUNTER_FIELDS = (*_value._COUNTER_FIELDS, field)
    if "extra_provider_cost_usd" not in _value._MONEY_FIELDS:
        _value._MONEY_FIELDS = (*_value._MONEY_FIELDS, "extra_provider_cost_usd")

    original_delta = _value._receipt_delta
    if not hasattr(original_delta, "__entroly_value_projection_original__"):
        def receipt_delta(receipt: Any) -> dict[str, Any]:
            out = original_delta(receipt)
            meta = _state.receipt_meta(str(getattr(receipt, "receipt_id", "")))
            rows = meta.get("value_contributions", [])
            out["attribution_events"] = sum(
                int(row.get("events", 0) or 0) for row in rows
            )
            out["attribution_reconciled_requests"] = int(
                bool(meta.get("attribution_reconciled"))
            )
            out["extra_provider_calls"] = int(meta.get("extra_provider_calls", 0) or 0)
            out["extra_provider_tokens"] = int(meta.get("extra_provider_tokens", 0) or 0)
            extra_cost = meta.get("extra_provider_cost_micro_usd")
            out["extra_provider_priced_requests"] = int(extra_cost is not None)
            out["extra_provider_cost_usd"] = (
                round(int(extra_cost) / 1_000_000.0, 6)
                if extra_cost is not None else 0.0
            )
            out["_value_attribution"] = _row_map(rows)
            return out
        receipt_delta.__entroly_value_projection_original__ = original_delta
        _value._receipt_delta = receipt_delta

    original_accumulate = _value._accumulate
    if not hasattr(original_accumulate, "__entroly_value_projection_original__"):
        def accumulate(target: dict[str, Any], row: Mapping[str, Any]) -> None:
            original_accumulate(target, row)
            _merge_rows(target, row.get("_value_attribution"))
        accumulate.__entroly_value_projection_original__ = original_accumulate
        _value._accumulate = accumulate

    original_finalize = _value._finalize_metrics
    if not hasattr(original_finalize, "__entroly_value_projection_original__"):
        def finalize(row: dict[str, Any]) -> dict[str, Any]:
            out = original_finalize(row)
            raw = out.pop("_value_attribution", {})
            rows = list(raw.values()) if isinstance(raw, Mapping) else []
            out["value_by_source"] = sorted(
                (dict(item) for item in rows),
                key=lambda item: (
                    -abs(int(item.get("tokens", 0) or 0)),
                    -abs(int(item.get("micro_usd", 0) or 0)),
                    str(item.get("source") or ""),
                ),
            )
            out["net_value_after_observed_extra_provider_cost_usd"] = round(
                float(out.get("total_ai_value_protected_usd", 0.0) or 0.0)
                - float(out.get("extra_provider_cost_usd", 0.0) or 0.0),
                6,
            )
            return out
        finalize.__entroly_value_projection_original__ = original_finalize
        _value._finalize_metrics = finalize

    original_snapshot = _value.build_traffic_value_snapshot
    if not hasattr(original_snapshot, "__entroly_value_projection_original__"):
        def snapshot(*args: Any, **kwargs: Any) -> dict[str, Any]:
            out = original_snapshot(*args, **kwargs)
            out["attribution_schema_version"] = _state.ATTRIBUTION_SCHEMA
            truth = out.setdefault("truth", {})
            truth["value_attribution"] = (
                "Contribution rows explain the canonical headline; only rows "
                "marked headline_included participate in headline token accounting."
            )
            truth["extra_provider_cost"] = (
                "Additional provider work observed under the outer request is a "
                "cost debit only when provider usage and auditable pricing exist."
            )
            return out
        snapshot.__entroly_value_projection_original__ = original_snapshot
        _value.build_traffic_value_snapshot = snapshot

    _install_cli_and_stats()
    _install_prometheus()
    _INSTALLED = True


install_value_projection()
try:
    from . import proxy_value_dashboard as _proxy_value_dashboard  # noqa: F401
except ImportError:
    pass

__all__ = ["install_value_projection"]
