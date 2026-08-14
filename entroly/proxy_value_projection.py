"""Project canonical request value attribution into product surfaces."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import proxy_traffic_session as _state
from . import proxy_traffic_value as _value

_INSTALLED = False


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

    _INSTALLED = True


install_value_projection()

__all__ = ["install_value_projection"]
