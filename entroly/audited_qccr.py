"""Audited QCCR entry point shared by guarded callers and benchmarks.

The native engine returns a full audit envelope. This module validates that
envelope, preserves ingestion identities, and attaches only the compact
certificate to selected fragments. Candidate telemetry remains in the envelope
so production payloads do not accidentally multiply it per fragment.
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

from .qccr import _load_rank_weights, _rust_select, logical_source
from .sufficiency_contract import SufficiencyCertificate


def _fragment_tokens(fragment: dict[str, Any]) -> int:
    for key in ("token_count", "tokens"):
        value = fragment.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(math.ceil(value))
    return math.ceil(len(str(fragment.get("content") or "")) / 4)


def _identity_envelope(
    fragments: Sequence[dict[str, Any]], token_budget: int, *, reason: str
) -> dict[str, Any]:
    selected = [dict(fragment) for fragment in fragments]
    total = sum(_fragment_tokens(fragment) for fragment in selected)
    metrics = {
        "captured_mass": 1.0,
        "shadow_price": 0.0,
        "residual_risk": 0.0,
        "cutoff_ambiguity": 0.0,
        "query_coverage": 1.0,
        "boundary_exposure": 0.0,
        "budget_saturation": total / max(token_budget, 1),
        "source_span_integrity": True,
        "excluded_positive_candidates": 0,
        "verdict": "sufficient",
        "scope": "candidate_units",
        "reasons": [reason],
        "signal_availability": {
            "identity_preservation": True,
            "semantic_calibration": False,
            "task_oracle": False,
        },
        "calibration_version": None,
        "calibration_dataset_fingerprint": None,
    }
    return {
        "selected": selected,
        "candidates": [],
        "metrics": metrics,
        "requested_budget": token_budget,
        "raw_tokens": total,
        "emitted_tokens": total,
        "selection_mode": "identity",
    }


def select_with_audit(
    fragments: Sequence[dict[str, Any]], token_budget: int, query: str = ""
) -> dict[str, Any]:
    """Return selected fragments plus exact candidate/span audit telemetry."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    original = [dict(fragment) for fragment in fragments]
    if not original:
        return _identity_envelope([], token_budget, reason="empty input")

    raw_tokens = sum(_fragment_tokens(fragment) for fragment in original)
    if raw_tokens <= token_budget:
        return _identity_envelope(
            original,
            token_budget,
            reason="input already fits; no evidence was removed",
        )
    if not query:
        envelope = _identity_envelope(
            original,
            token_budget,
            reason="no query was available for conditioned selection",
        )
        envelope["metrics"]["verdict"] = "uncertain"
        envelope["metrics"]["scope"] = "unavailable"
        envelope["selection_mode"] = "identity_no_query"
        return envelope

    slim: list[dict[str, Any]] = []
    source_fragment_ids: dict[str, list[str]] = {}
    for index, raw in enumerate(original):
        source = logical_source(str(raw.get("source") or ""))
        fragment_id = str(raw.get("fragment_id") or raw.get("id") or f"input::{index}")
        source_fragment_ids.setdefault(source, []).append(fragment_id)
        slim.append(
            {
                "id": str(raw.get("id") or ""),
                "fragment_id": fragment_id,
                "source": source,
                "content": str(raw.get("content") or ""),
                "start_byte": raw.get("start_byte"),
                "end_byte": raw.get("end_byte"),
                "feedback_multiplier": float(
                    raw.get("feedback_multiplier", 1.0) or 1.0
                ),
            }
        )

    try:
        encoded = _rust_select(
            json.dumps(slim, ensure_ascii=False),
            int(token_budget),
            query,
            json.dumps(_load_rank_weights()),
            "[]",
            True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "installed entroly_core is older than the audited QCCR contract; "
            "rebuild the native extension from this checkout"
        ) from exc

    envelope = json.loads(encoded)
    if not isinstance(envelope, dict):
        raise RuntimeError("audited QCCR returned a non-object envelope")
    selected = envelope.get("selected")
    metrics = envelope.get("metrics")
    candidates = envelope.get("candidates")
    if not isinstance(selected, list) or not isinstance(metrics, dict):
        raise RuntimeError("audited QCCR envelope is missing selected/metrics")
    if not isinstance(candidates, list):
        envelope["candidates"] = []

    certificate = SufficiencyCertificate.from_mapping(metrics)
    compact_certificate = dict(metrics)
    compact_certificate.update(certificate.to_dict())
    for fragment in selected:
        if not isinstance(fragment, dict):
            continue
        source = str(fragment.get("source") or "")
        ids = source_fragment_ids.get(source, [])
        if ids:
            fragment["source_fragment_ids"] = list(dict.fromkeys(ids))
    if selected and isinstance(selected[0], dict):
        selected[0]["_sufficiency"] = compact_certificate
    envelope["selected"] = selected
    envelope["metrics"] = compact_certificate
    return envelope


def select(
    fragments: Sequence[dict[str, Any]], token_budget: int, query: str = ""
) -> list[dict[str, Any]]:
    """Selector-compatible facade used by ``select_guarded``."""
    return list(select_with_audit(fragments, token_budget, query)["selected"])
