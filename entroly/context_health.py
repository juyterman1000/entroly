"""Evidence-classified aggregate context health for the local dashboard.

The dashboard must not turn architectural capabilities into outcome claims.
This module reports only counters that can be derived from local receipts,
the optimization ledger, and the value tracker.  It never reads or returns
prompt, code, path, query, model, receipt, or pseudonymous identifiers.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "entroly.context-health.v1"


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _signed_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _nonnegative_float(value: object) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if number >= 0.0 and number < float("inf") else 0.0


def default_ledger_path() -> Path:
    configured = os.environ.get("ENTROLY_OPTIMIZATION_LEDGER", "").strip()
    if configured:
        return Path(configured).expanduser()
    state = Path(os.environ.get("ENTROLY_DIR", ".entroly")).expanduser()
    return state / "optimization-ledger.sqlite3"


def _ledger_summary(path: Path) -> tuple[dict[str, int], bool]:
    if not path.is_file():
        return {}, False
    from .optimization_ledger import OptimizationLedger

    return OptimizationLedger(path).summary().as_dict(), True


def _session_summary(index: Any) -> dict[str, Any]:
    listed = index.list_sessions(limit=100)
    sessions = listed.get("sessions", []) if isinstance(listed, Mapping) else []
    diagnostics = listed.get("diagnostics", []) if isinstance(listed, Mapping) else []
    result = {
        "available": bool(sessions),
        "sessions_total": _nonnegative_int(
            listed.get("total", len(sessions)) if isinstance(listed, Mapping) else 0
        ),
        "sessions_sampled": 0,
        "valid_sessions": 0,
        "invalid_sessions": 0,
        "receipts_sampled": 0,
        "selected_tokens": 0,
        "omitted_tokens": 0,
        "omitted_items": 0,
        "recoverable_omitted_items": 0,
        "diagnostics": min(len(diagnostics), 20),
    }
    for summary in sessions:
        if not isinstance(summary, Mapping):
            continue
        session = index.get_session(str(summary.get("key", "")))
        if not isinstance(session, Mapping):
            continue
        result["sessions_sampled"] += 1
        integrity = session.get("integrity", {})
        if isinstance(integrity, Mapping) and integrity.get("valid") is True:
            result["valid_sessions"] += 1
        else:
            result["invalid_sessions"] += 1
        receipts = session.get("receipts", [])
        if not isinstance(receipts, list):
            continue
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                continue
            result["receipts_sampled"] += 1
            result["selected_tokens"] += _nonnegative_int(
                receipt.get("selected_tokens")
            )
            result["omitted_tokens"] += _nonnegative_int(receipt.get("omitted_tokens"))
            omitted = receipt.get("omitted", [])
            if not isinstance(omitted, list):
                continue
            for item in omitted:
                if not isinstance(item, Mapping):
                    continue
                result["omitted_items"] += 1
                if item.get("recoverable") is True:
                    result["recoverable_omitted_items"] += 1

    sampled = result["sessions_sampled"]
    omitted_items = result["omitted_items"]
    result["integrity_pct"] = (
        round(result["valid_sessions"] * 100.0 / sampled, 1) if sampled else None
    )
    result["recoverability_pct"] = (
        round(result["recoverable_omitted_items"] * 100.0 / omitted_items, 1)
        if omitted_items
        else None
    )
    return result


def build_context_health(
    *,
    index: Any | None = None,
    ledger_path: str | Path | None = None,
    value_confidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-blind, locally derived context-health snapshot."""
    if index is None:
        from .context_sessions import ContextSessionIndex

        index = ContextSessionIndex()
    if value_confidence is None:
        from .value_tracker import get_tracker

        tracker = get_tracker()
        tracker.reload_if_changed()
        value_confidence = tracker.get_confidence()

    sessions = _session_summary(index)
    path = Path(ledger_path).expanduser() if ledger_path else default_ledger_path()
    ledger, ledger_available = _ledger_summary(path)

    lifetime = value_confidence.get("lifetime", {})
    if not isinstance(lifetime, Mapping):
        lifetime = {}
    provider_tokens = _nonnegative_int(lifetime.get("tokens_saved"))
    provider_cost = _nonnegative_float(lifetime.get("cost_saved_usd"))
    unsupported_blocked = _nonnegative_int(lifetime.get("hallucinations_blocked"))
    gross = _nonnegative_int(ledger.get("measured_gross_tokens"))
    reexpanded = _nonnegative_int(ledger.get("measured_reexpanded_tokens"))
    # Re-expansion can exceed gross reduction. Preserve that negative result:
    # hiding a token-negative or cost-negative session would make the dashboard
    # a marketing counter rather than an accounting surface.
    net = _signed_int(ledger.get("measured_net_tokens"))
    net_micro_usd = _signed_int(ledger.get("measured_net_micro_usd"))
    recovery_tax_pct = round(reexpanded * 100.0 / gross, 1) if gross else None

    integrity_status = "measured" if sessions["sessions_sampled"] else "unavailable"
    recovery_status = (
        "measured" if ledger_available and gross else
        "available_no_measured_events" if ledger_available else
        "unavailable"
    )
    share_parts = ["Entroly Context Health"]
    if provider_tokens:
        share_parts.append(f"{provider_tokens:,} provider-bound input tokens reduced")
    if provider_cost:
        share_parts.append(f"${provider_cost:.4f} modeled input cost avoided")
    if sessions["sessions_sampled"]:
        share_parts.append(
            f"{sessions['valid_sessions']}/{sessions['sessions_sampled']} receipt chains verified"
        )
    if recovery_tax_pct is not None:
        share_parts.append(f"{recovery_tax_pct:.1f}% measured recovery tax")
    if unsupported_blocked:
        share_parts.append(f"{unsupported_blocked} unsupported claims blocked")
    share_parts.append("local aggregate only; no prompts, code, paths, or queries")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.time(),
        "privacy": {
            "content_blind": True,
            "contains_user_content": False,
            "contains_identifiers": False,
            "leaves_machine": False,
        },
        "value": {
            "provider_bound_tokens_reduced": provider_tokens,
            "modeled_provider_cost_avoided_usd": round(provider_cost, 6),
            "modeled_cost_basis": "configured input rates; not a provider invoice",
            "ledger_status": recovery_status,
            "measured_gross_tokens": gross,
            "measured_reexpanded_tokens": reexpanded,
            "measured_net_tokens": net,
            "measured_net_usd": round(net_micro_usd / 1_000_000.0, 6),
            "recovery_tax_pct": recovery_tax_pct,
        },
        "evidence": sessions,
        "protections": {
            "confusion": {
                "label": "Unsupported-claim protection",
                "status": "observed" if unsupported_blocked else "no_events_observed",
                "unsupported_claims_blocked": unsupported_blocked,
                "scope": "WITNESS suppressions only; not proof that every answer is correct",
            },
            "rot": {
                "label": "Recoverability and re-expansion",
                "status": recovery_status,
                "recoverability_pct": sessions["recoverability_pct"],
                "recovery_tax_pct": recovery_tax_pct,
                "scope": "omitted-evidence recovery; not a universal context-rot cure",
            },
            "drift": {
                "label": "Receipt-chain integrity",
                "status": integrity_status,
                "integrity_pct": sessions["integrity_pct"],
                "invalid_sessions": sessions["invalid_sessions"],
                "source_freshness": "unavailable",
                "scope": "receipt-chain drift only; source freshness is not measured here",
            },
        },
        "share": {
            "safe_to_share": True,
            "text": " • ".join(share_parts),
        },
        "limitations": [
            "Provider cost is modeled from configured rates and is not an invoice.",
            "A zero counter means no event was observed, not that the risk was eliminated.",
            "Source freshness and counterfactual answer quality are not inferred.",
            "Receipt aggregates are bounded to the newest 100 locally discoverable sessions.",
        ],
    }


__all__ = ["SCHEMA_VERSION", "build_context_health", "default_ledger_path"]
