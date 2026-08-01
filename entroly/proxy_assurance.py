"""Opt-in fail-closed assurance controller for provider proxy messages.

This module deliberately sits outside the compatibility proxy pipeline. When
enabled, it applies the public assured-message SDK to standard string-content
chat messages. Unsupported/multimodal shapes, invalid profiles, and runtime
errors preserve the request byte-for-byte at the Python object level and emit
bounded diagnostic headers rather than forwarding a partially transformed
request.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .assurance_sdk import AssuredMessagesResult, compress_messages_assured
from .assurance_telemetry import AssuranceLedger
from .sufficiency_calibration import CalibrationProfile

_ALLOWED_MODES = {"off", "candidate_units", "semantic"}
_ALLOWED_FALLBACKS = {"original", "selected"}
_MAX_PROFILE_BYTES = 2_000_000


def _bounded_header(value: Any, *, limit: int = 240) -> str:
    rendered = str(value).replace("\r", " ").replace("\n", " ").strip()
    return rendered[:limit]


def load_calibration_profile(path: str | Path) -> CalibrationProfile:
    """Load a bounded JSON profile without accepting unknown fields."""
    candidate = Path(path).expanduser().resolve(strict=True)
    if not candidate.is_file():
        raise ValueError("assurance profile must be an existing file")
    if candidate.stat().st_size > _MAX_PROFILE_BYTES:
        raise ValueError("assurance profile exceeds the bounded size limit")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("assurance profile must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("assurance profile must contain a JSON object")
    allowed = {item.name for item in fields(CalibrationProfile)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("assurance profile contains unknown fields")
    if "calibration_membership" in payload:
        membership = payload["calibration_membership"]
        if not isinstance(membership, list) or not all(
            isinstance(item, str) for item in membership
        ):
            raise ValueError("calibration_membership must be a string array")
        payload["calibration_membership"] = tuple(membership)
    try:
        return CalibrationProfile(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("assurance profile does not match the current schema") from exc


@dataclass(frozen=True)
class ProxyAssuranceOutcome:
    body: dict[str, Any]
    headers: dict[str, str]
    enabled: bool
    changed: bool


class ProxyAssuranceController:
    """Apply assured message compression under an explicit proxy policy."""

    def __init__(
        self,
        *,
        mode: str = "off",
        budget_tokens: int = 0,
        budget_fraction: float = 0.15,
        preserve_last_n: int = 4,
        fallback: str = "original",
        max_expansions: int = 2,
        calibration_profile: CalibrationProfile | None = None,
        ledger: AssuranceLedger | None = None,
        init_error: str = "",
    ) -> None:
        normalized_mode = str(mode).strip().lower() or "off"
        if normalized_mode not in _ALLOWED_MODES:
            raise ValueError("assurance mode must be off, candidate_units, or semantic")
        if budget_tokens < 0:
            raise ValueError("assurance budget_tokens must be non-negative")
        if not 0 < budget_fraction <= 1:
            raise ValueError("assurance budget_fraction must be in (0, 1]")
        if preserve_last_n < 1:
            raise ValueError("assurance preserve_last_n must be positive")
        if fallback not in _ALLOWED_FALLBACKS:
            raise ValueError("assurance fallback must be original or selected")
        if max_expansions < 0:
            raise ValueError("assurance max_expansions must be non-negative")
        self.mode = normalized_mode
        self.budget_tokens = int(budget_tokens)
        self.budget_fraction = float(budget_fraction)
        self.preserve_last_n = int(preserve_last_n)
        self.fallback = fallback
        self.max_expansions = int(max_expansions)
        self.calibration_profile = calibration_profile
        self.ledger = ledger
        self.init_error = _bounded_header(init_error)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_config(cls, config: Any) -> "ProxyAssuranceController":
        mode = str(getattr(config, "assurance_mode", "off") or "off").strip().lower()
        profile: CalibrationProfile | None = None
        ledger: AssuranceLedger | None = None
        errors: list[str] = []
        profile_path = str(getattr(config, "assurance_profile_path", "") or "").strip()
        if profile_path:
            try:
                profile = load_calibration_profile(profile_path)
            except (OSError, ValueError) as exc:
                errors.append(f"profile:{type(exc).__name__}")
        ledger_path = str(getattr(config, "assurance_ledger_path", "") or "").strip()
        if ledger_path:
            try:
                ledger = AssuranceLedger(ledger_path)
            except (OSError, ValueError) as exc:
                errors.append(f"ledger:{type(exc).__name__}")
        return cls(
            mode=mode,
            budget_tokens=int(getattr(config, "assurance_budget_tokens", 0) or 0),
            budget_fraction=float(
                getattr(config, "assurance_budget_fraction", 0.15) or 0.15
            ),
            preserve_last_n=int(getattr(config, "assurance_preserve_last_n", 4) or 4),
            fallback=str(getattr(config, "assurance_fallback", "original") or "original"),
            max_expansions=int(getattr(config, "assurance_max_expansions", 2) or 2),
            calibration_profile=profile,
            ledger=ledger,
            init_error=",".join(errors),
        )

    def _budget(self, context_window: int) -> int:
        if self.budget_tokens > 0:
            return self.budget_tokens
        return max(256, int(math.floor(max(1, context_window) * self.budget_fraction)))

    def _bypass(
        self,
        body: Mapping[str, Any],
        decision: str,
        *,
        error: str = "",
    ) -> ProxyAssuranceOutcome:
        headers = {
            "X-Entroly-Assurance": "true",
            "X-Entroly-Assurance-Decision": _bounded_header(decision),
            "X-Entroly-Assurance-Changed": "false",
        }
        if error:
            headers["X-Entroly-Assurance-Error"] = _bounded_header(error)
        return ProxyAssuranceOutcome(dict(body), headers, True, False)

    def apply(
        self,
        body: Mapping[str, Any],
        *,
        query: str,
        context_window: int,
    ) -> ProxyAssuranceOutcome:
        if not self.enabled:
            return ProxyAssuranceOutcome(dict(body), {}, False, False)
        if self.init_error:
            return self._bypass(body, "BYPASS_INIT_ERROR", error=self.init_error)
        messages = body.get("messages")
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            return self._bypass(body, "BYPASS_UNSUPPORTED_SHAPE")
        # The current SDK contract selects whole string-content messages. A
        # multimodal/tool-call block must never disappear merely because it is
        # not rankable by this path.
        if any(not isinstance(message.get("content"), str) for message in messages):
            return self._bypass(body, "BYPASS_MULTIMODAL_OR_STRUCTURED")
        required_scope = "semantic" if self.mode == "semantic" else "candidate_units"
        try:
            result: AssuredMessagesResult = compress_messages_assured(
                messages,
                budget=self._budget(context_window),
                query=query,
                preserve_last_n=self.preserve_last_n,
                required_scope=required_scope,
                calibration_profile=self.calibration_profile,
                fallback=self.fallback,
                max_expansions=self.max_expansions,
                ledger=self.ledger,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._bypass(
                body,
                "BYPASS_RUNTIME_ERROR",
                error=type(exc).__name__,
            )
        updated = dict(body)
        updated["messages"] = [dict(message) for message in result.messages]
        receipt = result.receipt
        attempts = receipt.get("attempts")
        certificate_scope = "unavailable"
        certificate_verdict = "uncertain"
        if isinstance(attempts, list) and attempts:
            final_attempt = attempts[-1]
            if isinstance(final_attempt, dict):
                certificate_scope = str(
                    final_attempt.get("certificate_scope") or certificate_scope
                )
                certificate_verdict = str(
                    final_attempt.get("certificate_verdict") or certificate_verdict
                )
        headers = {
            "X-Entroly-Assurance": "true",
            "X-Entroly-Assurance-Decision": _bounded_header(
                receipt.get("decision", "unknown")
            ),
            "X-Entroly-Assurance-Required-Scope": required_scope,
            "X-Entroly-Assurance-Certificate": _bounded_header(
                f"{certificate_verdict}/{certificate_scope}"
            ),
            "X-Entroly-Assurance-Changed": str(result.changed).lower(),
            "X-Entroly-Assurance-Budget-Compliant": str(
                result.budget_compliant
            ).lower(),
            "X-Entroly-Assurance-Tokens": _bounded_header(
                f"{result.original_tokens}->{result.delivered_tokens}"
            ),
        }
        return ProxyAssuranceOutcome(updated, headers, True, result.changed)


__all__ = [
    "ProxyAssuranceController",
    "ProxyAssuranceOutcome",
    "load_calibration_profile",
]
