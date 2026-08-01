"""Fail-closed caller policy for query-conditioned context selection.

The selector ranks and extracts context. This module owns the separate policy
question: whether the caller may trust that output, should retry with a larger
budget, or must return the original input with an explicit receipt.

Nothing in this module claims semantic sufficiency by itself. It accepts a
certificate only when both its verdict and evidence scope satisfy the caller's
policy. Missing or weaker evidence is ``uncertain``, never an implicit pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from .sufficiency_contract import CertificateScope, SufficiencyCertificate, parse_scope


class GuardDecision(str, Enum):
    BYPASS_EMPTY = "BYPASS_EMPTY"
    BYPASS_NO_QUERY = "BYPASS_NO_QUERY"
    BYPASS_ALREADY_FITS = "BYPASS_ALREADY_FITS"
    COMPRESSED_CERTIFIED = "COMPRESSED_CERTIFIED"
    EXPANDED_CERTIFIED = "EXPANDED_CERTIFIED"
    BYPASS_UNCERTIFIED = "BYPASS_UNCERTIFIED"
    UNCERTIFIED_BUDGET_ENFORCED = "UNCERTIFIED_BUDGET_ENFORCED"


class SufficiencyNotCertifiedError(RuntimeError):
    """Raised when strict mode forbids both fallback choices."""


@dataclass(frozen=True)
class GuardedSelectionReceipt:
    """Auditable decision record for one guarded selection."""

    decision: GuardDecision
    requested_budget: int
    final_budget: int
    raw_tokens: int
    delivered_tokens: int
    required_scope: CertificateScope
    exact_identity: bool
    budget_compliant: bool
    input_sha256: str
    output_sha256: str
    attempts: tuple[dict[str, Any], ...] = field(default=())
    reasons: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "requested_budget": self.requested_budget,
            "final_budget": self.final_budget,
            "raw_tokens": self.raw_tokens,
            "delivered_tokens": self.delivered_tokens,
            "required_scope": self.required_scope.value,
            "exact_identity": self.exact_identity,
            "budget_compliant": self.budget_compliant,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "attempts": [dict(attempt) for attempt in self.attempts],
            "reasons": list(self.reasons),
        }


def estimate_fragment_tokens(fragment: dict[str, Any]) -> int:
    """Conservative orchestration estimate; never reused as provider usage."""
    for key in ("token_count", "tokens"):
        value = fragment.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(math.ceil(value))
    content = str(fragment.get("content") or "")
    return max(0, math.ceil(len(content) / 4))


def estimate_total_tokens(fragments: Sequence[dict[str, Any]]) -> int:
    return sum(estimate_fragment_tokens(fragment) for fragment in fragments)


def _content_hash(fragments: Sequence[dict[str, Any]]) -> str:
    """Hash ordered source/content bytes; metadata cannot fake identity."""
    digest = hashlib.sha256()
    for fragment in fragments:
        source = str(fragment.get("source") or "").encode("utf-8")
        content = str(fragment.get("content") or "").encode("utf-8")
        digest.update(len(source).to_bytes(8, "big"))
        digest.update(source)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _extract_certificate(
    fragments: Sequence[dict[str, Any]],
) -> SufficiencyCertificate:
    for fragment in fragments:
        value = fragment.get("_sufficiency")
        if isinstance(value, dict):
            return SufficiencyCertificate.from_mapping(value)
    return SufficiencyCertificate.from_mapping(None)


def _attempt_record(
    budget: int,
    selected: Sequence[dict[str, Any]],
    certificate: SufficiencyCertificate,
) -> dict[str, Any]:
    return {
        "budget": int(budget),
        "delivered_tokens": estimate_total_tokens(selected),
        "certificate_verdict": certificate.verdict.value,
        "certificate_scope": certificate.scope.value,
        "certificate_reasons": list(certificate.reasons),
    }


def select_guarded(
    fragments: Sequence[dict[str, Any]],
    token_budget: int,
    query: str = "",
    *,
    max_expansions: int = 2,
    expansion_factor: float = 1.5,
    min_expansion_tokens: int = 64,
    required_scope: str | CertificateScope = CertificateScope.SEMANTIC,
    fallback: str = "original",
    selector: Callable[..., list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], GuardedSelectionReceipt]:
    """Apply identity/accept/expand/fallback policy around a selector.

    ``fallback="original"`` is quality-first and may exceed the requested
    compression budget; the receipt reports that fact. ``fallback="selected"``
    obeys the transport budget but labels the output uncertified. ``raise``
    returns neither and exposes all attempts in the exception payload.
    """
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_expansions < 0:
        raise ValueError("max_expansions must be non-negative")
    if max_expansions and expansion_factor <= 1.0:
        raise ValueError("expansion_factor must be > 1 when expansion is enabled")
    if min_expansion_tokens < 1:
        raise ValueError("min_expansion_tokens must be positive")
    if fallback not in {"original", "selected", "raise"}:
        raise ValueError("fallback must be 'original', 'selected', or 'raise'")

    required = (
        required_scope
        if isinstance(required_scope, CertificateScope)
        else parse_scope(required_scope)
    )
    if required is CertificateScope.UNAVAILABLE and str(required_scope) != "unavailable":
        raise ValueError(f"unknown required_scope: {required_scope!r}")

    original = list(fragments)
    raw_tokens = estimate_total_tokens(original)
    input_hash = _content_hash(original)

    def finish(
        output: list[dict[str, Any]],
        *,
        decision: GuardDecision,
        final_budget: int,
        attempts: list[dict[str, Any]],
        reasons: Sequence[str] = (),
        exact_identity: bool = False,
    ) -> tuple[list[dict[str, Any]], GuardedSelectionReceipt]:
        delivered = estimate_total_tokens(output)
        return output, GuardedSelectionReceipt(
            decision=decision,
            requested_budget=int(token_budget),
            final_budget=int(final_budget),
            raw_tokens=raw_tokens,
            delivered_tokens=delivered,
            required_scope=required,
            exact_identity=exact_identity,
            budget_compliant=delivered <= token_budget,
            input_sha256=input_hash,
            output_sha256=_content_hash(output),
            attempts=tuple(attempts),
            reasons=tuple(str(reason) for reason in reasons),
        )

    if not original:
        return finish(
            [],
            decision=GuardDecision.BYPASS_EMPTY,
            final_budget=token_budget,
            attempts=[],
            exact_identity=True,
        )
    if not query:
        return finish(
            original,
            decision=GuardDecision.BYPASS_NO_QUERY,
            final_budget=token_budget,
            attempts=[],
            reasons=("query-conditioned sufficiency is unavailable without a query",),
            exact_identity=True,
        )
    if raw_tokens <= token_budget:
        return finish(
            original,
            decision=GuardDecision.BYPASS_ALREADY_FITS,
            final_budget=token_budget,
            attempts=[],
            reasons=("identity dominates compression when input already fits",),
            exact_identity=True,
        )

    if selector is None:
        from .qccr import select as selector

    current_budget = int(token_budget)
    attempts: list[dict[str, Any]] = []
    last_selected: list[dict[str, Any]] = []
    last_certificate = SufficiencyCertificate.from_mapping(None)

    for attempt_index in range(max_expansions + 1):
        selected = list(selector(original, token_budget=current_budget, query=query))
        certificate = _extract_certificate(selected)
        attempts.append(_attempt_record(current_budget, selected, certificate))
        last_selected = selected
        last_certificate = certificate

        if certificate.satisfies(required):
            decision = (
                GuardDecision.COMPRESSED_CERTIFIED
                if attempt_index == 0
                else GuardDecision.EXPANDED_CERTIFIED
            )
            return finish(
                selected,
                decision=decision,
                final_budget=current_budget,
                attempts=attempts,
                exact_identity=_content_hash(selected) == input_hash,
            )

        if attempt_index >= max_expansions or current_budget >= raw_tokens:
            break
        next_budget = min(
            raw_tokens,
            max(
                current_budget + min_expansion_tokens,
                int(math.ceil(current_budget * expansion_factor)),
            ),
        )
        if next_budget <= current_budget:
            break
        current_budget = next_budget

    failure_reasons = list(last_certificate.reasons)
    if not last_certificate.satisfies(required):
        failure_reasons.append(
            f"certificate {last_certificate.verdict.value}/{last_certificate.scope.value} "
            f"does not satisfy required scope {required.value}"
        )

    if fallback == "original":
        return finish(
            original,
            decision=GuardDecision.BYPASS_UNCERTIFIED,
            final_budget=current_budget,
            attempts=attempts,
            reasons=failure_reasons,
            exact_identity=True,
        )
    if fallback == "selected":
        return finish(
            last_selected,
            decision=GuardDecision.UNCERTIFIED_BUDGET_ENFORCED,
            final_budget=current_budget,
            attempts=attempts,
            reasons=failure_reasons,
            exact_identity=_content_hash(last_selected) == input_hash,
        )

    raise SufficiencyNotCertifiedError(
        json.dumps(
            {
                "decision": "RAISE_UNCERTIFIED",
                "requested_budget": token_budget,
                "final_budget": current_budget,
                "required_scope": required.value,
                "attempts": attempts,
                "reasons": failure_reasons,
            },
            sort_keys=True,
        )
    )
