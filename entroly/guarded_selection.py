"""Fail-closed orchestration for QCCR selection.

``select_guarded`` is intentionally separate from ``entroly.qccr.select``.
The selector's contract is ranking and budgeted extraction. This module owns
the caller policy: identity bypass, sufficiency acceptance, bounded expansion,
and an explicit fallback when answer preservation cannot be certified.

The default policy favours quality over a hard compression budget. If semantic
sufficiency cannot be certified, it returns the original fragments byte-for-
byte and records that the requested budget was not met. Callers that must obey
a hard transport limit can choose ``fallback="selected"``; the receipt then
marks the output uncertified rather than pretending it is safe.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


_SCOPE_RANK = {
    "unavailable": 0,
    "optimizer_proxy": 1,
    "file_retrieval": 1,
    "candidate_units": 2,
    "semantic": 3,
    "answer_preservation": 3,
}


class SufficiencyNotCertifiedError(RuntimeError):
    """Raised when ``fallback="raise"`` and no attempt can be certified."""


@dataclass(frozen=True)
class GuardedSelectionReceipt:
    """Auditable decision record for one guarded selection."""

    decision: str
    requested_budget: int
    final_budget: int
    raw_tokens: int
    delivered_tokens: int
    required_scope: str
    exact_identity: bool
    budget_compliant: bool
    input_sha256: str
    output_sha256: str
    attempts: tuple[dict[str, Any], ...] = field(default=())
    reasons: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "requested_budget": self.requested_budget,
            "final_budget": self.final_budget,
            "raw_tokens": self.raw_tokens,
            "delivered_tokens": self.delivered_tokens,
            "required_scope": self.required_scope,
            "exact_identity": self.exact_identity,
            "budget_compliant": self.budget_compliant,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "attempts": [dict(attempt) for attempt in self.attempts],
            "reasons": list(self.reasons),
        }


def estimate_fragment_tokens(fragment: dict[str, Any]) -> int:
    """Conservative token estimate used only for orchestration accounting."""
    for key in ("token_count", "tokens"):
        value = fragment.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(math.ceil(value))
    content = str(fragment.get("content") or "")
    return max(0, math.ceil(len(content) / 4))


def estimate_total_tokens(fragments: Sequence[dict[str, Any]]) -> int:
    return sum(estimate_fragment_tokens(fragment) for fragment in fragments)


def _content_hash(fragments: Sequence[dict[str, Any]]) -> str:
    """Hash ordered source/content bytes; metadata changes do not fake identity."""
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
) -> dict[str, Any] | None:
    for fragment in fragments:
        value = fragment.get("_sufficiency")
        if isinstance(value, dict):
            return value
    return None


def _scope_satisfies(actual: str, required: str) -> bool:
    return _SCOPE_RANK.get(actual, 0) >= _SCOPE_RANK.get(required, 10)


def _certificate_is_acceptable(
    certificate: dict[str, Any] | None,
    required_scope: str,
) -> bool:
    if certificate is None:
        return False
    return (
        certificate.get("verdict") == "sufficient"
        and _scope_satisfies(
            str(certificate.get("scope") or "unavailable"),
            required_scope,
        )
    )


def _attempt_record(
    budget: int,
    selected: Sequence[dict[str, Any]],
    certificate: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "budget": int(budget),
        "delivered_tokens": estimate_total_tokens(selected),
        "certificate_verdict": (
            str(certificate.get("verdict")) if certificate else "missing"
        ),
        "certificate_scope": (
            str(certificate.get("scope")) if certificate else "unavailable"
        ),
        "certificate_reasons": (
            list(certificate.get("reasons") or []) if certificate else []
        ),
    }


def select_guarded(
    fragments: Sequence[dict[str, Any]],
    token_budget: int,
    query: str = "",
    *,
    max_expansions: int = 2,
    expansion_factor: float = 1.5,
    min_expansion_tokens: int = 64,
    required_scope: str = "semantic",
    fallback: str = "original",
    selector: Callable[..., list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], GuardedSelectionReceipt]:
    """Select context under an explicit accept/expand/bypass policy.

    Decisions:
    - ``bypass_already_fits``: identity is strictly better than compression.
    - ``compressed``: the requested budget produced an acceptable certificate.
    - ``expanded``: a larger budget was needed before certification.
    - ``bypass_uncertified``: quality-first fallback to the original.
    - ``uncertified_budget_enforced``: caller required selected output despite
      the missing/degraded certificate.

    ``required_scope="semantic"`` deliberately rejects today's
    ``optimizer_proxy`` certificate. This prevents a retrieval-score proxy from
    being marketed as proof that the final answer span survived trimming.
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

    original = list(fragments)
    raw_tokens = estimate_total_tokens(original)
    input_hash = _content_hash(original)

    def finish(
        output: list[dict[str, Any]],
        *,
        decision: str,
        final_budget: int,
        attempts: list[dict[str, Any]],
        reasons: Sequence[str] = (),
        exact_identity: bool = False,
    ) -> tuple[list[dict[str, Any]], GuardedSelectionReceipt]:
        delivered = estimate_total_tokens(output)
        receipt = GuardedSelectionReceipt(
            decision=decision,
            requested_budget=int(token_budget),
            final_budget=int(final_budget),
            raw_tokens=raw_tokens,
            delivered_tokens=delivered,
            required_scope=required_scope,
            exact_identity=exact_identity,
            budget_compliant=delivered <= token_budget,
            input_sha256=input_hash,
            output_sha256=_content_hash(output),
            attempts=tuple(attempts),
            reasons=tuple(str(reason) for reason in reasons),
        )
        return output, receipt

    if not original:
        return finish(
            [],
            decision="bypass_empty",
            final_budget=token_budget,
            attempts=[],
            exact_identity=True,
        )

    if not query:
        return finish(
            original,
            decision="bypass_no_query",
            final_budget=token_budget,
            attempts=[],
            reasons=("no query-conditioned compression was possible",),
            exact_identity=True,
        )

    if raw_tokens <= token_budget:
        return finish(
            original,
            decision="bypass_already_fits",
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
    last_certificate: dict[str, Any] | None = None

    for attempt_index in range(max_expansions + 1):
        selected = list(
            selector(
                original,
                token_budget=current_budget,
                query=query,
            )
        )
        certificate = _extract_certificate(selected)
        attempts.append(_attempt_record(current_budget, selected, certificate))
        last_selected = selected
        last_certificate = certificate

        if _certificate_is_acceptable(certificate, required_scope):
            return finish(
                selected,
                decision=("compressed" if attempt_index == 0 else "expanded"),
                final_budget=current_budget,
                attempts=attempts,
                reasons=(),
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

    failure_reasons = (
        list(last_certificate.get("reasons") or [])
        if last_certificate
        else ["selector emitted no sufficiency certificate"]
    )
    actual_scope = (
        str(last_certificate.get("scope") or "unavailable")
        if last_certificate
        else "unavailable"
    )
    if not _scope_satisfies(actual_scope, required_scope):
        failure_reasons.append(
            f"certificate scope {actual_scope!r} does not satisfy "
            f"required scope {required_scope!r}"
        )

    if fallback == "original":
        return finish(
            original,
            decision="bypass_uncertified",
            final_budget=current_budget,
            attempts=attempts,
            reasons=failure_reasons,
            exact_identity=True,
        )

    if fallback == "selected":
        return finish(
            last_selected,
            decision="uncertified_budget_enforced",
            final_budget=current_budget,
            attempts=attempts,
            reasons=failure_reasons,
            exact_identity=_content_hash(last_selected) == input_hash,
        )

    raise SufficiencyNotCertifiedError(
        json.dumps(
            {
                "decision": "raise_uncertified",
                "requested_budget": token_budget,
                "final_budget": current_budget,
                "required_scope": required_scope,
                "attempts": attempts,
                "reasons": failure_reasons,
            },
            sort_keys=True,
        )
    )
