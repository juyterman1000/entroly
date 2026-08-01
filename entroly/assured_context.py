"""Public opt-in API for assurance-gated context selection.

This is deliberately separate from the compatibility SDK path. Structural
assurance is available from exact candidate/span evidence. Semantic assurance
requires a validated held-out calibration profile; without one the controller
expands and ultimately returns the original context in quality-first mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from . import audited_qccr
from .guarded_selection import GuardedSelectionReceipt, select_guarded
from .sufficiency_calibration import CalibrationProfile, certify_with_profile
from .sufficiency_contract import CertificateScope


@dataclass(frozen=True)
class AssuredSelection:
    selected: tuple[dict[str, Any], ...]
    receipt: GuardedSelectionReceipt
    audits: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [dict(fragment) for fragment in self.selected],
            "receipt": self.receipt.to_dict(),
            "audits": [dict(audit) for audit in self.audits],
        }


def select_assured(
    fragments: Sequence[dict[str, Any]],
    token_budget: int,
    query: str,
    *,
    required_scope: str | CertificateScope = CertificateScope.SEMANTIC,
    calibration_profile: CalibrationProfile | None = None,
    fallback: str = "original",
    max_expansions: int = 2,
    expansion_factor: float = 1.5,
    min_expansion_tokens: int = 64,
) -> AssuredSelection:
    """Select under an explicit evidence-scope and fallback contract."""
    audits: list[dict[str, Any]] = []

    def selector(
        current: Sequence[dict[str, Any]], *, token_budget: int, query: str
    ) -> list[dict[str, Any]]:
        envelope = audited_qccr.select_with_audit(current, token_budget, query)
        selected = [dict(fragment) for fragment in envelope["selected"]]
        metrics = dict(envelope["metrics"])
        if calibration_profile is not None:
            certificate = certify_with_profile(metrics, calibration_profile)
            metrics.update(certificate.to_dict())
            envelope["metrics"] = metrics
            if selected:
                selected[0]["_sufficiency"] = metrics
        audits.append(envelope)
        return selected

    selected, receipt = select_guarded(
        fragments,
        token_budget,
        query,
        max_expansions=max_expansions,
        expansion_factor=expansion_factor,
        min_expansion_tokens=min_expansion_tokens,
        required_scope=required_scope,
        fallback=fallback,
        selector=selector,
    )
    return AssuredSelection(tuple(selected), receipt, tuple(audits))


def select_structurally_assured(
    fragments: Sequence[dict[str, Any]],
    token_budget: int,
    query: str,
    *,
    fallback: str = "original",
    max_expansions: int = 2,
) -> AssuredSelection:
    """Accept exact candidate-unit assurance without a semantic claim."""
    return select_assured(
        fragments,
        token_budget,
        query,
        required_scope=CertificateScope.CANDIDATE_UNITS,
        fallback=fallback,
        max_expansions=max_expansions,
    )
