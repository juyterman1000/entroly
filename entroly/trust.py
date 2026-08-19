"""Thin Python surface for Entroly's shared Rust Trust Engine.

No evidence scoring, profile policy, commitment, or criticality semantics live
here. Python validates native capability, converts JSON, and returns Rust-owned
results.
"""

from __future__ import annotations

import json
from typing import Any

from .native_status import native_status, native_status_message

_TRUST_STATUS = native_status(("TrustEngine",))
_RustTrustEngine = (
    getattr(_TRUST_STATUS.module, "TrustEngine", None) if _TRUST_STATUS.ok else None
)


class TrustEngineUnavailableError(RuntimeError):
    """Raised when the native shared Trust Engine is unavailable."""


def _require_native() -> type:
    if _RustTrustEngine is None:
        raise TrustEngineUnavailableError(
            native_status_message(_TRUST_STATUS, feature="the Entroly Trust Engine")
        )
    return _RustTrustEngine


class TrustEngine:
    """Evidence-bounded trust facade backed entirely by shared Rust semantics."""

    __slots__ = ("_inner",)

    def __init__(self, profile: str = "rag") -> None:
        self._inner = _require_native()(profile)

    def assess_claim(self, evidence: str, claim: str) -> dict[str, Any]:
        return json.loads(str(self._inner.assess_claim_json(evidence, claim)))

    def file_criticality(self, path: str) -> str:
        return str(self._inner.file_criticality(path))

    def has_safety_signal(self, content: str) -> bool:
        return bool(self._inner.has_safety_signal(content))


__all__ = ["TrustEngine", "TrustEngineUnavailableError"]
