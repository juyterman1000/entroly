"""Ergonomic opt-in SDK for evidence-assured text and message compression.

The compatibility SDK remains unchanged. These functions return both output and
an auditable receipt, and default to semantic, quality-first behavior: without a
validated semantic calibration profile, uncertain compression returns the
original input rather than silently weakening evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .assurance_telemetry import AssuranceLedger
from .assured_context import AssuredSelection, select_assured
from .domain_assurance import DomainAssuredCompression, compress_domain_assured
from .evidence_locked_compression import detect_heavy_content_type
from .sufficiency_calibration import CalibrationProfile
from .sufficiency_contract import CertificateScope, parse_scope
from .text_features import protected_input_reason

_DOMAIN_TYPES = {
    "json", "json_text", "logs", "log", "shell", "traceback",
    "stacktrace", "code", "table",
}


def _estimate_tokens(text: str) -> int:
    return max(0, math.ceil(len(text) / 4))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fragment_text(text: str, *, source: str, max_chars: int = 1600) -> list[dict[str, Any]]:
    if max_chars < 128:
        raise ValueError("max_chars must be at least 128")
    fragments: list[dict[str, Any]] = []
    start = 0
    encoded_prefix = 0
    while start < len(text):
        provisional_end = min(len(text), start + max_chars)
        end = provisional_end
        if provisional_end < len(text):
            window = text[start:provisional_end]
            boundaries = [
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
            ]
            boundary = max(boundaries)
            if boundary >= max_chars // 2:
                marker_width = 2 if window[boundary : boundary + 2] in {". ", "? ", "! ", "\n\n"} else 1
                end = start + boundary + marker_width
        chunk = text[start:end]
        chunk_bytes = chunk.encode("utf-8")
        fragments.append(
            {
                "fragment_id": f"{source}::{encoded_prefix}",
                "source": source,
                "content": chunk,
                "start_byte": encoded_prefix,
                "end_byte": encoded_prefix + len(chunk_bytes),
                "token_count": _estimate_tokens(chunk),
            }
        )
        encoded_prefix += len(chunk_bytes)
        start = end
    return fragments


@dataclass(frozen=True)
class AssuredTextResult:
    text: str
    mode: str
    receipt: dict[str, Any]
    audits: tuple[dict[str, Any], ...] = ()

    @property
    def changed(self) -> bool:
        return not bool(self.receipt.get("exact_identity"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode,
            "receipt": dict(self.receipt),
            "audits": [dict(audit) for audit in self.audits],
        }


@dataclass(frozen=True)
class AssuredMessagesResult:
    messages: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]
    audits: tuple[dict[str, Any], ...]
    original_tokens: int
    delivered_tokens: int
    budget_compliant: bool

    @property
    def changed(self) -> bool:
        return not bool(self.receipt.get("exact_identity"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [dict(message) for message in self.messages],
            "receipt": dict(self.receipt),
            "audits": [dict(audit) for audit in self.audits],
            "original_tokens": self.original_tokens,
            "delivered_tokens": self.delivered_tokens,
            "budget_compliant": self.budget_compliant,
        }


def _normalise_scope(scope: str | CertificateScope) -> CertificateScope:
    if isinstance(scope, CertificateScope):
        return scope
    parsed = parse_scope(scope)
    if parsed is CertificateScope.UNAVAILABLE and scope != "unavailable":
        raise ValueError(f"unknown required_scope: {scope!r}")
    return parsed


def _identity_text_result(
    text: str,
    *,
    budget: int,
    reason: str,
    content_type: str,
    source_path: str | Path | None,
    ledger: AssuranceLedger | None,
    started: float,
) -> AssuredTextResult:
    tokens = _estimate_tokens(text)
    digest = _sha256(text)
    decision = {
        "already_fits": "BYPASS_ALREADY_FITS",
        "instruction_file_full_fidelity": "BYPASS_INSTRUCTION_FILE",
        "short_input_full_fidelity": "BYPASS_SHORT_INPUT",
    }.get(reason, "BYPASS_PROTECTED")
    receipt: dict[str, Any] = {
        "decision": decision,
        "requested_budget": budget,
        "raw_tokens": tokens,
        "delivered_tokens": tokens,
        "exact_identity": True,
        "budget_compliant": tokens <= budget,
        "input_sha256": digest,
        "output_sha256": digest,
        "attempts": [],
        "reasons": [reason],
        "content_type": content_type,
    }
    if source_path is not None:
        receipt["source_path_sha256"] = _sha256(str(source_path))
    latency = (time.perf_counter() - started) * 1000
    if ledger is not None:
        ledger.record_selection(
            receipt,
            latency_ms=latency,
            metadata={
                "sdk_surface": "compress_assured",
                "content_type": content_type,
                "protected_reason": reason,
            },
        )
    return AssuredTextResult(text, "identity", receipt)


def compress_assured(
    text: str,
    *,
    query: str,
    budget: int,
    content_type: str = "auto",
    required_scope: str | CertificateScope = CertificateScope.SEMANTIC,
    calibration_profile: CalibrationProfile | None = None,
    fallback: str = "original",
    max_expansions: int = 2,
    ledger: AssuranceLedger | None = None,
    source_path: str | Path | None = None,
    protect_short_inputs: bool = True,
) -> AssuredTextResult:
    """Compress text under an explicit assurance and fallback contract."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    required = _normalise_scope(required_scope)
    started = time.perf_counter()
    detected = detect_heavy_content_type(text) if content_type == "auto" else content_type
    detected = str(detected or "text").lower()
    if detected == "stacktrace":
        detected = "traceback"

    protection = protected_input_reason(
        text,
        budget_tokens=budget,
        content_type=detected,
        source_path=source_path,
    )
    if protection == "short_input_full_fidelity" and not protect_short_inputs:
        protection = None
    if protection is not None:
        return _identity_text_result(
            text,
            budget=budget,
            reason=protection,
            content_type=detected,
            source_path=source_path,
            ledger=ledger,
            started=started,
        )

    if detected in _DOMAIN_TYPES and required.value in {
        CertificateScope.CANDIDATE_UNITS.value,
        CertificateScope.FILE_RETRIEVAL.value,
        CertificateScope.OPTIMIZER_PROXY.value,
        CertificateScope.UNAVAILABLE.value,
    }:
        domain: DomainAssuredCompression = compress_domain_assured(
            text,
            query=query,
            budget_tokens=budget,
            fallback="compressed" if fallback == "selected" else fallback,
        )
        receipt = domain.receipt.to_dict()
        latency = (time.perf_counter() - started) * 1000
        if ledger is not None:
            ledger.record_domain(
                receipt,
                latency_ms=latency,
                metadata={"sdk_surface": "compress_assured", "content_type": detected},
            )
        return AssuredTextResult(domain.text, "domain", receipt)

    fragments = _fragment_text(text, source="sdk:text")
    selection: AssuredSelection = select_assured(
        fragments,
        budget,
        query,
        required_scope=required,
        calibration_profile=calibration_profile,
        fallback=fallback,
        max_expansions=max_expansions,
    )
    output = "\n".join(
        str(fragment.get("content") or "") for fragment in selection.selected
    )
    receipt = selection.receipt.to_dict()
    latency = (time.perf_counter() - started) * 1000
    if ledger is not None:
        ledger.record_selection(
            receipt,
            latency_ms=latency,
            metadata={"sdk_surface": "compress_assured", "content_type": detected},
        )
    return AssuredTextResult(output, "audited_qccr", receipt, selection.audits)


def compress_file_assured(
    path: str | Path,
    *,
    query: str,
    budget: int,
    workspace: str | Path | None = None,
    max_file_bytes: int = 8_000_000,
    content_type: str = "auto",
    required_scope: str | CertificateScope = CertificateScope.SEMANTIC,
    calibration_profile: CalibrationProfile | None = None,
    fallback: str = "original",
    max_expansions: int = 2,
    ledger: AssuranceLedger | None = None,
    protect_short_inputs: bool = True,
) -> AssuredTextResult:
    """Read and compress a UTF-8 file under workspace and preservation guards."""
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")
    raw_path = Path(path).expanduser()
    if workspace is not None:
        root = Path(workspace).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace must be an existing directory")
        candidate = (root / raw_path).resolve(strict=True) if not raw_path.is_absolute() else raw_path.resolve(strict=True)
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("file path escapes workspace") from exc
        source_path: str | Path = relative
    else:
        candidate = raw_path.resolve(strict=True)
        source_path = candidate.name
    if not candidate.is_file():
        raise ValueError("path must be an existing file")
    size = candidate.stat().st_size
    if size > max_file_bytes:
        raise ValueError(f"file exceeds {max_file_bytes:,} bytes")
    try:
        text = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("file must be UTF-8 text") from exc
    result = compress_assured(
        text,
        query=query,
        budget=budget,
        content_type=content_type,
        required_scope=required_scope,
        calibration_profile=calibration_profile,
        fallback=fallback,
        max_expansions=max_expansions,
        ledger=ledger,
        source_path=source_path,
        protect_short_inputs=protect_short_inputs,
    )
    receipt = dict(result.receipt)
    receipt["source_file_sha256"] = _sha256(text)
    receipt["source_bytes"] = size
    return AssuredTextResult(result.text, result.mode, receipt, result.audits)


def _infer_query(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _message_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        _estimate_tokens(str(message.get("content") or ""))
        for message in messages
        if isinstance(message.get("content"), str)
    )


def compress_messages_assured(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    query: str | None = None,
    preserve_last_n: int = 4,
    required_scope: str | CertificateScope = CertificateScope.SEMANTIC,
    calibration_profile: CalibrationProfile | None = None,
    fallback: str = "original",
    max_expansions: int = 2,
    ledger: AssuranceLedger | None = None,
) -> AssuredMessagesResult:
    """Compress older messages while preserving recent messages verbatim."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if preserve_last_n < 1:
        raise ValueError("preserve_last_n must be positive")
    original = [dict(message) for message in messages]
    original_tokens = _message_tokens(original)
    if not original:
        return AssuredMessagesResult((), {"decision": "BYPASS_EMPTY", "exact_identity": True}, (), 0, 0, True)
    if original_tokens <= budget:
        receipt = {
            "decision": "BYPASS_ALREADY_FITS",
            "requested_budget": budget,
            "raw_tokens": original_tokens,
            "delivered_tokens": original_tokens,
            "exact_identity": True,
            "budget_compliant": True,
            "input_sha256": _sha256(json.dumps(original, sort_keys=True, ensure_ascii=False)),
            "output_sha256": _sha256(json.dumps(original, sort_keys=True, ensure_ascii=False)),
            "attempts": [],
            "reasons": ["identity dominates compression when messages already fit"],
        }
        if ledger is not None:
            ledger.record_selection(receipt, latency_ms=0.0, metadata={"sdk_surface": "compress_messages_assured"})
        return AssuredMessagesResult(tuple(original), receipt, (), original_tokens, original_tokens, True)

    effective_preserve = min(preserve_last_n, len(original))
    recent = original[-effective_preserve:]
    older = original[:-effective_preserve]
    recent_tokens = _message_tokens(recent)
    if not older or recent_tokens >= budget:
        receipt = {
            "decision": "BYPASS_UNCERTIFIED",
            "requested_budget": budget,
            "raw_tokens": original_tokens,
            "delivered_tokens": original_tokens,
            "exact_identity": True,
            "budget_compliant": original_tokens <= budget,
            "input_sha256": _sha256(json.dumps(original, sort_keys=True, ensure_ascii=False)),
            "output_sha256": _sha256(json.dumps(original, sort_keys=True, ensure_ascii=False)),
            "attempts": [],
            "reasons": ["recent verbatim messages consume the available budget"],
        }
        if ledger is not None:
            ledger.record_selection(receipt, latency_ms=0.0, metadata={"sdk_surface": "compress_messages_assured"})
        return AssuredMessagesResult(tuple(original), receipt, (), original_tokens, original_tokens, original_tokens <= budget)

    conditioned_query = (query or _infer_query(original)).strip()
    fragments: list[dict[str, Any]] = []
    source_to_message: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, message in enumerate(older):
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        role = str(message.get("role") or "unknown")
        source = f"message:{index}:{role}"
        source_to_message[source] = (index, message)
        fragments.append(
            {
                "fragment_id": source,
                "source": source,
                "content": content,
                "start_byte": 0,
                "end_byte": len(content.encode("utf-8")),
                "token_count": _estimate_tokens(content),
            }
        )

    started = time.perf_counter()
    selection = select_assured(
        fragments,
        max(1, budget - recent_tokens),
        conditioned_query,
        required_scope=_normalise_scope(required_scope),
        calibration_profile=calibration_profile,
        fallback=fallback,
        max_expansions=max_expansions,
    )
    rebuilt: list[tuple[int, dict[str, Any]]] = []
    for fragment in selection.selected:
        source = str(fragment.get("source") or "")
        mapped = source_to_message.get(source)
        if mapped is None:
            continue
        index, original_message = mapped
        updated = dict(original_message)
        updated["content"] = str(fragment.get("content") or "")
        rebuilt.append((index, updated))
    rebuilt.sort(key=lambda item: item[0])
    output = [message for _, message in rebuilt] + recent
    delivered_tokens = _message_tokens(output)
    receipt = selection.receipt.to_dict()
    receipt["overall_original_tokens"] = original_tokens
    receipt["overall_delivered_tokens"] = delivered_tokens
    receipt["overall_budget_compliant"] = delivered_tokens <= budget
    latency = (time.perf_counter() - started) * 1000
    if ledger is not None:
        ledger.record_selection(
            receipt,
            latency_ms=latency,
            metadata={"sdk_surface": "compress_messages_assured", "preserve_last_n": effective_preserve},
        )
    return AssuredMessagesResult(
        tuple(output),
        receipt,
        selection.audits,
        original_tokens,
        delivered_tokens,
        delivered_tokens <= budget,
    )


__all__ = [
    "AssuredMessagesResult",
    "AssuredTextResult",
    "compress_assured",
    "compress_file_assured",
    "compress_messages_assured",
]
