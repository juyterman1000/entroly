"""Fail-closed validation for workload-specific context compression.

The existing evidence-locked compressor owns extraction. This module owns the
caller contract: validate the emitted representation with a domain oracle,
accept only when it remains useful and budget-positive, otherwise return the
original or explicitly label a hard-budget result uncertified.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable

from .evidence_locked_compression import (
    CompressionResult,
    compress_evidence_locked,
    estimate_tokens,
)
from .text_features import query_terms, text_terms

_CRITICAL_RE = re.compile(
    r"(?i)\b(?:error|fatal|failed?|exception|traceback|panic|denied|"
    r"timeout|segfault|assertion|exit(?:\s+code)?\s*[:=]?\s*[1-9][0-9]*)\b"
)
_SYMBOL_RE = re.compile(
    r"(?m)^\s*(?:async\s+)?(?:def|class|fn|struct|enum|interface|function|"
    r"trait|type)\s+([A-Za-z_][A-Za-z0-9_]*)"
)

class DomainDecision(str, Enum):
    BYPASS_ALREADY_FITS = "bypass_already_fits"
    BYPASS_UNCHANGED = "bypass_unchanged"
    COMPRESSED_VALIDATED = "compressed_validated"
    BYPASS_INVALID = "bypass_invalid"
    BYPASS_NEGATIVE_SAVINGS = "bypass_negative_savings"
    UNCERTIFIED_BUDGET_ENFORCED = "uncertified_budget_enforced"


@dataclass(frozen=True)
class DomainValidation:
    content_type: str
    valid: bool
    query_coverage: float
    critical_items: int
    critical_items_retained: int
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DomainAssuranceReceipt:
    decision: DomainDecision
    content_type: str
    requested_budget: int
    original_tokens: int
    emitted_tokens: int
    budget_compliant: bool
    exact_identity: bool
    input_sha256: str
    output_sha256: str
    validation: DomainValidation
    compressor_receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        payload["validation"] = self.validation.to_dict()
        return payload


@dataclass(frozen=True)
class DomainAssuredCompression:
    text: str
    receipt: DomainAssuranceReceipt

    @property
    def changed(self) -> bool:
        return not self.receipt.exact_identity


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coverage(query: str, emitted: str) -> float:
    terms = query_terms(query)
    if not terms:
        return 1.0
    emitted_terms = text_terms(emitted)
    return len(terms & emitted_terms) / len(terms)


def _normalise_line(line: str) -> str:
    return " ".join(line.lower().split())


def _critical_lines(text: str) -> list[str]:
    return [
        _normalise_line(line)
        for line in text.splitlines()
        if _CRITICAL_RE.search(line) and line.strip()
    ]


def _walk_json(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, (*path, str(index)))
    else:
        yield path, value


def _json_relevant_values(value: Any, query: str) -> list[str]:
    terms = query_terms(query)
    if not terms:
        return []
    relevant: list[str] = []
    for path, scalar in _walk_json(value):
        rendered = str(scalar)
        haystack = " ".join((*path, rendered)).lower()
        if any(term in haystack for term in terms):
            normalised = " ".join(rendered.lower().split())
            if normalised and len(normalised) <= 500:
                relevant.append(normalised)
    return list(dict.fromkeys(relevant))[:64]


def _parse_json_documents(text: str) -> list[Any]:
    """Parse one or more whitespace-separated JSON values deterministically."""
    decoder = json.JSONDecoder()
    documents: list[Any] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, end = decoder.raw_decode(text, index)
        documents.append(value)
        index = end
    if not documents:
        raise ValueError("no JSON documents found")
    return documents


def _validate_json(original: str, emitted: str, query: str, *, content_type: str = "json") -> DomainValidation:
    reasons: list[str] = []
    checks = {
        "original_json_valid": False,
        "emitted_json_valid": False,
        "summary_marker": False,
    }
    try:
        original_documents = _parse_json_documents(original)
        checks["original_json_valid"] = True
    except (ValueError, json.JSONDecodeError):
        reasons.append("original payload was not valid JSON or a JSON document stream")
        return DomainValidation(
            content_type, False, _coverage(query, emitted), 0, 0, checks, tuple(reasons)
        )
    try:
        emitted_documents = _parse_json_documents(emitted)
        checks["emitted_json_valid"] = True
    except (ValueError, json.JSONDecodeError):
        reasons.append("compressed payload is not valid JSON")
        return DomainValidation(
            content_type, False, _coverage(query, emitted), 0, 0, checks, tuple(reasons)
        )

    checks["summary_marker"] = any(
        isinstance(document, dict) and isinstance(document.get("elc"), str)
        for document in emitted_documents
    )
    if not checks["summary_marker"]:
        reasons.append("compressed JSON is missing the evidence-locked summary marker")

    relevant: list[str] = []
    for document in original_documents:
        relevant.extend(_json_relevant_values(document, query))
    relevant = list(dict.fromkeys(relevant))[:64]
    rendered = json.dumps(
        emitted_documents, ensure_ascii=False, sort_keys=True
    ).casefold()
    retained = sum(value.casefold() in rendered for value in relevant)
    if relevant and retained == 0:
        reasons.append("no query-relevant JSON scalar survived")
    coverage = _coverage(query, emitted)
    if query and coverage < 0.5:
        reasons.append(f"query-term coverage is {coverage:.1%}")
    valid = all(checks.values()) and (not relevant or retained > 0) and coverage >= 0.5
    return DomainValidation(
        content_type,
        valid,
        coverage,
        len(relevant),
        retained,
        checks,
        tuple(reasons),
    )


def _validate_line_evidence(
    original: str, emitted: str, query: str, content_type: str
) -> DomainValidation:
    critical = _critical_lines(original)
    normalised_output = _normalise_line(emitted)
    retained = sum(line in normalised_output for line in critical)
    coverage = _coverage(query, emitted)
    checks = {
        "non_empty": bool(emitted.strip()),
        "critical_evidence_preserved": retained == len(critical),
        "query_coverage": not query or coverage >= 0.5,
    }
    reasons = []
    if not checks["non_empty"]:
        reasons.append("compressed output is empty")
    if not checks["critical_evidence_preserved"]:
        reasons.append(f"retained {retained}/{len(critical)} critical lines")
    if not checks["query_coverage"]:
        reasons.append(f"query-term coverage is {coverage:.1%}")
    return DomainValidation(
        content_type,
        all(checks.values()),
        coverage,
        len(critical),
        retained,
        checks,
        tuple(reasons),
    )


def _validate_code(original: str, emitted: str, query: str) -> DomainValidation:
    relevant_query_terms = query_terms(query)
    symbols = list(dict.fromkeys(_SYMBOL_RE.findall(original)))
    relevant_symbols = [
        symbol for symbol in symbols if symbol.casefold() in relevant_query_terms or any(
            term in symbol.casefold() for term in relevant_query_terms
        )
    ]
    retained = sum(re.search(rf"\b{re.escape(symbol)}\b", emitted) is not None for symbol in relevant_symbols)
    coverage = _coverage(query, emitted)
    checks = {
        "non_empty": bool(emitted.strip()),
        "query_symbols_preserved": not relevant_symbols or retained == len(relevant_symbols),
        "query_coverage": not query or coverage >= 0.5,
        "line_bounded": not emitted or emitted.endswith(("\n", ".", ";", "}", ")", "]")),
    }
    # Excerpts commonly end at a complete line without punctuation. Accept that
    # shape when every emitted line came from an original line verbatim.
    emitted_lines = [line.strip() for line in emitted.splitlines() if line.strip()]
    original_lines = {_normalise_line(line) for line in original.splitlines() if line.strip()}
    if emitted_lines and all(
        line.startswith("...") or _normalise_line(line) in original_lines
        for line in emitted_lines
    ):
        checks["line_bounded"] = True
    reasons = []
    if not checks["query_symbols_preserved"]:
        reasons.append(f"retained {retained}/{len(relevant_symbols)} query-relevant symbols")
    if not checks["query_coverage"]:
        reasons.append(f"query-term coverage is {coverage:.1%}")
    if not checks["line_bounded"]:
        reasons.append("compressed code appears to end inside an atomic line")
    if not checks["non_empty"]:
        reasons.append("compressed code is empty")
    return DomainValidation(
        "code",
        all(checks.values()),
        coverage,
        len(relevant_symbols),
        retained,
        checks,
        tuple(reasons),
    )


def validate_domain_output(
    original: str, emitted: str, *, content_type: str, query: str = ""
) -> DomainValidation:
    kind = content_type.lower()
    if kind in {"json", "json_text", "jsonl"}:
        try:
            _parse_json_documents(original)
        except (ValueError, json.JSONDecodeError):
            if kind == "json":
                return _validate_json(original, emitted, query, content_type=kind)
            return _validate_line_evidence(original, emitted, query, kind)
        return _validate_json(original, emitted, query, content_type=kind)
    if kind in {"log", "logs", "shell", "traceback"}:
        return _validate_line_evidence(original, emitted, query, kind)
    if kind == "code":
        return _validate_code(original, emitted, query)
    return _validate_line_evidence(original, emitted, query, kind or "text")


def compress_domain_assured(
    text: str,
    *,
    query: str = "",
    budget_tokens: int,
    fallback: str = "original",
) -> DomainAssuredCompression:
    """Compress and accept only after a domain-specific validation oracle."""
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    if fallback not in {"original", "compressed", "raise"}:
        raise ValueError("fallback must be 'original', 'compressed', or 'raise'")

    original_tokens = estimate_tokens(text)
    input_hash = _sha256(text)
    if original_tokens <= budget_tokens:
        validation = DomainValidation(
            "identity", True, 1.0, 0, 0, {"identity": True}, ("input already fits",)
        )
        receipt = DomainAssuranceReceipt(
            DomainDecision.BYPASS_ALREADY_FITS,
            "identity",
            budget_tokens,
            original_tokens,
            original_tokens,
            True,
            True,
            input_hash,
            input_hash,
            validation,
            {},
        )
        return DomainAssuredCompression(text, receipt)

    result: CompressionResult = compress_evidence_locked(
        text, query=query, budget_tokens=budget_tokens
    )
    emitted = result.compressed
    emitted_tokens = estimate_tokens(emitted)
    content_type = result.receipt.content_type
    validation = validate_domain_output(
        text, emitted, content_type=content_type, query=query
    )
    budget_compliant = emitted_tokens <= budget_tokens
    positive_savings = emitted_tokens < original_tokens

    if not result.changed:
        decision = DomainDecision.BYPASS_UNCHANGED
        output = text
    elif validation.valid and budget_compliant and positive_savings:
        decision = DomainDecision.COMPRESSED_VALIDATED
        output = emitted
    elif not positive_savings:
        decision = DomainDecision.BYPASS_NEGATIVE_SAVINGS
        output = text
    elif fallback == "compressed":
        decision = DomainDecision.UNCERTIFIED_BUDGET_ENFORCED
        output = emitted
    elif fallback == "raise":
        raise RuntimeError(
            json.dumps(
                {
                    "decision": "raise_invalid_domain_compression",
                    "content_type": content_type,
                    "budget_compliant": budget_compliant,
                    "positive_savings": positive_savings,
                    "validation": validation.to_dict(),
                },
                sort_keys=True,
            )
        )
    else:
        decision = DomainDecision.BYPASS_INVALID
        output = text

    exact_identity = output == text
    receipt = DomainAssuranceReceipt(
        decision=decision,
        content_type=content_type,
        requested_budget=budget_tokens,
        original_tokens=original_tokens,
        emitted_tokens=estimate_tokens(output),
        budget_compliant=estimate_tokens(output) <= budget_tokens,
        exact_identity=exact_identity,
        input_sha256=input_hash,
        output_sha256=_sha256(output),
        validation=validation,
        compressor_receipt=result.receipt.as_dict(),
    )
    return DomainAssuredCompression(output, receipt)
