"""Typed models for Entroly Context Receipts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "context-receipt.v1"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _str_or_default(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _int_or_default(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return coerced if math.isfinite(coerced) else default


def _bool_or_default(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def _mapping_items(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _list_or_empty(value) if isinstance(item, Mapping)]


def text_fingerprint(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class DocumentRecord:
    document_id: str
    source_path: str
    title: str
    fingerprint: str
    token_count: int
    byte_count: int
    chunk_ids: list[str]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | object) -> "DocumentRecord":
        payload = data if isinstance(data, Mapping) else {}
        return cls(
            document_id=_str_or_default(payload.get("document_id")),
            source_path=_str_or_default(payload.get("source_path")),
            title=_str_or_default(payload.get("title")),
            fingerprint=_str_or_default(payload.get("fingerprint")),
            token_count=_int_or_default(payload.get("token_count")),
            byte_count=_int_or_default(payload.get("byte_count")),
            chunk_ids=_list_of_str(payload.get("chunk_ids", [])),
        )


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    source_path: str
    title: str
    section_heading: str | None
    page_number: int | None
    chunk_index: int
    byte_start: int
    byte_end: int
    token_start: int
    token_end: int
    token_count: int
    fingerprint: str
    text: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | object) -> "DocumentChunk":
        payload = data if isinstance(data, Mapping) else {}
        return cls(
            chunk_id=_str_or_default(payload.get("chunk_id")),
            document_id=_str_or_default(payload.get("document_id")),
            source_path=_str_or_default(payload.get("source_path")),
            title=_str_or_default(payload.get("title")),
            section_heading=_optional_str(payload.get("section_heading")),
            page_number=(
                None
                if payload.get("page_number") is None
                else _int_or_default(payload.get("page_number"))
            ),
            chunk_index=_int_or_default(payload.get("chunk_index")),
            byte_start=_int_or_default(payload.get("byte_start")),
            byte_end=_int_or_default(payload.get("byte_end")),
            token_start=_int_or_default(payload.get("token_start")),
            token_end=_int_or_default(payload.get("token_end")),
            token_count=_int_or_default(payload.get("token_count")),
            fingerprint=_str_or_default(payload.get("fingerprint")),
            text=_str_or_default(payload.get("text")),
        )


@dataclass
class ContextIndex:
    schema_version: str
    documents: list[DocumentRecord]
    chunks: list[DocumentChunk]
    chunk_token_limit: int
    chunk_overlap: int
    source_fingerprints: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | object) -> "ContextIndex":
        payload = data if isinstance(data, Mapping) else {}
        return cls(
            schema_version=_str_or_default(
                payload.get("schema_version"), SCHEMA_VERSION
            ),
            documents=[
                DocumentRecord.from_dict(d)
                for d in _mapping_items(payload.get("documents", []))
            ],
            chunks=[
                DocumentChunk.from_dict(c)
                for c in _mapping_items(payload.get("chunks", []))
            ],
            chunk_token_limit=_int_or_default(payload.get("chunk_token_limit"), 360),
            chunk_overlap=_int_or_default(payload.get("chunk_overlap"), 32),
            source_fingerprints=_dict_or_empty(payload.get("source_fingerprints", {})),
        )


@dataclass
class RankedChunk:
    chunk_id: str
    lexical_score: float
    semantic_score: float
    rerank_score: float
    final_score: float
    reasons: list[str]


@dataclass
class DependencyLink:
    source_chunk_id: str
    target_chunk_id: str | None
    relation_type: str
    evidence: str
    source_document_id: str
    target_document_id: str | None
    resolved: bool
    warning: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DependencyLink":
        return cls(
            source_chunk_id=_str_or_default(data.get("source_chunk_id")),
            target_chunk_id=_optional_str(data.get("target_chunk_id")),
            relation_type=_str_or_default(data.get("relation_type")),
            evidence=_str_or_default(data.get("evidence")),
            source_document_id=_str_or_default(data.get("source_document_id")),
            target_document_id=_optional_str(data.get("target_document_id")),
            resolved=_bool_or_default(data.get("resolved", False)),
            warning=_optional_str(data.get("warning")),
        )


@dataclass
class SelectedContextItem:
    chunk_id: str
    source_path: str
    section_heading: str | None
    page_number: int | None
    byte_start: int
    byte_end: int
    token_start: int
    token_end: int
    token_count: int
    score: float
    reasons: list[str]
    dependencies_included: list[str]
    dependencies_missing: list[str]
    fingerprint: str
    text: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectedContextItem":
        return cls(
            chunk_id=_str_or_default(data.get("chunk_id")),
            source_path=_str_or_default(data.get("source_path")),
            section_heading=_optional_str(data.get("section_heading")),
            page_number=(
                None
                if data.get("page_number") is None
                else _int_or_default(data.get("page_number"))
            ),
            byte_start=_int_or_default(data.get("byte_start")),
            byte_end=_int_or_default(data.get("byte_end")),
            token_start=_int_or_default(data.get("token_start")),
            token_end=_int_or_default(data.get("token_end")),
            token_count=_int_or_default(data.get("token_count")),
            score=_float_or_default(data.get("score")),
            reasons=_list_of_str(data.get("reasons", [])),
            dependencies_included=_list_of_str(data.get("dependencies_included", [])),
            dependencies_missing=_list_of_str(data.get("dependencies_missing", [])),
            fingerprint=_str_or_default(data.get("fingerprint")),
            text=_str_or_default(data.get("text")),
        )


@dataclass
class OmittedContextItem:
    chunk_id: str
    source_path: str
    section_heading: str | None
    page_number: int | None
    token_count: int
    score: float
    reasons: list[str]
    omission_reason: str
    fingerprint: str
    text_preview: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OmittedContextItem":
        return cls(
            chunk_id=_str_or_default(data.get("chunk_id")),
            source_path=_str_or_default(data.get("source_path")),
            section_heading=_optional_str(data.get("section_heading")),
            page_number=(
                None
                if data.get("page_number") is None
                else _int_or_default(data.get("page_number"))
            ),
            token_count=_int_or_default(data.get("token_count")),
            score=_float_or_default(data.get("score")),
            reasons=_list_of_str(data.get("reasons", [])),
            omission_reason=_str_or_default(data.get("omission_reason")),
            fingerprint=_str_or_default(data.get("fingerprint")),
            text_preview=_str_or_default(data.get("text_preview")),
        )


@dataclass
class CompressionRatio:
    source_tokens: int
    selected_tokens: int
    tokens_saved: int
    selected_to_source_ratio: float
    source_to_selected_ratio: float
    reduction_pct: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompressionRatio":
        return cls(
            source_tokens=_int_or_default(data.get("source_tokens")),
            selected_tokens=_int_or_default(data.get("selected_tokens")),
            tokens_saved=_int_or_default(data.get("tokens_saved")),
            selected_to_source_ratio=_float_or_default(
                data.get("selected_to_source_ratio")
            ),
            source_to_selected_ratio=_float_or_default(
                data.get("source_to_selected_ratio"), 1.0
            ),
            reduction_pct=_float_or_default(data.get("reduction_pct")),
        )


@dataclass
class ContextReceipt:
    receipt_id: str
    schema_version: str
    query: str
    token_budget: int
    selected_context: list[SelectedContextItem]
    omitted_context: list[OmittedContextItem]
    dependency_links: list[DependencyLink]
    ranking_reasons: dict[str, list[str]]
    compression_ratio: CompressionRatio
    source_fingerprints: dict[str, Any]
    risk_summary: dict[str, Any]
    warnings: list[str]
    reproducibility_hash: str
    outcome_links: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | object) -> "ContextReceipt":
        payload = data if isinstance(data, Mapping) else {}
        return cls(
            receipt_id=_str_or_default(payload.get("receipt_id")),
            schema_version=_str_or_default(
                payload.get("schema_version"), SCHEMA_VERSION
            ),
            query=_str_or_default(payload.get("query")),
            token_budget=_int_or_default(payload.get("token_budget")),
            selected_context=[
                SelectedContextItem.from_dict(x)
                for x in _mapping_items(payload.get("selected_context", []))
            ],
            omitted_context=[
                OmittedContextItem.from_dict(x)
                for x in _mapping_items(payload.get("omitted_context", []))
            ],
            dependency_links=[
                DependencyLink.from_dict(x)
                for x in _mapping_items(payload.get("dependency_links", []))
            ],
            ranking_reasons={
                str(k): _list_of_str(v)
                for k, v in _dict_or_empty(payload.get("ranking_reasons", {})).items()
            },
            compression_ratio=CompressionRatio.from_dict(
                _dict_or_empty(payload.get("compression_ratio", {}))
            ),
            source_fingerprints=_dict_or_empty(payload.get("source_fingerprints", {})),
            risk_summary=_dict_or_empty(payload.get("risk_summary", {})),
            warnings=_list_of_str(payload.get("warnings", [])),
            reproducibility_hash=_str_or_default(payload.get("reproducibility_hash")),
            outcome_links=_mapping_items(payload.get("outcome_links", [])),
        )
