"""Typed models for Entroly Context Receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "context-receipt.v1"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


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
    def from_dict(cls, data: dict[str, Any]) -> "DocumentRecord":
        return cls(**data)


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
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


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
    def from_dict(cls, data: dict[str, Any]) -> "ContextIndex":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            documents=[DocumentRecord.from_dict(d) for d in data.get("documents", [])],
            chunks=[DocumentChunk.from_dict(c) for c in data.get("chunks", [])],
            chunk_token_limit=int(data.get("chunk_token_limit", 360)),
            chunk_overlap=int(data.get("chunk_overlap", 32)),
            source_fingerprints=dict(data.get("source_fingerprints", {})),
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


@dataclass
class CompressionRatio:
    source_tokens: int
    selected_tokens: int
    tokens_saved: int
    selected_to_source_ratio: float
    source_to_selected_ratio: float
    reduction_pct: float


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
    def from_dict(cls, data: dict[str, Any]) -> "ContextReceipt":
        return cls(
            receipt_id=data["receipt_id"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            query=data["query"],
            token_budget=int(data["token_budget"]),
            selected_context=[SelectedContextItem(**x) for x in data.get("selected_context", [])],
            omitted_context=[OmittedContextItem(**x) for x in data.get("omitted_context", [])],
            dependency_links=[DependencyLink(**x) for x in data.get("dependency_links", [])],
            ranking_reasons={str(k): list(v) for k, v in data.get("ranking_reasons", {}).items()},
            compression_ratio=CompressionRatio(**data["compression_ratio"]),
            source_fingerprints=dict(data.get("source_fingerprints", {})),
            risk_summary=dict(data.get("risk_summary", {})),
            warnings=list(data.get("warnings", [])),
            reproducibility_hash=data["reproducibility_hash"],
            outcome_links=list(data.get("outcome_links", [])),
        )
