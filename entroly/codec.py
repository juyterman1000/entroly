"""Common contract for content-specific compression codecs.

A compressed representation can always point back to the complete original
source. Recovery is delegated to Entroly's hardened scoped retrieval store,
which already provides inter-process locking, bounded persistence, atomic
updates and exact source spans.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from .source_span import SourceSpan

CODEC_CONTRACT_VERSION = "2"


def _to_bytes(data: bytes | str) -> bytes:
    """Encode text for hashing and length checks, tolerating lone surrogates.

    Text arriving from a model, a tool or a file on disk is not guaranteed to
    be well-formed Unicode. Plain ``.encode("utf-8")`` raises on an unpaired
    surrogate, which turns hostile input into a crash on the trust-critical
    path -- the digest is computed before any codec decision is made.

    ``surrogatepass`` round-trips such input exactly, so the byte-exact
    recovery contract still holds for content that is not valid UTF-8.
    """
    return data.encode("utf-8", "surrogatepass") if isinstance(data, str) else data


def content_digest(data: bytes | str) -> str:
    """Return ``sha256:<hex>`` over exact bytes. See `_to_bytes`."""
    return "sha256:" + hashlib.sha256(_to_bytes(data)).hexdigest()


@dataclass(frozen=True)
class RecoveryReference:
    """Verifiable pointer to the complete original source bytes."""

    digest: str
    byte_length: int
    item_count: int = 0
    note: str = ""
    receipt_id: str = ""
    span_id: str = ""

    def verify(self, recovered: bytes | str) -> bool:
        """True only when the bytes match BOTH the digest and the length.

        Checking the digest alone left `byte_length` unauthenticated: a store
        could return content of any size and still verify, so the field was
        decoration rather than a check. Both are compared here.
        """
        raw = _to_bytes(recovered)
        return len(raw) == self.byte_length and content_digest(raw) == self.digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "byte_length": self.byte_length,
            "item_count": self.item_count,
            "note": self.note,
            "receipt_id": self.receipt_id,
            "span_id": self.span_id,
        }


@dataclass(frozen=True)
class Representation:
    """One candidate representation of one context item."""

    representation_id: str
    source_id: str
    content_type: str
    text: str
    token_cost: int
    codec: str
    codec_version: str
    source_sha256: str
    protected_evidence: tuple[str, ...] = ()
    distortion_risk: float = 0.0
    recovery: RecoveryReference | None = None
    span: SourceSpan | None = None
    estimated_relevance: float | None = None
    dependency_coverage: float | None = None

    def verify_protected_evidence(self) -> tuple[str, ...]:
        return tuple(e for e in self.protected_evidence if e not in self.text)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "representation_id": self.representation_id,
            "source_id": self.source_id,
            "content_type": self.content_type,
            "token_cost": self.token_cost,
            "codec": self.codec,
            "codec_version": self.codec_version,
            "contract_version": CODEC_CONTRACT_VERSION,
            "source_sha256": self.source_sha256,
            "protected_evidence": list(self.protected_evidence),
            "distortion_risk": round(self.distortion_risk, 4),
            "recovery": self.recovery.to_dict() if self.recovery else None,
            "estimated_relevance": self.estimated_relevance,
            "dependency_coverage": self.dependency_coverage,
        }
        if self.span is not None:
            out["span"] = {
                "source_path": self.span.source_path,
                "source_digest": self.span.source_digest,
                "byte_start": self.span.byte_start,
                "byte_end": self.span.byte_end,
                "representation": self.span.representation,
            }
        return out


@dataclass
class SupportDecision:
    supported: bool
    confidence: float = 0.0
    reason: str = ""

    def __bool__(self) -> bool:
        return self.supported


@runtime_checkable
class ContextCodec(Protocol):
    name: str
    version: str

    def supports(self, text: str, content_type: str = "") -> SupportDecision: ...

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]: ...


class RecoveryStore:
    """Adapter over Entroly's hardened scoped compression-retrieval store.

    Every compressed representation stores one exact span covering the complete
    original source. This deliberately trades some local storage for a simple,
    auditable invariant: ``recover(reference) == original_text``.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        scope_id: str | None = None,
        max_bytes: int | None = None,
    ) -> None:
        from .compression_retrieval_store_secure import CompressionRetrievalStore

        scope = scope_id or f"codec:{Path.cwd().resolve()}"
        self._store = CompressionRetrievalStore(
            path,
            scope_id=scope,
            require_scope=True,
            max_bytes=max_bytes,
        )
        self._by_digest: dict[str, RecoveryReference] = {}

    def put(
        self,
        content: str,
        *,
        item_count: int = 0,
        note: str = "",
    ) -> RecoveryReference:
        digest = content_digest(content)
        line_count = max(1, len(content.splitlines(keepends=True)))
        receipt = {
            "codec_contract_version": CODEC_CONTRACT_VERSION,
            "original_tokens": estimate_tokens(content),
            "compressed_tokens": 0,
            "omitted_spans": [
                {
                    "start_line": 1,
                    "end_line": line_count,
                    "reason": "codec_original_source",
                }
            ],
        }
        stored = self._store.put(
            original_text=content,
            compressed_text="",
            receipt=receipt,
            metadata={
                "codec_recovery": True,
                "codec_digest": digest,
                "codec_byte_length": len(content.encode("utf-8")),
                "codec_item_count": int(item_count),
                "codec_note": note,
            },
        )
        if len(stored.spans) != 1:
            raise RuntimeError("codec recovery must persist exactly one source span")
        span = stored.spans[0]
        ref = RecoveryReference(
            digest=digest,
            byte_length=len(_to_bytes(content)),
            item_count=int(item_count),
            note=note,
            receipt_id=stored.receipt_id,
            span_id=span.span_id,
        )
        self._by_digest[digest] = ref
        return ref

    def get(self, ref: RecoveryReference | str) -> str | None:
        resolved = ref
        if isinstance(ref, str):
            resolved = self._by_digest.get(ref)
            if resolved is None:
                return None
        if not resolved.receipt_id or not resolved.span_id:
            return None
        span = self._store.get_span(resolved.receipt_id, resolved.span_id)
        return None if span is None else span.content

    def recover(self, ref: RecoveryReference) -> str:
        content = self.get(ref)
        if content is None:
            raise KeyError(f"no recovery entry for {ref.digest}")
        if not ref.verify(content):
            raise ValueError(
                f"recovered content does not match {ref.digest} -- the store is "
                "corrupt or the reference was forged"
            )
        return content

    def __len__(self) -> int:
        return len(self._by_digest)


@dataclass
class CodecRegistry:
    """Select the positively supporting codec with highest confidence."""

    codecs: list[ContextCodec] = field(default_factory=list)

    def register(self, codec: ContextCodec) -> None:
        self.codecs.append(codec)

    def select(self, text: str, content_type: str = "") -> ContextCodec | None:
        best: tuple[float, ContextCodec] | None = None
        for codec in self.codecs:
            decision = codec.supports(text, content_type)
            if decision and (best is None or decision.confidence > best[0]):
                best = (decision.confidence, codec)
        return best[1] if best else None

    def representations(
        self, text: str, source_id: str = "", content_type: str = "", **options: Any
    ) -> list[Representation]:
        codec = self.select(text, content_type)
        if codec is None:
            return []
        return codec.representations(text, source_id=source_id, **options)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def verify_all(
    representations: Iterable[Representation],
) -> dict[str, tuple[str, ...]]:
    broken = {}
    for rep in representations:
        missing = rep.verify_protected_evidence()
        if missing:
            broken[rep.representation_id] = missing
    return broken
