"""Common contract for content-specific compression codecs.

Entroly already had several codecs -- JSON/XML/CSV/markdown/log/stacktrace in
``universal_compress``, shell output in ``shell_codec`` -- each with its own
signature, its own return shape, and no way for a caller to ask *what did you
throw away, and can I get it back*. Both codecs measured in this module's tests
elided real content and reported only a count:

    "... (40 items)"                      39 records, unrecoverable
    "connection pool exhausted  [x200]"   199 lines, unrecoverable

That is the gap this closes. A codec now returns ``Representation`` objects
carrying provenance and, when it drops anything, a ``RecoveryReference`` whose
digest resolves to the exact original bytes.

Design notes
------------

* **Codecs do not judge sufficiency.** A codec reports what it did
  (``distortion_risk``, ``protected_evidence``, ``omitted_bytes``) and never
  whether the result is enough for a task. That decision belongs to the
  sufficiency controller, which sees the whole selection; a codec sees one item.
  ``Representation`` deliberately has no ``sufficient`` field.

* **Provenance composes, it is not reinvented.** ``entroly.source_span``
  already defines a validated ``SourceSpan`` (canonical path, whole-source
  digest, byte offsets, fragment digest) and ``entroly.context_receipts`` uses
  the same vocabulary. A ``Representation`` carries an optional ``SourceSpan``
  rather than a parallel scheme, so a representation of a file region is
  verifiable by the machinery receipts already use.

* **Recovery is content-addressed.** The reference is the SHA-256 of the
  omitted bytes, so a caller can verify what came back is what was dropped
  without trusting the store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from .source_span import SourceSpan

CODEC_CONTRACT_VERSION = "1"


def content_digest(data: bytes | str) -> str:
    """`sha256:<hex>` over UTF-8 bytes. The recovery key and its own checksum."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RecoveryReference:
    """A verifiable pointer to content a codec removed.

    ``digest`` is the SHA-256 of the omitted bytes, so recovery is checkable
    against the reference itself. A store that returns the wrong content is
    caught by ``verify``, not trusted.
    """

    digest: str
    byte_length: int
    item_count: int = 0
    note: str = ""

    def verify(self, recovered: bytes | str) -> bool:
        return content_digest(recovered) == self.digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "byte_length": self.byte_length,
            "item_count": self.item_count,
            "note": self.note,
        }


@dataclass(frozen=True)
class Representation:
    """One way a codec can present one context item.

    A codec may offer several (full, elided, reference-only); choosing among
    them is the caller's job, which is why cost and risk are reported rather
    than resolved here.
    """

    representation_id: str
    source_id: str
    content_type: str
    text: str
    token_cost: int
    codec: str
    codec_version: str
    source_sha256: str
    # Substrings the codec asserts it preserved verbatim. Checkable: see
    # `verify_protected_evidence`. This is a claim about THIS text, not a claim
    # that the text answers anything.
    protected_evidence: tuple[str, ...] = ()
    # Codec's own estimate of how much meaning it altered, 0.0 (verbatim) to
    # 1.0. NOT a sufficiency judgement -- a lossless excerpt of the wrong
    # material has distortion 0.0 and is useless.
    distortion_risk: float = 0.0
    recovery: RecoveryReference | None = None
    span: SourceSpan | None = None
    estimated_relevance: float | None = None
    dependency_coverage: float | None = None

    def verify_protected_evidence(self) -> tuple[str, ...]:
        """Protected substrings that are NOT actually present. Empty is good."""
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
    """Whether a codec claims an item, and how strongly."""

    supported: bool
    confidence: float = 0.0
    reason: str = ""

    def __bool__(self) -> bool:
        return self.supported


@runtime_checkable
class ContextCodec(Protocol):
    """Produce candidate representations of one item. Never judge sufficiency."""

    name: str
    version: str

    def supports(self, text: str, content_type: str = "") -> SupportDecision: ...

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]: ...


# ── Recovery store ──────────────────────────────────────────────────────────


class RecoveryStore:
    """Content-addressed store for what codecs removed.

    In-memory by default; ``path`` adds a JSON sidecar so recovery survives the
    process. Deliberately dumb -- the digest IS the key, so a corrupted or
    swapped entry fails ``RecoveryReference.verify`` at read time.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._mem: dict[str, str] = {}
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            try:
                self._mem.update(json.loads(self._path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

    def put(self, content: str, *, item_count: int = 0, note: str = "") -> RecoveryReference:
        ref = RecoveryReference(
            digest=content_digest(content),
            byte_length=len(content.encode("utf-8")),
            item_count=item_count,
            note=note,
        )
        self._mem[ref.digest] = content
        self._flush()
        return ref

    def get(self, ref: RecoveryReference | str) -> str | None:
        digest = ref.digest if isinstance(ref, RecoveryReference) else ref
        return self._mem.get(digest)

    def recover(self, ref: RecoveryReference) -> str:
        """Return the exact omitted content, or raise if it cannot be verified."""
        content = self.get(ref)
        if content is None:
            raise KeyError(f"no recovery entry for {ref.digest}")
        if not ref.verify(content):
            raise ValueError(
                f"recovered content does not match {ref.digest} -- the store is "
                f"corrupt or the entry was replaced"
            )
        return content

    def __len__(self) -> int:
        return len(self._mem)

    def _flush(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._mem), encoding="utf-8")
        except OSError:
            pass


# ── Registry ────────────────────────────────────────────────────────────────


@dataclass
class CodecRegistry:
    """Pick the codec that claims an item most confidently.

    Unknown content must degrade to something safe rather than be rewritten by
    a codec that does not understand it, so selection requires a positive
    support decision and falls back to `None` (caller keeps the original).
    """

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


def verify_all(representations: Iterable[Representation]) -> dict[str, tuple[str, ...]]:
    """Representation id -> protected substrings it claimed but does not contain."""
    broken = {}
    for rep in representations:
        missing = rep.verify_protected_evidence()
        if missing:
            broken[rep.representation_id] = missing
    return broken
