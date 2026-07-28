"""Authoritative, auditable context transformation pipeline.

This module converges Entroly's existing typed tool-output compression,
Evidence-Locked Compression (ELC), outbound redaction, exact-result reuse,
recovery storage, and receipts behind one deterministic contract.

The pipeline deliberately coordinates existing algorithms instead of adding a
parallel compressor. Deadlines are cooperative: the pipeline checks them at
stage boundaries and degrades deterministically, but it does not claim to
preempt arbitrary Python code that is already executing.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from .evidence_locked_compression import compress_evidence_locked, estimate_tokens
from .provider_policy import GatewayRedactionPolicy

_PIPELINE_VERSION = "1"
_ALREADY_TRANSFORMED_MARKERS = (
    "[entroly-elc:",
    "[entroly-ref:",
    "[entroly-recovery:",
    "[aged-pruned]",
)
_SENSITIVE_METADATA_TERMS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _safe_receipt_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Return bounded scalar metadata without credential-shaped fields."""
    safe: dict[str, object] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)
        lowered = key.lower()
        if any(term in lowered for term in _SENSITIVE_METADATA_TERMS):
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            safe[key] = value
    return safe


def _authorization_scope_hash(metadata: Mapping[str, object]) -> str:
    value = metadata.get("authorization_scope")
    return _sha256(str(value)) if value not in (None, "") else ""


@dataclass(frozen=True, slots=True)
class ContentEnvelope:
    """One transformable context object plus its execution identity."""

    content: str
    source: str = ""
    query: str = ""
    content_type: str | None = None
    tool_name: str = ""
    command: str = ""
    workspace: str = ""
    cwd: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def has_scoped_execution_identity(self) -> bool:
        """Whether exact reuse is safely scoped to an execution boundary."""
        scope_present = bool(self.workspace or _authorization_scope_hash(self.metadata))
        operation_present = bool(self.command or self.tool_name)
        return scope_present and operation_present

    def exact_identity(self, safe_content_sha256: str) -> str:
        payload = {
            "workspace": self.workspace,
            "cwd": self.cwd,
            "command_sha256": _sha256(self.command) if self.command else "",
            "tool_name": self.tool_name,
            "source": self.source,
            "authorization_scope_sha256": _authorization_scope_hash(self.metadata),
            "environment_fingerprint": str(
                self.metadata.get("environment_fingerprint", "")
            ),
            "source_version": str(self.metadata.get("source_version", "")),
            "safe_content_sha256": safe_content_sha256,
            "pipeline_version": _PIPELINE_VERSION,
        }
        return _sha256(_stable_json(payload))


@dataclass(frozen=True, slots=True)
class TransformPolicy:
    """Request-level policy shared by SDK, proxy, MCP, and plugins."""

    token_budget: int = 1200
    deadline_ms: float = 20.0
    min_savings: float = 0.08
    redact_sensitive: bool = False
    allow_exact_reference: bool = True
    prefer_typed_formatter: bool = True
    preserve_exact_json: bool = False
    fail_open: bool = True

    def __post_init__(self) -> None:
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if not 0.0 <= self.min_savings < 1.0:
            raise ValueError("min_savings must be in [0, 1)")

    @property
    def fingerprint(self) -> str:
        return _sha256(_stable_json(asdict(self)))


@dataclass(frozen=True, slots=True)
class TransformStage:
    name: str
    elapsed_ms: float
    outcome: str
    changed: bool = False
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextReceiptV1:
    """Versioned proof of one context transformation."""

    receipt_id: str
    version: str
    policy_sha256: str
    source: str
    content_type: str
    algorithm: str
    input_sha256: str
    safe_input_sha256: str
    output_sha256: str
    original_tokens: int
    transmitted_tokens: int
    gross_saved_tokens: int
    savings_ratio: float
    deadline_ms: float
    elapsed_ms: float
    deadline_exceeded: bool
    redacted: bool
    redaction_counts: dict[str, int]
    recovery_receipt_id: str | None
    recovery_span_ids: tuple[str, ...]
    exact_reference_of: str | None
    stages: tuple[TransformStage, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["stages"] = [asdict(stage) for stage in self.stages]
        return data


@dataclass(frozen=True, slots=True)
class TransformResult:
    content: str
    changed: bool
    receipt: ContextReceiptV1


@dataclass(frozen=True, slots=True)
class _ExactEntry:
    receipt_id: str
    recovery_receipt_id: str
    recovery_span_id: str
    original_tokens: int


class ExactResultCache:
    """Bounded process-local index for exact repeated command results."""

    def __init__(self, max_entries: int = 512) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._items: OrderedDict[str, _ExactEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> _ExactEntry | None:
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
            return item

    def put(self, key: str, value: _ExactEntry) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._items), "max_entries": self._max_entries}


class ContextTransformPipeline:
    """Coordinate existing Entroly mechanisms under one auditable contract."""

    def __init__(
        self,
        *,
        retrieval_store: Any | None = None,
        exact_cache: ExactResultCache | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._retrieval_store = retrieval_store
        self._exact_cache = exact_cache or ExactResultCache()
        self._clock = clock

    def transform(
        self,
        envelope: ContentEnvelope,
        policy: TransformPolicy | None = None,
    ) -> TransformResult:
        policy = policy or TransformPolicy()
        started = self._clock()
        stages: list[TransformStage] = []
        raw = envelope.content
        input_hash = _sha256(raw)

        safe, redacted, redaction_counts = self._redact(
            raw,
            policy=policy,
            stages=stages,
            started=started,
        )
        safe_hash = _sha256(safe)
        exact_key = envelope.exact_identity(safe_hash)
        exact_reuse_enabled = (
            policy.allow_exact_reference
            and envelope.has_scoped_execution_identity
            and self._retrieval_store is not None
        )

        if self._deadline_exceeded(started, policy):
            stages.append(
                TransformStage(
                    name="deadline",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="expired_after_redaction",
                )
            )
            return self._finish(
                envelope=envelope,
                policy=policy,
                started=started,
                stages=stages,
                raw=raw,
                safe=safe,
                output=safe,
                algorithm="deadline_fallback",
                input_hash=input_hash,
                safe_hash=safe_hash,
                redacted=redacted,
                redaction_counts=redaction_counts,
                recovery=None,
                exact_reference_of=None,
            )

        exact = self._exact_cache.get(exact_key) if exact_reuse_enabled else None
        if exact is not None:
            output = (
                f"[entroly-ref:{exact.recovery_receipt_id}:{exact.recovery_span_id}]\n"
                f"Exact repeat of receipt {exact.receipt_id}; "
                f"{exact.original_tokens} original tokens remain recoverable."
            )
            stages.append(
                TransformStage(
                    name="exact_reuse",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="hit",
                    changed=True,
                    details={"referenced_receipt_id": exact.receipt_id},
                )
            )
            return self._finish(
                envelope=envelope,
                policy=policy,
                started=started,
                stages=stages,
                raw=raw,
                safe=safe,
                output=output,
                algorithm="exact_reference",
                input_hash=input_hash,
                safe_hash=safe_hash,
                redacted=redacted,
                redaction_counts=redaction_counts,
                recovery=(exact.recovery_receipt_id, (exact.recovery_span_id,)),
                exact_reference_of=exact.receipt_id,
            )

        if any(marker in safe for marker in _ALREADY_TRANSFORMED_MARKERS):
            stages.append(
                TransformStage(
                    name="idempotency",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="already_transformed",
                )
            )
            output = safe
            algorithm = "identity"
            already_transformed = True
        else:
            output, algorithm = self._compress(
                safe,
                envelope=envelope,
                policy=policy,
                started=started,
                stages=stages,
            )
            already_transformed = False

        should_seed_exact_reuse = exact_reuse_enabled and not already_transformed
        should_persist_changed_output = output != safe and not already_transformed
        recovery = self._persist_recovery(
            original=safe,
            transformed=output,
            envelope=envelope,
            policy=policy,
            algorithm=algorithm,
            stages=stages,
            started=started,
            enabled=should_seed_exact_reuse or should_persist_changed_output,
        )

        result = self._finish(
            envelope=envelope,
            policy=policy,
            started=started,
            stages=stages,
            raw=raw,
            safe=safe,
            output=output,
            algorithm=algorithm,
            input_hash=input_hash,
            safe_hash=safe_hash,
            redacted=redacted,
            redaction_counts=redaction_counts,
            recovery=recovery,
            exact_reference_of=None,
        )

        if recovery is not None and should_seed_exact_reuse:
            recovery_receipt_id, span_ids = recovery
            if span_ids:
                self._exact_cache.put(
                    exact_key,
                    _ExactEntry(
                        receipt_id=result.receipt.receipt_id,
                        recovery_receipt_id=recovery_receipt_id,
                        recovery_span_id=span_ids[0],
                        original_tokens=estimate_tokens(safe),
                    ),
                )
        return result

    def _compress(
        self,
        content: str,
        *,
        envelope: ContentEnvelope,
        policy: TransformPolicy,
        started: float,
        stages: list[TransformStage],
    ) -> tuple[str, str]:
        if estimate_tokens(content) <= policy.token_budget:
            stages.append(
                TransformStage(
                    name="budget_gate",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="already_within_budget",
                )
            )
            return content, "identity"

        if policy.preserve_exact_json and self._is_json(content):
            stages.append(
                TransformStage(
                    name="exact_json_policy",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="preserved",
                )
            )
            return content, "identity"

        candidate = content
        if policy.prefer_typed_formatter and not self._deadline_exceeded(started, policy):
            try:
                from .proxy_transform import compress_tool_output

                typed, codec, savings = compress_tool_output(candidate)
                changed = typed != candidate and savings >= policy.min_savings
                stages.append(
                    TransformStage(
                        name="typed_formatter",
                        elapsed_ms=self._elapsed_ms(started),
                        outcome=codec if changed else "no_gain",
                        changed=changed,
                        details={"savings_ratio": round(float(savings), 6)},
                    )
                )
                if changed and estimate_tokens(typed) <= policy.token_budget:
                    return typed, f"typed:{codec}"
                if changed:
                    candidate = typed
            except Exception as error:
                stages.append(
                    TransformStage(
                        name="typed_formatter",
                        elapsed_ms=self._elapsed_ms(started),
                        outcome="error_fail_open",
                        details={"error_type": type(error).__name__},
                    )
                )

        if self._deadline_exceeded(started, policy):
            stages.append(
                TransformStage(
                    name="deadline",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="expired_after_typed_formatter",
                )
            )
            return self._bounded_fallback(candidate, policy.token_budget), "deadline_fallback"

        try:
            elc = compress_evidence_locked(
                candidate,
                query=envelope.query,
                budget_tokens=policy.token_budget,
                content_type=envelope.content_type,
                min_savings=policy.min_savings,
            )
            stages.append(
                TransformStage(
                    name="evidence_locked_compression",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="compressed" if elc.changed else "no_gain",
                    changed=elc.changed,
                    details={
                        "content_type": elc.receipt.content_type,
                        "savings_ratio": round(elc.receipt.savings_ratio, 6),
                    },
                )
            )
            if elc.changed:
                return elc.with_receipt_header(), "elc"
        except Exception as error:
            stages.append(
                TransformStage(
                    name="evidence_locked_compression",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="error_fail_open",
                    details={"error_type": type(error).__name__},
                )
            )

        if policy.fail_open:
            return candidate, "identity"
        return self._bounded_fallback(candidate, policy.token_budget), "bounded_fallback"

    def _redact(
        self,
        content: str,
        *,
        policy: TransformPolicy,
        stages: list[TransformStage],
        started: float,
    ) -> tuple[str, bool, dict[str, int]]:
        if not policy.redact_sensitive:
            stages.append(
                TransformStage(
                    name="redaction",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="disabled",
                )
            )
            return content, False, {}
        redacted, receipt = GatewayRedactionPolicy(enabled=True).redact_text(content)
        counts = receipt.counts
        stages.append(
            TransformStage(
                name="redaction",
                elapsed_ms=self._elapsed_ms(started),
                outcome="redacted" if receipt.changed else "clean",
                changed=receipt.changed,
                details={"counts": counts},
            )
        )
        return redacted, receipt.changed, counts

    def _persist_recovery(
        self,
        *,
        original: str,
        transformed: str,
        envelope: ContentEnvelope,
        policy: TransformPolicy,
        algorithm: str,
        stages: list[TransformStage],
        started: float,
        enabled: bool,
    ) -> tuple[str, tuple[str, ...]] | None:
        store = self._retrieval_store
        if not enabled:
            stages.append(
                TransformStage(
                    name="recovery_store",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="not_required",
                )
            )
            return None
        if store is None:
            stages.append(
                TransformStage(
                    name="recovery_store",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="unconfigured",
                )
            )
            return None
        if self._deadline_exceeded(started, policy):
            stages.append(
                TransformStage(
                    name="recovery_store",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="skipped_deadline",
                )
            )
            return None
        try:
            line_count = max(1, original.count("\n") + 1)
            receipt = {
                "version": _PIPELINE_VERSION,
                "original_tokens": estimate_tokens(original),
                "compressed_tokens": estimate_tokens(transformed),
                "algorithm": algorithm,
                "omitted_spans": [
                    {
                        "start_line": 1,
                        "end_line": line_count,
                        "line_count": line_count,
                        "reason": "context_pipeline_full_recovery",
                    }
                ],
            }
            stored = store.put(
                original_text=original,
                compressed_text=transformed,
                receipt=receipt,
                metadata={
                    "source": envelope.source,
                    "tool_name": envelope.tool_name,
                    "command_sha256": _sha256(envelope.command) if envelope.command else "",
                    "workspace_sha256": _sha256(envelope.workspace) if envelope.workspace else "",
                    "authorization_scope_sha256": _authorization_scope_hash(
                        envelope.metadata
                    ),
                    "pipeline_version": _PIPELINE_VERSION,
                },
            )
            span_ids = tuple(span.span_id for span in stored.spans)
            stages.append(
                TransformStage(
                    name="recovery_store",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="stored",
                    changed=True,
                    details={
                        "receipt_id": stored.receipt_id,
                        "span_count": len(span_ids),
                    },
                )
            )
            return stored.receipt_id, span_ids
        except Exception as error:
            stages.append(
                TransformStage(
                    name="recovery_store",
                    elapsed_ms=self._elapsed_ms(started),
                    outcome="error_fail_open",
                    details={"error_type": type(error).__name__},
                )
            )
            return None

    def _finish(
        self,
        *,
        envelope: ContentEnvelope,
        policy: TransformPolicy,
        started: float,
        stages: list[TransformStage],
        raw: str,
        safe: str,
        output: str,
        algorithm: str,
        input_hash: str,
        safe_hash: str,
        redacted: bool,
        redaction_counts: dict[str, int],
        recovery: tuple[str, tuple[str, ...]] | None,
        exact_reference_of: str | None,
    ) -> TransformResult:
        elapsed = self._elapsed_ms(started)
        original_tokens = estimate_tokens(safe)
        transmitted_tokens = estimate_tokens(output)
        saved = max(0, original_tokens - transmitted_tokens)
        recovery_receipt_id = recovery[0] if recovery is not None else None
        recovery_span_ids = recovery[1] if recovery is not None else ()
        receipt_payload = {
            "version": _PIPELINE_VERSION,
            "source": envelope.source,
            "algorithm": algorithm,
            "safe_input_sha256": safe_hash,
            "output_sha256": _sha256(output),
            "policy_sha256": policy.fingerprint,
            "recovery_receipt_id": recovery_receipt_id,
            "exact_reference_of": exact_reference_of,
        }
        receipt_id = _sha256(_stable_json(receipt_payload))[:24]
        metadata = _safe_receipt_metadata(envelope.metadata)
        authorization_scope_hash = _authorization_scope_hash(envelope.metadata)
        if authorization_scope_hash:
            metadata["authorization_scope_sha256"] = authorization_scope_hash
        receipt = ContextReceiptV1(
            receipt_id=receipt_id,
            version=_PIPELINE_VERSION,
            policy_sha256=policy.fingerprint,
            source=envelope.source,
            content_type=envelope.content_type or "auto",
            algorithm=algorithm,
            input_sha256=input_hash,
            safe_input_sha256=safe_hash,
            output_sha256=_sha256(output),
            original_tokens=original_tokens,
            transmitted_tokens=transmitted_tokens,
            gross_saved_tokens=saved,
            savings_ratio=0.0 if original_tokens == 0 else saved / original_tokens,
            deadline_ms=policy.deadline_ms,
            elapsed_ms=elapsed,
            deadline_exceeded=elapsed > policy.deadline_ms,
            redacted=redacted,
            redaction_counts=dict(redaction_counts),
            recovery_receipt_id=recovery_receipt_id,
            recovery_span_ids=tuple(recovery_span_ids),
            exact_reference_of=exact_reference_of,
            stages=tuple(stages),
            metadata=metadata,
        )
        return TransformResult(content=output, changed=output != raw, receipt=receipt)

    def _elapsed_ms(self, started: float) -> float:
        return round((self._clock() - started) * 1000.0, 6)

    def _deadline_exceeded(self, started: float, policy: TransformPolicy) -> bool:
        return (self._clock() - started) * 1000.0 >= policy.deadline_ms

    @staticmethod
    def _bounded_fallback(content: str, token_budget: int) -> str:
        max_chars = max(1, token_budget * 4)
        if len(content) <= max_chars:
            return content
        marker = "\n...[deadline-bounded; exact source may require upstream recovery]"
        if max_chars <= len(marker):
            return content[:max_chars]
        return content[: max_chars - len(marker)].rstrip() + marker

    @staticmethod
    def _is_json(content: str) -> bool:
        stripped = content.strip()
        if not stripped or stripped[0] not in "[{":
            return False
        try:
            json.loads(stripped)
        except Exception:
            return False
        return True


__all__ = [
    "ContentEnvelope",
    "ContextReceiptV1",
    "ContextTransformPipeline",
    "ExactResultCache",
    "TransformPolicy",
    "TransformResult",
    "TransformStage",
]
