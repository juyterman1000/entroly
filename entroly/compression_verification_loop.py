"""Verification loop for recoverable compressed prompts.

Flow:

    compressed prompt -> model answer -> verifier -> retrieve missing span -> retry once

The implementation is model-agnostic: callers pass a model callable and optional
verifier callable. Entroly supplies deterministic fallback verification so the
loop is useful even without WITNESS/EICV dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .compression_retrieval_store import CompressionRetrievalStore, StoredSpan

ModelCall = Callable[[list[dict[str, Any]]], str]
Verifier = Callable[[str, list[dict[str, Any]]], dict[str, Any]]


@dataclass(slots=True)
class VerificationLoopResult:
    answer: str
    retried: bool
    verifier_result: dict[str, Any]
    retrieved_spans: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def answer_with_retrieval_verification(
    messages: list[dict[str, Any]],
    *,
    model_call: ModelCall,
    retrieval_store: CompressionRetrievalStore,
    receipts: list[dict[str, Any]],
    verifier: Verifier | None = None,
    query: str = "",
    max_spans: int = 3,
) -> VerificationLoopResult:
    """Run answer generation, verify, retrieve omitted spans if needed, retry once."""
    answer = model_call(messages)
    check = _verify(answer, messages, verifier=verifier)
    if not _needs_retrieval(check):
        return VerificationLoopResult(answer=answer, retried=False, verifier_result=check)

    spans = _select_recovery_spans(
        retrieval_store,
        receipts=receipts,
        query=query or _reason_query(check),
        limit=max_spans,
    )
    if not spans:
        return VerificationLoopResult(answer=answer, retried=False, verifier_result=check)

    retry_messages = _append_recovered_context(messages, spans)
    retry_answer = model_call(retry_messages)
    retry_check = _verify(retry_answer, retry_messages, verifier=verifier)
    return VerificationLoopResult(
        answer=retry_answer,
        retried=True,
        verifier_result=retry_check,
        retrieved_spans=[span.as_dict() for span in spans],
    )


def _verify(answer: str, messages: list[dict[str, Any]], *, verifier: Verifier | None) -> dict[str, Any]:
    if verifier is not None:
        try:
            result = verifier(answer, messages)
            if isinstance(result, dict):
                return result
        except Exception as exc:
            return {"status": "verifier_error", "needs_retrieval": True, "reason": str(exc)}
    lower = answer.lower()
    uncertain = any(
        marker in lower
        for marker in (
            "not enough context",
            "insufficient context",
            "cannot determine",
            "can't determine",
            "unclear",
            "unknown",
        )
    )
    return {
        "status": "fallback_uncertainty_check",
        "needs_retrieval": uncertain,
        "reason": "answer expressed uncertainty" if uncertain else "answer did not express uncertainty",
    }


def _needs_retrieval(check: dict[str, Any]) -> bool:
    if bool(check.get("needs_retrieval")):
        return True
    status = str(check.get("status", "")).lower()
    risk = str(check.get("risk", "")).lower()
    return status in {"failed", "hallucinated", "unsupported"} or risk in {"high", "critical"}


def _reason_query(check: dict[str, Any]) -> str:
    return str(check.get("reason") or check.get("claim") or check.get("missing_evidence") or "")


def _select_recovery_spans(
    store: CompressionRetrievalStore,
    *,
    receipts: list[dict[str, Any]],
    query: str,
    limit: int,
) -> list[StoredSpan]:
    spans: list[StoredSpan] = []
    seen: set[tuple[str, str]] = set()

    # First use explicit receipt/span ids from compression receipts.
    for receipt in receipts:
        retrieval = receipt.get("retrieval")
        if not isinstance(retrieval, dict):
            continue
        receipt_id = str(retrieval.get("receipt_id", ""))
        for span_id in retrieval.get("span_ids", []) or []:
            span = store.get_span(receipt_id, str(span_id))
            if span is None:
                continue
            key = (span.receipt_id, span.span_id)
            if key not in seen:
                spans.append(span)
                seen.add(key)
            if len(spans) >= limit:
                return spans

    # Then search by verifier reason/query.
    if query:
        for span in store.search(query, limit=limit):
            key = (span.receipt_id, span.span_id)
            if key not in seen:
                spans.append(span)
                seen.add(key)
            if len(spans) >= limit:
                break
    return spans


def _append_recovered_context(messages: list[dict[str, Any]], spans: list[StoredSpan]) -> list[dict[str, Any]]:
    context = "\n\n".join(
        f"[Recovered compressed span {span.span_id} lines {span.start_line}-{span.end_line}]\n{span.content}"
        for span in spans
    )
    return [
        *messages,
        {
            "role": "tool",
            "name": "entroly_recovered_compressed_context",
            "content": context,
        },
    ]


__all__ = ["VerificationLoopResult", "answer_with_retrieval_verification"]
