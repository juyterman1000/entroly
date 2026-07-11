"""Local JSONL bridge for the Entroly OpenClaw context-engine plugin."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO

from .context_receipts.retrieval import tokenize
from .sdk import compress

BRIDGE_SCHEMA = "entroly.openclaw.bridge.v1"
RECEIPT_SCHEMA = "entroly.openclaw.receipt.v1"
DEFAULT_TOKEN_BUDGET = 50_000
DEFAULT_PRESERVE_LAST_N = 4


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _estimate_value_tokens(value: Any) -> int:
    if isinstance(value, str):
        return max(1, (len(value) + 3) // 4) if value else 0
    if value is None or isinstance(value, (bool, int, float)):
        return 1
    if isinstance(value, list):
        return sum(_estimate_value_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _estimate_value_tokens(key) + _estimate_value_tokens(item)
            for key, item in value.items()
        )
    return _estimate_value_tokens(str(value))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate full message tokens, including structured tool payloads."""
    return sum(_estimate_value_tokens(message) + 3 for message in messages)


def _protected_message(message: dict[str, Any]) -> bool:
    """Return whether a message must remain byte-for-byte equivalent."""
    if message.get("role") in {"system", "developer"}:
        return True
    content = message.get("content")
    return not isinstance(content, str)


def _message_relevance(
    messages: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    query_terms = sorted(set(tokenize(query)))
    if not query_terms:
        return [
            {"score": 0.0, "matched_terms": [], "token_count": 0}
            for _ in messages
        ]

    documents = [tokenize(str(message.get("content", ""))) for message in messages]
    document_frequency: dict[str, int] = defaultdict(int)
    for terms in documents:
        for term in set(terms):
            document_frequency[term] += 1
    document_count = max(1, len(documents))
    average_length = max(
        1.0, sum(len(terms) for terms in documents) / document_count
    )

    ranked: list[dict[str, Any]] = []
    for index, terms in enumerate(documents):
        frequencies = Counter(terms)
        document_length = max(1, len(terms))
        matched = sorted(term for term in query_terms if frequencies.get(term, 0))
        score = 0.0
        for term in matched:
            frequency = frequencies[term]
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(
                1.0
                + (document_count - frequency_docs + 0.5)
                / (frequency_docs + 0.5)
            )
            normalization = 1.0 - 0.75 + 0.75 * document_length / average_length
            score += inverse_frequency * (frequency * 2.2) / (
                frequency + 1.2 * normalization
            )
        coverage = len(matched) / max(1, len(query_terms))
        pin_eligible = True
        security_flags: list[str] = []
        pin_blocked_reason: str | None = None
        if matched:
            try:
                from .context_firewall import scan

                scan_result = scan(
                    str(messages[index].get("content", "")),
                    source=f"openclaw_message_{index}",
                    check_repetition=False,
                )
                security_flags = sorted(
                    {
                        f"{threat.severity}:{threat.threat_type}"
                        for threat in scan_result.threats
                    }
                )
                pin_eligible = scan_result.is_safe
                if not pin_eligible:
                    pin_blocked_reason = "context_firewall"
            except Exception:
                pin_eligible = False
                security_flags = ["critical:scanner_error"]
                pin_blocked_reason = "context_firewall_error"
        ranked.append(
            {
                "score": round(score * (1.0 + coverage), 6),
                "matched_terms": matched,
                "token_count": len(terms),
                "pin_eligible": pin_eligible,
                "security_flags": security_flags,
                "pin_blocked_reason": pin_blocked_reason,
            }
        )
    return ranked


def _query_for_request(
    request: dict[str, Any], messages: list[dict[str, Any]]
) -> str:
    prompt = request.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    for message in reversed(messages):
        if message.get("role") not in {"user", "human"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _compress_message_bodies(
    messages: list[dict[str, Any]],
    *,
    content_budget: int,
    distill: bool,
    query: str,
) -> tuple[list[dict[str, Any]], set[int], list[dict[str, Any]]]:
    text_tokens = [
        max(1, _estimate_value_tokens(message.get("content", ""))) for message in messages
    ]
    relevance = _message_relevance(messages, query)
    reserve_for_compression = min(int(content_budget * 0.35), len(messages) * 8)
    pin_budget = min(
        int(content_budget * 0.65), max(0, content_budget - reserve_for_compression)
    )
    pinned: set[int] = set()
    for index in sorted(
        range(len(messages)),
        key=lambda item: (-relevance[item]["score"], text_tokens[item], item),
    ):
        if (
            relevance[index]["score"] <= 0
            or not relevance[index]["matched_terms"]
            or not relevance[index]["pin_eligible"]
        ):
            continue
        if text_tokens[index] <= pin_budget:
            pinned.add(index)
            pin_budget -= text_tokens[index]

    unpinned = [index for index in range(len(messages)) if index not in pinned]
    remaining_budget = max(
        0, content_budget - sum(text_tokens[index] for index in pinned)
    )
    allocations = {index: text_tokens[index] for index in pinned}
    if unpinned:
        distributable = max(0, remaining_budget - len(unpinned))
        max_score = max((relevance[index]["score"] for index in unpinned), default=0.0)
        weights = {
            index: math.sqrt(text_tokens[index])
            * (1.0 + 2.0 * relevance[index]["score"] / max(1.0, max_score))
            for index in unpinned
        }
        total_weight = max(1.0, sum(weights.values()))
        for index in unpinned:
            allocations[index] = 1 + int(
                distributable * weights[index] / total_weight
            )

    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        raw_content = message.get("content", "")
        if not isinstance(raw_content, str):
            # Structured content (lists, dicts) must not be coerced to repr.
            result.append(copy.deepcopy(message))
            continue
        if index in pinned:
            result.append(copy.deepcopy(message))
            continue
        content = raw_content
        if distill and message.get("role") == "assistant":
            try:
                from .proxy_transform import distill_response

                content, _, _ = distill_response(content, mode="full")
            except Exception:
                pass
        compressed_message = copy.deepcopy(message)
        compressed_message["content"] = compress(
            content, budget=max(1, allocations[index])
        )
        result.append(compressed_message)
    for index, item in enumerate(relevance):
        item["allocated_tokens"] = allocations[index]
    return result, pinned, relevance


def _safe_session_name(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_id).strip("-.")
    return (safe or "session")[:80]


def _receipt_directory(request: dict[str, Any]) -> Path:
    configured = request.get("receipt_dir")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser().resolve()
    workspace = request.get("workspace_dir")
    root = Path(workspace).expanduser() if isinstance(workspace, str) and workspace else Path.cwd()
    return (root / ".entroly" / "receipts" / "openclaw").resolve()


def _write_receipt(
    request: dict[str, Any],
    source_messages: list[dict[str, Any]],
    assembled_messages: list[dict[str, Any]],
    *,
    source_tokens: int,
    assembled_tokens: int,
    warnings: list[str],
    evidence: dict[int, dict[str, Any]],
    strategy: str,
) -> tuple[str, str | None]:
    source_hash = hashlib.sha256(_canonical_json(source_messages).encode("utf-8")).hexdigest()
    assembled_hash = hashlib.sha256(
        _canonical_json(assembled_messages).encode("utf-8")
    ).hexdigest()
    message_decisions = []
    for index, (source, assembled) in enumerate(
        zip(source_messages, assembled_messages)
    ):
        source_content = source.get("content")
        assembled_content = assembled.get("content")
        source_content_json = _canonical_json(source_content)
        assembled_content_json = _canonical_json(assembled_content)
        evidence_item = evidence.get(index, {})
        is_pinned = bool(evidence_item.get("evidence_pinned"))
        message_decisions.append(
            {
                "message_index": index,
                "role": source.get("role"),
                "action": (
                    "evidence_pinned"
                    if is_pinned
                    else "preserved"
                    if source_content_json == assembled_content_json
                    else "compressed"
                ),
                "source_content_sha256": hashlib.sha256(
                    source_content_json.encode("utf-8")
                ).hexdigest(),
                "assembled_content_sha256": hashlib.sha256(
                    assembled_content_json.encode("utf-8")
                ).hexdigest(),
                "source_chars": len(source_content) if isinstance(source_content, str) else None,
                "assembled_chars": (
                    len(assembled_content) if isinstance(assembled_content, str) else None
                ),
                "relevance_score": evidence_item.get("score"),
                "matched_query_terms": evidence_item.get("matched_terms", []),
                "allocated_tokens": evidence_item.get("allocated_tokens"),
                "pin_eligible": evidence_item.get("pin_eligible"),
                "security_flags": evidence_item.get("security_flags", []),
                "pin_blocked_reason": evidence_item.get("pin_blocked_reason"),
            }
        )
    identity = _canonical_json(
        {
            "source_hash": source_hash,
            "assembled_hash": assembled_hash,
            "budget": request.get("token_budget"),
            "query": request.get("prompt", ""),
        }
    )
    receipt_id = "ocr_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    tokens_saved = max(0, source_tokens - assembled_tokens)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "session_id": str(request.get("session_id") or ""),
        "query": str(request.get("prompt") or ""),
        "model": request.get("model"),
        "token_budget": request.get("token_budget"),
        "source_tokens_estimated": source_tokens,
        "assembled_tokens_estimated": assembled_tokens,
        "tokens_saved_estimated": tokens_saved,
        "reduction_pct_estimated": round(
            (tokens_saved / source_tokens * 100.0) if source_tokens else 0.0, 2
        ),
        "source_message_count": len(source_messages),
        "assembled_message_count": len(assembled_messages),
        "source_sha256": source_hash,
        "assembled_sha256": assembled_hash,
        "changed": source_hash != assembled_hash,
        "message_decisions": message_decisions,
        "assembly_strategy": strategy,
        "evidence_pinned_count": sum(
            1 for item in evidence.values() if item.get("evidence_pinned")
        ),
        "evidence_pin_blocked_count": sum(
            1 for item in evidence.values() if item.get("pin_blocked_reason")
        ),
        "recovery_source": "openclaw_transcript_unmodified",
        "warnings": warnings,
        "local_only": True,
    }
    if request.get("write_receipt", True) is False:
        return receipt_id, None

    directory = _receipt_directory(request)
    directory.mkdir(parents=True, exist_ok=True)
    session_name = _safe_session_name(str(request.get("session_id") or "session"))
    destination = directory / f"{session_name}-{receipt_id}.json"
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return receipt_id, str(destination)


def assemble(request: dict[str, Any]) -> dict[str, Any]:
    raw_messages = request.get("messages")
    if not isinstance(raw_messages, list) or not all(
        isinstance(message, dict) for message in raw_messages
    ):
        raise ValueError("messages must be a list of objects")

    messages: list[dict[str, Any]] = copy.deepcopy(raw_messages)
    budget = request.get("token_budget", DEFAULT_TOKEN_BUDGET)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise ValueError("token_budget must be a positive number")
    budget = max(1, int(budget))
    preserve_last_n = request.get("preserve_last_n", DEFAULT_PRESERVE_LAST_N)
    if isinstance(preserve_last_n, bool) or not isinstance(preserve_last_n, (int, float)):
        raise ValueError("preserve_last_n must be a non-negative integer")
    preserve_last_n = max(0, int(preserve_last_n))

    source_tokens = estimate_messages_tokens(messages)
    evidence_pinning = request.get("evidence_pinning", True) is not False
    query = _query_for_request(request, messages) if evidence_pinning else ""
    strategy = (
        "query_aware_evidence_pinning"
        if evidence_pinning
        else "uniform_budget_compression"
    )
    warnings = [
        "Token counts are deterministic estimates, not provider-billed usage."
    ]
    evidence: dict[int, dict[str, Any]] = {}
    protected_indexes = {
        index for index, message in enumerate(messages) if _protected_message(message)
    }
    if preserve_last_n:
        protected_indexes.update(range(max(0, len(messages) - preserve_last_n), len(messages)))
    protected_tokens = estimate_messages_tokens(
        [messages[index] for index in sorted(protected_indexes)]
    )

    if source_tokens <= budget:
        assembled = messages
    elif protected_tokens >= budget:
        assembled = messages
        warnings.append(
            "Protected system, structured, and recent messages exceed the token budget; "
            "Entroly returned the exact original context."
        )
    else:
        compressible_indexes = [
            index for index in range(len(messages)) if index not in protected_indexes
        ]
        compressible = [messages[index] for index in compressible_indexes]
        fixed_tokens = estimate_messages_tokens(
            [{**message, "content": ""} for message in compressible]
        )
        content_budget = budget - protected_tokens - fixed_tokens
        if content_budget <= 0:
            assembled = messages
            warnings.append(
                "Protected messages and message metadata exceed the token budget; "
                "Entroly returned the exact original context."
            )
        else:
            compressed, pinned, relevance = _compress_message_bodies(
                compressible,
                content_budget=content_budget,
                distill=bool(request.get("distill", True)),
                query=query,
            )
            assembled = copy.deepcopy(messages)
            for local_index, (index, compressed_message) in enumerate(
                zip(compressible_indexes, compressed, strict=True)
            ):
                assembled[index] = compressed_message
                evidence[index] = {
                    **relevance[local_index],
                    "evidence_pinned": local_index in pinned,
                }

    assembled_tokens = estimate_messages_tokens(assembled)
    blocked_count = sum(
        1 for item in evidence.values() if item.get("pin_blocked_reason")
    )
    if blocked_count:
        warnings.append(
            f"Context firewall blocked verbatim evidence pinning for {blocked_count} "
            "message(s); those messages remained subject to normal compression."
        )
    if assembled_tokens > budget and assembled != messages:
        warnings.append(
            "Structured message overhead kept the assembled estimate above budget; "
            "no protected message was modified."
        )

    receipt_id, receipt_path = _write_receipt(
        request,
        messages,
        assembled,
        source_tokens=source_tokens,
        assembled_tokens=assembled_tokens,
        warnings=warnings,
        evidence=evidence,
        strategy=strategy,
    )
    pinned_indexes = sorted(
        index for index, item in evidence.items() if item.get("evidence_pinned")
    )
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "messages": assembled,
        "estimated_tokens": assembled_tokens,
        "source_tokens": source_tokens,
        "tokens_saved": max(0, source_tokens - assembled_tokens),
        "changed": assembled != messages,
        "receipt_id": receipt_id,
        "receipt_path": receipt_path,
        "assembly_strategy": strategy,
        "evidence_pinned": len(pinned_indexes),
        "evidence_pin_blocked": blocked_count,
        "pinned_message_indexes": pinned_indexes,
        "warnings": warnings,
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = request.get("operation")
    if operation == "health":
        return {"schema_version": BRIDGE_SCHEMA, "ok": True, "status": "ready"}
    if operation == "assemble":
        return assemble(request)
    raise ValueError(f"unsupported operation: {operation!r}")


def serve(input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> int:
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        request_id: Any = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            request_id = payload.get("request_id")
            response = handle_request(payload)
        except Exception as error:
            response = {
                "schema_version": BRIDGE_SCHEMA,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        response["request_id"] = request_id
        output_stream.write(_canonical_json(response) + "\n")
        output_stream.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="store_true", help="serve newline-delimited JSON")
    args = parser.parse_args(argv)
    if not args.jsonl:
        parser.error("--jsonl is required")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
