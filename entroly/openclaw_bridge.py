"""Local JSONL bridge for the Entroly OpenClaw context-engine plugin."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TextIO

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


def _compress_message_bodies(
    messages: list[dict[str, Any]], *, content_budget: int, distill: bool
) -> list[dict[str, Any]]:
    text_tokens = [
        max(1, _estimate_value_tokens(message.get("content", ""))) for message in messages
    ]
    total_text_tokens = max(1, sum(text_tokens))
    result: list[dict[str, Any]] = []
    for message, token_count in zip(messages, text_tokens, strict=True):
        raw_content = message.get("content", "")
        if not isinstance(raw_content, str):
            # Structured content (lists, dicts) must not be coerced to repr.
            result.append(copy.deepcopy(message))
            continue
        content = raw_content
        if distill and message.get("role") == "assistant":
            try:
                from .proxy_transform import distill_response

                content, _, _ = distill_response(content, mode="full")
            except Exception:
                pass
        allocation = max(1, int(content_budget * token_count / total_text_tokens))
        compressed_message = copy.deepcopy(message)
        compressed_message["content"] = compress(content, budget=allocation)
        result.append(compressed_message)
    return result


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
        message_decisions.append(
            {
                "message_index": index,
                "role": source.get("role"),
                "action": (
                    "preserved"
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
    warnings = [
        "Token counts are deterministic estimates, not provider-billed usage."
    ]
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
            compressed = _compress_message_bodies(
                compressible,
                content_budget=content_budget,
                distill=bool(request.get("distill", True)),
            )
            assembled = copy.deepcopy(messages)
            for index, compressed_message in zip(
                compressible_indexes, compressed, strict=True
            ):
                assembled[index] = compressed_message

    assembled_tokens = estimate_messages_tokens(assembled)
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
