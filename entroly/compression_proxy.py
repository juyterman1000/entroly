"""High-efficiency Evidence-Locked Compression proxy surface.

This module is the product-facing bridge between Entroly's compression
algorithms and LLM API payloads. It is intentionally provider-light: callers can
use it from the existing proxy, SDKs, MCP tools, or benchmarks without starting
an HTTP server.

Design principle:

    Compress aggressively around evidence, never through evidence.

The proxy surface preserves normal user/assistant text by default and compresses
heavy tool/function payloads using Evidence-Locked Compression receipts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .evidence_locked_compression import compress_evidence_locked, estimate_tokens


@dataclass(slots=True)
class ProxyCompressionReceipt:
    provider: str
    mode: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    savings_ratio: float
    compressed_blocks: int
    receipts: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ProxyCompressionResult:
    body: dict[str, Any]
    receipt: ProxyCompressionReceipt
    changed: bool

    def headers(self) -> dict[str, str]:
        """Headers suitable for an HTTP proxy response/debug surface."""
        return {
            "x-entroly-compression-mode": self.receipt.mode,
            "x-entroly-original-tokens": str(self.receipt.original_tokens),
            "x-entroly-compressed-tokens": str(self.receipt.compressed_tokens),
            "x-entroly-tokens-saved": str(self.receipt.tokens_saved),
            "x-entroly-savings-ratio": f"{self.receipt.savings_ratio:.4f}",
            "x-entroly-compressed-blocks": str(self.receipt.compressed_blocks),
        }


def compress_proxy_payload(
    body: dict[str, Any],
    *,
    provider: str = "openai",
    query: str = "",
    budget_tokens: int = 1200,
    mode: str = "elc",
    include_receipt_header: bool = True,
    compress_user_messages: bool = False,
) -> ProxyCompressionResult:
    """Compress heavy payload blocks before they reach an LLM provider.

    Supported shapes:
    - OpenAI chat/completions: ``messages`` with string content or text blocks.
    - Anthropic messages: ``messages`` with list content and ``tool_result`` blocks.
    - OpenAI Responses-style ``input`` arrays with message/content blocks.

    By default, only tool/function/tool_result payloads are compressed. User
    messages are preserved unless ``compress_user_messages=True`` because the
    user's latest request is usually the semantic target, not background noise.
    """
    if mode not in {"elc", "off"}:
        raise ValueError("mode must be 'elc' or 'off'")
    if mode == "off":
        tokens = _estimate_body_tokens(body)
        receipt = ProxyCompressionReceipt(provider, mode, tokens, tokens, 0, 0.0, 0, [])
        return ProxyCompressionResult(dict(body), receipt, changed=False)

    original_tokens = _estimate_body_tokens(body)
    new_body = dict(body)
    receipts: list[dict[str, object]] = []
    compressed_blocks = 0

    if isinstance(body.get("messages"), list):
        messages, changed_count, block_receipts = _compress_messages(
            body["messages"],
            query=query,
            budget_tokens=budget_tokens,
            include_receipt_header=include_receipt_header,
            compress_user_messages=compress_user_messages,
        )
        if changed_count:
            new_body["messages"] = messages
            compressed_blocks += changed_count
            receipts.extend(block_receipts)

    if "input" in body and isinstance(body.get("input"), list):
        new_input, changed_count, block_receipts = _compress_input_items(
            body["input"],
            query=query,
            budget_tokens=budget_tokens,
            include_receipt_header=include_receipt_header,
            compress_user_messages=compress_user_messages,
        )
        if changed_count:
            new_body["input"] = new_input
            compressed_blocks += changed_count
            receipts.extend(block_receipts)

    compressed_tokens = _estimate_body_tokens(new_body)
    saved = max(0, original_tokens - compressed_tokens)
    receipt = ProxyCompressionReceipt(
        provider=provider,
        mode="elc",
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        tokens_saved=saved,
        savings_ratio=0.0 if original_tokens == 0 else saved / original_tokens,
        compressed_blocks=compressed_blocks,
        receipts=receipts,
    )
    return ProxyCompressionResult(new_body, receipt, changed=compressed_blocks > 0)


def _compress_messages(
    messages: list[Any],
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
    compress_user_messages: bool,
) -> tuple[list[Any], int, list[dict[str, object]]]:
    out: list[Any] = []
    changed_count = 0
    receipts: list[dict[str, object]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        role = str(msg.get("role", ""))
        content = msg.get("content")
        should_compress_role = role in {"tool", "function"} or (
            compress_user_messages and role == "user"
        )

        if isinstance(content, str) and should_compress_role:
            new_text, changed, receipt = _compress_text_block(
                content,
                query=query,
                budget_tokens=budget_tokens,
                include_receipt_header=include_receipt_header,
            )
            if changed:
                new_msg = dict(msg)
                new_msg["content"] = new_text
                out.append(new_msg)
                changed_count += 1
                receipts.append(receipt)
                continue

        if isinstance(content, list):
            new_content, changed, block_receipts = _compress_content_blocks(
                content,
                query=query,
                budget_tokens=budget_tokens,
                include_receipt_header=include_receipt_header,
                compress_user_text=compress_user_messages and role == "user",
            )
            if changed:
                new_msg = dict(msg)
                new_msg["content"] = new_content
                out.append(new_msg)
                changed_count += len(block_receipts)
                receipts.extend(block_receipts)
                continue

        out.append(msg)
    return out, changed_count, receipts


def _compress_input_items(
    items: list[Any],
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
    compress_user_messages: bool,
) -> tuple[list[Any], int, list[dict[str, object]]]:
    out: list[Any] = []
    changed_count = 0
    receipts: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        role = str(item.get("role", ""))
        if isinstance(item.get("content"), list):
            new_content, changed, block_receipts = _compress_content_blocks(
                item["content"],
                query=query,
                budget_tokens=budget_tokens,
                include_receipt_header=include_receipt_header,
                compress_user_text=compress_user_messages and role == "user",
            )
            if changed:
                new_item = dict(item)
                new_item["content"] = new_content
                out.append(new_item)
                changed_count += len(block_receipts)
                receipts.extend(block_receipts)
                continue
        out.append(item)
    return out, changed_count, receipts


def _compress_content_blocks(
    blocks: list[Any],
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
    compress_user_text: bool,
) -> tuple[list[Any], bool, list[dict[str, object]]]:
    out: list[Any] = []
    changed = False
    receipts: list[dict[str, object]] = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(block)
            continue
        block_type = str(block.get("type", ""))
        is_tool_result = block_type == "tool_result"
        is_text = block_type in {"text", "input_text"}
        if is_tool_result:
            value = block.get("content", "")
            if isinstance(value, str):
                new_text, did_change, receipt = _compress_text_block(
                    value,
                    query=query,
                    budget_tokens=budget_tokens,
                    include_receipt_header=include_receipt_header,
                )
                if did_change:
                    new_block = dict(block)
                    new_block["content"] = new_text
                    out.append(new_block)
                    changed = True
                    receipts.append(receipt)
                    continue
        elif is_text and compress_user_text and isinstance(block.get("text"), str):
            new_text, did_change, receipt = _compress_text_block(
                block["text"],
                query=query,
                budget_tokens=budget_tokens,
                include_receipt_header=include_receipt_header,
            )
            if did_change:
                new_block = dict(block)
                new_block["text"] = new_text
                out.append(new_block)
                changed = True
                receipts.append(receipt)
                continue
        out.append(block)
    return out, changed, receipts


def _compress_text_block(
    text: str,
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
) -> tuple[str, bool, dict[str, object]]:
    result = compress_evidence_locked(text, query=query, budget_tokens=budget_tokens)
    if not result.changed:
        return text, False, result.receipt.as_dict()
    rendered = result.with_receipt_header() if include_receipt_header else result.compressed
    return rendered, True, result.receipt.as_dict()


def _estimate_body_tokens(body: dict[str, Any]) -> int:
    return estimate_tokens(_stable_body_text(body))


def _stable_body_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


__all__ = [
    "ProxyCompressionReceipt",
    "ProxyCompressionResult",
    "compress_proxy_payload",
]
